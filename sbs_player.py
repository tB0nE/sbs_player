import sys
import os
os.environ["OPENCV_VIDEO_DEBUG"] = "0"
os.environ["QT_QPA_PLATFORM"] = "xcb"
import time
import queue
import threading
import argparse
import subprocess
import numpy as np
import cv2
import torch

V2_MODELS = [
    "depth-anything/Depth-Anything-V2-Small-hf",
    "depth-anything/Depth-Anything-V2-Base-hf",
    "depth-anything/Depth-Anything-V2-Large-hf",
]

DA3_MODELS = [
    "depth-anything/DA3-SMALL",
    "depth-anything/DA3-BASE",
    "depth-anything/DA3-LARGE-1.1",
    "depth-anything/DA3-GIANT-1.1",
    "depth-anything/DA3MONO-LARGE",
    "depth-anything/DA3METRIC-LARGE",
]

ALL_MODELS = V2_MODELS + DA3_MODELS


class SBSVideoPlayer:
    def __init__(self, video_path, model_name="depth-anything/Depth-Anything-V2-Base-hf", max_shift=20, buffer_size=15, inference_size=518, precision="fp16"):
        self.video_path = video_path
        self.model_name = model_name
        self.max_shift = max_shift
        self.buffer_size = buffer_size
        self.inference_size = inference_size
        self.precision = precision
        self.is_da3 = model_name in DA3_MODELS

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Info] Using device: {self.device}")

        print(f"[Info] Loading model: {self.model_name} (precision: {self.precision})...")

        if self.is_da3:
            self._load_da3_model()
        else:
            self._load_v2_model()

        print("[Info] Model loaded successfully.")

        self.frame_queue = queue.Queue(maxsize=self.buffer_size)
        self.sbs_queue = queue.Queue(maxsize=self.buffer_size)

        self.running = True
        self.play = True
        self.fullscreen = False
        self.fps_history = []

    def _load_v2_model(self):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        if self.precision == "int8":
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name, load_in_8bit=True)
        else:
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
            self.model.to(self.device)
            if self.precision == "fp16":
                self.model.half()
        self.model.eval()

    def _load_da3_model(self):
        from depth_anything_3.api import DepthAnything3
        self.da3_model = DepthAnything3.from_pretrained(self.model_name)
        self.da3_model = self.da3_model.to(device=self.device)
        self.da3_model.eval()

    def video_reader_thread(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[Error] Could not open video file: {self.video_path}")
            self.running = False
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[Info] Input Video FPS: {fps}")

        while self.running:
            if not self.play:
                time.sleep(0.05)
                continue

            if self.frame_queue.full():
                time.sleep(0.01)
                continue

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            self.frame_queue.put(frame)

        cap.release()

    def depth_and_warp_thread(self):
        while self.running:
            if self.frame_queue.empty():
                time.sleep(0.01)
                continue

            frame = self.frame_queue.get()
            h, w = frame.shape[:2]

            if self.is_da3:
                normalized_depth = self._infer_da3(frame, h, w)
            else:
                normalized_depth = self._infer_v2(frame, h, w)

            right_eye = self.warp_right_eye(frame, normalized_depth)

            left_half = cv2.resize(frame, (w // 2, h))
            right_half = cv2.resize(right_eye, (w // 2, h))
            sbs_frame = np.hstack((left_half, right_half))

            self.sbs_queue.put(sbs_frame)

    def _infer_v2(self, frame, h, w):
        scale = self.inference_size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        nh = (nh // 14) * 14
        nw = (nw // 14) * 14
        if nh == 0: nh = 14
        if nw == 0: nw = 14

        small_frame = cv2.resize(frame, (nw, nh))
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        inputs = self.image_processor(images=rgb_frame, return_tensors="pt").to(self.device)
        if self.precision == "fp16":
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        depth_small = predicted_depth.squeeze().cpu().float().numpy()
        prediction = cv2.resize(depth_small, (w, h), interpolation=cv2.INTER_CUBIC)

        depth_min, depth_max = prediction.min(), prediction.max()
        if depth_max - depth_min > 0:
            return (prediction - depth_min) / (depth_max - depth_min)
        return np.zeros_like(prediction)

    def _infer_da3(self, frame, h, w):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        use_autocast = self.precision == "fp16" and torch.cuda.is_available()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=use_autocast):
            prediction = self.da3_model.inference(
                [rgb_frame],
                process_res=self.inference_size,
                process_res_method="upper_bound_resize",
                export_format="mini_npz",
            )

        depth_map = prediction.depth[0]

        if depth_map.shape[0] != h or depth_map.shape[1] != w:
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_CUBIC)

        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max - depth_min > 0:
            return (depth_map - depth_min) / (depth_max - depth_min)
        return np.zeros_like(depth_map)

    def warp_right_eye(self, frame, depth):
        h, w, c = frame.shape
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        shift = depth * self.max_shift
        map_x_warped = map_x - shift

        warped = cv2.remap(frame, map_x_warped, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        mask = (map_x_warped < 0).astype(np.uint8) * 255

        if np.any(mask):
            warped = cv2.inpaint(warped, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        return warped

    def _get_display_size(self):
        try:
            result = subprocess.run(
                ["xrandr", "--current"],
                capture_output=True, text=True, timeout=2
            )
            for line in result.stdout.splitlines():
                if "current" in line:
                    idx = line.index("current") + len("current")
                    rest = line[idx:].strip()
                    dims = rest.split(",")[0].strip()
                    w, h = dims.split("x")
                    return (int(w.strip()), int(h.strip()))
        except Exception:
            pass
        return (1920, 1080)

    def _toggle_fullscreen(self, window_name):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 540)

    def _on_mouse(self, event, x, y, flags, param):
        pass

    def run(self):
        reader = threading.Thread(target=self.video_reader_thread, daemon=True)
        processor = threading.Thread(target=self.depth_and_warp_thread, daemon=True)

        reader.start()
        processor.start()

        print("[Info] Buffering frames...")
        time.sleep(2.0)

        window_name = "2D to 3D SBS Player"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.setMouseCallback(window_name, self._on_mouse, param=window_name)

        self.display_w, self.display_h = self._get_display_size()
        print(f"[Info] Display resolution: {self.display_w}x{self.display_h}")

        print("[Info] Control keys:")
        print("  'q' / ESC : Quit")
        print("  'space'   : Pause/Resume")
        print("  '+' / '=' : Increase 3D Depth strength")
        print("  '-'       : Decrease 3D Depth strength")
        print("  'f'       : Toggle fullscreen")

        last_frame_time = time.time()

        while self.running:
            if not self.sbs_queue.empty():
                sbs_frame = self.sbs_queue.get()

                now = time.time()
                self.fps_history.append(now)
                self.fps_history = [t for t in self.fps_history if now - t < 1.0]
                fps = len(self.fps_history)

                gpu_util = torch.cuda.utilization() if torch.cuda.is_available() else 0
                vram_used = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
                vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0

                cv2.putText(sbs_frame, f"FPS: {fps}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(sbs_frame, f"Depth: {self.max_shift}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(sbs_frame, f"GPU: {gpu_util}%", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(sbs_frame, f"VRAM: {vram_used:.1f}/{vram_reserved:.1f} GB", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                display_frame = cv2.resize(sbs_frame, (self.display_w, self.display_h), interpolation=cv2.INTER_LINEAR)

                cv2.imshow(window_name, display_frame)

                elapsed = time.time() - last_frame_time
                delay = max(1, int((1.0 / 30.0 - elapsed) * 1000))
                last_frame_time = time.time()

                key = cv2.waitKey(delay) & 0xFF
                if key == ord('q') or key == 27:
                    self.running = False
                elif key == ord(' '):
                    self.play = not self.play
                    print(f"[Info] {'Paused' if not self.play else 'Resumed'}")
                elif key == ord('+') or key == ord('='):
                    self.max_shift = min(100, self.max_shift + 2)
                    print(f"[Info] Depth strength increased to: {self.max_shift}")
                elif key == ord('-'):
                    self.max_shift = max(0, self.max_shift - 2)
                    print(f"[Info] Depth strength decreased to: {self.max_shift}")
                elif key == ord('f'):
                    self._toggle_fullscreen(window_name)
            else:
                time.sleep(0.01)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D to 3D SBS Video Player using Depth-Anything")
    parser.add_argument("video", type=str, help="Path to input 2D video file")
    parser.add_argument("--model", type=str, default="depth-anything/Depth-Anything-V2-Base-hf",
                        choices=ALL_MODELS,
                        help="Depth model to use (V2 or DA3)")
    parser.add_argument("--strength", type=int, default=20, help="Stereo shift strength (max pixels)")
    parser.add_argument("--inference-size", type=int, default=518, help="Longest side resolution for depth model input")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"],
                        help="Model precision: fp16 (default), fp32, int8 (V2 only, via bitsandbytes). DA3 uses autocast for fp16.")

    args = parser.parse_args()

    player = SBSVideoPlayer(
        video_path=args.video,
        model_name=args.model,
        max_shift=args.strength,
        inference_size=args.inference_size,
        precision=args.precision
    )
    player.run()
