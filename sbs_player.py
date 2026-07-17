import sys
import os
os.environ["OPENCV_VIDEO_DEBUG"] = "0"
os.environ["QT_QPA_PLATFORM"] = "xcb"
import time
import queue
import threading
import argparse
import subprocess
import ctypes
import numpy as np
import cv2
import torch
import av
import sounddevice as sd
import pynvml
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QSlider, QLabel, QPushButton, QComboBox, QCheckBox, QFileDialog, QSplitter, QListWidget)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap, QKeyEvent

# Dynamic library loader to resolve CUDA and TensorRT shared libraries
def load_cuda_libs():
    site_packages = [p for p in sys.path if 'site-packages' in p and os.path.exists(p)]
    loaded = []
    libs_to_load = [
        ('nvidia/cuda_runtime/lib', 'libcudart.so.12'),
        ('nvidia/nvjitlink/lib', 'libnvjitlink.so.12'),
        ('nvidia/cublas/lib', 'libcublasLt.so.12'),
        ('nvidia/cublas/lib', 'libcublas.so.12'),
        ('nvidia/cufft/lib', 'libcufft.so.11'),
        ('nvidia/curand/lib', 'libcurand.so.10'),
        ('nvidia/cudnn/lib', 'libcudnn.so.9'),
        ('tensorrt_libs', 'libnvinfer.so.10'),
        ('tensorrt_libs', 'libnvonnxparser.so.10'),
        ('tensorrt_libs', 'libnvinfer_plugin.so.10'),
    ]
    for sub, name in libs_to_load:
        found = False
        for sp in site_packages:
            path = os.path.join(sp, sub, name)
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(name)
                    found = True
                    break
                except Exception:
                    pass
            else:
                dir_path = os.path.join(sp, sub)
                if os.path.exists(dir_path):
                    for f in os.listdir(dir_path):
                        if f.startswith(name):
                            try:
                                ctypes.CDLL(os.path.join(dir_path, f), mode=ctypes.RTLD_GLOBAL)
                                loaded.append(f)
                                found = True
                                break
                            except Exception:
                                pass
                    if found:
                        break
        if not found:
            print(f"[Warning] Could not find library: {os.path.join(sub, name)}")
    return loaded

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
    def __init__(self, video_path, model_name="depth-anything/Depth-Anything-V2-Large-hf", max_shift=16, buffer_size=15, inference_size=518, precision="fp16", use_trt=True, benchmark=False, alpha=0.3, convergence=-10.0, edge_softness=20.0, depth_gamma=0.2, sharpen=14.0, artifact_smoothing=1.0):
        self.video_path = video_path
        self.model_name = model_name
        self.max_shift = max_shift
        self.buffer_size = buffer_size
        self.inference_size = inference_size
        self.precision = precision
        self.use_trt = use_trt and (model_name in V2_MODELS)
        self.is_da3 = model_name in DA3_MODELS
        self.video_fps = 30.0
        self.benchmark = benchmark
        self.alpha = alpha
        self.convergence = convergence
        self.edge_softness = edge_softness
        self.depth_gamma = depth_gamma
        self.sharpen = sharpen
        self.artifact_smoothing = artifact_smoothing
        self.prev_depth_gpu = None
        self.last_inference_ms = 0.0
        self.last_preprocess_ms = 0.0
        self.last_model_ms = 0.0
        self.last_postprocess_ms = 0.0
        self.last_warp_ms = 0.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Info] Using device: {self.device}")

        # Resolve paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.checkpoints_dir = os.path.join(self.script_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        if self.use_trt:
            print("[Info] Resolving CUDA and TensorRT library paths...")
            loaded_libs = load_cuda_libs()
            print(f"[Info] Loaded {len(loaded_libs)} CUDA/TensorRT libraries successfully.")
            
            self._load_v2_trt_model()
        elif self.is_da3:
            self._load_da3_model()
        else:
            self._load_v2_pytorch_model()

        self.frame_queue = queue.Queue(maxsize=60)
        self.depth_queue = queue.Queue(maxsize=60)
        self.sbs_queue = queue.Queue(maxsize=90)
        self.audio_queue = queue.Queue(maxsize=300)

        self.running = True
        self.play = True
        self.fullscreen = False
        self.fps_history = []
        self._reset_temporal_depth = False
        self._reset_temporal_warp = False
        self.loop_video = False
        self.video_ended = False

        # Audio state
        self.current_audio_data = np.zeros((0, 2), dtype=np.float32)
        self.audio_samples_played = 0
        self.audio_sample_rate = 44100
        self.audio_latency_frames = 0
        self.seek_audio_target = None
        self.seek_video_target = None
        self.has_audio = False
        self.volume = 1.0
        self.sys_clock_start = time.time()
        self.sys_clock_pause_time = time.time()

        # Warp cache
        self._map_x = None
        self._map_y = None
        self._cached_w = 0
        self._cached_h = 0

        # Filter flags
        self.use_smoothing = True
        self.use_edge_softness = True
        self.use_artifact_smoothing = True
        self.use_sharpen = True
        self.hq_artifact_smoothing = False
        self.use_frame_doubler = False

        self.pipeline_latency = 0.5
        self.latency_alpha = 0.05
        self.seek_epoch = 0
        self._seek_accept_behind = False

        self.last_sbs_frame = None
        self.last_timestamp_ms = None
        self.last_padded_tensor = None
        self.rife_context = None
        self.rife_engine = None

        # Load video metadata (try av first for HEVC support, fall back to cv2)
        if self.video_path:
            try:
                container = av.open(self.video_path)
                vid = container.streams.video[0]
                self.total_frames = vid.frames if vid.frames else 0
                self.video_fps = float(vid.average_rate) if vid.average_rate else 0.0
                if self.video_fps <= 0:
                    self.video_fps = float(vid.guessed_rate) if vid.guessed_rate else 24.0
                if vid.duration:
                    self.duration_sec = float(vid.duration * vid.time_base)
                elif container.duration:
                    self.duration_sec = float(container.duration / av.time_base)
                elif self.total_frames > 0 and self.video_fps > 0:
                    self.duration_sec = self.total_frames / self.video_fps
                else:
                    self.duration_sec = 0.0
                if self.total_frames == 0 and self.duration_sec > 0:
                    self.total_frames = int(self.duration_sec * self.video_fps)
                self.width = vid.width
                self.height = vid.height
                container.close()
                print(f"[Info] Video metadata (av): {self.width}x{self.height} @ {self.video_fps:.2f}fps, {self.total_frames} frames")
            except Exception:
                cap = cv2.VideoCapture(self.video_path)
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_fps = cap.get(cv2.CAP_PROP_FPS)
                if self.video_fps <= 0:
                    self.video_fps = 24.0
                self.duration_sec = self.total_frames / self.video_fps
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
        else:
            self.total_frames = 0
            self.video_fps = 30.0
            self.duration_sec = 0.0
            self.width = 1920
            self.height = 1080

        # Load config
        self.load_config()

        if self.use_trt:
            self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=torch.float16).view(1, 3, 1, 1)
            self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=torch.float16).view(1, 3, 1, 1)

    @property
    def reset_temporal(self):
        return self._reset_temporal_depth or self._reset_temporal_warp

    @reset_temporal.setter
    def reset_temporal(self, val):
        if val:
            self._reset_temporal_depth = True
            self._reset_temporal_warp = True
        else:
            self._reset_temporal_depth = False
            self._reset_temporal_warp = False

    def load_config(self):
        config_path = os.path.expanduser("~/.config/sbs_player/config.json")
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.max_shift = config.get("max_shift", self.max_shift)
                self.convergence = config.get("convergence", self.convergence)
                self.edge_softness = config.get("edge_softness", self.edge_softness)
                self.depth_gamma = config.get("depth_gamma", self.depth_gamma)
                self.sharpen = config.get("sharpen", self.sharpen)
                self.artifact_smoothing = config.get("artifact_smoothing", self.artifact_smoothing)
                self.alpha = config.get("alpha", self.alpha)
                self.volume = config.get("volume", self.volume)
                self.model_name = config.get("model_name", self.model_name)
                self.use_smoothing = config.get("use_smoothing", self.use_smoothing)
                self.use_edge_softness = config.get("use_edge_softness", self.use_edge_softness)
                self.use_artifact_smoothing = config.get("use_artifact_smoothing", self.use_artifact_smoothing)
                self.use_sharpen = config.get("use_sharpen", self.use_sharpen)
                self.hq_artifact_smoothing = config.get("hq_artifact_smoothing", self.hq_artifact_smoothing)
                self.use_frame_doubler = config.get("use_frame_doubler", self.use_frame_doubler)
                # Ensure model matches our config
                self.use_trt = self.model_name in V2_MODELS
                self.is_da3 = self.model_name in DA3_MODELS
                print("[Info] Loaded configuration from:", config_path)
            except Exception as e:
                print("[Warning] Failed to load config:", e)

    def save_config(self):
        config_path = os.path.expanduser("~/.config/sbs_player/config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        try:
            import json
            config = {
                "max_shift": self.max_shift,
                "convergence": self.convergence,
                "edge_softness": self.edge_softness,
                "depth_gamma": self.depth_gamma,
                "sharpen": self.sharpen,
                "artifact_smoothing": self.artifact_smoothing,
                "alpha": self.alpha,
                "volume": self.volume,
                "model_name": self.model_name,
                "use_smoothing": self.use_smoothing,
                "use_edge_softness": self.use_edge_softness,
                "use_artifact_smoothing": self.use_artifact_smoothing,
                "use_sharpen": self.use_sharpen,
                "hq_artifact_smoothing": self.hq_artifact_smoothing,
                "use_frame_doubler": self.use_frame_doubler
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            print("[Info] Saved configuration to:", config_path)
        except Exception as e:
            print("[Warning] Failed to save config:", e)

    def _load_v2_pytorch_model(self):
        print(f"[Info] Loading PyTorch model: {self.model_name} (precision: {self.precision})...")
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
        print("[Info] PyTorch Model loaded successfully.")

    def _load_v2_trt_model(self):
        model_short_name = self.model_name.split("/")[-1]
        onnx_filename = f"{model_short_name}_{self.inference_size}.onnx"
        self.onnx_path = os.path.join(self.checkpoints_dir, onnx_filename)

        if not os.path.exists(self.onnx_path):
            print(f"[Info] ONNX model not found. Exporting {self.model_name} to ONNX (this is done once)...")
            from transformers import AutoModelForDepthEstimation
            pytorch_model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
            pytorch_model.eval()
            dummy_input = torch.randn(1, 3, self.inference_size, self.inference_size)
            torch.onnx.export(
                pytorch_model,
                (dummy_input,),
                self.onnx_path,
                input_names=['pixel_values'],
                output_names=['predicted_depth'],
                opset_version=17,
            )
            print(f"[Info] Exported ONNX model saved to: {self.onnx_path}")
            del pytorch_model
            torch.cuda.empty_cache()

        import tensorrt as trt
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        engine_path = os.path.join(self.checkpoints_dir, f"{model_short_name}_{self.inference_size}_fp16.engine")
        if os.path.exists(engine_path):
            print(f"[Info] Loading cached TRT engine: {engine_path}")
            runtime = trt.Runtime(self.trt_logger)
            with open(engine_path, 'rb') as f:
                self.trt_engine = runtime.deserialize_cuda_engine(f.read())
            print("[Info] TRT engine loaded.")
        else:
            print(f"[Info] Building TRT engine from ONNX (first time, takes 1-5 minutes)...")
            builder = trt.Builder(self.trt_logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, self.trt_logger)

            with open(self.onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        print(f"[Error] TRT Parser: {parser.get_error(i)}")
                    raise RuntimeError("Failed to parse ONNX model")

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
            if self.precision == "fp16":
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    print("[Info] FP16 enabled for TRT engine.")
                else:
                    print("[Warning] Platform does not support fast FP16.")

            profile = builder.create_optimization_profile()
            profile.set_shape('pixel_values',
                (1, 3, self.inference_size, self.inference_size),
                (1, 3, self.inference_size, self.inference_size),
                (1, 3, self.inference_size, self.inference_size))
            config.add_optimization_profile(profile)

            print("[Info] Building engine (this may take a few minutes)...")
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TRT engine")

            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)
            print(f"[Info] TRT engine saved to: {engine_path}")

            runtime = trt.Runtime(self.trt_logger)
            self.trt_engine = runtime.deserialize_cuda_engine(serialized_engine)
            print("[Info] TRT engine built and loaded.")

        self.trt_context = self.trt_engine.create_execution_context()
        self.trt_stream = torch.cuda.Stream()

        self.trt_d_input = torch.empty(1, 3, self.inference_size, self.inference_size, dtype=torch.float16, device=self.device)
        self.trt_d_output = torch.empty(1, self.inference_size, self.inference_size, dtype=torch.float16, device=self.device)

        self.trt_context.set_tensor_address('pixel_values', int(self.trt_d_input.data_ptr()))
        self.trt_context.set_tensor_address('predicted_depth', int(self.trt_d_output.data_ptr()))

        print("[Info] TRT context initialized with GPU buffers.")

    def _load_rife_trt_model(self):
        # Calculate padded dimensions to multiple of 32
        self.rife_w = ((self.width + 31) // 32) * 32
        self.rife_h = ((self.height + 31) // 32) * 32
        
        onnx_filename = "rife49.onnx"
        self.rife_onnx_path = os.path.join(self.checkpoints_dir, onnx_filename)
        
        # 1. Download ONNX if needed
        if not os.path.exists(self.rife_onnx_path):
            print("[Info] RIFE ONNX model not found. Downloading...")
            import urllib.request
            url = 'https://huggingface.co/yuvraj108c/rife-onnx/resolve/main/rife49_ensemble_True_scale_1_sim.onnx'
            urllib.request.urlretrieve(url, self.rife_onnx_path)
            print("[Info] Download complete.")
            
        # 2. Check cached engine
        engine_filename = f"rife49_{self.rife_w}_{self.rife_h}_fp16.engine"
        engine_path = os.path.join(self.checkpoints_dir, engine_filename)
        
        import tensorrt as trt
        if not hasattr(self, 'trt_logger') or self.trt_logger is None:
            self.trt_logger = trt.Logger(trt.Logger.WARNING)
            
        if os.path.exists(engine_path):
            print(f"[Info] Loading cached RIFE TRT engine: {engine_path}")
            runtime = trt.Runtime(self.trt_logger)
            with open(engine_path, 'rb') as f:
                self.rife_engine = runtime.deserialize_cuda_engine(f.read())
            print("[Info] RIFE TRT engine loaded successfully.")
        else:
            print(f"[Info] Building RIFE TRT engine from ONNX for resolution {self.rife_w}x{self.rife_h} (takes ~30-60 seconds)...")
            builder = trt.Builder(self.trt_logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, self.trt_logger)
            
            with open(self.rife_onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        print(f"[Error] RIFE TRT Parser: {parser.get_error(i)}")
                    raise RuntimeError("Failed to parse RIFE ONNX model")
                    
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                
            profile = builder.create_optimization_profile()
            profile.set_shape('img0', (1, 3, self.rife_h, self.rife_w), (1, 3, self.rife_h, self.rife_w), (1, 3, self.rife_h, self.rife_w))
            profile.set_shape('img1', (1, 3, self.rife_h, self.rife_w), (1, 3, self.rife_h, self.rife_w), (1, 3, self.rife_h, self.rife_w))
            profile.set_shape('timestep', (1,), (1,), (1,))
            config.add_optimization_profile(profile)
            
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError("Failed to build RIFE TRT engine")
                
            with open(engine_path, 'wb') as f:
                f.write(serialized)
            print(f"[Info] RIFE TRT engine saved to: {engine_path}")
            
            runtime = trt.Runtime(self.trt_logger)
            self.rife_engine = runtime.deserialize_cuda_engine(serialized)
            
        # 3. Create context and allocate buffers
        self.rife_context = self.rife_engine.create_execution_context()
        self.rife_stream = torch.cuda.Stream()
        
        self.rife_d_img0 = torch.empty(1, 3, self.rife_h, self.rife_w, dtype=torch.float16, device=self.device)
        self.rife_d_img1 = torch.empty(1, 3, self.rife_h, self.rife_w, dtype=torch.float16, device=self.device)
        self.rife_d_timestep = torch.tensor([0.5], dtype=torch.float16, device=self.device)
        self.rife_d_output = torch.empty(1, 3, self.rife_h, self.rife_w, dtype=torch.float16, device=self.device)
        
        self.rife_context.set_tensor_address('img0', int(self.rife_d_img0.data_ptr()))
        self.rife_context.set_tensor_address('img1', int(self.rife_d_img1.data_ptr()))
        self.rife_context.set_tensor_address('timestep', int(self.rife_d_timestep.data_ptr()))
        self.rife_context.set_tensor_address('output', int(self.rife_d_output.data_ptr()))
        
        print("[Info] RIFE TRT context initialized with GPU buffers.")

    def _load_da3_model(self):
        print(f"[Info] Loading DA3 model: {self.model_name}...")
        from depth_anything_3.api import DepthAnything3
        self.da3_model = DepthAnything3.from_pretrained(self.model_name)
        self.da3_model = self.da3_model.to(device=self.device)
        self.da3_model.eval()
        print("[Info] DA3 Model loaded successfully.")

    def flush_queues(self):
        while not self.frame_queue.empty():
            try: self.frame_queue.get_nowait()
            except queue.Empty: break
        while not self.depth_queue.empty():
            try: self.depth_queue.get_nowait()
            except queue.Empty: break
        while not self.sbs_queue.empty():
            try: self.sbs_queue.get_nowait()
            except queue.Empty: break

    def video_reader_thread(self):
        try:
            self._video_reader_av()
            return
        except Exception as e:
            print(f"[Warning] PyAV reader failed ({e}), falling back to OpenCV reader.")

        self._video_reader_cv2()

    def _video_reader_av(self):
        container = av.open(self.video_path)
        video_stream = container.streams.video[0]
        avg_rate = float(video_stream.average_rate) if video_stream.average_rate else 0.0
        if avg_rate <= 0:
            avg_rate = float(video_stream.guessed_rate) if video_stream.guessed_rate else 30.0
        self.video_fps = avg_rate
        time_base_sec = float(video_stream.time_base)

        print(f"[Info] PyAV reader: {video_stream.codec_context.codec.name} ({video_stream.width}x{video_stream.height} @ {self.video_fps:.2f}fps)")

        frame_interval = 1.0 / self.video_fps if not self.benchmark else 0.0
        last_read_time = 0.0
        first_frame = True
        demuxer = container.demux(video_stream)

        while self.running:
            if self.seek_video_target is not None:
                self.seek_epoch += 1
                target_offset = int(self.seek_video_target * av.time_base)
                print(f"[Seek] AV reader seeking to {self.seek_video_target:.1f}s (offset={target_offset})")
                try:
                    container.seek(target_offset, backward=True)
                except Exception as e:
                    print(f"[Error] AV seek failed: {e}")
                self.flush_queues()
                self.seek_video_target = None
                first_frame = True
                demuxer = container.demux(video_stream)

            if not self.play:
                time.sleep(0.05)
                continue

            if self.frame_queue.full():
                time.sleep(0.01)
                continue

            now = time.monotonic()
            if not first_frame and frame_interval > 0:
                elapsed = now - last_read_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                    now = time.monotonic()

            try:
                packet = next(demuxer)
            except StopIteration:
                if self.loop_video:
                    container.seek(0, stream=video_stream, backward=True)
                    demuxer = container.demux(video_stream)
                    self.reset_temporal = True
                    self.seek_audio_target = 0.0
                    first_frame = True
                    continue
                else:
                    self.video_ended = True
                    self.play = False
                    continue
            except Exception:
                time.sleep(0.01)
                continue

            try:
                for frame in packet.decode():
                    frame_np = frame.to_ndarray(format='bgr24')
                    if frame.pts is not None:
                        timestamp_ms = float(frame.pts * video_stream.time_base * 1000)
                    elif frame.time is not None:
                        timestamp_ms = frame.time * 1000.0
                    else:
                        timestamp_ms = 0.0

                    entry_time = time.time()
                    self.frame_queue.put((frame_np, timestamp_ms, entry_time, self.seek_epoch))
                    last_read_time = now
                    if first_frame:
                        print(f"[Seek] AV reader produced first post-seek frame: ts={timestamp_ms:.0f}ms epoch={self.seek_epoch}")
                    first_frame = False
            except Exception:
                continue

        container.close()

    def _video_reader_cv2(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[Error] Could not open video file: {self.video_path}")
            self.running = False
            return

        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        if self.video_fps <= 0:
            self.video_fps = 30.0
        print(f"[Info] OpenCV reader (FPS: {self.video_fps})")

        frame_interval = 1.0 / self.video_fps if not self.benchmark else 0.0
        last_read_time = 0.0

        while self.running:
            if self.seek_video_target is not None:
                self.seek_epoch += 1
                cap.set(cv2.CAP_PROP_POS_MSEC, self.seek_video_target * 1000.0)
                self.flush_queues()
                self.reset_temporal = True
                self.seek_video_target = None

            if not self.play:
                time.sleep(0.05)
                continue

            if self.frame_queue.full():
                time.sleep(0.01)
                continue

            now = time.monotonic()
            if frame_interval > 0:
                elapsed = now - last_read_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                    now = time.monotonic()

            ret, frame = cap.read()
            last_read_time = now
            if not ret:
                if self.loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.reset_temporal = True
                    self.seek_audio_target = 0.0
                else:
                    self.video_ended = True
                    self.play = False
                continue

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            entry_time = time.time()
            self.frame_queue.put((frame, timestamp_ms, entry_time, self.seek_epoch))

        cap.release()

    def depth_inference_thread(self):
        if self.use_trt:
            self._depth_loop_trt()
        elif self.is_da3:
            self._depth_loop_da3()
        else:
            self._depth_loop_pytorch()

    def _depth_loop_trt(self):
        while self.running:
            if self.frame_queue.empty():
                time.sleep(0.001)
                continue

            if self._reset_temporal_depth:
                self.prev_depth_gpu = None
                self._reset_temporal_depth = False

            frame, timestamp_ms, entry_time, epoch = self.frame_queue.get()
            if epoch != self.seek_epoch:
                continue
            h, w = frame.shape[:2]

            t0 = time.perf_counter()

            # Upload and preprocess efficiently on GPU with minimal allocations
            frame_gpu = torch.from_numpy(frame).to(self.device, non_blocking=True)
            frame_gpu = frame_gpu.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float16)
            frame_gpu = frame_gpu.flip(1) # BGR -> RGB
            resized_gpu = torch.nn.functional.interpolate(frame_gpu, size=(self.inference_size, self.inference_size), mode='bilinear', align_corners=False)
            
            # In-place normalization directly into the pre-allocated input buffer
            self.trt_d_input.copy_(resized_gpu)
            self.trt_d_input.div_(255.0).sub_(self._mean).div_(self._std)

            self.last_preprocess_ms = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            with torch.cuda.stream(self.trt_stream):
                self.trt_context.execute_async_v3(stream_handle=self.trt_stream.cuda_stream)
            self.trt_stream.synchronize()
            self.last_model_ms = (time.perf_counter() - t1) * 1000.0

            t2 = time.perf_counter()
            depth = self.trt_d_output.clone()
            if self.prev_depth_gpu is None or self.prev_depth_gpu.shape != depth.shape or not isinstance(self.prev_depth_gpu, torch.Tensor) or self.prev_depth_gpu.dtype != depth.dtype:
                self.prev_depth_gpu = depth.clone()
            else:
                if self.use_smoothing:
                    depth.mul_(self.alpha).add_(self.prev_depth_gpu, alpha=1.0 - self.alpha)
                self.prev_depth_gpu.copy_(depth)

            depth_resized = torch.nn.functional.interpolate(depth.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            depth_min = depth_resized.min()
            depth_max = depth_resized.max()
            diff = depth_max - depth_min
            if diff > 0:
                normalized_depth = ((depth_resized - depth_min) / diff).pow(self.depth_gamma).float().cpu().numpy()
            else:
                normalized_depth = np.zeros((h, w), dtype=np.float32)
            self.last_postprocess_ms = (time.perf_counter() - t2) * 1000.0
            self.last_inference_ms = (time.perf_counter() - t0) * 1000.0

            self.depth_queue.put((frame, normalized_depth, timestamp_ms, entry_time, epoch))

    def _depth_loop_pytorch(self):
        while self.running:
            if self.frame_queue.empty():
                time.sleep(0.001)
                continue

            if self._reset_temporal_depth:
                self.prev_depth_gpu = None
                self._reset_temporal_depth = False

            frame, timestamp_ms, entry_time, epoch = self.frame_queue.get()
            if epoch != self.seek_epoch:
                continue
            h, w = frame.shape[:2]

            t0 = time.perf_counter()
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

            if self.prev_depth_gpu is None or self.prev_depth_gpu.shape != predicted_depth.shape or not isinstance(self.prev_depth_gpu, torch.Tensor):
                self.prev_depth_gpu = predicted_depth.clone()
            else:
                if self.use_smoothing:
                    predicted_depth = self.alpha * predicted_depth + (1.0 - self.alpha) * self.prev_depth_gpu
                self.prev_depth_gpu.copy_(predicted_depth)

            depth_small = predicted_depth.squeeze().cpu().float().numpy()
            prediction = cv2.resize(depth_small, (w, h), interpolation=cv2.INTER_CUBIC)

            depth_min, depth_max = prediction.min(), prediction.max()
            if depth_max - depth_min > 0:
                normalized_depth = np.power((prediction - depth_min) / (depth_max - depth_min), self.depth_gamma)
            else:
                normalized_depth = np.zeros_like(prediction)
            self.last_inference_ms = (time.perf_counter() - t0) * 1000.0

            self.depth_queue.put((frame, normalized_depth, timestamp_ms, entry_time, epoch))

    def _depth_loop_da3(self):
        while self.running:
            if self.frame_queue.empty():
                time.sleep(0.001)
                continue

            if self._reset_temporal_depth:
                self.prev_depth_gpu = None
                self._reset_temporal_depth = False

            frame, timestamp_ms, entry_time, epoch = self.frame_queue.get()
            if epoch != self.seek_epoch:
                continue
            h, w = frame.shape[:2]

            t0 = time.perf_counter()
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

            if self.prev_depth_gpu is None or self.prev_depth_gpu.shape != depth_map.shape or isinstance(self.prev_depth_gpu, torch.Tensor):
                self.prev_depth_gpu = depth_map.copy()
            else:
                if self.use_smoothing:
                    depth_map = self.alpha * depth_map + (1.0 - self.alpha) * self.prev_depth_gpu
                self.prev_depth_gpu = depth_map.copy()

            if depth_map.shape[0] != h or depth_map.shape[1] != w:
                depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_CUBIC)

            depth_min, depth_max = depth_map.min(), depth_map.max()
            if depth_max - depth_min > 0:
                normalized_depth = np.power((depth_map - depth_min) / (depth_max - depth_min), self.depth_gamma)
            else:
                normalized_depth = np.zeros_like(depth_map)
            self.last_inference_ms = (time.perf_counter() - t0) * 1000.0

            self.depth_queue.put((frame, normalized_depth, timestamp_ms, entry_time, epoch))

    def warp_thread(self):
        while self.running:
            if self.depth_queue.empty():
                time.sleep(0.001)
                continue

            if self._reset_temporal_warp:
                self.last_sbs_frame = None
                self.last_timestamp_ms = None
                self._reset_temporal_warp = False

            frame, normalized_depth, timestamp_ms, entry_time, epoch = self.depth_queue.get()
            if epoch != self.seek_epoch:
                continue
            h, w = frame.shape[:2]

            tw = time.perf_counter()
            right_eye = self.warp_right_eye(frame, normalized_depth)

            left_half = cv2.resize(frame, (w // 2, h))
            right_half = cv2.resize(right_eye, (w // 2, h))
            sbs_frame = np.hstack((left_half, right_half))
            self.last_warp_ms = (time.perf_counter() - tw) * 1000.0

            latency = time.time() - entry_time
            self.pipeline_latency = (1 - self.latency_alpha) * self.pipeline_latency + self.latency_alpha * latency

            if self.use_frame_doubler:
                if self.last_sbs_frame is not None and self.last_timestamp_ms is not None:
                    # Intermediate frame using RIFE
                    try:
                        sbs_h, sbs_w = sbs_frame.shape[:2]
                        pad_h = self.rife_h - sbs_h
                        pad_w = self.rife_w - sbs_w

                        with torch.cuda.stream(self.rife_stream):
                            t_img0 = torch.from_numpy(self.last_sbs_frame).to(self.device, dtype=torch.float16) / 255.0
                            t_img0 = t_img0.flip(-1).permute(2, 0, 1).unsqueeze(0).contiguous()
                            
                            t_img1 = torch.from_numpy(sbs_frame).to(self.device, dtype=torch.float16) / 255.0
                            t_img1 = t_img1.flip(-1).permute(2, 0, 1).unsqueeze(0).contiguous()
                            
                            if pad_h > 0 or pad_w > 0:
                                t_img0 = torch.nn.functional.pad(t_img0, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
                                t_img1 = torch.nn.functional.pad(t_img1, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
                                
                            self.rife_d_img0.copy_(t_img0)
                            self.rife_d_img1.copy_(t_img1)
                            self.rife_context.execute_async_v3(stream_handle=self.rife_stream.cuda_stream)
                        self.rife_stream.synchronize()
                        
                        inter_gpu = self.rife_d_output
                        if pad_h > 0 or pad_w > 0:
                            inter_gpu = inter_gpu[:, :, :sbs_h, :sbs_w]
                        inter_gpu = (inter_gpu.squeeze(0).permute(1, 2, 0).flip(-1) * 255.0).clamp(0, 255).to(torch.uint8).contiguous()
                        inter_frame = inter_gpu.cpu().numpy()
                    except Exception as e:
                        print(f"[Warning] RIFE inference failed: {e}. Falling back to crossfade.")
                        inter_frame = cv2.addWeighted(self.last_sbs_frame, 0.5, sbs_frame, 0.5, 0)
                        
                    inter_timestamp = (self.last_timestamp_ms + timestamp_ms) / 2.0
                    self.sbs_queue.put((inter_frame, inter_timestamp, epoch))
                
                self.sbs_queue.put((sbs_frame, timestamp_ms, epoch))
                self.last_sbs_frame = sbs_frame.copy()
                self.last_timestamp_ms = timestamp_ms
            else:
                self.last_sbs_frame = None
                self.last_timestamp_ms = None
                self.sbs_queue.put((sbs_frame, timestamp_ms, epoch))

    def warp_right_eye(self, frame, depth):
        h, w, c = frame.shape
        
        # 1. Apply Edge Softness to depth map
        if self.use_edge_softness and self.edge_softness > 0:
            ksize = int(self.edge_softness)
            if ksize % 2 == 0:
                ksize += 1
            depth = cv2.GaussianBlur(depth, (ksize, ksize), 0)

        # 2. Build or reuse cached warp maps
        if self._cached_w != w or self._cached_h != h or self._map_x is None:
            self._cached_w = w
            self._cached_h = h
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            self._map_x = map_x.astype(np.float32)
            self._map_y = map_y.astype(np.float32)

        # 3. Apply shift with convergence
        shift = depth * self.max_shift + self.convergence
        map_x_warped = self._map_x - shift

        # 4. Remap with BORDER_REPLICATE (removes need for expensive inpaint!)
        warped = cv2.remap(frame, map_x_warped, self._map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # 5. Apply Artifact Smoothing on disocclusion edges
        if self.use_artifact_smoothing and self.artifact_smoothing > 0:
            if self.hq_artifact_smoothing:
                # Full-resolution edge detection
                grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
                grad = cv2.magnitude(grad_x, grad_y)
                _, edge_mask = cv2.threshold(grad, 0.05, 255, cv2.THRESH_BINARY)
                edge_mask = edge_mask.astype(np.uint8)
            else:
                # Downsample depth map to 256x256 to make Sobel/magnitude extremely fast
                small_depth = cv2.resize(depth, (256, 256), interpolation=cv2.INTER_NEAREST)
                grad_x = cv2.Sobel(small_depth, cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(small_depth, cv2.CV_32F, 0, 1, ksize=3)
                grad = cv2.magnitude(grad_x, grad_y)
                _, edge_mask_small = cv2.threshold(grad, 0.05, 255, cv2.THRESH_BINARY)
                
                # Upscale mask back to full resolution
                edge_mask = cv2.resize(edge_mask_small, (w, h), interpolation=cv2.INTER_LINEAR)
                edge_mask = edge_mask.astype(np.uint8)
            
            # Dilate the mask to cover boundaries
            k_size = int(self.artifact_smoothing * 4) | 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            edge_mask = cv2.dilate(edge_mask, kernel)
            
            # Use fast box filter instead of GaussianBlur for image blurring
            blurred_warped = cv2.boxFilter(warped, -1, (k_size, k_size))
            
            # Vectorized binary copy (no slow float multiplication)
            np.copyto(warped, blurred_warped, where=(edge_mask > 127)[:, :, None])

        # 6. Apply post-warp sharpening
        if self.use_sharpen and self.sharpen > 0:
            blurred = cv2.GaussianBlur(warped, (0, 0), sigmaX=1.0)
            amount = self.sharpen / 10.0
            warped = cv2.addWeighted(warped, 1.0 + amount, blurred, -amount, 0)

        return warped

    def _audio_callback(self, outdata, frames, time_info, status):
        filled = 0
        if not self.play:
            outdata.fill(0)
            return

        while filled < frames:
            if len(self.current_audio_data) == 0:
                try:
                    self.current_audio_data = self.audio_queue.get_nowait()
                except queue.Empty:
                    outdata[filled:] = 0
                    break
            
            to_copy = min(frames - filled, len(self.current_audio_data))
            outdata[filled:filled+to_copy] = self.current_audio_data[:to_copy]
            self.current_audio_data = self.current_audio_data[to_copy:]
            filled += to_copy
            
        if self.volume != 1.0:
            outdata *= self.volume
            
        self.audio_samples_played += filled

    def get_audio_time(self):
        if not self.has_audio:
            if self.play:
                return time.time() - self.sys_clock_start
            else:
                return self.sys_clock_pause_time - self.sys_clock_start
        
        return max(0.0, (self.audio_samples_played - self.audio_latency_frames) / self.audio_sample_rate)

    def audio_thread_func(self):
        try:
            container = av.open(self.video_path)
            if not container.streams.audio:
                self.has_audio = False
                print("[Info] No audio stream found in video.")
                return
            
            self.has_audio = True
            audio_stream = container.streams.audio[0]
            self.audio_sample_rate = audio_stream.rate
            
            resampler = av.AudioResampler(
                format='fltp',
                layout='stereo',
                rate=self.audio_sample_rate
            )
            
            self.sd_stream = sd.OutputStream(
                samplerate=self.audio_sample_rate,
                channels=2,
                callback=self._audio_callback,
                dtype='float32'
            )
            self.sd_stream.start()
            self.audio_latency_frames = self.sd_stream.latency * self.audio_sample_rate
            
            while self.running:
                if self.seek_audio_target is not None:
                    target_pts = int(self.seek_audio_target * av.time_base)
                    container.seek(target_pts)
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.current_audio_data = np.zeros((0, 2), dtype=np.float32)
                    self.audio_samples_played = int(self.seek_audio_target * self.audio_sample_rate)
                    self.seek_audio_target = None
                
                if self.audio_queue.full():
                    time.sleep(0.01)
                    continue
                
                try:
                    packet = next(container.demux(audio_stream))
                except StopIteration:
                    time.sleep(0.05)
                    continue
                except Exception:
                    time.sleep(0.01)
                    continue
                
                for frame in packet.decode():
                    resampled_frames = resampler.resample(frame)
                    for r in resampled_frames:
                        data = r.to_ndarray()
                        interleaved = np.ascontiguousarray(data.T)
                        self.audio_queue.put(interleaved)
                        
            self.sd_stream.stop()
            self.sd_stream.close()
            
        except Exception as e:
            print(f"[Warning] Audio playback error: {e}")
            self.has_audio = False

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

    def start_threads(self):
        if self.use_frame_doubler and (not hasattr(self, 'rife_context') or self.rife_context is None):
            self._load_rife_trt_model()

        reader = threading.Thread(target=self.video_reader_thread, daemon=True)
        processor = threading.Thread(target=self.depth_inference_thread, daemon=True)
        warper = threading.Thread(target=self.warp_thread, daemon=True)
        audio = threading.Thread(target=self.audio_thread_func, daemon=True)

        reader.start()
        processor.start()
        warper.start()

        print("[Info] Priming pipeline...")
        t0 = time.time()
        while self.sbs_queue.empty() and self.running:
            time.sleep(0.05)
            if time.time() - t0 > 10.0:
                print("[Warning] Pipeline priming timed out, starting audio anyway.")
                break

        audio.start()
        print(f"[Info] Audio started (pipeline latency: {self.pipeline_latency:.3f}s)")

    def run_display_loop(self):
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
        print("  '['       : Decrease Temporal Alpha (more smoothing)")
        print("  ']'       : Increase Temporal Alpha (less smoothing)")
        print("  'c' / 'v' : Decrease/Increase Convergence")
        print("  'e' / 'r' : Decrease/Increase Edge Softness")
        print("  'g' / 'h' : Decrease/Increase Depth Gamma")
        print("  't' / 'y' : Decrease/Increase Sharpening")
        print("  'u' / 'i' : Decrease/Increase Artifact Smoothing")
        print("  'f'       : Toggle fullscreen")

        self.sys_clock_start = time.time()
        self.sys_clock_pause_time = time.time()

        while self.running:
            if not self.sbs_queue.empty():
                sbs_frame, timestamp_ms = self.sbs_queue.get()[0:2]

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

                backend = "TRT" if self.use_trt else ("DA3" if self.is_da3 else "PT")
                cv2.putText(sbs_frame, f"Infer: {self.last_inference_ms:.1f}ms [{backend}]", (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(sbs_frame, f"Alpha: {self.alpha:.2f}", (20, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(sbs_frame, f"Conv: {self.convergence:.1f} | Soft: {self.edge_softness:.1f}", (20, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(sbs_frame, f"Gamma: {self.depth_gamma:.2f} | Sharp: {self.sharpen:.1f} | Smooth: {self.artifact_smoothing:.1f}", (20, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if self.use_trt:
                    cv2.putText(sbs_frame, f"  pre:{self.last_preprocess_ms:.1f} model:{self.last_model_ms:.1f} post:{self.last_postprocess_ms:.1f} warp:{self.last_warp_ms:.1f}", (20, 280),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

                display_frame = cv2.resize(sbs_frame, (self.display_w, self.display_h), interpolation=cv2.INTER_LINEAR)

                cv2.imshow(window_name, display_frame)

                if self.benchmark:
                    delay = 1
                else:
                    audio_time = self.get_audio_time()
                    video_time = timestamp_ms / 1000.0
                    if video_time < audio_time - self.pipeline_latency - 0.015:
                        delay = 1  # far behind, display immediately
                    else:
                        delay = max(1, int((video_time - audio_time) * 1000.0))

                key = cv2.waitKey(delay) & 0xFF
                if key == ord('q') or key == 27:
                    self.running = False
                elif key == ord(' '):
                    self.play = not self.play
                    if not self.play:
                        self.sys_clock_pause_time = time.time()
                    else:
                        self.sys_clock_start += time.time() - self.sys_clock_pause_time
                    print(f"[Info] {'Paused' if not self.play else 'Resumed'}")
                elif key == ord('+') or key == ord('='):
                    self.max_shift = min(100, self.max_shift + 2)
                    print(f"[Info] Depth strength increased to: {self.max_shift}")
                elif key == ord('-'):
                    self.max_shift = max(0, self.max_shift - 2)
                    print(f"[Info] Depth strength decreased to: {self.max_shift}")
                elif key == ord('['):
                    self.alpha = max(0.05, self.alpha - 0.05)
                    print(f"[Info] Temporal Alpha decreased to: {self.alpha:.2f}")
                elif key == ord(']'):
                    self.alpha = min(1.0, self.alpha + 0.05)
                    print(f"[Info] Temporal Alpha increased to: {self.alpha:.2f}")
                elif key == ord('c'):
                    self.convergence = max(-100.0, self.convergence - 2.0)
                    print(f"[Info] Convergence decreased to: {self.convergence:.1f}")
                elif key == ord('v'):
                    self.convergence = min(100.0, self.convergence + 2.0)
                    print(f"[Info] Convergence increased to: {self.convergence:.1f}")
                elif key == ord('e'):
                    self.edge_softness = max(0.0, self.edge_softness - 2.0)
                    print(f"[Info] Edge Softness decreased to: {self.edge_softness:.1f}")
                elif key == ord('r'):
                    self.edge_softness = min(50.0, self.edge_softness + 2.0)
                    print(f"[Info] Edge Softness increased to: {self.edge_softness:.1f}")
                elif key == ord('g'):
                    self.depth_gamma = max(0.05, self.depth_gamma - 0.05)
                    print(f"[Info] Depth Gamma decreased to: {self.depth_gamma:.2f}")
                elif key == ord('h'):
                    self.depth_gamma = min(1.0, self.depth_gamma + 0.05)
                    print(f"[Info] Depth Gamma increased to: {self.depth_gamma:.2f}")
                elif key == ord('t'):
                    self.sharpen = max(0.0, self.sharpen - 1.0)
                    print(f"[Info] Sharpen decreased to: {self.sharpen:.1f}")
                elif key == ord('y'):
                    self.sharpen = min(30.0, self.sharpen + 1.0)
                    print(f"[Info] Sharpen increased to: {self.sharpen:.1f}")
                elif key == ord('u'):
                    self.artifact_smoothing = max(0.0, self.artifact_smoothing - 0.5)
                    print(f"[Info] Artifact Smoothing decreased to: {self.artifact_smoothing:.1f}")
                elif key == ord('i'):
                    self.artifact_smoothing = min(10.0, self.artifact_smoothing + 0.5)
                    print(f"[Info] Artifact Smoothing increased to: {self.artifact_smoothing:.1f}")
                elif key == ord('f'):
                    self._toggle_fullscreen(window_name)
            else:
                time.sleep(0.01)

        cv2.destroyAllWindows()

    def stop(self):
        self.running = False
        if hasattr(self, 'sd_stream') and self.sd_stream:
            try:
                self.sd_stream.stop()
                self.sd_stream.close()
            except Exception:
                pass

    def run(self):
        self.start_threads()
        self.run_display_loop()

class SBSPlayerGUI(QMainWindow):
    def __init__(self, player):
        super().__init__()
        self.player = player
        self.setWindowTitle("2D to 3D SBS Player")
        self.resize(1200, 800)
        self.is_seeking = False
        self.prev_volume = 100
        self.fullscreen_mode = False
        
        # Playlist state
        self.playlist = [self.player.video_path] if self.player.video_path else []
        self.current_playlist_idx = 0

        # Initialize NVML for accurate GPU VRAM stats
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_initialized = True
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.nvml_initialized = False
        
        self.init_ui()
        self.update_playlist_ui()
        
        self.current_gui_frame = None
        self.current_gui_ts = None
        self._stats_update_time = 0.0
        
        # Start background player threads only if a video is loaded
        if self.player.video_path:
            self.player.start_threads()
        
        # Frame timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.setSingleShot(True)
        self.timer.start(16)

    def init_ui(self):
        # Menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        open_action = file_menu.addAction("Open Video")
        open_action.triggered.connect(self.open_file)
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        # Main Layout splitter
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # Settings Panel (Left)
        self.settings_widget = QWidget()
        settings_layout = QVBoxLayout(self.settings_widget)
        settings_layout.setAlignment(Qt.AlignTop)
        settings_layout.setSpacing(10)

        # Sliders
        self.shift_slider = self.create_slider(settings_layout, "Depth Strength", 0, 100, int(self.player.max_shift),
            lambda v: setattr(self.player, 'max_shift', int(v)))
            
        self.conv_slider = self.create_slider(settings_layout, "Convergence", -100, 100, int(self.player.convergence),
            lambda v: setattr(self.player, 'convergence', float(v)))
            
        self.soft_slider = self.create_slider(settings_layout, "Edge Softness", 0, 50, int(self.player.edge_softness),
            lambda v: setattr(self.player, 'edge_softness', float(v)))
        self.soft_chk = QCheckBox("Enable Edge Softness")
        self.soft_chk.setChecked(self.player.use_edge_softness)
        self.soft_chk.stateChanged.connect(lambda state: (setattr(self.player, 'use_edge_softness', state == 2), self.player.save_config()))
        settings_layout.addWidget(self.soft_chk)
            
        self.gamma_slider = self.create_slider(settings_layout, "Depth Gamma", 5, 100, int(self.player.depth_gamma * 100),
            lambda v: setattr(self.player, 'depth_gamma', v / 100.0), scale=100.0)
            
        self.sharp_slider = self.create_slider(settings_layout, "Sharpen", 0, 30, int(self.player.sharpen),
            lambda v: setattr(self.player, 'sharpen', float(v)))
        self.sharp_chk = QCheckBox("Enable Sharpening")
        self.sharp_chk.setChecked(self.player.use_sharpen)
        self.sharp_chk.stateChanged.connect(lambda state: (setattr(self.player, 'use_sharpen', state == 2), self.player.save_config()))
        settings_layout.addWidget(self.sharp_chk)
            
        self.smooth_slider = self.create_slider(settings_layout, "Artifact Smoothing", 0, 100, int(self.player.artifact_smoothing * 10),
            lambda v: setattr(self.player, 'artifact_smoothing', v / 10.0), scale=10.0)
        self.smooth_chk = QCheckBox("Enable Artifact Smoothing")
        self.smooth_chk.setChecked(self.player.use_artifact_smoothing)
        self.smooth_chk.stateChanged.connect(lambda state: (setattr(self.player, 'use_artifact_smoothing', state == 2), self.player.save_config()))
        settings_layout.addWidget(self.smooth_chk)

        self.hq_smooth_chk = QCheckBox("High Quality Artifact Mask")
        self.hq_smooth_chk.setChecked(self.player.hq_artifact_smoothing)
        self.hq_smooth_chk.stateChanged.connect(lambda state: (setattr(self.player, 'hq_artifact_smoothing', state == 2), self.player.save_config()))
        settings_layout.addWidget(self.hq_smooth_chk)
            
        self.alpha_slider = self.create_slider(settings_layout, "Temporal Alpha", 5, 100, int(self.player.alpha * 100),
            lambda v: setattr(self.player, 'alpha', v / 100.0), scale=100.0)
        self.temporal_chk = QCheckBox("Enable Temporal Smoothing")
        self.temporal_chk.setChecked(self.player.use_smoothing)
        self.temporal_chk.stateChanged.connect(lambda state: (setattr(self.player, 'use_smoothing', state == 2), self.player.save_config()))
        settings_layout.addWidget(self.temporal_chk)

        # Model Selector
        settings_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(ALL_MODELS)
        self.model_combo.setCurrentText(self.player.model_name)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        settings_layout.addWidget(self.model_combo)

        # Loop Video Checkbox
        self.loop_checkbox = QCheckBox("Loop Current Video")
        self.loop_checkbox.setChecked(self.player.loop_video)
        self.loop_checkbox.stateChanged.connect(self.on_loop_changed)
        settings_layout.addWidget(self.loop_checkbox)

        # Frame Doubler Checkbox
        self.doubler_checkbox = QCheckBox("Enable Frame Doubler (to 60fps)")
        self.doubler_checkbox.setChecked(self.player.use_frame_doubler)
        self.doubler_checkbox.stateChanged.connect(self.on_doubler_changed)
        settings_layout.addWidget(self.doubler_checkbox)

        # Playlist ListWidget
        settings_layout.addWidget(QLabel("Playlist:"))
        self.playlist_widget = QListWidget()
        self.playlist_widget.itemDoubleClicked.connect(self.on_playlist_double_click)
        settings_layout.addWidget(self.playlist_widget)

        # Playlist buttons (Add/Remove)
        playlist_btns = QHBoxLayout()
        self.add_btn = QPushButton("Add File")
        self.add_btn.clicked.connect(self.on_playlist_add)
        playlist_btns.addWidget(self.add_btn)

        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self.on_playlist_remove)
        playlist_btns.addWidget(self.remove_btn)
        settings_layout.addLayout(playlist_btns)

        # Info Label (Stats)
        self.stats_label = QLabel()
        settings_layout.addWidget(self.stats_label)

        # Right Side (Video + Playback)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Video Label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(640, 360)
        right_layout.addWidget(self.video_label, 1)

        # Playback panel
        self.playback_widget = QWidget()
        playback_layout = QVBoxLayout(self.playback_widget)
        
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, int(self.player.duration_sec))
        self.seek_slider.sliderPressed.connect(self.on_seek_press)
        self.seek_slider.sliderReleased.connect(self.on_seek_release)
        playback_layout.addWidget(self.seek_slider)

        controls_layout = QHBoxLayout()
        self.prev_btn = QPushButton("|<")
        self.prev_btn.clicked.connect(self.on_playlist_prev)
        controls_layout.addWidget(self.prev_btn)

        self.play_button = QPushButton("Pause")
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)

        self.next_btn = QPushButton(">|")
        self.next_btn.clicked.connect(self.on_playlist_next)
        controls_layout.addWidget(self.next_btn)

        self.time_label = QLabel("00:00 / 00:00")
        controls_layout.addWidget(self.time_label)

        controls_layout.addStretch()

        self.mute_button = QPushButton("Mute")
        self.mute_button.clicked.connect(self.toggle_mute)
        controls_layout.addWidget(self.mute_button)

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setFixedWidth(120)
        self.volume_slider.valueChanged.connect(self.on_volume_change)
        controls_layout.addWidget(self.volume_slider)

        playback_layout.addLayout(controls_layout)
        right_layout.addWidget(self.playback_widget)

        # Add to splitter
        splitter.addWidget(self.settings_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 900])

        # Enable Drag & Drop
        self.setAcceptDrops(True)

    def create_slider(self, layout, label_text, min_val, max_val, init_val, callback, scale=1.0):
        h_layout = QHBoxLayout()
        display_val = init_val / scale if scale != 1.0 else init_val
        label = QLabel(f"{label_text}: {display_val:.2f}" if scale != 1.0 else f"{label_text}: {display_val}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(init_val)
        
        def on_change(val):
            real_val = val / scale if scale != 1.0 else val
            label.setText(f"{label_text}: {real_val:.2f}" if scale != 1.0 else f"{label_text}: {real_val}")
            callback(val)
            self.player.save_config()
            
        slider.valueChanged.connect(on_change)
        layout.addWidget(label)
        layout.addWidget(slider)
        return slider

    def update_frame(self):
        if self.player.video_ended:
            self.player.video_ended = False
            if len(self.playlist) > 1:
                self.on_playlist_next()
            else:
                self.seek_slider.setValue(0)
                self.player.seek_video_target = 0
                self.player.seek_audio_target = 0
                self.player.play = True
                self.play_button.setText("Pause")
            self.current_gui_frame = None
            self.current_gui_ts = None
            self.timer.start(16)
            return

        # Sync to audio time
        audio_time = self.player.get_audio_time()

        # Keep popping frames if they are in the past to catch up (frame dropping)
        while True:
            # If we don't have a frame cached, get one
            if self.current_gui_frame is None:
                if not self.player.sbs_queue.empty():
                    sbs_tuple = self.player.sbs_queue.get()
                    self.current_gui_frame, self.current_gui_ts = sbs_tuple[0], sbs_tuple[1]
                    if len(sbs_tuple) > 2 and sbs_tuple[2] != self.player.seek_epoch:
                        print(f"[Seek] GUI discarding frame with stale epoch {sbs_tuple[2]} (current: {self.player.seek_epoch})")
                        self.current_gui_frame = None
                        self.current_gui_ts = None
                        continue
                else:
                    break # Queue empty, nothing to do

            # Check if this frame is due
            video_time = self.current_gui_ts / 1000.0
            behind_tolerance = self.player.pipeline_latency + 0.015

            if self.player._seek_accept_behind:
                if video_time >= audio_time - behind_tolerance:
                    self.player._seek_accept_behind = False
                if video_time > audio_time + 0.015:
                    self.timer.start(16)
                    return
                if video_time < audio_time - behind_tolerance:
                    if not self.player.sbs_queue.empty():
                        sbs_tuple = self.player.sbs_queue.get()
                        self.current_gui_frame, self.current_gui_ts = sbs_tuple[0], sbs_tuple[1]
                        if len(sbs_tuple) > 2 and sbs_tuple[2] != self.player.seek_epoch:
                            self.current_gui_frame = None
                            self.current_gui_ts = None
                        continue
                break

            # If there's another frame in the queue, check if that one is also in the past.
            # If the next frame is also due (meaning we are behind), we drop the current one and get the next.
            if video_time < audio_time - behind_tolerance:
                if not self.player.sbs_queue.empty():
                    # Drop current frame, get next one
                    sbs_tuple = self.player.sbs_queue.get()
                    self.current_gui_frame, self.current_gui_ts = sbs_tuple[0], sbs_tuple[1]
                    if len(sbs_tuple) > 2 and sbs_tuple[2] != self.player.seek_epoch:
                        print(f"[Seek] GUI dropping stale frame epoch {sbs_tuple[2]} (current: {self.player.seek_epoch})")
                        self.current_gui_frame = None
                        self.current_gui_ts = None
                    continue
                else:
                    # Discard stale frame entirely - nothing else available yet
                    self.current_gui_frame = None
                    self.current_gui_ts = None
                    break

            # If the frame is in the future, we don't display it yet
            if video_time > audio_time + 0.015:
                if video_time > audio_time + 5.0:
                    # Frame is >5s ahead — stale from a backward seek, discard it
                    self.current_gui_frame = None
                    self.current_gui_ts = None
                    continue
                # Keep the frame cached, return to wait
                self.timer.start(16)
                return

            # Frame is within the display window, break and display it
            break

        # Display the frame if we have one
        if self.current_gui_frame is not None:
            sbs_frame = self.current_gui_frame
            timestamp_ms = self.current_gui_ts
            # Clear cache so we grab a new frame next time
            self.current_gui_frame = None
            self.current_gui_ts = None
            
            # Update FPS history
            now = time.time()
            self.player.fps_history.append(now)
            self.player.fps_history = [t for t in self.player.fps_history if now - t < 1.0]
            
            if not self.is_seeking:
                current_time = timestamp_ms / 1000.0
                self.seek_slider.setValue(int(current_time))
                self.time_label.setText(f"{self.format_time(current_time)} / {self.format_time(self.player.duration_sec)}")
            
            # Show stats
            gpu_util = torch.cuda.utilization() if torch.cuda.is_available() else 0
            if self.nvml_initialized:
                try:
                    info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                    vram_used = info.used / (1024 ** 3)
                    vram_total = info.total / (1024 ** 3)
                    vram_str = f"{vram_used:.2f} / {vram_total:.1f} GB"
                except Exception:
                    vram_str = "Error"
            else:
                vram_used = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
                vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0
                vram_str = f"{vram_used:.1f} / {vram_reserved:.1f} GB (PyTorch)"
            
            backend = "TRT" if self.player.use_trt else ("DA3" if self.player.is_da3 else "PT")
            self.stats_label.setText(
                f"FPS: {len(self.player.fps_history)}\n"
                f"GPU Util: {gpu_util}%\n"
                f"VRAM: {vram_str}\n"
                f"Infer Latency: {self.player.last_inference_ms:.1f}ms [{backend}]\n"
                f"Warp Latency: {self.player.last_warp_ms:.1f}ms"
            )
            self._stats_update_time = time.time()
            
            h, w, c = sbs_frame.shape
            bytes_per_line = c * w
            q_img = QImage(sbs_frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)
            self.video_label.setPixmap(pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

        if time.time() - self._stats_update_time > 0.3:
            self.stats_label.setText("")

        self.timer.start(16)

    def format_time(self, sec):
        m = int(sec) // 60
        s = int(sec) % 60
        return f"{m:02d}:{s:02d}"

    def toggle_play(self):
        self.player.play = not self.player.play
        if not self.player.play:
            self.player.sys_clock_pause_time = time.time()
            self.play_button.setText("Play")
        else:
            self.player.sys_clock_start += time.time() - self.player.sys_clock_pause_time
            self.play_button.setText("Pause")

    def toggle_mute(self):
        if self.player.volume > 0:
            self.prev_volume = int(self.player.volume * 100)
            self.volume_slider.setValue(0)
            self.mute_button.setText("Unmute")
        else:
            self.volume_slider.setValue(self.prev_volume)
            self.mute_button.setText("Mute")

    def on_volume_change(self, val):
        self.player.volume = val / 100.0
        if val == 0:
            self.mute_button.setText("Unmute")
        else:
            self.mute_button.setText("Mute")

    def update_playlist_ui(self):
        self.playlist_widget.clear()
        for path in self.playlist:
            self.playlist_widget.addItem(os.path.basename(path))
        if 0 <= self.current_playlist_idx < len(self.playlist):
            self.playlist_widget.setCurrentRow(self.current_playlist_idx)

    def on_playlist_add(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Add to Playlist", "", "Video Files (*.mp4 *.mkv *.avi *.mov)")
        if file_paths:
            for path in file_paths:
                if path not in self.playlist:
                    self.playlist.append(path)
            self.update_playlist_ui()

    def on_playlist_remove(self):
        current_row = self.playlist_widget.currentRow()
        if current_row >= 0 and current_row < len(self.playlist):
            if len(self.playlist) == 1:
                return
            
            self.playlist.pop(current_row)
            if self.current_playlist_idx == current_row:
                self.current_playlist_idx = min(self.current_playlist_idx, len(self.playlist) - 1)
                self.load_video(self.playlist[self.current_playlist_idx])
            elif self.current_playlist_idx > current_row:
                self.current_playlist_idx -= 1
                
            self.update_playlist_ui()

    def on_playlist_double_click(self, item):
        row = self.playlist_widget.row(item)
        if 0 <= row < len(self.playlist):
            self.current_playlist_idx = row
            self.load_video(self.playlist[row])
            self.update_playlist_ui()

    def on_playlist_prev(self):
        if len(self.playlist) <= 1:
            return
        self.current_playlist_idx = (self.current_playlist_idx - 1) % len(self.playlist)
        self.load_video(self.playlist[self.current_playlist_idx])
        self.update_playlist_ui()

    def on_playlist_next(self):
        if len(self.playlist) <= 1:
            return
        self.current_playlist_idx = (self.current_playlist_idx + 1) % len(self.playlist)
        self.load_video(self.playlist[self.current_playlist_idx])
        self.update_playlist_ui()

    def on_loop_changed(self, state):
        self.player.loop_video = (state == Qt.Checked or state == 2) # Qt6 Checked is 2

    def on_doubler_changed(self, state):
        self.player.use_frame_doubler = (state == Qt.Checked or state == 2)
        if self.player.use_frame_doubler and (not hasattr(self.player, 'rife_context') or self.player.rife_context is None):
            self.player._load_rife_trt_model()
        self.player.save_config()

    def on_seek_press(self):
        self.is_seeking = True

    def on_seek_release(self):
        self.is_seeking = False
        target = self.seek_slider.value()
        print(f"[Seek] target={target:.1f}s")
        self.player.play = True
        self.play_button.setText("Pause")
        self.player.seek_video_target = target
        self.player.seek_audio_target = target
        self.player.reset_temporal = True
        self.player.flush_queues()
        self.player._seek_accept_behind = True
        self.current_gui_frame = None
        self.current_gui_ts = None
        self.seek_slider.setValue(int(target))
        self.time_label.setText(f"{self.format_time(target)} / {self.format_time(self.player.duration_sec)}")

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.mkv *.avi *.mov)")
        if file_path:
            self.playlist = [file_path]
            self.current_playlist_idx = 0
            self.load_video(file_path)
            self.update_playlist_ui()

    def load_video(self, file_path):
        # Stop threads
        self.player.running = False
        time.sleep(0.5)
        
        # Clear stale state from previous video
        self.player.flush_queues()
        self.player.seek_epoch += 1
        self.player._seek_accept_behind = True
        self.current_gui_frame = None
        self.current_gui_ts = None
        
        # Load new video
        self.player.video_path = file_path
        self.player.running = True
        self.player.play = True
        self.player.reset_temporal = True
        
        # Refresh metadata
        try:
            container = av.open(file_path)
            vid = container.streams.video[0]
            self.player.total_frames = vid.frames if vid.frames else 0
            self.player.video_fps = float(vid.average_rate) if vid.average_rate else 0.0
            if self.player.video_fps <= 0:
                self.player.video_fps = float(vid.guessed_rate) if vid.guessed_rate else 24.0
            if vid.duration:
                self.player.duration_sec = float(vid.duration * vid.time_base)
            elif container.duration:
                self.player.duration_sec = float(container.duration / av.time_base)
            elif self.player.total_frames > 0 and self.player.video_fps > 0:
                self.player.duration_sec = self.player.total_frames / self.player.video_fps
            else:
                self.player.duration_sec = 0.0
            if self.player.total_frames == 0 and self.player.duration_sec > 0:
                self.player.total_frames = int(self.player.duration_sec * self.player.video_fps)
            self.player.width = vid.width
            self.player.height = vid.height
            container.close()
        except Exception:
            cap = cv2.VideoCapture(file_path)
            self.player.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.player.video_fps = cap.get(cv2.CAP_PROP_FPS)
            if self.player.video_fps <= 0:
                self.player.video_fps = 24.0
            self.player.duration_sec = self.player.total_frames / self.player.video_fps
            self.player.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.player.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        
        # Force re-compile RIFE if width/height changed and doubler is active
        if self.player.use_frame_doubler:
            self.player._load_rife_trt_model()
        
        self.seek_slider.setRange(0, int(self.player.duration_sec))
        self.seek_slider.setValue(0)
        self.play_button.setText("Pause")
        
        self.player.start_threads()

    def on_model_changed(self, model_name):
        # Stop threads
        self.player.running = False
        time.sleep(0.5)
        
        # Reload model
        self.player.model_name = model_name
        self.player.use_trt = model_name in V2_MODELS
        self.player.is_da3 = model_name in DA3_MODELS
        
        if self.player.use_trt:
            self.player._load_v2_trt_model()
        elif self.player.is_da3:
            self.player._load_da3_model()
        else:
            self.player._load_v2_pytorch_model()
            
        self.player.running = True
        self.player.start_threads()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            dropped_paths = []
            for url in urls:
                file_path = url.toLocalFile()
                if os.path.exists(file_path):
                    dropped_paths.append(file_path)
            if dropped_paths:
                for path in dropped_paths:
                    if path not in self.playlist:
                        self.playlist.append(path)
                first_dropped_idx = self.playlist.index(dropped_paths[0])
                self.current_playlist_idx = first_dropped_idx
                self.load_video(dropped_paths[0])
                self.update_playlist_ui()

    def toggle_fullscreen(self):
        self.fullscreen_mode = not self.fullscreen_mode
        if self.fullscreen_mode:
            self.settings_widget.hide()
            self.playback_widget.hide()
            self.menuBar().hide()
            self.showFullScreen()
        else:
            self.settings_widget.show()
            self.playback_widget.show()
            self.menuBar().show()
            self.showNormal()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            self.toggle_play()
        elif key == Qt.Key_F:
            self.toggle_fullscreen()
        elif key == Qt.Key_D:
            self.doubler_checkbox.setChecked(not self.doubler_checkbox.isChecked())
        elif key == Qt.Key_Q or key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            self.player.max_shift = min(100, self.player.max_shift + 2)
            self.shift_slider.setValue(self.player.max_shift)
        elif key == Qt.Key_Minus:
            self.player.max_shift = max(0, self.player.max_shift - 2)
            self.shift_slider.setValue(self.player.max_shift)
        elif key == Qt.Key_BracketLeft:
            self.player.alpha = max(0.05, self.player.alpha - 0.05)
            self.alpha_slider.setValue(int(self.player.alpha * 100))
        elif key == Qt.Key_BracketRight:
            self.player.alpha = min(1.0, self.player.alpha + 0.05)
            self.alpha_slider.setValue(int(self.player.alpha * 100))
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.player.save_config()
        self.player.stop()
        event.accept()
        QApplication.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D to 3D SBS Video Player using Depth-Anything")
    parser.add_argument("video", nargs="?", default=None, help="Path to input 2D video file (optional)")
    parser.add_argument("--model", type=str, default="depth-anything/Depth-Anything-V2-Large-hf",
                        choices=ALL_MODELS,
                        help="Depth model to use (V2 or DA3)")
    parser.add_argument("--strength", type=int, default=16, help="Stereo shift strength (max pixels)")
    parser.add_argument("--inference-size", type=int, default=518, help="Longest side resolution for depth model input")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"],
                        help="Model precision: fp16 (default), fp32, int8 (V2 only, via bitsandbytes). DA3 uses autocast for fp16.")
    parser.add_argument("--no-trt", action="store_true", help="Disable TensorRT acceleration (V2 models only)")
    parser.add_argument("--benchmark", action="store_true", help="Disable FPS cap to measure raw inference throughput")
    parser.add_argument("--alpha", type=float, default=0.3, help="Temporal smoothing factor alpha (0.05-1.0, lower is smoother)")
    parser.add_argument("--convergence", type=float, default=-10.0, help="Focal plane shift (positive=pop-out, negative=sink-in)")
    parser.add_argument("--edge-softness", type=float, default=20.0, help="GaussianBlur depth edge softening strength")
    parser.add_argument("--depth-gamma", type=float, default=0.2, help="Depth power factor for gamma correction")
    parser.add_argument("--sharpen", type=float, default=14.0, help="Post-warp unsharp mask strength")
    parser.add_argument("--artifact-smoothing", type=float, default=1.0, help="Warping artifact edge smoothing strength")
    parser.add_argument("--no-gui", action="store_true", help="Run in OpenCV console display mode instead of PySide6 GUI")

    args = parser.parse_args()

    player = SBSVideoPlayer(
        video_path=args.video,
        model_name=args.model,
        max_shift=args.strength,
        inference_size=args.inference_size,
        precision=args.precision,
        use_trt=not args.no_trt,
        benchmark=args.benchmark,
        alpha=args.alpha,
        convergence=args.convergence,
        edge_softness=args.edge_softness,
        depth_gamma=args.depth_gamma,
        sharpen=args.sharpen,
        artifact_smoothing=args.artifact_smoothing
    )

    if args.no_gui:
        if not args.video:
            print("[Error] --no-gui mode requires a video file.")
            sys.exit(1)
        player.run()
    else:
        app = QApplication(sys.argv)
        gui = SBSPlayerGUI(player)
        gui.show()
        sys.exit(app.exec())
