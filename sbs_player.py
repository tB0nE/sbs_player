#!/usr/bin/env python3
import sys
import os
os.environ["OPENCV_VIDEO_DEBUG"] = "0"
import time
import queue
import threading
import argparse
import subprocess
import ctypes
from collections import deque
import numpy as np
import cv2
import torch
import av
import sounddevice as sd
import pynvml
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QSlider, QLabel, QPushButton, QComboBox, QCheckBox, QFileDialog, QListWidget)
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
        self.max_shift = max_shift
        self.buffer_size = buffer_size
        self.inference_size = inference_size
        self.precision = precision
        self.benchmark = benchmark
        self.alpha = alpha
        self.convergence = convergence
        self.edge_softness = edge_softness
        self.depth_gamma = depth_gamma
        self.sharpen = sharpen
        self.artifact_smoothing = artifact_smoothing
        self.video_fps = 30.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Info] Using device: {self.device}")

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.checkpoints_dir = os.path.join(self.script_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Load persisted config first, then apply CLI overrides
        self.model_name = model_name
        self.use_trt = use_trt and (model_name in V2_MODELS)
        self.is_da3 = model_name in DA3_MODELS
        self.volume = 1.0
        self.use_smoothing = True
        self.use_edge_softness = True
        self.use_artifact_smoothing = True
        self.use_sharpen = True
        self.hq_artifact_smoothing = False
        self.use_frame_doubler = False
        self.load_config()

        # CLI args override persisted config
        self.model_name = model_name
        self.use_trt = use_trt and (model_name in V2_MODELS)
        self.is_da3 = model_name in DA3_MODELS

        # Select and load backend from resolved configuration
        if self.use_trt:
            print("[Info] Resolving CUDA and TensorRT library paths...")
            loaded_libs = load_cuda_libs()
            print(f"[Info] Loaded {len(loaded_libs)} CUDA/TensorRT libraries successfully.")
            self._load_v2_trt_model()
        elif self.is_da3:
            self._load_da3_model()
        else:
            self._load_v2_pytorch_model()

        self.prev_depth_gpu = None
        self.last_inference_ms = 0.0
        self.last_preprocess_ms = 0.0
        self.last_model_ms = 0.0
        self.last_postprocess_ms = 0.0
        self.last_warp_ms = 0.0

        self.frame_queue = queue.Queue(maxsize=4)
        self.depth_queue = queue.Queue(maxsize=4)
        self.sbs_queue = queue.Queue(maxsize=6)
        self.audio_queue = queue.Queue(maxsize=64)

        self.running = True
        self.play = True
        self.fullscreen = False
        self.fps_history = deque()
        self._reset_temporal_depth = False
        self._reset_temporal_warp = False
        self.loop_video = False
        self.video_ended = False
        self._reader_eof = False

        # Audio state
        self.current_audio_data = np.zeros((0, 2), dtype=np.float32)
        self.audio_samples_played = 0
        self.audio_sample_rate = 44100
        self.audio_latency_frames = 0
        self.seek_audio_target = None
        self.seek_video_target = None
        self.has_audio = False
        self.sys_clock_start = time.time()
        self.sys_clock_pause_time = time.time()
        self._last_valid_audio_time = 0.0

        # Warp cache
        self._map_x = None
        self._map_y = None
        self._cached_w = 0
        self._cached_h = 0

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

    def _trt_dtype_to_torch(self, trt_dtype):
        import tensorrt as trt
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32:   torch.int32,
            trt.int8:    torch.int8,
            trt.bool:    torch.bool,
        }
        if trt_dtype in mapping:
            return mapping[trt_dtype]
        raise ValueError(f"Unsupported TRT tensor dtype: {trt_dtype}")

    def _load_v2_trt_model(self):
        model_short_name = self.model_name.split("/")[-1]
        onnx_filename = f"{model_short_name}_{self.inference_size}.onnx"
        self.onnx_path = os.path.join(self.checkpoints_dir, onnx_filename)

        if not os.path.exists(self.onnx_path):
            print(f"[Info] ONNX model not found at: {self.onnx_path}")
            print(f"[Info] Attempting to export from PyTorch (requires transformers)...")
            try:
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
            except ImportError:
                print(f"[Error] The ONNX model file was not found and 'transformers' is not installed.")
                print(f"[Error] Please download the ONNX model manually:")
                print(f"[Error]   Place it at: {self.onnx_path}")
                print(f"[Error]   Or run: pip install transformers && re-launch the app")
                raise RuntimeError("ONNX model not found and transformers not installed")
            except Exception as e:
                print(f"[Error] Failed to export ONNX model: {e}")
                print(f"[Error] Please download the ONNX model manually to: {self.onnx_path}")
                raise RuntimeError(f"Failed to export ONNX model: {e}")

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

        # Inspect engine I/O tensor metadata and allocate matching buffers
        import tensorrt as trt
        num_io = self.trt_engine.num_io_tensors
        found_input = False
        found_output = False

        for i in range(num_io):
            name = self.trt_engine.get_tensor_name(i)
            mode = self.trt_engine.get_tensor_mode(name)
            dtype = self.trt_engine.get_tensor_dtype(name)
            shape = tuple(self.trt_engine.get_tensor_shape(name))
            td = self._trt_dtype_to_torch(dtype)

            if mode == trt.TensorIOMode.INPUT:
                if name == 'pixel_values':
                    found_input = True
                    self.trt_d_input = torch.empty(shape, dtype=td, device=self.device)
                    self.trt_context.set_tensor_address(name, int(self.trt_d_input.data_ptr()))
            elif mode == trt.TensorIOMode.OUTPUT:
                if name == 'predicted_depth':
                    found_output = True
                    self.trt_d_output = torch.empty(shape, dtype=td, device=self.device)
                    self.trt_context.set_tensor_address(name, int(self.trt_d_output.data_ptr()))

        if not found_input or not found_output:
            raise RuntimeError(
                f"TRT engine tensor mismatch: input={'found' if found_input else 'missing'}, "
                f"output={'found' if found_output else 'missing'}"
            )
        print(f"[Info] TRT I/O buffers: input {self.trt_d_input.dtype} {list(self.trt_d_input.shape)}, "
              f"output {self.trt_d_output.dtype} {list(self.trt_d_output.shape)}")

        input_dtype = self.trt_d_input.dtype
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=input_dtype).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=input_dtype).view(1, 3, 1, 1)

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
            
        # 3. Create context and allocate buffers from engine metadata
        self.rife_context = self.rife_engine.create_execution_context()
        self.rife_stream = torch.cuda.Stream()

        rife_inputs = {'img0': None, 'img1': None, 'timestep': None}
        rife_output = None

        num_io = self.rife_engine.num_io_tensors
        for i in range(num_io):
            name = self.rife_engine.get_tensor_name(i)
            mode = self.rife_engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT and name in rife_inputs:
                dtype = self.rife_engine.get_tensor_dtype(name)
                shape = tuple(self.rife_engine.get_tensor_shape(name))
                td = self._trt_dtype_to_torch(dtype)
                tensor = torch.empty(shape, dtype=td, device=self.device)
                rife_inputs[name] = tensor
                self.rife_context.set_tensor_address(name, int(tensor.data_ptr()))
            elif mode == trt.TensorIOMode.OUTPUT and name == 'output':
                dtype = self.rife_engine.get_tensor_dtype(name)
                shape = tuple(self.rife_engine.get_tensor_shape(name))
                td = self._trt_dtype_to_torch(dtype)
                rife_output = torch.empty(shape, dtype=td, device=self.device)
                self.rife_context.set_tensor_address(name, int(rife_output.data_ptr()))

        if any(v is None for v in rife_inputs.values()) or rife_output is None:
            raise RuntimeError(f"RIFE engine tensor mismatch: missing inputs or output")

        self.rife_d_img0 = rife_inputs['img0']
        self.rife_d_img1 = rife_inputs['img1']
        self.rife_d_timestep = rife_inputs['timestep']
        self.rife_d_output = rife_output
        
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
                target_sec = self.seek_video_target
                target_offset = int(target_sec * av.time_base)
                print(f"[Seek] AV reader seeking to {target_sec:.1f}s (offset={target_offset})")
                try:
                    container.seek(target_offset, backward=True)
                except Exception as e:
                    print(f"[Error] AV seek failed: {e}")
                self.flush_queues()
                self.seek_video_target = None
                self._seek_target_ms = target_sec * 1000.0
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
                    self._reader_eof = False
                    self.seek_audio_target = 0.0
                    first_frame = True
                    continue
                else:
                    self._reader_eof = True
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

                    if hasattr(self, '_seek_target_ms') and timestamp_ms < self._seek_target_ms - 1.0:
                        continue

                    if hasattr(self, '_seek_target_ms'):
                        del self._seek_target_ms

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
                    self._reader_eof = False
                    self.seek_audio_target = 0.0
                else:
                    self._reader_eof = True
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
            try:
                frame, timestamp_ms, entry_time, epoch = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if self._reset_temporal_depth:
                self.prev_depth_gpu = None
                self._reset_temporal_depth = False
            if epoch != self.seek_epoch:
                continue
            h, w = frame.shape[:2]

            t0 = time.perf_counter()

            # Pre-shrink on CPU before H2D: aspect-preserving resize + pad to square
            h, w = frame.shape[:2]
            scale = self.inference_size / max(h, w)
            nh, nw = int(h * scale), int(w * scale)
            small_frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            pad_h = self.inference_size - nh
            pad_w = self.inference_size - nw
            if pad_h > 0 or pad_w > 0:
                small_frame = cv2.copyMakeBorder(small_frame, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            frame_gpu = torch.from_numpy(small_frame).to(self.device, non_blocking=True)
            frame_gpu = frame_gpu.permute(2, 0, 1).unsqueeze(0).to(dtype=self.trt_d_input.dtype)
            frame_gpu = frame_gpu.flip(1)  # BGR -> RGB

            # Normalize directly into pre-allocated TRT input buffer
            self.trt_d_input.copy_(frame_gpu)
            self.trt_d_input.div_(255.0).sub_(self._mean).div_(self._std)

            self.last_preprocess_ms = (time.perf_counter() - t0) * 1000.0

            pre_done = torch.cuda.Event(enable_timing=True)
            pre_done.record()
            self.trt_stream.wait_event(pre_done)
            with torch.cuda.stream(self.trt_stream):
                self.trt_context.execute_async_v3(stream_handle=self.trt_stream.cuda_stream)
                infer_done = torch.cuda.Event(enable_timing=True)
                infer_done.record(self.trt_stream)
            infer_done.synchronize()
            self.last_model_ms = pre_done.elapsed_time(infer_done)

            t2 = time.perf_counter()
            depth = self.trt_d_output.clone()
            if self.prev_depth_gpu is None or self.prev_depth_gpu.shape != depth.shape or not isinstance(self.prev_depth_gpu, torch.Tensor) or self.prev_depth_gpu.dtype != depth.dtype:
                self.prev_depth_gpu = depth.clone()
            else:
                if self.use_smoothing:
                    depth.mul_(self.alpha).add_(self.prev_depth_gpu, alpha=1.0 - self.alpha)
                self.prev_depth_gpu.copy_(depth)

            # Download small depth map, upscale to full resolution on CPU
            depth_small = depth.float().cpu().numpy()
            while depth_small.ndim > 2:
                depth_small = depth_small.squeeze(0)
            if depth_small.size == 0 or h == 0 or w == 0:
                self.depth_queue.put((frame, np.zeros((h, w), dtype=np.float32), timestamp_ms, entry_time, epoch))
                continue
            # Percentile-based normalization (resists single outlier "pumping")
            depth_min = np.percentile(depth_small, 5)
            depth_max = np.percentile(depth_small, 95)
            diff = depth_max - depth_min
            if diff > 0:
                depth_norm_small = np.clip(np.power((depth_small - depth_min) / diff, self.depth_gamma), 0, 1).astype(np.float32)
            else:
                depth_norm_small = np.zeros((self.inference_size, self.inference_size), dtype=np.float32)
            normalized_depth = cv2.resize(depth_norm_small, (w, h), interpolation=cv2.INTER_LINEAR)
            self.last_postprocess_ms = (time.perf_counter() - t2) * 1000.0
            self.last_inference_ms = (time.perf_counter() - t0) * 1000.0

            self.depth_queue.put((frame, normalized_depth, timestamp_ms, entry_time, epoch))

    def _depth_loop_pytorch(self):
        while self.running:
            try:
                frame, timestamp_ms, entry_time, epoch = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if self._reset_temporal_depth:
                self.prev_depth_gpu = None
                self._reset_temporal_depth = False
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
            try:
                frame, timestamp_ms, entry_time, epoch = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if self._reset_temporal_depth:
                self.prev_depth_gpu = None
                self._reset_temporal_depth = False
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
            try:
                frame, normalized_depth, timestamp_ms, entry_time, epoch = self.depth_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if self._reset_temporal_warp:
                self.last_sbs_frame = None
                self.last_timestamp_ms = None
                self._reset_temporal_warp = False
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
                should_skip_rife = (self.pipeline_latency > 1.0 / self.video_fps) if self.video_fps > 0 else False
                if self.last_sbs_frame is not None and self.last_timestamp_ms is not None and not should_skip_rife:
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
        if self.has_audio:
            t = max(0.0, (self.audio_samples_played - self.audio_latency_frames) / self.audio_sample_rate)
            self._last_valid_audio_time = t
            return t
        return self._last_valid_audio_time

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
                    target_sec = self.seek_audio_target
                    target_pts = int(target_sec * av.time_base)
                    container.seek(target_pts)
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.current_audio_data = np.zeros((0, 2), dtype=np.float32)
                    self.audio_samples_played = int(target_sec * self.audio_sample_rate)
                    self._audio_seek_target_sec = target_sec
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
                
                try:
                    decoded_frames = packet.decode()
                    for frame in decoded_frames:
                        if hasattr(self, '_audio_seek_target_sec') and self._audio_seek_target_sec is not None:
                            frame_sec = float(frame.pts * audio_stream.time_base) if frame.pts else 0.0
                            if frame_sec + 0.1 < self._audio_seek_target_sec:
                                continue
                            self._audio_seek_target_sec = None
                        resampled_frames = resampler.resample(frame)
                        for r in resampled_frames:
                            data = r.to_ndarray()
                            interleaved = np.ascontiguousarray(data.T)
                            self.audio_queue.put(interleaved)
                except Exception:
                    pass
                        
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
        self._threads = []

        reader = threading.Thread(target=self.video_reader_thread, daemon=False)
        processor = threading.Thread(target=self.depth_inference_thread, daemon=False)
        warper = threading.Thread(target=self.warp_thread, daemon=False)
        audio = threading.Thread(target=self.audio_thread_func, daemon=False)

        reader.start()
        processor.start()
        warper.start()
        audio.start()

        self._threads = [reader, processor, warper, audio]
        print("[Info] All threads started.")

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
        if hasattr(self, '_threads'):
            for t in self._threads:
                t.join(timeout=2.0)
            self._threads.clear()
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
        self.setWindowTitle("Nightfall Player")
        self.resize(1200, 800)
        self.is_seeking = False
        self._seek_setting = False
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
        self._telemetry_cache = ""
        self._telemetry_last_update = 0.0
        
        # Start background player threads only if a video is loaded
        if self.player.video_path:
            self.player.start_threads()
        
        # Frame timer
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.update_frame)
        self.timer.setSingleShot(True)
        self.timer.start(16)

        self._config_save_timer = QTimer(self)
        self._config_save_timer.setSingleShot(True)
        self._config_save_timer.timeout.connect(self._do_save_config)
        self._config_dirty = False

    def _do_save_config(self):
        if self._config_dirty:
            self._config_dirty = False
            self.player.save_config()

    def _mark_config_dirty(self):
        self._config_dirty = True
        self._config_save_timer.start(500)

    def init_ui(self):
        self.setStyleSheet("""
            /* ── Base ─────────────────────────────── */
            QMainWindow, QWidget {
                background-color: #0d0d1a;
                color: #c8c8d4;
                font-family: "Inter", "Segoe UI", sans-serif;
                font-size: 13px;
            }

            /* ── Panels ────────────────────────────── */
            QWidget#settings_widget, QWidget#playlist_panel {
                background: #12122a;
                border: 1px solid #1e1e3a;
                border-radius: 8px;
                margin: 4px;
            }

            /* ── Headers ───────────────────────────── */
            QLabel { color: #c8c8d4; }
            QLabel b { color: #e0e0f0; }

            /* ── Buttons ───────────────────────────── */
            QPushButton {
                background: #1a1a3a;
                color: #c8c8d4;
                border: 1px solid #2a2a4a;
                border-radius: 6px;
                padding: 6px 14px;
            }
            QPushButton:hover { background: #252550; border-color: #3a3a6a; }
            QPushButton:pressed { background: #e94560; color: #fff; border-color: #e94560; }
            QPushButton:checked { background: #e94560; color: #fff; }

            /* Transport buttons */
            QPushButton#prev_btn, QPushButton#play_button,
            QPushButton#next_btn, QPushButton#mute_button,
            QPushButton#playlist_toggle_controls_btn,
            QPushButton#settings_toggle_controls_btn {
                font-size: 11px;
                padding: 4px 10px;
            }

            /* Close buttons (× «) */
            QPushButton[text=\"×\"], QPushButton[text=\"«\"] {
                background: transparent;
                border: none;
                font-size: 16px;
                font-weight: bold;
                padding: 0;
                color: #888;
            }
            QPushButton[text=\"×\"]:hover, QPushButton[text=\"«\"]:hover {
                color: #e94560;
                background: #1e1e3a;
                border-radius: 4px;
            }

            /* ── Sliders ───────────────────────────── */
            QSlider::groove:horizontal {
                background: #1e1e3a;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #e94560;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover { background: #ff5a7a; }
            QSlider::sub-page:horizontal { background: #e94560; border-radius: 3px; }

            /* ── Seek Slider ───────────────────────── */
            QSlider#seek_slider::groove:horizontal { height: 8px; }
            QSlider#seek_slider::handle:horizontal {
                width: 14px; height: 14px;
                margin: -3px 0;
                background: #fff;
                border: 2px solid #e94560;
            }

            /* ── Checkboxes ────────────────────────── */
            QCheckBox { spacing: 8px; }
            QCheckBox::indicator {
                width: 18px; height: 18px;
                border: 2px solid #2a2a4a;
                border-radius: 4px;
                background: #0d0d1a;
            }
            QCheckBox::indicator:checked {
                background: #e94560;
                border-color: #e94560;
            }

            /* ── Combobox ──────────────────────────── */
            QComboBox {
                background: #1a1a3a;
                border: 1px solid #2a2a4a;
                border-radius: 6px;
                padding: 6px 10px;
                color: #c8c8d4;
            }
            QComboBox:hover { border-color: #3a3a6a; }
            QComboBox::drop-down { border: none; width: 24px; }
            QComboBox QAbstractItemView {
                background: #12122a;
                border: 1px solid #1e1e3a;
                border-radius: 4px;
                selection-background-color: #e94560;
            }

            /* ── List Widget ───────────────────────── */
            QListWidget {
                background: #0d0d1a;
                border: 1px solid #1e1e3a;
                border-radius: 6px;
                padding: 4px;
                outline: 0;
            }
            QListWidget::item {
                padding: 8px 10px;
                border-radius: 4px;
                margin: 2px 0;
            }
            QListWidget::item:selected {
                background: #e94560;
                color: #fff;
            }
            QListWidget::item:hover { background: #1a1a3a; }
            QListWidget::item:selected:hover { background: #e94560; }

            /* ── Scrollbars ────────────────────────── */
            QScrollBar:vertical {
                background: #0d0d1a;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #2a2a4a;
                min-height: 30px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover { background: #3a3a6a; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

            /* ── Splitter ──────────────────────────── */
            QSplitter::handle {
                background: #1e1e3a;
                width: 2px;
            }

            /* ── Tooltips ──────────────────────────── */
            QToolTip {
                background: #12122a;
                color: #c8c8d4;
                border: 1px solid #1e1e3a;
                border-radius: 4px;
                padding: 4px 8px;
            }

            /* ── Stats / Time Labels ───────────────── */
            QLabel#stats_label {
                color: #777;
                font-size: 11px;
                font-family: "JetBrains Mono", "Fira Code", monospace;
                padding: 8px 4px;
            }
            QLabel#time_label {
                color: #999;
                font-size: 11px;
                padding: 0 8px;
            }

            /* ── Volume Slider ─────────────────────── */
            QSlider#volume_slider::groove:horizontal {
                height: 4px;
                background: #1e1e3a;
                border-radius: 2px;
            }
            QSlider#volume_slider::handle:horizontal {
                width: 12px; height: 12px;
                margin: -4px 0;
                background: #e94560;
                border-radius: 6px;
            }
            QSlider#volume_slider::sub-page:horizontal {
                background: #e94560;
                border-radius: 2px;
            }
        """)

        # ── Widget Creation ──────────────────────────────────────────
        self.settings_widget = QWidget()
        self.settings_widget.setObjectName("settings_widget")
        settings_layout = QVBoxLayout(self.settings_widget)
        settings_layout.setContentsMargins(4, 4, 4, 4)
        settings_layout.setSpacing(8)

        settings_header = QHBoxLayout()
        settings_header.addWidget(QLabel("<b>Settings</b>"))
        settings_header.addStretch()
        self.settings_toggle_btn = QPushButton("×")
        self.settings_toggle_btn.setFixedSize(24, 24)
        self.settings_toggle_btn.clicked.connect(self.toggle_settings)
        settings_header.addWidget(self.settings_toggle_btn)
        settings_layout.addLayout(settings_header)

        # Sliders
        self.shift_slider = self.create_slider(settings_layout, "Depth Strength", 0, 100, int(self.player.max_shift),
            lambda v: setattr(self.player, 'max_shift', int(v)))
            
        self.conv_slider = self.create_slider(settings_layout, "Convergence", -100, 100, int(self.player.convergence),
            lambda v: setattr(self.player, 'convergence', float(v)))
            
        self.soft_slider = self.create_slider(settings_layout, "Edge Softness", 0, 50, int(self.player.edge_softness),
            lambda v: setattr(self.player, 'edge_softness', float(v)))
        self.soft_chk = QCheckBox("Enable Edge Softness")
        self.soft_chk.setChecked(self.player.use_edge_softness)
        self.soft_chk.stateChanged.connect(lambda state: (setattr(self.player, 'use_edge_softness', state == 2), self._mark_config_dirty()))
        settings_layout.addWidget(self.soft_chk)
            
        self.gamma_slider = self.create_slider(settings_layout, "Depth Gamma", 5, 100, int(self.player.depth_gamma * 100),
            lambda v: setattr(self.player, 'depth_gamma', v / 100.0), scale=100.0)
            
        self.sharp_slider = self.create_slider(settings_layout, "Sharpen", 0, 30, int(self.player.sharpen),
            lambda v: setattr(self.player, 'sharpen', float(v)))
        self.sharp_chk = QCheckBox("Enable Sharpening")
        self.sharp_chk.setChecked(self.player.use_sharpen)
        self.sharp_chk.stateChanged.connect(lambda state: (setattr(self.player, 'use_sharpen', state == 2), self._mark_config_dirty()))
        settings_layout.addWidget(self.sharp_chk)
            
        self.smooth_slider = self.create_slider(settings_layout, "Artifact Smoothing", 0, 100, int(self.player.artifact_smoothing * 10),
            lambda v: setattr(self.player, 'artifact_smoothing', v / 10.0), scale=10.0)
        self.smooth_chk = QCheckBox("Enable Artifact Smoothing")
        self.smooth_chk.setChecked(self.player.use_artifact_smoothing)
        self.smooth_chk.stateChanged.connect(lambda state: (setattr(self.player, 'use_artifact_smoothing', state == 2), self._mark_config_dirty()))
        settings_layout.addWidget(self.smooth_chk)

        self.hq_smooth_chk = QCheckBox("High Quality Artifact Mask")
        self.hq_smooth_chk.setChecked(self.player.hq_artifact_smoothing)
        self.hq_smooth_chk.stateChanged.connect(lambda state: (setattr(self.player, 'hq_artifact_smoothing', state == 2), self._mark_config_dirty()))
        settings_layout.addWidget(self.hq_smooth_chk)
            
        self.alpha_slider = self.create_slider(settings_layout, "Temporal Alpha", 5, 100, int(self.player.alpha * 100),
            lambda v: setattr(self.player, 'alpha', v / 100.0), scale=100.0)
        self.temporal_chk = QCheckBox("Enable Temporal Smoothing")
        self.temporal_chk.setChecked(self.player.use_smoothing)
        self.temporal_chk.stateChanged.connect(lambda state: (setattr(self.player, 'use_smoothing', state == 2), self._mark_config_dirty()))
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

        # Right Side (Video + Playback) removed — video is now in main layout directly

        # Video Label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("QLabel { background-color: #050510; border: 1px solid #1e1e3a; border-radius: 4px; }")
        self.video_label.setMinimumSize(640, 360)

        # Playback panel
        self.playback_widget = QWidget()
        playback_layout = QVBoxLayout(self.playback_widget)
        
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setObjectName("seek_slider")
        self.seek_slider.setSingleStep(1)
        self.seek_slider.setRange(0, int(self.player.duration_sec))
        self.seek_slider.installEventFilter(self)
        self.seek_slider.sliderPressed.connect(self.on_seek_press)
        self.seek_slider.sliderReleased.connect(self.on_seek_release)
        self.seek_slider.valueChanged.connect(self.on_seek_value_changed)
        playback_layout.addWidget(self.seek_slider)

        controls_layout = QHBoxLayout()
        self.prev_btn = QPushButton("|<")
        self.prev_btn.setObjectName("prev_btn")
        self.prev_btn.clicked.connect(self.on_playlist_prev)
        controls_layout.addWidget(self.prev_btn)

        self.play_button = QPushButton("Pause")
        self.play_button.setObjectName("play_button")
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)

        self.next_btn = QPushButton(">|")
        self.next_btn.setObjectName("next_btn")
        self.next_btn.clicked.connect(self.on_playlist_next)
        controls_layout.addWidget(self.next_btn)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setObjectName("time_label")
        controls_layout.addWidget(self.time_label)

        controls_layout.addStretch()

        self.playlist_toggle_controls_btn = QPushButton("Playlist ◂")
        self.playlist_toggle_controls_btn.setObjectName("playlist_toggle_controls_btn")
        self.playlist_toggle_controls_btn.clicked.connect(self.toggle_playlist)
        controls_layout.addWidget(self.playlist_toggle_controls_btn)

        self.settings_toggle_controls_btn = QPushButton("Settings ▸")
        self.settings_toggle_controls_btn.setObjectName("settings_toggle_controls_btn")
        self.settings_toggle_controls_btn.clicked.connect(self.toggle_settings)
        controls_layout.addWidget(self.settings_toggle_controls_btn)

        self.mute_button = QPushButton("Mute")
        self.mute_button.setObjectName("mute_button")
        self.mute_button.clicked.connect(self.toggle_mute)
        controls_layout.addWidget(self.mute_button)

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setObjectName("volume_slider")
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setFixedWidth(120)
        self.volume_slider.valueChanged.connect(self.on_volume_change)
        controls_layout.addWidget(self.volume_slider)

        playback_layout.addLayout(controls_layout)

        # Playlist Panel (Right, collapsible)
        self.playlist_panel = QWidget()
        self.playlist_panel.setObjectName("playlist_panel")
        playlist_panel_layout = QVBoxLayout(self.playlist_panel)
        playlist_panel_layout.setContentsMargins(4, 4, 4, 4)

        playlist_header = QHBoxLayout()
        playlist_header.addWidget(QLabel("<b>Playlist</b>"))
        playlist_header.addStretch()
        self.playlist_toggle_btn = QPushButton("×")
        self.playlist_toggle_btn.setFixedSize(24, 24)
        self.playlist_toggle_btn.clicked.connect(self.toggle_playlist)
        playlist_header.addWidget(self.playlist_toggle_btn)
        playlist_panel_layout.addLayout(playlist_header)

        self.playlist_widget = QListWidget()
        self.playlist_widget.itemDoubleClicked.connect(self.on_playlist_double_click)
        playlist_panel_layout.addWidget(self.playlist_widget)

        playlist_btns = QHBoxLayout()
        self.add_btn = QPushButton("Add File")
        self.add_btn.clicked.connect(self.on_playlist_add)
        playlist_btns.addWidget(self.add_btn)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self.on_playlist_remove)
        playlist_btns.addWidget(self.remove_btn)
        playlist_panel_layout.addLayout(playlist_btns)

        self.stats_label = QLabel()
        self.stats_label.setObjectName("stats_label")
        playlist_panel_layout.addWidget(self.stats_label)

        # ── Layout Assembly ──────────────────────────────────────────
        center = QWidget()
        self.setCentralWidget(center)
        main_layout = QVBoxLayout(center)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        video_row = QHBoxLayout()
        video_row.setContentsMargins(0, 0, 0, 0)
        video_row.setSpacing(0)
        video_row.addWidget(self.settings_widget)
        video_row.addWidget(self.video_label, 1)
        video_row.addWidget(self.playlist_panel)
        main_layout.addLayout(video_row, 1)
        main_layout.addWidget(self.playback_widget)

        # Initial state: settings hidden, playlist shown
        self.settings_widget.setFixedWidth(300)
        self.playlist_panel.setFixedWidth(250)
        self.settings_widget.hide()
        self.settings_toggle_btn.setText("«")
        self.settings_toggle_controls_btn.setText("Settings ▸")

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
            self._mark_config_dirty()
            
        slider.valueChanged.connect(on_change)
        layout.addWidget(label)
        layout.addWidget(slider)
        return slider

    def update_frame(self):
        if self.player._reader_eof and self.player.frame_queue.empty() and self.player.depth_queue.empty() and self.player.sbs_queue.empty() and self.current_gui_frame is None:
            self.player._reader_eof = False
            if len(self.playlist) > 1:
                self.on_playlist_next()
            else:
                self._seek_setting = True
                self.seek_slider.setValue(0)
                self._seek_setting = False
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

            if not self.player.has_audio:
                break

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
            while self.player.fps_history and now - self.player.fps_history[0] > 1.0:
                self.player.fps_history.popleft()
            
            if not self.is_seeking:
                current_time = timestamp_ms / 1000.0
                self._seek_setting = True
                self.seek_slider.setValue(int(current_time))
                self._seek_setting = False
                self.time_label.setText(f"{self.format_time(current_time)} / {self.format_time(self.player.duration_sec)}")
            
            # Show stats (rate-limited to 2 Hz)
            now_ts = time.time()
            if now_ts - self._telemetry_last_update >= 0.5:
                self._telemetry_last_update = now_ts
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
                self._telemetry_cache = (
                    f"FPS: {len(self.player.fps_history)}\n"
                    f"GPU Util: {gpu_util}%\n"
                    f"VRAM: {vram_str}\n"
                    f"Infer Latency: {self.player.last_inference_ms:.1f}ms [{backend}]\n"
                    f"Warp Latency: {self.player.last_warp_ms:.1f}ms"
                )
            if self._telemetry_cache:
                self.stats_label.setText(self._telemetry_cache)
            self._stats_update_time = now_ts
            
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

    def toggle_playlist(self):
        if self.playlist_panel.isVisible():
            self.playlist_panel.hide()
            self.playlist_toggle_btn.setText("«")
            self.playlist_toggle_controls_btn.setText("Playlist ▸")
        else:
            self.playlist_panel.show()
            self.playlist_toggle_btn.setText("×")
            self.playlist_toggle_controls_btn.setText("Playlist ◂")

    def toggle_settings(self):
        if self.settings_widget.isVisible():
            self.settings_widget.hide()
            self.settings_toggle_btn.setText("«")
            self.settings_toggle_controls_btn.setText("Settings ▸")
        else:
            self.settings_widget.show()
            self.settings_toggle_btn.setText("×")
            self.settings_toggle_controls_btn.setText("Settings ◂")

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
            self._run_rife_build()
        self._mark_config_dirty()

    def _run_rife_build(self):
        if getattr(self, '_rife_building', False):
            return
        self._rife_building = True
        self.stats_label.setText("Building RIFE engine...")
        threading.Thread(target=self._rife_build_worker, daemon=True).start()

    def _rife_build_worker(self):
        try:
            self.player._load_rife_trt_model()
        except Exception as e:
            print(f"[Error] RIFE build failed: {e}")
        self._rife_building = False
        QTimer.singleShot(0, self._on_rife_ready)

    def _on_rife_ready(self):
        self._rife_building = False
        self.stats_label.setText("")

    def on_seek_press(self):
        self.is_seeking = True

    def on_seek_value_changed(self, value):
        if self._seek_setting:
            return
        if not self.is_seeking:
            self._do_seek(value)
        else:
            self.time_label.setText(f"{self.format_time(value)} / {self.format_time(self.player.duration_sec)}")

    def _do_seek(self, target):
        self.is_seeking = True
        print(f"[Seek] target={target:.1f}s")
        self.player.seek_epoch += 1
        self.player.play = True
        self.play_button.setText("Pause")
        self.player.seek_video_target = target
        self.player.seek_audio_target = target
        self.player.reset_temporal = True
        self.player._reader_eof = False
        self.player.flush_queues()
        self.player._seek_accept_behind = True
        self.current_gui_frame = None
        self.current_gui_ts = None
        self._seek_setting = True
        self.seek_slider.setValue(int(target))
        self._seek_setting = False
        self.time_label.setText(f"{self.format_time(target)} / {self.format_time(self.player.duration_sec)}")
        self.is_seeking = False

    def on_seek_release(self):
        if self.is_seeking:
            self._do_seek(self.seek_slider.value())

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.mkv *.avi *.mov)")
        if file_path:
            self.playlist = [file_path]
            self.current_playlist_idx = 0
            self.load_video(file_path)
            self.update_playlist_ui()

    def load_video(self, file_path):
        self.player.stop()
        
        # Clear stale state from previous video
        self.player.flush_queues()
        self.player.seek_epoch += 1
        self.player._seek_accept_behind = True
        self.current_gui_frame = None
        self.current_gui_ts = None
        
        # Load new video
        self.player.video_path = file_path
        self.player._reader_eof = False
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
            self._run_rife_build()
        
        self.seek_slider.setRange(0, int(self.player.duration_sec))
        self._seek_setting = True
        self.seek_slider.setValue(0)
        self._seek_setting = False
        self.play_button.setText("Pause")
        
        self.player.start_threads()

    def on_model_changed(self, model_name):
        self.player.stop()
        self.player.model_name = model_name
        self.player.use_trt = model_name in V2_MODELS
        self.player.is_da3 = model_name in DA3_MODELS
        self.player.running = True
        
        self.stats_label.setText("Loading model...")
        threading.Thread(target=self._model_load_worker, daemon=True).start()

    def _model_load_worker(self):
        try:
            if self.player.use_trt:
                self.player._load_v2_trt_model()
            elif self.player.is_da3:
                self.player._load_da3_model()
            else:
                self.player._load_v2_pytorch_model()
        except Exception as e:
            print(f"[Error] Model load failed: {e}")
        QTimer.singleShot(0, self._on_model_ready)

    def _on_model_ready(self):
        self.stats_label.setText("")
        if self.player.video_path:
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
            self.playlist_panel.hide()
            self.playback_widget.hide()
            self.showFullScreen()
        else:
            if self.settings_toggle_btn.text() == "×":
                self.settings_widget.show()
            if self.playlist_toggle_btn.text() == "×":
                self.playlist_panel.show()
            self.playback_widget.show()
            self.showNormal()

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        from PySide6.QtWidgets import QStyle, QStyleOptionSlider
        if obj == self.seek_slider and event.type() == QEvent.MouseButtonPress:
            opt = QStyleOptionSlider()
            self.seek_slider.initStyleOption(opt)
            handle_pos = QStyle.sliderPositionFromValue(
                self.seek_slider.minimum(), self.seek_slider.maximum(),
                self.seek_slider.value(), self.seek_slider.width(), opt.upsideDown
            )
            click_pos = int(event.position().x())
            if abs(click_pos - handle_pos) > 12:
                value = QStyle.sliderValueFromPosition(
                    self.seek_slider.minimum(), self.seek_slider.maximum(),
                    click_pos, self.seek_slider.width(), opt.upsideDown
                )
                self._seek_setting = True
                self.seek_slider.setValue(int(value))
                self._seek_setting = False
                self._do_seek(value)
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()
        if mods == Qt.ControlModifier and key == Qt.Key_O:
            self.open_file()
        elif key == Qt.Key_Space:
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
        self._mark_config_dirty()
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
