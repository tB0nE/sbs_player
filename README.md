# Nightfall Player

Convert any 2D video to side-by-side 3D in real time using AI depth estimation — a companion app for the Nightfall streaming platform.

Uses [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) via [TensorRT](https://developer.nvidia.com/tensorrt) for GPU-accelerated inference on NVIDIA GPUs.

https://github.com/user-attachments/assets/7b13c4a1-8f3e-4a9d-b5c2-1e0d8f6a3b2c

## Requirements

- **NVIDIA GPU** with CUDA 12+ drivers
- **Linux** (tested on Fedora/Bazzite, should work on Ubuntu/Arch)
- **Python 3.10+** and **pip**

## Quick Install

```bash
curl -fsSL https://raw.githubusercontent.com/tB0nE/sbs_player/master/install.sh | bash
```

This installs everything to `~/sbs_player/` — a dedicated Python environment, all dependencies, and your chosen AI model.

First launch builds the TensorRT engine for your GPU (takes 60-90s one-time, cached after).

## Usage

```bash
# Launch the GUI
~/sbs_player/sbs_player

# Play a video directly
~/sbs_player/sbs_player video.mp4

# Play with HEVC/H.265 support
~/sbs_player/sbs_player video.mkv
```

After install, find **SBS 3D Player** in your application menu. Double-click `.mp4`/`.mkv` files to open them directly.

## Controls

| Key | Action |
|---|---|
| `Space` | Play / Pause |
| `+` / `-` | Adjust 3D depth strength |
| `[` / `]` | Temporal smoothing |
| `c` / `v` | Convergence (focal plane) |
| `e` / `r` | Edge softness |
| `g` / `h` | Depth gamma |
| `t` / `y` | Post-warp sharpen |
| `f` | Toggle fullscreen |

## CLI Options

```
python sbs_player.py [video] [options]

  --model MODEL           Depth model: Small, Base, or Large (default: Large)
  --strength N            3D depth shift in pixels (default: 16)
  --inference-size N      Model input resolution (default: 518)
  --precision fp16|fp32   Inference precision (default: fp16)
  --no-trt                Disable TensorRT, use PyTorch instead
  --benchmark             Disable FPS cap
  --no-gui                OpenCV console mode instead of Qt GUI
```

## Supported Formats

- **Video:** MP4, MKV, AVI, MOV
- **Codecs:** H.264, H.265/HEVC, and any FFmpeg-supported format
- **Audio:** AAC, AC3, and any FFmpeg-supported codec

## How It Works

1. Each video frame is passed through Depth Anything V2 (ONNX → TensorRT engine)
2. The depth map guides a per-pixel horizontal shift, creating a synthetic right-eye view
3. Left (original) and right (synthesized) views are composited side-by-side
4. Audio plays in sync, with dynamic frame dropping to maintain A/V sync

## License

MIT
