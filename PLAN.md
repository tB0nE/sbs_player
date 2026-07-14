# SBS Player v2 — Improvement Plan

## Overview

Transform the current prototype into a polished, native-Linux 2D-to-3D SBS video player with temporal stability, audio, and a proper GUI.

---

## Phase 1: Temporal Smoothing (Depth Stabilization)

### Problem
Single-image depth estimators like Depth Anything V2 process each frame independently. This causes visible flickering/jitter in the depth map between consecutive frames, which translates to uncomfortable flickering in the 3D stereo output.

### Solution: Exponential Moving Average (EMA)

Apply a per-pixel temporal filter that blends the current raw depth prediction with the previous smoothed depth map:

```
D_t = alpha * P_t + (1 - alpha) * D_{t-1}
```

- `D_t` = smoothed depth at time t
- `P_t` = raw depth prediction for current frame
- `D_{t-1}` = smoothed depth from previous frame
- `alpha` = smoothing factor (0.0 to 1.0)

### Implementation Details

- Maintain a persistent `torch.Tensor` on GPU for `D_{t-1}` (same shape as output depth map)
- After each frame's raw depth inference, compute EMA on GPU before normalization
- Reset `D_{t-1}` on seek/loop to avoid stale ghosting from previous scene
- Default `alpha = 0.3` (70% weight on previous frame = strong smoothing)
- Runtime adjustable via `[` and `]` keys
- Display current alpha value on HUD

### Alpha Values Guide

| alpha | Behavior |
|-------|----------|
| 1.0   | No smoothing (current behavior) |
| 0.5   | Light smoothing, very responsive |
| 0.3   | Moderate smoothing (recommended default) |
| 0.1   | Heavy smoothing, noticeable lag on fast motion |

### Future Upgrade: Motion-Corrected EMA

Basic EMA causes ghosting when objects move because it blends pixel values at fixed coordinates rather than tracking objects. The upgrade path:

1. Compute dense optical flow between frame t-1 and t using `cv2.cuda_FarnebackOpticalFlow` (GPU-accelerated)
2. Warp `D_{t-1}` using the flow field to align it with the current frame's geometry
3. Then apply EMA: `D_t = alpha * P_t + (1 - alpha) * warp(D_{t-1}, flow)`

This is significantly more complex and may add 3-5ms per frame, so we start with basic EMA and upgrade later if ghosting is problematic.

---

## Phase 2: Enhanced Stereo Parameters

### Problem
Our current stereo generation is minimal — we only have `max_shift` (pixel displacement). Oku3D/Video-Stereo-Converter exposes 7 parameters that significantly improve 3D quality and comfort.

### New Parameters

#### 2.1 Convergence (Focal Plane Shift)
- **Current**: We shift pixels only in one direction (right eye = left shift), meaning everything appears to recede behind the screen
- **New**: `convergence` shifts the zero-parallax plane. Positive values make objects pop out of the screen, negative makes them sink in
- **Implementation**: `map_x_warped = map_x - (shift + convergence_pixels)` where `convergence_pixels = convergence * (w / 100)`
- **Default**: -10.0 (slight sink-in, comfortable for extended viewing)
- **Range**: -50.0 to +50.0
- **Key**: `c`/`v` to decrease/increase

#### 2.2 Edge Softness
- **Problem**: Hard depth edges create visible tearing/artifacts at object boundaries during warping
- **Solution**: Apply Gaussian blur to the depth map before warping, controlled by `edge_softness` parameter
- **Implementation**: `cv2.GaussianBlur(depth, (0, 0), sigmaX=edge_softness/10.0)` — sigma scales with parameter
- **Default**: 20.0
- **Range**: 0.0 to 30.0
- **Key**: `e`/`r` to decrease/increase

#### 2.3 Depth Gamma
- **Problem**: Linear depth maps often have too much detail in far distances and not enough in near objects
- **Solution**: Apply gamma correction to compress/expand depth ranges: `depth_corrected = depth ^ gamma`
- **Effect**: gamma < 1.0 compresses far distances (makes near objects pop more), gamma > 1.0 expands far distances
- **Default**: 0.2
- **Range**: 0.1 to 1.0
- **Key**: `g`/`h` to decrease/increase

#### 2.4 Sharpen (Post-Warp Unsharp Mask)
- **Problem**: The warp + resize pipeline softens the image
- **Solution**: Apply unsharp mask after warping to recover sharpness
- **Implementation**: `sharpened = original + amount * (original - cv2.GaussianBlur(original, (0,0), sigmaX=1.0))`
- **Default**: 14.0
- **Range**: 0.0 to 30.0
- **Key**: `t`/`y` to decrease/increase

#### 2.5 Artifact Smoothing
- **Problem**: At disocclusion boundaries (where warping reveals previously hidden areas), harsh edges appear
- **Solution**: Additional Gaussian blur on the warped right-eye image, focused on the disocclusion mask region
- **Default**: 1.0
- **Range**: 0.0 to 5.0
- **Key**: `u`/`i` to decrease/increase

#### 2.6 Super-Sampling (Internal Upscale)
- **Problem**: Warping at native resolution can miss sub-pixel details
- **Solution**: Upscale the frame by `super_sampling` factor before warping, then downscale back to output resolution
- **Trade-off**: Higher quality but much more GPU work (2x super-sampling = 4x pixels)
- **Default**: 1.0 (disabled for real-time performance)
- **Range**: 1.0 to 3.0
- **Key**: `j`/`k` to decrease/increase

### Parameter Storage

All parameters stored in a `StereoConfig` dataclass and persisted to `~/.config/sbs_player/config.json`. Runtime changes auto-save on exit.

---

## Phase 3: Audio Support

### Problem
Currently the player is video-only. Watching a movie without audio is obviously unusable.

### Approach: ffpyplayer

Use **ffpyplayer** (Python bindings for libffplayer/FFmpeg) for audio decoding and output. This is the most compatible approach for Linux audio.

### Implementation Details

1. **Audio Reader Thread**: Separate thread that reads audio packets from the video file using ffpyplayer or PyAV
2. **Audio Output**: Use `simpleaudio` or `pyaudio` to stream decoded PCM audio to the sound card
3. **A/V Sync Strategy**:
   - Track video PTS (presentation timestamp) from OpenCV's `cap.get(cv2.CAP_PROP_POS_MSEC)`
   - Track audio PTS from the audio decoder
   - If audio leads video: sleep the audio thread briefly
   - If video leads audio: skip video frames to catch up
   - Target sync tolerance: ±20ms (imperceptible to humans)
4. **Volume Control**: Multiply PCM samples by volume factor before writing to audio device
   - Default: 100%
   - Keys: `9`/`0` to decrease/increase volume
   - `m` to toggle mute

### Alternative: PyAV

If ffpyplayer proves problematic, **PyAV** (Pythonic FFmpeg bindings) can decode both audio and video streams. We would:
- Use PyAV for video decoding (replace OpenCV's VideoCapture — also gets us proper PTS)
- Use PyAV for audio decoding
- Use sounddevice or pyaudio for audio output

This is a bigger refactor but gives us better control over timestamps and seeking.

### Fallback: External Audio

Simplest approach: launch `ffplay` or `mpv` in audio-only mode alongside our player, with a manual sync offset. Not ideal but works as a quick proof-of-concept.

---

## Phase 4: Qt GUI

### Problem
The current OpenCV-based UI is keyboard-only, has no sliders, no file picker, no seek bar, and the HUD text is crude. It's fine for development but painful for actual use.

### Framework: PyQt6 or PySide6

Both are mature, well-documented, and support GPU-accelerated rendering. PySide6 is the official Qt for Python binding and has a more permissive license (LGPL). PyQt6 requires a commercial license for closed-source use. We'll use **PySide6**.

### Window Layout

```
+--------------------------------------------------+
|  Menu Bar: File | View | Settings | Help         |
+--------------------------------------------------+
|                                                   |
|                                                   |
|              Video Display Area                   |
|           (SBS output, fullscreen-capable)        |
|                                                   |
|                                                   |
+--------------------------------------------------+
|  Seek Bar: [=========|========================]  |
|  00:12:34 / 01:45:00                              |
+--------------------------------------------------+
|  Transport:  [|<] [<<] [>] [>>] [>|] [🔇]       |
+--------------------------------------------------+
|  Stereo Controls (collapsible panel):             |
|                                                   |
|  Disparity:    [=========|===========]  50        |
|  Convergence:  [=====|===============]  -10       |
|  Edge Soft:    [=========|===========]  20        |
|  Depth Gamma:  [|====================]  0.2       |
|  Sharpen:      [=============|======]  14        |
|  Art. Smooth:  [|====================]  1.0       |
|  Temporal:     [====|=================]  0.3      |
|  Super-sample: [|====================]  1.0       |
|                                                   |
|  Model: [Depth-Anything-V2-Large-hf ▼]           |
|  Precision: [FP16 ▼]  [x] TensorRT               |
+--------------------------------------------------+
|  Status: FPS: 24 | Infer: 23ms | GPU: 45% | ...  |
+--------------------------------------------------+
```

### Video Display Widget

- Custom `QWidget` that receives numpy arrays and renders them via `QImage` → `QPainter`
- For performance: use `QOpenGLWidget` with texture upload from GPU memory (avoid CPU round-trip)
- Fullscreen mode: hide all controls, video fills screen, mouse hover at bottom shows controls overlay
- Mouse click on video: toggle play/pause
- Double-click: toggle fullscreen

### Control Panel

- All stereo parameters as `QSlider` widgets with labels showing current values
- Real-time update: slider changes immediately affect the next processed frame
- Collapsible: can be hidden to maximize video area
- Preset buttons: "Mild 3D", "Medium 3D", "Strong 3D" that set parameter combinations

### Transport Controls

- Play/Pause button
- Seek bar (QSlider mapped to video position in milliseconds)
- Step forward/backward (single frame)
- Speed control (0.5x, 1.0x, 1.5x, 2.0x)
- Volume slider + mute toggle

### File Handling

- File → Open: `QFileDialog` to select video file
- Drag-and-drop onto window to open
- Recent files menu (persisted in config)
- Auto-detect display resolution on startup

### Keyboard Shortcuts (Preserved)

All current shortcuts work plus new ones:
- `Space`: Play/Pause
- `f`: Fullscreen
- `q`/ESC: Quit
- `m`: Mute
- `9`/`0`: Volume down/up
- `[`/`]`: Temporal smoothing down/up
- Arrow keys: Seek ±5 seconds
- `c`/`v`: Convergence down/up
- `s`: Save current settings to config

---

## Phase 5: Polish & UX

### 5.1 Seeking Support

- Current limitation: video reader thread reads sequentially; seeking requires resetting `CAP_PROP_POS_FRAMES`
- Implementation:
  - Add a `seek_target` variable (thread-safe via `threading.Event`)
  - When user seeks: set `seek_target`, pause the depth+warp threads, flush queues, seek the VideoCapture, resume
  - Clear `D_{t-1}` (temporal smoothing buffer) on seek to prevent ghosting from previous position
  - Use PyAV instead of OpenCV for better seek performance (optional, Phase 3 dependency)

### 5.2 Subtitle Support

- Extract subtitle streams from video using PyAV or `ffprobe`
- Render SRT/ASS subtitles onto both left and right halves of the SBS frame
- Position subtitles at the screen plane (zero parallax) for comfortable reading
- Toggle subtitles on/off with `b` key

### 5.3 Configuration Persistence

- Config file: `~/.config/sbs_player/config.json`
- Save on exit, load on startup
- Stores:
  - All stereo parameters (disparity, convergence, edge softness, etc.)
  - Window size and position
  - Last opened file
  - Recent files list
  - Model selection
  - Volume level
  - Fullscreen state

### 5.4 Drag-and-Drop

- Accept video file drops on the Qt window
- Validate file format before loading
- Show loading indicator while model initializes

### 5.5 Error Handling

- Graceful handling of: missing GPU, missing model files, corrupted video, unsupported codecs
- User-friendly error dialogs instead of console errors
- Auto-fallback: if TensorRT fails, fall back to PyTorch with a notification

---

## Implementation Order & Estimates

| Phase | Description | Effort | Priority |
|-------|-------------|--------|----------|
| 1 | Temporal Smoothing (EMA) | ~1 hour | Critical |
| 2 | Enhanced Stereo Parameters | ~2 hours | High |
| 3 | Audio Support | ~2-3 hours | High |
| 4 | Qt GUI | ~4-6 hours | Medium |
| 5 | Polish & UX | ~2-3 hours | Low |

Phases 1 and 2 can be done immediately against the current codebase. Phase 3 may benefit from switching to PyAV for video decoding (which also helps with seeking in Phase 5). Phase 4 is the largest effort but also the most transformative for usability.

---

## Technical Dependencies

| Library | Purpose | Phase |
|---------|---------|-------|
| PySide6 | Qt GUI framework | 4 |
| PyAV | Video/audio decoding, seeking, subtitles | 3, 5 |
| sounddevice or pyaudio | Audio output | 3 |
| ffpyplayer | Alternative audio decoding/output | 3 |

All are pip-installable and Linux-compatible.
