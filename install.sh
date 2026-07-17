#!/usr/bin/env bash
set -euo pipefail

# Nightfall Player — Linux Installer
# Installs to ~/sbs_player/ with a dedicated Python venv
# Creates start menu entry and file associations

INSTALL_DIR="$HOME/sbs_player"
VENV_DIR="$INSTALL_DIR/venv"
CHECKPOINTS_DIR="$INSTALL_DIR/checkpoints"
APP_URL="https://raw.githubusercontent.com/tB0nE/sbs_player/master/sbs_player.py"
RELEASE_URL="https://github.com/tB0nE/sbs_player/releases/download/v1.0.0"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

MODEL_ARG="${1:-}"
if [ -n "$MODEL_ARG" ]; then
    case "$MODEL_ARG" in
        small|base|large|skip) ;;
        *) MODEL_ARG="" ;;
    esac
fi

banner() { echo -e "${CYAN}==>${NC} $*"; }
ok()     { echo -e "${GREEN}  ✓${NC} $*"; }
warn()   { echo -e "${YELLOW}  !${NC} $*"; }
err()    { echo -e "${RED}  ✗${NC} $*"; exit 1; }

# ── Preflight ──────────────────────────────────────────────
banner "Nightfall Player Installer"

if ! command -v python3 &>/dev/null; then
    err "python3 not found. Install Python 3.10+ first."
fi

PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
banner "Python $PYVER detected"

if ! command -v pip3 &>/dev/null && ! python3 -m pip --version &>/dev/null; then
    err "pip not found. Install python3-pip first."
fi

if ! command -v nvidia-smi &>/dev/null; then
    warn "nvidia-smi not found. SBS Player requires an NVIDIA GPU with CUDA drivers."
    warn "Install will continue but the app will not work without a supported GPU."
fi

# ── Create directories ─────────────────────────────────────
banner "Creating install directory: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR" "$CHECKPOINTS_DIR"

# ── Python virtual environment ─────────────────────────────
if [ -d "$VENV_DIR" ]; then
    banner "Existing venv found, reusing"
else
    banner "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"

banner "Upgrading pip"
"$PIP" install --upgrade pip --quiet

# ── Install dependencies ───────────────────────────────────
banner "Installing PyTorch (CUDA 12)..."
"$PIP" install "torch>=2.1" "torchvision>=0.16" --index-url https://download.pytorch.org/whl/cu128 --quiet

banner "Installing TensorRT..."
"$PIP" install "tensorrt>=10" "tensorrt-cu12>=10" "tensorrt-cu12-libs>=10" \
    "nvidia-cudnn-cu12>=9" "nvidia-cublas-cu12" "nvidia-cuda-runtime-cu12" \
    "nvidia-curand-cu12" "nvidia-cufft-cu12" "nvidia-cuda-nvrtc-cu12" \
    --quiet

banner "Installing application dependencies..."
"$PIP" install \
    "numpy>=1.26" \
    "opencv-python>=4.10" \
    "av>=12" \
    "PySide6>=6.5" \
    "sounddevice>=0.4" \
    "nvidia-ml-py>=12" \
    --quiet

banner "Installing optional dependencies (PyTorch/DA3 fallback)..."
"$PIP" install "transformers>=4.40" --quiet || warn "transformers install failed (--no-trt fallback won't work)"
"$PIP" install "depth_anything_3" --quiet 2>/dev/null || warn "DA3 model support not available (pip install depth_anything_3 manually if needed)"

# ── Download app code ──────────────────────────────────────
banner "Downloading sbs_player.py..."
curl -fsSL "$APP_URL" -o "$INSTALL_DIR/sbs_player.py" || err "Failed to download app code"
ok "App code downloaded"

# ── Download ONNX model ────────────────────────────────────
if [ -n "$MODEL_ARG" ]; then
    # Non-interactive mode (CLI arg or NIGHTFALL_MODEL env var)
    MODEL_CHOICE="$MODEL_ARG"
elif [ -n "${NIGHTFALL_MODEL:-}" ]; then
    MODEL_CHOICE="${NIGHTFALL_MODEL}"
elif [ -t 0 ]; then
    echo ""
    echo "Select model to download:"
    echo "  1) Small  (~50MB,  fastest)"
    echo "  2) Base   (~200MB, balanced, recommended)"
    echo "  3) Large  (~400MB, best quality)"
    echo "  4) Skip   (I already have models)"
    read -r -p "Choice [2]: " MODEL_CHOICE
    MODEL_CHOICE=${MODEL_CHOICE:-2}
else
    # Piped install, default to Base
    MODEL_CHOICE="2"
    echo "[Info] Non-interactive install, defaulting to Base model"
fi

case "$MODEL_CHOICE" in
    1) MODEL_NAME="Depth-Anything-V2-Small-hf"; MODEL_SIZE="50MB" ;;
    2) MODEL_NAME="Depth-Anything-V2-Base-hf";  MODEL_SIZE="200MB" ;;
    3) MODEL_NAME="Depth-Anything-V2-Large-hf"; MODEL_SIZE="400MB" ;;
    4) banner "Skipping model download" ;;
    *) err "Invalid choice" ;;
esac

if [ "$MODEL_CHOICE" != "4" ]; then
    MODEL_FILE="${MODEL_NAME}_518.onnx"
    MODEL_DEST="$CHECKPOINTS_DIR/$MODEL_FILE"

    if [ -f "$MODEL_DEST" ]; then
        banner "Model already downloaded: $MODEL_FILE"
    else
        banner "Downloading $MODEL_NAME ($MODEL_SIZE)..."
        curl -fsSL "$RELEASE_URL/$MODEL_FILE" -o "$MODEL_DEST" || {
            warn "Failed to download from release. You can manually place the ONNX file at:"
            warn "  $MODEL_DEST"
        }
        ok "Model downloaded"
    fi
fi

# ── Create launcher script ─────────────────────────────────
banner "Creating launcher script..."
cat > "$INSTALL_DIR/sbs_player" << LAUNCHER
#!/usr/bin/env bash
INSTALL_DIR="\$HOME/sbs_player"
PYVER=\$("\$INSTALL_DIR/venv/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
export LD_LIBRARY_PATH="\$INSTALL_DIR/venv/lib/python\${PYVER}/site-packages/tensorrt_libs:\$INSTALL_DIR/venv/lib/python\${PYVER}/site-packages/nvidia/cudnn/lib:\$INSTALL_DIR/venv/lib/python\${PYVER}/site-packages/nvidia/cuda_runtime/lib:\$INSTALL_DIR/venv/lib/python\${PYVER}/site-packages/nvidia/cublas/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
exec "\$INSTALL_DIR/venv/bin/python" "\$INSTALL_DIR/sbs_player.py" "\$@"
LAUNCHER

chmod +x "$INSTALL_DIR/sbs_player"
ok "Launcher created: $INSTALL_DIR/sbs_player"

# ── Register file associations ────────────────────────────
banner "Setting up file associations..."

xdg-mime default sbs_player.desktop \
    video/mp4 video/x-matroska video/x-msvideo \
    video/quicktime video/webm video/x-ms-wmv 2>/dev/null || \
    warn "xdg-mime not available. File associations may need manual setup."

# ── Create desktop entry ──────────────────────────────────
banner "Creating application menu entry..."

APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$APPS_DIR"

cat > "$APPS_DIR/sbs_player.desktop" << DESKTOP
[Desktop Entry]
Name=Nightfall Player
Comment=AI-powered 2D to 3D video player
Exec=$INSTALL_DIR/sbs_player %f
Icon=nightfall-player
Terminal=false
Type=Application
Categories=AudioVideo;Player;Video;
MimeType=video/mp4;video/x-matroska;video/x-msvideo;video/quicktime;video/x-ms-wmv;video/webm;
StartupNotify=true
DESKTOP

update-desktop-database "$APPS_DIR" 2>/dev/null || true
ok "Desktop entry created"

# ── Done ──────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Nightfall Player installed successfully!    ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo "  Location:   $INSTALL_DIR"
echo "  Launcher:   $INSTALL_DIR/sbs_player"
echo "  Config:     ~/.config/sbs_player/config.json"
echo "  Models:     $CHECKPOINTS_DIR"
echo ""
echo "  Usage:"
echo "    $INSTALL_DIR/sbs_player video.mkv    Play a file"
echo "    $INSTALL_DIR/sbs_player              Launch without video"
echo ""
echo "  Or find 'Nightfall Player' in your application menu."
echo "  Double-click .mkv/.mp4 files to open them in the player."
echo ""
echo -e "  ${YELLOW}First run will build the TRT engine for your GPU (60-90s one-time).${NC}"
echo ""
