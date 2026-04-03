#!/bin/bash
# ================================================================
# setup_cluster.sh — Deploy RLPD experiments to a SLURM cluster.
#
# What it does:
#   1. Downloads MuJoCo 210 if missing
#   2. Copies the rlpd/ library from upstream if missing
#   3. Creates a conda env with pinned, tested dependencies
#   4. Installs Adroit binary envs (mjrl, mj_envs, datasets)
#   5. Verifies imports and compilation
#
# Usage (on a GPU node, not login node):
#   bash setup_cluster.sh
#
# If cloned from GitHub, run from the repo directory.
# ================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
CONDA_ENV="${CONDA_ENV:-rlpd}"
MUJOCO_DIR="$HOME/.mujoco"

echo "============================================"
echo "RLPD Cluster Setup"
echo "  Repo:   $REPO_DIR"
echo "  Env:    conda:$CONDA_ENV"
echo "============================================"
echo ""

# --- 1. Prerequisites ---
echo "[1/6] Prerequisites..."

if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found." >&2
  echo "  Install miniconda: https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi
echo "  conda OK"

if ! command -v git &>/dev/null; then
  echo "ERROR: git not found" >&2; exit 1
fi

if ! command -v gcc &>/dev/null; then
  echo "ERROR: gcc not found. mujoco-py needs a C compiler." >&2
  echo "  Try: module load gcc" >&2
  exit 1
fi
echo "  gcc OK"

if command -v nvidia-smi &>/dev/null; then
  GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  echo "  GPU: $GPU_INFO"
else
  echo "  WARNING: No GPU detected. Run on a GPU node for full verification."
fi

# --- 2. MuJoCo ---
echo ""
echo "[2/6] MuJoCo..."

if [ -d "$MUJOCO_DIR/mujoco210" ]; then
  echo "  mujoco210 found"
else
  echo "  Downloading mujoco210..."
  mkdir -p "$MUJOCO_DIR"
  wget -q https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
    -O "$MUJOCO_DIR/mujoco210-linux-x86_64.tar.gz" || {
    echo "ERROR: MuJoCo download failed. Check network." >&2; exit 1
  }
  tar xzf "$MUJOCO_DIR/mujoco210-linux-x86_64.tar.gz" -C "$MUJOCO_DIR"
  rm -f "$MUJOCO_DIR/mujoco210-linux-x86_64.tar.gz"
  echo "  Installed to $MUJOCO_DIR/mujoco210"
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$MUJOCO_DIR/mujoco210/bin"
[ -d /usr/lib/nvidia ] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/nvidia"
[ -d /usr/lib64/nvidia ] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib64/nvidia"

# --- 3. RLPD library ---
echo ""
echo "[3/6] RLPD library..."

if [ -d "$REPO_DIR/rlpd" ]; then
  echo "  rlpd/ already exists"
else
  echo "  Cloning ikostrikov/rlpd (library only)..."
  UPSTREAM=$(mktemp -d)
  git clone -q https://github.com/ikostrikov/rlpd.git "$UPSTREAM" || {
    rm -rf "$UPSTREAM"
    echo "ERROR: git clone failed. Check network." >&2; exit 1
  }
  mkdir -p "$REPO_DIR"
  cp -r "$UPSTREAM/rlpd" "$REPO_DIR/rlpd"
  rm -rf "$UPSTREAM"
  echo "  Copied rlpd/ to $REPO_DIR"
fi

# --- 4. Conda environment + dependencies ---
echo ""
echo "[4/6] Conda environment..."

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list 2>/dev/null | grep -qw "^${CONDA_ENV}"; then
  echo "  Env '$CONDA_ENV' exists, activating..."
  conda activate "$CONDA_ENV"
  # Verify Python version
  PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
  case "$PY_VER" in
    3.10|3.11) echo "  Python $PY_VER OK" ;;
    *)
      echo "  WARNING: Python $PY_VER detected. This was tested on 3.10."
      echo "  If you see errors, recreate: conda env remove -n $CONDA_ENV && rerun setup."
      ;;
  esac
else
  echo "  Creating conda env (python 3.10)..."
  conda create -n "$CONDA_ENV" python=3.10 -y -q
  conda activate "$CONDA_ENV"
fi

# Verify activation
if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]; then
  echo "ERROR: conda activate failed" >&2; exit 1
fi

echo "  Installing build dependencies..."
conda install -y -q patchelf glew mesalib 2>/dev/null || \
  conda install -y -q -c conda-forge patchelf glew mesalib 2>/dev/null || \
  echo "  WARNING: conda build deps install failed — mujoco-py may not compile"

echo "  Installing numpy + Cython..."
pip install -q numpy==1.26.4 "Cython<3" || { echo "ERROR: numpy/Cython install failed" >&2; exit 1; }

echo "  Installing mujoco-py..."
pip install -q mujoco-py==2.1.2.14 || { echo "ERROR: mujoco-py install failed" >&2; exit 1; }

echo "  Compiling mujoco-py extensions (first import)..."
python -c "import mujoco_py" || {
  echo "ERROR: mujoco-py compilation failed." >&2
  echo "  Check: LD_LIBRARY_PATH includes mujoco210/bin and nvidia libs" >&2
  echo "  Check: patchelf and glew are installed (conda install patchelf glew mesalib)" >&2
  exit 1
}

echo "  Installing JAX (CUDA 12)..."
pip install -q "jax[cuda12]==0.4.30" || {
  echo "  JAX CUDA 12 failed, trying CUDA 11..."
  pip install -q "jax[cuda11_pip]==0.4.30" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html || {
    echo "  WARNING: GPU JAX failed. Installing CPU JAX (training will be very slow)."
    pip install -q "jax==0.4.30" "jaxlib==0.4.30" || { echo "ERROR: JAX install failed" >&2; exit 1; }
  }
}

echo "  Installing remaining dependencies..."
pip install -q \
  flax==0.8.5 \
  optax==0.2.3 \
  tensorflow-probability==0.23.0 \
  gym==0.23.1 \
  "dm-control==1.0.20" \
  mujoco==2.3.7 \
  ml-collections==0.1.1 \
  absl-py==2.1.0 \
  scipy==1.13.1 \
  tqdm==4.66.4 \
  wandb==0.17.5 \
  imageio==2.34.2 \
  "moviepy==1.0.3" \
  gdown \
  || { echo "ERROR: dependency install failed" >&2; exit 1; }

echo "  Installing d4rl..."
pip install -q "d4rl @ git+https://github.com/Farama-Foundation/d4rl@master" || \
  echo "  WARNING: d4rl install had issues (may still work)"

# --- 5. Adroit binary envs ---
echo ""
echo "[5/6] Adroit binary environments..."

if python -c "import mjrl" 2>/dev/null; then
  echo "  mjrl OK"
else
  echo "  Installing mjrl..."
  [ ! -d "$HOME/mjrl" ] && git clone -q https://github.com/aravindr93/mjrl "$HOME/mjrl"
  pip install -q -e "$HOME/mjrl"
fi

if python -c "import mj_envs" 2>/dev/null; then
  echo "  mj_envs OK"
else
  echo "  Installing mj_envs..."
  if [ ! -d "$HOME/mj_envs" ]; then
    git clone -q --recursive https://github.com/philipjball/mj_envs.git "$HOME/mj_envs"
    cd "$HOME/mj_envs" && git submodule update --remote 2>/dev/null || true
  fi
  pip install -q -e "$HOME/mj_envs" --no-deps
fi

if [ -d "$HOME/.datasets/awac-data" ] && ls "$HOME/.datasets/awac-data/"*.npy &>/dev/null; then
  echo "  Adroit datasets OK"
else
  echo "  Downloading Adroit datasets..."
  mkdir -p "$HOME/.datasets"
  gdown "https://drive.google.com/uc?id=1yUdJnGgYit94X_AvV6JJP5Y3Lx2JF30Y" \
    -O "$HOME/.datasets/awac_dext.zip" --fuzzy -q 2>/dev/null
  if [ -f "$HOME/.datasets/awac_dext.zip" ]; then
    unzip -qo "$HOME/.datasets/awac_dext.zip" -d "$HOME/.datasets/awac-data/"
    rm -f "$HOME/.datasets/awac_dext.zip"
    echo "  Datasets installed"
  else
    echo "  WARNING: Dataset download failed. Download manually from:"
    echo "    https://drive.google.com/file/d/1yUdJnGgYit94X_AvV6JJP5Y3Lx2JF30Y"
    echo "  Unzip into ~/.datasets/awac-data/"
  fi
fi

# --- 6. Verify ---
echo ""
echo "[6/6] Verifying..."

ERRORS=0
cd "$REPO_DIR"

python -c "import jax; print('  JAX', jax.__version__, '| devices:', jax.devices())" || {
  echo "  ERROR: JAX import failed"; ERRORS=$((ERRORS+1)); }

python -c "import mujoco_py; print('  mujoco_py OK')" || {
  echo "  ERROR: mujoco_py failed"; ERRORS=$((ERRORS+1)); }

python -c "
from rlpd.networks import Ensemble, MLP, StateActionValue, subsample_ensemble
from sac_learner_v2 import SACLearnerV2
print('  SACLearnerV2 OK')
" || { echo "  ERROR: SACLearnerV2 import failed"; ERRORS=$((ERRORS+1)); }

python -c "
import gym; import d4rl
from rlpd.data.binary_datasets import BinaryDataset
env = gym.make('pen-binary-v0')
obs = env.reset()
print('  pen-binary-v0 OK (obs shape:', obs.shape, ')')
" || { echo "  WARNING: pen-binary-v0 env test failed"; }

echo ""
echo "============================================"
if [ "$ERRORS" -eq 0 ]; then
  echo "SETUP COMPLETE"
else
  echo "SETUP COMPLETE WITH $ERRORS ERROR(S)"
fi
echo ""
echo "Next steps:"
echo "  cd $REPO_DIR"
echo "  bash submit_all.sh --dry-run"
echo "  bash submit_all.sh"
echo "============================================"
