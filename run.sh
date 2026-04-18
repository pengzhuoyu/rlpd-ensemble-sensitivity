#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=rlpd
#SBATCH --output=rlpd_%j.txt
#SBATCH --error=rlpd_%j_err.txt

# ================================================================
# SLURM job script for RLPD experiments.
# Called by submit_all.sh — you don't run this directly.
#
# Cluster-specific:
#   If your cluster uses -G instead of --gres, edit line 2.
#   Add --partition/--account to the SBATCH header as needed.
# ================================================================
set -e

# --- Resolve repo directory ---
# SLURM_SUBMIT_DIR is set by sbatch to the directory where sbatch was called.
# Fall back to script location for interactive use.
if [ -n "$SLURM_SUBMIT_DIR" ]; then
  REPO="$SLURM_SUBMIT_DIR"
else
  REPO="$(cd "$(dirname "$0")" && pwd)"
fi
cd "$REPO"

# --- Activate conda ---
CONDA_BASE="$(conda info --base 2>/dev/null || echo "")"
if [ -z "$CONDA_BASE" ]; then
  for d in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
    [ -f "$d/etc/profile.d/conda.sh" ] && { CONDA_BASE="$d"; break; }
  done
fi
if [ -z "$CONDA_BASE" ]; then
  echo "ERROR: Cannot find conda" >&2; exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate rlpd

# Verify activation
if [ "$CONDA_DEFAULT_ENV" != "rlpd" ]; then
  echo "ERROR: conda activate rlpd failed (active env: ${CONDA_DEFAULT_ENV:-none})" >&2
  exit 1
fi

# --- Environment variables ---
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin"
[ -d /usr/lib/nvidia ] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/nvidia"
[ -d /usr/lib64/nvidia ] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib64/nvidia"

# --- Run experiments ---
TRAIN_SCRIPT="train_abc.py"
if [ "${DIAG:-0}" = "1" ]; then
  TRAIN_SCRIPT="train_diagnostic.py"
fi

if [ -z "$EXPERIMENTS" ]; then
  echo "ERROR: EXPERIMENTS variable is empty. Nothing to run." >&2
  exit 1
fi

IFS='|' read -ra RUNS <<< "$EXPERIMENTS"
for run in "${RUNS[@]}"; do
  IFS=',' read -r env seed nqs minqs dropout maxsteps specnorm <<< "$run"

  DROP_FLAG=""
  if [ "$dropout" != "0" ] && [ -n "$dropout" ]; then
    DROP_FLAG="--config.critic_dropout_rate=$dropout"
  fi
  SPEC_FLAG=""
  if [ "$specnorm" != "0" ] && [ -n "$specnorm" ]; then
    SPEC_FLAG="--config.spec_norm_coef=$specnorm"
  fi

  echo ""
  echo "=========================================="
  echo "RUN: env=$env seed=$seed nq=$nqs mq=$minqs drop=$dropout specnorm=${specnorm:-0} steps=$maxsteps diag=${DIAG:-0} — $(date)"
  echo "=========================================="

  python "$TRAIN_SCRIPT" \
    --env_name="$env" \
    --max_steps="$maxsteps" \
    --config=configs/rlpd_config.py \
    --config.backup_entropy=False \
    --config.hidden_dims="(256, 256, 256)" \
    --config.num_min_qs="$minqs" \
    --config.num_qs="$nqs" \
    --config.critic_layer_norm=True \
    $DROP_FLAG \
    $SPEC_FLAG \
    --bootstrap_mask=False \
    --independent_targets=False \
    --critic_reset_step=0 \
    --seed="$seed" \
    --results_dir=results

  echo "FINISHED — $(date)"
done
echo "=== ALL DONE $(date) ==="
