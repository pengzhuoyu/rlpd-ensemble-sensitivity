#!/bin/bash
# ================================================================
# submit_all.sh — Submit all experiments from experiments.txt
#
# Usage:
#   bash submit_all.sh                  # submit everything
#   bash submit_all.sh --dry-run        # preview without submitting
#   bash submit_all.sh --diag-only      # only DIAG runs
#   bash submit_all.sh --standard-only  # only standard runs
#   bash submit_all.sh --retry          # re-submit failed/missing runs
#   bash submit_all.sh --no-pair        # one run per job (no pairing)
#
# Must be run from the repo directory.
# ================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENTS_FILE="${SCRIPT_DIR}/experiments.txt"
RESULTS_DIR="results"
DRY_RUN=false
DIAG_ONLY=false
STANDARD_ONLY=false
RETRY_MODE=false
NO_PAIR=false

for arg in "$@"; do
  case "$arg" in
    --dry-run)       DRY_RUN=true ;;
    --diag-only)     DIAG_ONLY=true ;;
    --standard-only) STANDARD_ONLY=true ;;
    --retry)         RETRY_MODE=true ;;
    --no-pair)       NO_PAIR=true ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

if [ ! -f "$EXPERIMENTS_FILE" ]; then
  echo "ERROR: $EXPERIMENTS_FILE not found" >&2; exit 1
fi

if ! command -v sbatch &>/dev/null && [ "$DRY_RUN" = false ]; then
  echo "ERROR: sbatch not found. Use --dry-run to preview locally." >&2; exit 1
fi

if [ ! -f "run.sh" ]; then
  echo "ERROR: run.sh not found. Run from the repo directory." >&2; exit 1
fi

# --- Parse experiments.txt ---
STANDARD_RUNS=""
DIAG_RUNS=""
N_STANDARD=0
N_DIAG=0

while IFS= read -r line; do
  line="${line%%#*}"
  line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [ -z "$line" ] && continue
  if [ "${line%% *}" = "DIAG" ]; then
    run="${line#DIAG }"
    DIAG_RUNS="${DIAG_RUNS:+$DIAG_RUNS
}$run"
    N_DIAG=$((N_DIAG + 1))
  else
    STANDARD_RUNS="${STANDARD_RUNS:+$STANDARD_RUNS
}$line"
    N_STANDARD=$((N_STANDARD + 1))
  fi
done < "$EXPERIMENTS_FILE"

echo "Parsed experiments.txt: $N_STANDARD standard + $N_DIAG diagnostic = $((N_STANDARD + N_DIAG)) total"

# --- Check if a run already completed (for --retry) ---
# Matches the directory naming used by train_abc.py / train_diagnostic.py:
#   {env}_nq{nqs}_mq{minqs}_{drop_tag}_{spec_tag}_s{seed}     (standard)
#   {env}_diag_nq{nqs}_mq{minqs}_{drop_tag}_{spec_tag}_s{seed} (diagnostic)
already_done() {
  local cfg="$1" diag="$2"
  IFS=',' read -r env seed nqs minqs dropout maxsteps specnorm <<< "$cfg"
  local drop_tag="nodrop"
  if [ "$dropout" != "0" ] && [ -n "$dropout" ]; then
    drop_tag="drop${dropout}"
  fi
  local spec_tag="nosn"
  if [ "$specnorm" != "0" ] && [ -n "$specnorm" ]; then
    spec_tag="sn${specnorm}"
  fi
  local prefix=""
  [ "$diag" = "1" ] && prefix="diag_"
  local dir="$RESULTS_DIR/${env}_${prefix}nq${nqs}_mq${minqs}_${drop_tag}_${spec_tag}_s${seed}"
  [ -f "$dir/summary.json" ]
}

# --- Submit one job ---
submit_count=0
skip_count=0

submit_batch() {
  local experiments="$1" diag="$2"
  if [ "$DRY_RUN" = true ]; then
    [ "$diag" = "1" ] \
      && echo "  [DRY RUN] DIAG=1 EXPERIMENTS=\"$experiments\" sbatch run.sh" \
      || echo "  [DRY RUN] EXPERIMENTS=\"$experiments\" sbatch run.sh"
    submit_count=$((submit_count + 1))
    return
  fi

  # Export as shell variables so --export=ALL carries them into SLURM.
  # Cannot use --export=EXPERIMENTS="..." because SLURM splits on commas,
  # which destroys the comma-separated experiment format.
  export EXPERIMENTS="$experiments"
  export DIAG="$diag"
  local output
  output=$(sbatch --export=ALL run.sh 2>&1)

  if echo "$output" | grep -q "Submitted"; then
    echo "  $output"
    submit_count=$((submit_count + 1))
  else
    echo "  ERROR: $output" >&2
  fi
}

# --- Pair runs by nqs (similar runtime), preferring same env ---
pair_runs() {
  local runs_file used_file
  runs_file=$(mktemp); cat > "$runs_file"
  used_file=$(mktemp); touch "$used_file"
  local n; n=$(wc -l < "$runs_file" | tr -d ' ')

  local i=1
  while [ "$i" -le "$n" ]; do
    grep -q "^${i}$" "$used_file" 2>/dev/null && { i=$((i+1)); continue; }
    local run_i; run_i=$(sed -n "${i}p" "$runs_file")
    local env_i nqs_i; IFS=',' read -r env_i _ nqs_i _ _ _ _ <<< "$run_i"
    local best=0 best_same_env=false

    local j=$((i+1))
    while [ "$j" -le "$n" ]; do
      grep -q "^${j}$" "$used_file" 2>/dev/null && { j=$((j+1)); continue; }
      local run_j; run_j=$(sed -n "${j}p" "$runs_file")
      local env_j nqs_j; IFS=',' read -r env_j _ nqs_j _ _ _ _ <<< "$run_j"
      if [ "$nqs_i" = "$nqs_j" ]; then
        [ "$best" = "0" ] && best=$j
        [ "$env_i" = "$env_j" ] && [ "$best_same_env" = false ] && { best=$j; best_same_env=true; }
      fi
      j=$((j+1))
    done

    if [ "$best" != "0" ]; then
      echo "${run_i}|$(sed -n "${best}p" "$runs_file")"
      echo -e "$i\n$best" >> "$used_file"
    else
      echo "$run_i"; echo "$i" >> "$used_file"
    fi
    i=$((i+1))
  done
  rm -f "$runs_file" "$used_file"
}

# --- Process a batch ---
process_runs() {
  local runs_str="$1" diag="$2"
  [ -z "$runs_str" ] && { echo "  Nothing to submit."; return; }

  local filtered="" local_skip=0
  while IFS= read -r run; do
    [ -z "$run" ] && continue
    if [ "$RETRY_MODE" = true ] && already_done "$run" "$diag"; then
      local_skip=$((local_skip + 1)); continue
    fi
    filtered="${filtered:+$filtered
}$run"
  done <<< "$runs_str"

  [ "$local_skip" -gt 0 ] && { echo "  Skipping $local_skip completed runs"; skip_count=$((skip_count + local_skip)); }
  [ -z "$filtered" ] && { echo "  Nothing to submit (all done)."; return; }

  local n_filtered; n_filtered=$(echo "$filtered" | wc -l | tr -d ' ')
  local jobs
  [ "$NO_PAIR" = true ] && jobs="$filtered" || jobs=$(echo "$filtered" | pair_runs)
  local n_jobs; n_jobs=$(echo "$jobs" | wc -l | tr -d ' ')
  echo "  Submitting $n_jobs jobs ($n_filtered runs)..."
  echo ""
  while IFS= read -r job; do
    [ -z "$job" ] && continue
    submit_batch "$job" "$diag"
  done <<< "$jobs"
}

# --- Submit ---
if [ "$DIAG_ONLY" = false ]; then
  echo ""; echo "=========================================="; echo "STANDARD RUNS ($N_STANDARD total)"; echo "=========================================="
  process_runs "$STANDARD_RUNS" "0"
fi
if [ "$STANDARD_ONLY" = false ]; then
  echo ""; echo "=========================================="; echo "DIAG RUNS ($N_DIAG total)"; echo "=========================================="
  process_runs "$DIAG_RUNS" "1"
fi

echo ""; echo "=========================================="
echo "DONE — $submit_count jobs submitted"
[ "$skip_count" -gt 0 ] && echo "  $skip_count runs skipped (already done)"
echo ""; echo "Monitor: squeue -u \$USER"; echo "         bash check_progress.sh"
echo "=========================================="
