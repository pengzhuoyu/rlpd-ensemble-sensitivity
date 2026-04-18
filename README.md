# RLPD Critic Ensemble Sensitivity

How do critic ensemble size, min-Q subsetting, and dropout rate affect online RL fine-tuning performance? This repo runs controlled experiments on the [RLPD](https://github.com/ikostrikov/rlpd) backbone across Adroit binary manipulation tasks.

Built on `SACLearnerV2`, a modified SAC agent with configurable:
- **Ensemble size** (num_qs): 2, 4, 6, 10 Q-heads
- **Min-Q subset** (num_min_qs): how many heads to min over for targets
- **Dropout** (DroQ-style): implicit ensembling via per-head stochastic masks
- **Bootstrap masks**: Bernoulli(0.5) per-head sample weighting
- **Independent targets**: per-head target Q (no shared min)
- **Critic reset**: re-initialize critic at a given step

## Quick start

```bash
git clone <this-repo> rlpd_experiments
scp -r rlpd_experiments/ user@cluster:~/
ssh user@cluster

# On a GPU node:
bash ~/rlpd_experiments/setup_cluster.sh

cd ~/rlpd_experiments
bash submit_all.sh --dry-run   # preview
bash submit_all.sh             # submit all 50 runs
bash check_progress.sh         # monitor
```

## Setup

**Requirements**: SLURM cluster with CUDA GPUs, conda, git.

`setup_cluster.sh` handles everything automatically:
- Clones upstream [ikostrikov/rlpd](https://github.com/ikostrikov/rlpd) (library only)
- Creates a conda env (`rlpd`, Python 3.10) with pinned dependencies
- Installs MuJoCo 210, d4rl, mjrl, mj_envs
- Downloads Adroit binary datasets
- Deploys experiment files

Run on a **GPU node** (not login node) so JAX can verify CUDA:

```bash
# Get an interactive GPU node (syntax varies by cluster):
salloc --gres=gpu:1 -c 4 --mem=16G -t 0-1   # most clusters
salloc -G a100:1 -c 4 --mem=16G -t 0-1       # ASU Sol

# Run setup:
bash ~/rlpd_experiments/setup_cluster.sh
```

If your cluster uses a different GPU request syntax, edit line 2 of `run.sh`.

See [requirements.txt](requirements.txt) for exact dependency versions.

## Running experiments

All experiments are defined in [experiments.txt](experiments.txt). Format:

```
env,seed,nqs,minqs,dropout,maxsteps
```

| Field | Values |
|-------|--------|
| env | pen-binary-v0, door-binary-v0 |
| seed | 0-4 |
| nqs | 2, 4, 6, 10 |
| minqs | 1, 2 |
| dropout | 0, 0.005, 0.01, 0.02, 0.05, 0.1 |
| maxsteps | 1000000 |

Lines prefixed with `DIAG` enable per-head Q-value diagnostics.

```bash
cd ~/rlpd_experiments

bash submit_all.sh                  # submit all
bash submit_all.sh --standard-only  # 40 standard runs only
bash submit_all.sh --diag-only      # 10 diagnostic runs only
bash submit_all.sh --retry          # re-submit only failed/missing
bash submit_all.sh --no-pair        # one run per GPU (no pairing)
```

Runs are automatically paired by ensemble size onto the same GPU to halve job count (50 runs -> ~28 jobs).

## Monitoring

```bash
bash check_progress.sh         # queue + results + errors
bash check_progress.sh --short # just counts
```

## Spectral Norm Switch

Spectral norm is controlled by `config.spec_norm_coef`. In
`experiments.txt`, use the optional seventh column:

```text
# env,seed,nqs,minqs,dropout,maxsteps,spec_norm_coef
halfcheetah-medium-v2,0,2,2,0,250000,0      # off
halfcheetah-medium-v2,0,2,2,0,250000,1.0    # on, per-layer coef = 1.0
```

Leaving the seventh column blank keeps the old behavior. Run directories
include `nosn` or `sn<coef>` so spectral-norm and non-spectral-norm runs do
not overwrite each other.

## Results

Each run writes to `results/<run_name>/`:

| File | Contents |
|------|----------|
| `online_log.csv` | Eval scores at every checkpoint |
| `summary.json` | Final score, peak score, wall time |
| `diagnostic.csv` | (DIAG only) Per-head Q-values, pairwise correlation, OOD overestimation, effective rank |

New smoothing diagnostics:

| Column | Meaning |
|--------|---------|
| `head_var` | Mean over diagnostic points of `Var_i[Q_i(s,a)]` |
| `grad_var` | Mean over diagnostic points of cross-head action-gradient variance |
| `sharpness_single` | Mean single-head perturbation sharpness |
| `sharpness_ens` / `roughness` | Ensemble-mean perturbation sharpness |
| `smooth_gain` | `sharpness_single - sharpness_ens` |
| `smooth_ratio` | `sharpness_single / sharpness_ens` |
| `single_head_grad_sharpness` | Mean over heads and diagnostic points of `||grad_a Q_i||^2` |
| `ensemble_grad_sharpness` | Mean over diagnostic points of `||grad_a mean_i Q_i||^2` |
| `grad_smooth_gain` | `single_head_grad_sharpness - ensemble_grad_sharpness` |
| `grad_smooth_ratio` | `single_head_grad_sharpness / ensemble_grad_sharpness` |

## Repo structure

```
rlpd_experiments/
  sac_learner_v2.py     # SAC agent with ensemble critics
  train_abc.py          # Standard training script
  train_diagnostic.py   # Training with Q-head diagnostics
  diagnostic.py         # Diagnostic metrics (pairwise corr, rank, OOD)
  configs/
    td_config.py        # Base hyperparameters
    sac_config.py       # SAC-specific config
    rlpd_config.py      # RLPD defaults (10 heads, min_qs=2, layer_norm)
  experiments.txt       # Run list (40 standard + 10 diagnostic)
  run.sh                # SLURM job script
  submit_all.sh         # Batch submission with pairing
  check_progress.sh     # Progress monitor
  setup_cluster.sh      # One-shot cluster deployment
  requirements.txt      # Pinned dependency versions
```

## Key design decisions

**Target critic matches subsample size.** The target critic's `apply_fn` is built with `num_min_qs` heads (matching the subsampled params it receives). Its `params` store all `num_qs` heads (EMA'd from the full critic), and `subsample_ensemble` slices them down before each forward pass. This ensures the vmap axis size matches the param count, which is critical when dropout splits RNG keys per head.

**Dropout masks are independent per head.** The upstream RLPD `Ensemble` uses `nn.vmap` with `split_rngs={"dropout": True}`, giving each Q-head a different dropout mask. This is essential for DroQ-style implicit ensembling.

**Eval uses the actor only.** During evaluation episodes, only `agent.eval_actions()` is called (deterministic actor mode). The critic is never queried, so dropout noise cannot corrupt eval scores.

**Diagnostic Q-values are deterministic.** `diagnostic.py` calls the critic with `training=False`, disabling dropout for reproducible measurements on fixed (s,a) pairs.

## License

Experiment code is provided as-is. The RLPD library (`rlpd/` directory) is from [ikostrikov/rlpd](https://github.com/ikostrikov/rlpd) — see that repo for its license.
