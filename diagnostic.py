"""diagnostic.py — Multi-head Q diagnostics for RLPD ablation runs.

Metrics computed on a fixed buffer of (s, a) pairs:
  - Pairwise correlation, |Qi - Qj|, ensemble std (head diversity)
  - q_mean_expert / q_mean_random / overestimation (OOD gap)
  - Effective rank, condition number (representation quality)
  - Roughness: variance of Q under small action perturbations
  - Mask variance: variance of Q across dropout RNG samples (DroQ-style)
"""
import csv
import os
import numpy as np
import jax
import jax.numpy as jnp
from itertools import combinations


def setup_diag_buffer(ds, env, n=1000, seed=42):
    """Sample a fixed set of (s, a) pairs from the offline dataset."""
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(ds), size=n)
    obs = ds.dataset_dict["observations"][idx]
    act = ds.dataset_dict["actions"][idx]
    act_rand = np.stack([env.action_space.sample() for _ in range(n)])
    return {
        "obs": jnp.array(obs, dtype=jnp.float32),
        "act_expert": jnp.array(act, dtype=jnp.float32),
        "act_random": jnp.array(act_rand, dtype=jnp.float32),
    }


def _get_qs(agent, obs, actions):
    """Deterministic Q-values for all heads. Used for non-dropout diagnostics."""
    key = jax.random.PRNGKey(0)
    return np.array(agent.critic.apply_fn(
        {"params": agent.critic.params}, obs, actions, False,
        rngs={"dropout": key}))


def _pairwise_stats(qs):
    """Mean / max / min of pairwise |Qi - Qj| and corr across heads.
    qs: [num_qs, n_samples]."""
    num_qs = qs.shape[0]
    if num_qs < 2:
        return 0.0, 1.0, 1.0, 1.0, 0.0

    diffs = []
    corrs = []
    for i, j in combinations(range(num_qs), 2):
        diffs.append(float(np.mean(np.abs(qs[i] - qs[j]))))
        c = np.corrcoef(qs[i], qs[j])[0, 1]
        if np.isnan(c):
            c = 1.0
        corrs.append(float(c))

    mean_diff = float(np.mean(diffs))
    mean_corr = float(np.mean(corrs))
    max_corr = float(np.max(corrs))
    min_corr = float(np.min(corrs))
    std_diff = float(np.std(diffs)) if len(diffs) > 1 else 0.0
    return mean_diff, mean_corr, max_corr, min_corr, std_diff


def _compute_rank(agent, obs, act, input_dim):
    """Jacobian-based effective rank. EXPENSIVE — call sparingly."""
    def q_head0(sa):
        o = sa[:input_dim]
        a = sa[input_dim:]
        key = jax.random.PRNGKey(0)
        qs = agent.critic.apply_fn(
            {"params": agent.critic.params},
            o[None], a[None], False,
            rngs={"dropout": key})
        return qs[0, 0]

    n_rank = min(200, obs.shape[0])
    sa = jnp.concatenate([obs[:n_rank], act[:n_rank]], axis=-1)
    jac_fn = jax.vmap(jax.grad(q_head0))
    jacs = np.array(jac_fn(sa))
    s = np.linalg.svd(jacs, compute_uv=False)
    s = np.maximum(s, 1e-10)
    s_norm = s / s.sum()
    eff_rank = float(np.exp(-np.sum(s_norm * np.log(s_norm))))
    cond_number = float(s[0] / s[min(len(s) - 1, 49)])
    return eff_rank, cond_number


def _compute_roughness(agent, obs, act, n_perturb=100, sigma=0.05, seed=42):
    """Roughness of Q surface around (s, a) pairs.

    For each (s, a), sample n_perturb actions a' = a + eps, eps ~ N(0, sigma^2).
    Compute Q(s, a') for each, then return mean over states of var over perturbations.
    Uses training=False (deterministic, no dropout) for reproducibility.
    """
    key = jax.random.PRNGKey(seed)
    eps = jax.random.normal(key, (n_perturb,) + act.shape) * sigma
    act_pert = act[None] + eps  # [n_perturb, n_samples, act_dim]

    def single(a_batch):
        qs = agent.critic.apply_fn(
            {"params": agent.critic.params}, obs, a_batch, False,
            rngs={"dropout": jax.random.PRNGKey(0)})
        return qs.mean(axis=0)  # mean over heads -> [n_samples]

    q_perturb = jax.vmap(single)(act_pert)  # [n_perturb, n_samples]
    var_per_state = np.array(q_perturb).var(axis=0)  # [n_samples]
    return float(var_per_state.mean())


def _compute_mask_var(agent, obs, act, n_samples=10, seed=123):
    """Variance of Q across dropout RNG samples (training=True).

    Measures the implicit-ensemble diversity from DroQ-style dropout.
    Returns 0 for runs without dropout (deterministic forward pass).
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), n_samples)

    def single(k):
        qs = agent.critic.apply_fn(
            {"params": agent.critic.params}, obs, act, True,
            rngs={"dropout": k})
        return qs.mean(axis=0)  # mean over heads -> [n_samples]

    q_samples = jax.vmap(single)(keys)  # [n_samples, n_obs]
    return float(np.array(q_samples).var(axis=0).mean())


def compute_roughness_only(agent, diag_buf):
    """Lightweight roughness computation for use by train_abc.py."""
    return _compute_roughness(agent, diag_buf["obs"], diag_buf["act_expert"])


def run_diagnostic(agent, diag_buf, step, diag_path):
    obs = diag_buf["obs"]
    act_ex = diag_buf["act_expert"]
    act_rnd = diag_buf["act_random"]

    # 1. Per-head Q on expert actions — pairwise stats
    qs_ex = _get_qs(agent, obs, act_ex)
    num_qs = qs_ex.shape[0]
    mean_diff, mean_corr, max_corr, min_corr, std_diff = _pairwise_stats(qs_ex)

    head_means = [float(np.mean(qs_ex[k])) for k in range(num_qs)]
    q_mean_expert = float(np.mean(qs_ex))

    # 2. OOD overestimation
    qs_rnd = _get_qs(agent, obs, act_rnd)
    q_mean_random = float(np.mean(qs_rnd))
    overestimation = q_mean_random - q_mean_expert

    # 3. Roughness — variance under action perturbation (every diag step)
    try:
        roughness = _compute_roughness(agent, obs, act_ex)
    except Exception:
        roughness = -1.0

    # 4. Mask variance — variance under dropout RNG sampling (every diag step)
    try:
        mask_var = _compute_mask_var(agent, obs, act_ex)
    except Exception:
        mask_var = -1.0

    # 5. Feature rank — EXPENSIVE, only every 50k
    eff_rank = -1.0
    cond_number = -1.0
    if step % 50000 == 0:
        try:
            input_dim = obs.shape[-1]
            eff_rank, cond_number = _compute_rank(
                agent, obs, act_ex, input_dim)
        except Exception:
            pass

    # 6. Q-value magnitude
    q_max = float(np.max(qs_ex))
    q_min = float(np.min(qs_ex))
    q_ensemble_std = float(np.mean(np.std(qs_ex, axis=0)))

    row = {
        "step": step,
        "num_qs": num_qs,
        "mean_pairwise_diff": round(mean_diff, 4),
        "mean_pairwise_corr": round(mean_corr, 6),
        "max_pairwise_corr": round(max_corr, 6),
        "min_pairwise_corr": round(min_corr, 6),
        "q_mean_expert": round(q_mean_expert, 4),
        "q_mean_random": round(q_mean_random, 4),
        "overestimation": round(overestimation, 4),
        "roughness": round(roughness, 6),
        "mask_var": round(mask_var, 6),
        "eff_rank": round(eff_rank, 2),
        "cond_number": round(cond_number, 2),
        "q_max": round(q_max, 4),
        "q_min": round(q_min, 4),
        "q_ensemble_std": round(q_ensemble_std, 4),
    }
    for k in range(min(num_qs, 10)):
        row["q_head_{}".format(k)] = round(head_means[k], 4)

    file_exists = os.path.exists(diag_path)
    with open(diag_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            w.writeheader()
        w.writerow(row)

    print("[DIAG {:>7}] nq={} pw_diff={:.3f} pw_corr={:.4f} "
          "Qexp={:.2f} Qrnd={:.2f} overest={:.2f} "
          "rough={:.4f} mask_var={:.4f}".format(
              step, num_qs, mean_diff, mean_corr,
              q_mean_expert, q_mean_random, overestimation,
              roughness, mask_var), flush=True)
    return row
