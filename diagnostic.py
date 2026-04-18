"""diagnostic.py -- Multi-head Q diagnostics for RLPD ablation runs.

Metrics computed on a fixed buffer of (s, a) pairs:
- Pairwise correlation, |Qi - Qj|, ensemble std (head diversity)
- q_mean_expert / q_mean_random / overestimation (OOD gap)
- Effective rank, condition number (representation quality)
- Roughness: variance of Q under small action perturbations
- Grad norm: ||grad_a Qbar|| on the ensemble-averaged critic
- Grad variation: change in grad_a Qbar under small action perturbation
- Mask variance: variance of Q across dropout RNG samples (DroQ-style)
- Head residual variance: Var_i[Qi(s,a)] relative to ensemble mean
- Gradient residual variance: Var_i[grad_a Qi(s,a)] across heads
- Gradient sharpness: squared action-gradient norm of single-head vs ensemble Q
- Smoothing gain: single-head sharpness minus ensemble sharpness
"""

import csv
import os

import numpy as np
import jax
import jax.numpy as jnp
from itertools import combinations


def _critic_variables(agent):
    """Variables dict for critic.apply_fn, including spectral-norm stats."""
    if getattr(agent, "use_spec_norm", False):
        return {
            "params": agent.critic.params,
            "batch_stats": agent.critic.batch_stats,
        }
    return {"params": agent.critic.params}


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
        _critic_variables(agent), obs, actions, False,
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


def _compute_head_var(qs):
    """Mean over samples of Var_i[Qi(s,a)].

    This implements the operational decomposition Q_i = Qbar + delta_i:
    head_var = E_{(s,a)}[mean_i delta_i(s,a)^2].
    """
    return float(np.mean(np.var(qs, axis=0)))


def _compute_head_grad_metrics(agent, obs, act):
    """Action-gradient metrics for all critic heads.

    Returns a dict with:
      grad_var:
        E_s[mean_i ||grad_a Qi - mean_j grad_a Qj||^2].
      single_head_grad_sharpness:
        E_s[mean_i ||grad_a Qi||^2].
      ensemble_grad_sharpness:
        E_s[||mean_i grad_a Qi||^2].

    The final two are the small-perturbation counterparts of
    sharpness_single and sharpness_ens, since Var_eps[Q(a+eps)] is
    approximately sigma^2 ||grad_a Q||^2 for small isotropic eps.
    """
    num_qs = agent.num_qs

    def q_head_sums(act_batch, obs_batch):
        qs = agent.critic.apply_fn(
            _critic_variables(agent),
            obs_batch, act_batch, False,
            rngs={"dropout": jax.random.PRNGKey(0)})
        # One scalar per head. jacrev gives per-head action gradients.
        return qs.sum(axis=1)

    grads = jax.jacrev(q_head_sums, argnums=0)(act, obs)
    grads = grads[:num_qs]  # [num_qs, n_samples, act_dim]
    grad_mean = grads.mean(axis=0, keepdims=True)
    grad_var = jnp.sum((grads - grad_mean) ** 2, axis=-1).mean()
    single_head_grad_sharpness = jnp.sum(grads ** 2, axis=-1).mean()
    ensemble_grad_sharpness = jnp.sum(
        jnp.squeeze(grad_mean, axis=0) ** 2, axis=-1).mean()
    grad_smooth_gain = (
        single_head_grad_sharpness - ensemble_grad_sharpness)
    grad_smooth_ratio = (
        single_head_grad_sharpness
        / jnp.maximum(ensemble_grad_sharpness, 1e-12))
    return {
        "grad_var": float(grad_var),
        "single_head_grad_sharpness": float(single_head_grad_sharpness),
        "ensemble_grad_sharpness": float(ensemble_grad_sharpness),
        "grad_smooth_gain": float(grad_smooth_gain),
        "grad_smooth_ratio": float(grad_smooth_ratio),
    }


def _compute_grad_var(agent, obs, act):
    """Back-compat wrapper for the cross-head gradient variance."""
    return _compute_head_grad_metrics(agent, obs, act)["grad_var"]


def _compute_rank(agent, obs, act, input_dim):
    """Jacobian-based effective rank. EXPENSIVE -- call sparingly."""
    def q_head0(sa):
        o = sa[:input_dim]
        a = sa[input_dim:]
        key = jax.random.PRNGKey(0)
        qs = agent.critic.apply_fn(
            _critic_variables(agent),
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
    Compute Qbar(s, a') for each (mean over heads), then return mean over
    states of variance over perturbations. Uses training=False (deterministic,
    no dropout) for reproducibility.
    """
    key = jax.random.PRNGKey(seed)
    eps = jax.random.normal(key, (n_perturb,) + act.shape) * sigma
    act_pert = act[None] + eps  # [n_perturb, n_samples, act_dim]

    def single(a_batch):
        qs = agent.critic.apply_fn(
            _critic_variables(agent), obs, a_batch, False,
            rngs={"dropout": jax.random.PRNGKey(0)})
        return qs.mean(axis=0)  # mean over heads -> [n_samples]

    q_perturb = jax.vmap(single)(act_pert)  # [n_perturb, n_samples]
    var_per_state = np.array(q_perturb).var(axis=0)  # [n_samples]
    return float(var_per_state.mean())


def _compute_smoothing_metrics(
        agent, obs, act, n_perturb=100, sigma=0.05, seed=42):
    """Single-head vs ensemble sharpness under shared perturbations.

    Returns:
      sharpness_single: mean_i E_s Var_eps[Qi(s, a + eps)]
      sharpness_ens:    E_s Var_eps[mean_i Qi(s, a + eps)]
      smooth_gain:      sharpness_single - sharpness_ens
      smooth_ratio:     sharpness_single / sharpness_ens
    """
    key = jax.random.PRNGKey(seed)
    eps = jax.random.normal(key, (n_perturb,) + act.shape) * sigma
    act_pert = act[None] + eps  # [n_perturb, n_samples, act_dim]

    def single(a_batch):
        return agent.critic.apply_fn(
            _critic_variables(agent), obs, a_batch, False,
            rngs={"dropout": jax.random.PRNGKey(0)})

    q_perturb = jax.vmap(single)(act_pert)
    # [n_perturb, num_qs, n_samples]
    per_head_var = jnp.var(q_perturb, axis=0)
    sharpness_single = float(per_head_var.mean())

    q_ens = q_perturb.mean(axis=1)  # [n_perturb, n_samples]
    sharpness_ens = float(jnp.var(q_ens, axis=0).mean())
    smooth_gain = sharpness_single - sharpness_ens
    smooth_ratio = sharpness_single / max(sharpness_ens, 1e-12)
    return sharpness_single, sharpness_ens, smooth_gain, smooth_ratio


def _compute_grad_metrics(agent, obs, act, n_perturb=50, sigma=0.05, seed=42):
    """Gradient-based action-space metrics on the ensemble-averaged critic.

    Returns (grad_norm, grad_variation):
      grad_norm:      mean over (s,a) of ||grad_a Qbar(s,a)||_2
                      Direct measurement of action-gradient magnitude.
      grad_variation: mean over (s,a,eps) of
                      ||grad_a Qbar(s,a+eps) - grad_a Qbar(s,a)||_2
                      How much the gradient field changes under small action
                      perturbations (curvature-like signal).

    Uses n_perturb=50 (half of roughness) because each perturbation requires
    a vmapped backward pass. Uses training=False -- no dropout.

    Implementation note: we define a scalar loss = sum over batch of Qbar.
    Because grad of a sum equals the sum of per-element grads (and each
    per-element Qbar depends only on its own row of the action batch), a
    single grad_a of this scalar returns the per-sample action gradient
    stacked along the batch axis. This avoids inner vmap over samples.
    """
    def q_mean_sum(act_batch, obs_batch):
        qs = agent.critic.apply_fn(
            _critic_variables(agent),
            obs_batch, act_batch, False,
            rngs={"dropout": jax.random.PRNGKey(0)})
        # qs: [num_qs, n_samples]; Qbar per sample = qs.mean(axis=0)
        return qs.mean(axis=0).sum()

    grad_fn = jax.grad(q_mean_sum, argnums=0)  # -> [n_samples, act_dim]

    grad_a = grad_fn(act, obs)
    grad_norm = float(jnp.linalg.norm(grad_a, axis=-1).mean())

    # Perturb actions; compute gradient at each perturbation; measure
    # L2 distance from unperturbed gradient.
    key = jax.random.PRNGKey(seed)
    eps_all = jax.random.normal(key, (n_perturb,) + act.shape) * sigma

    def one_pert(eps_k):
        grad_p = grad_fn(act + eps_k, obs)
        return jnp.linalg.norm(grad_p - grad_a, axis=-1)  # [n_samples]

    diffs = jax.vmap(one_pert)(eps_all)  # [n_perturb, n_samples]
    grad_variation = float(np.array(diffs).mean())
    return grad_norm, grad_variation


def _compute_mask_var(agent, obs, act, n_samples=10, seed=123):
    """Variance of Q across dropout RNG samples (training=True).

    Measures the implicit-ensemble diversity from DroQ-style dropout.
    Returns 0 for runs without dropout (deterministic forward pass).
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), n_samples)

    def single(k):
        qs = agent.critic.apply_fn(
            _critic_variables(agent), obs, act, True,
            rngs={"dropout": k})
        return qs.mean(axis=0)  # mean over heads -> [n_obs]

    q_samples = jax.vmap(single)(keys)  # [n_samples, n_obs]
    return float(np.array(q_samples).var(axis=0).mean())


def compute_roughness_only(agent, diag_buf):
    """Lightweight roughness computation for use by train_abc.py.

    Back-compat shim: existing callers that only need roughness can keep
    using this. New callers should use compute_sharpness_bundle below to
    get roughness + grad_norm + grad_variation in one call.
    """
    return _compute_roughness(agent, diag_buf["obs"], diag_buf["act_expert"])


def compute_sharpness_bundle(agent, diag_buf):
    """Roughness + grad_norm + grad_variation for the 50k-step probe.

    Grad metrics add ~2-3x the cost of roughness alone but remain cheap
    relative to the training step. Returns a dict; each key defaults to
    -1.0 on failure so the caller can safely round() each value without
    special-casing.
    """
    obs = diag_buf["obs"]
    act = diag_buf["act_expert"]
    try:
        qs = _get_qs(agent, obs, act)
        head_var = _compute_head_var(qs)
    except Exception:
        head_var = -1.0
    try:
        (sharpness_single, roughness, smooth_gain,
         smooth_ratio) = _compute_smoothing_metrics(agent, obs, act)
    except Exception:
        sharpness_single = -1.0
        roughness = -1.0
        smooth_gain = -1.0
        smooth_ratio = -1.0
    try:
        grad_norm, grad_variation = _compute_grad_metrics(agent, obs, act)
    except Exception:
        grad_norm, grad_variation = -1.0, -1.0
    try:
        head_grad_metrics = _compute_head_grad_metrics(agent, obs, act)
    except Exception:
        head_grad_metrics = {
            "grad_var": -1.0,
            "single_head_grad_sharpness": -1.0,
            "ensemble_grad_sharpness": -1.0,
            "grad_smooth_gain": -1.0,
            "grad_smooth_ratio": -1.0,
        }
    return {
        "head_var": head_var,
        **head_grad_metrics,
        "roughness": roughness,
        "sharpness_single": sharpness_single,
        "smooth_gain": smooth_gain,
        "smooth_ratio": smooth_ratio,
        "grad_norm": grad_norm,
        "grad_variation": grad_variation,
    }


def run_diagnostic(agent, diag_buf, step, diag_path):
    obs = diag_buf["obs"]
    act_ex = diag_buf["act_expert"]
    act_rnd = diag_buf["act_random"]

    # 1. Per-head Q on expert actions -- pairwise stats
    qs_ex = _get_qs(agent, obs, act_ex)
    num_qs = qs_ex.shape[0]
    mean_diff, mean_corr, max_corr, min_corr, std_diff = _pairwise_stats(qs_ex)
    head_means = [float(np.mean(qs_ex[k])) for k in range(num_qs)]
    q_mean_expert = float(np.mean(qs_ex))
    head_var = _compute_head_var(qs_ex)

    # 2. OOD overestimation
    qs_rnd = _get_qs(agent, obs, act_rnd)
    q_mean_random = float(np.mean(qs_rnd))
    overestimation = q_mean_random - q_mean_expert

    # 3. Smoothing metrics -- single-head vs ensemble sharpness
    try:
        (sharpness_single, roughness, smooth_gain,
         smooth_ratio) = _compute_smoothing_metrics(agent, obs, act_ex)
    except Exception:
        sharpness_single = -1.0
        roughness = -1.0
        smooth_gain = -1.0
        smooth_ratio = -1.0

    # 4. Gradient metrics -- direct ||grad_a Qbar|| and its variation
    try:
        grad_norm, grad_variation = _compute_grad_metrics(agent, obs, act_ex)
    except Exception:
        grad_norm, grad_variation = -1.0, -1.0

    # 4b. Gradient residual variance across heads
    try:
        head_grad_metrics = _compute_head_grad_metrics(agent, obs, act_ex)
    except Exception:
        head_grad_metrics = {
            "grad_var": -1.0,
            "single_head_grad_sharpness": -1.0,
            "ensemble_grad_sharpness": -1.0,
            "grad_smooth_gain": -1.0,
            "grad_smooth_ratio": -1.0,
        }

    # 5. Mask variance -- variance under dropout RNG sampling
    try:
        mask_var = _compute_mask_var(agent, obs, act_ex)
    except Exception:
        mask_var = -1.0

    # 6. Feature rank -- EXPENSIVE, only every 50k
    eff_rank = -1.0
    cond_number = -1.0
    if step % 50000 == 0:
        try:
            input_dim = obs.shape[-1]
            eff_rank, cond_number = _compute_rank(
                agent, obs, act_ex, input_dim)
        except Exception:
            pass

    # 7. Q-value magnitude
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
        "head_var": round(head_var, 6),
        "grad_var": round(head_grad_metrics["grad_var"], 6),
        "single_head_grad_sharpness": round(
            head_grad_metrics["single_head_grad_sharpness"], 6),
        "ensemble_grad_sharpness": round(
            head_grad_metrics["ensemble_grad_sharpness"], 6),
        "grad_smooth_gain": round(
            head_grad_metrics["grad_smooth_gain"], 6),
        "grad_smooth_ratio": round(
            head_grad_metrics["grad_smooth_ratio"], 6),
        "roughness": round(roughness, 6),
        "sharpness_single": round(sharpness_single, 6),
        "sharpness_ens": round(roughness, 6),
        "smooth_gain": round(smooth_gain, 6),
        "smooth_ratio": round(smooth_ratio, 6),
        "grad_norm": round(grad_norm, 6),
        "grad_variation": round(grad_variation, 6),
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
          "rough={:.4f} singleS={:.4f} gain={:.4f} "
          "head_var={:.4f} grad_var={:.4f} "
          "|gradQ|={:.4f} dgradQ={:.4f} mask_var={:.4f}".format(
              step, num_qs, mean_diff, mean_corr,
              q_mean_expert, q_mean_random, overestimation,
              roughness, sharpness_single, smooth_gain,
              head_var, head_grad_metrics["grad_var"],
              grad_norm, grad_variation, mask_var), flush=True)

    return row
