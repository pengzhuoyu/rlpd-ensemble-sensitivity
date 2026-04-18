"""train_abc.py -- Test directions A (bootstrap masks), B (independent
targets), C (plasticity reset). All on RLPD backbone.

Updated: logs roughness + grad_norm + grad_variation every 50k steps
via compute_sharpness_bundle (see diagnostic.py).
"""

import os
import csv
import json
import time

import d4rl
import d4rl.gym_mujoco
import d4rl.locomotion
import gym
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

import wandb

from sac_learner_v2 import SACLearnerV2
from rlpd.data import ReplayBuffer
from rlpd.data.d4rl_datasets import D4RLDataset

try:
    from rlpd.data.binary_datasets import BinaryDataset
except Exception:
    pass

from rlpd.evaluation import evaluate
from rlpd.wrappers import wrap_gym

from diagnostic import setup_diag_buffer, compute_sharpness_bundle

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "rlpd_abc", "wandb project.")
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Seed.")
flags.DEFINE_integer("eval_episodes", 20, "Eval episodes.")
flags.DEFINE_integer("log_interval", 1000, "Log interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("max_steps", 300000, "Max steps.")
flags.DEFINE_integer("start_training", 5000, "Start training.")
flags.DEFINE_boolean("tqdm", True, "Progress bar.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_integer("utd_ratio", 20, "UTD ratio.")
flags.DEFINE_string("results_dir", "results", "Results dir.")
flags.DEFINE_boolean("bootstrap_mask", False, "A: bootstrap masks.")
flags.DEFINE_boolean("independent_targets", False,
                     "B: per-head targets.")
flags.DEFINE_integer("critic_reset_step", 0,
                     "C: reset critic at this step. 0 = disabled.")
flags.DEFINE_integer("actor_delay", 1,
                     "Update actor every N env steps. 1 = every step.")
config_flags.DEFINE_config_file(
    "config", "configs/sac_config.py",
    "Config.", lock_config=False)


def combine(one_dict, other_dict):
    """Concatenate two batch dicts and shuffle."""
    combined = {}
    first_key = next(
        k for k, v in one_dict.items() if not isinstance(v, dict))
    n_total = (one_dict[first_key].shape[0]
               + other_dict[first_key].shape[0])
    perm = np.random.permutation(n_total)
    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            combined[k] = np.concatenate(
                [v, other_dict[k]], axis=0)[perm]
    return combined


def make_tag():
    """Build a descriptive tag from flags."""
    parts = []
    if FLAGS.bootstrap_mask:
        parts.append("bmask")
    if FLAGS.independent_targets:
        parts.append("indep")
    if FLAGS.critic_reset_step > 0:
        parts.append("reset{}k".format(
            FLAGS.critic_reset_step // 1000))
    return "_".join(parts) if parts else "baseline"


def main(_):
    tag = make_tag()
    wandb.init(
        project=FLAGS.project_name,
        mode=os.environ.get("WANDB_MODE", "online"))
    wandb.config.update(FLAGS)

    kwargs = dict(FLAGS.config)
    kwargs.pop("model_cls")

    nqs = kwargs.get("num_qs", 10)
    mqs = kwargs.get("num_min_qs", nqs)
    ln = kwargs.get("critic_layer_norm", True)
    drop = kwargs.get("critic_dropout_rate", None)
    drop_tag = "drop{}".format(drop) if drop else "nodrop"
    spec = kwargs.get("spec_norm_coef", None)
    spec_tag = "sn{}".format(spec) if spec is not None else "nosn"

    # Run name encodes every config that distinguishes runs, so paired runs
    # in the same SLURM job never overwrite each other's results.
    parts = [FLAGS.env_name, "nq{}".format(nqs), "mq{}".format(mqs),
             drop_tag, spec_tag, "s{}".format(FLAGS.seed)]
    if tag != "baseline":
        parts.insert(-1, tag)
    run_name = "_".join(parts)

    log_dir = os.path.join(FLAGS.results_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    print("")
    print("=" * 60)
    print("RLPD A/B/C Experiment")
    print("Env: {} | Seed: {} | nq={} ln={}".format(
        FLAGS.env_name, FLAGS.seed, nqs, ln))
    print("Dropout={} | SpectralNorm={}".format(drop, spec))
    print("A(bmask)={} B(indep)={} C(reset)={}".format(
        FLAGS.bootstrap_mask, FLAGS.independent_targets,
        FLAGS.critic_reset_step))
    print("Tag: {} | Dir: {}".format(tag, log_dir))
    print("=" * 60)
    print("", flush=True)

    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)

    if "binary" in FLAGS.env_name:
        ds = BinaryDataset(env, include_bc_data=True)
    else:
        ds = D4RLDataset(env)

    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.seed(FLAGS.seed + 42)

    agent = SACLearnerV2.create(
        FLAGS.seed,
        env.observation_space,
        env.action_space,
        bootstrap_mask=FLAGS.bootstrap_mask,
        independent_targets=FLAGS.independent_targets,
        **kwargs)

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space,
        FLAGS.max_steps)
    replay_buffer.seed(FLAGS.seed)

    # Fixed (s, a) buffer for sharpness probes (computed every 50k steps).
    diag_buf = setup_diag_buffer(ds, env)

    log_rows = []
    log_path = os.path.join(log_dir, "online_log.csv")
    t_start = time.time()
    last_info = {}
    did_reset = False

    observation, done = env.reset(), False
    for i in tqdm.tqdm(
            range(0, FLAGS.max_steps + 1),
            smoothing=0.1,
            disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)

        next_observation, reward, done, info = env.step(action)
        mask = 1.0 if (
            not done or "TimeLimit.truncated" in info) else 0.0
        replay_buffer.insert(dict(
            observations=observation,
            actions=action,
            rewards=reward,
            masks=mask,
            dones=done,
            next_observations=next_observation))
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log(
                    {"training/{}".format(decode[k]): v}, step=i)

        if i >= FLAGS.start_training:
            # Direction C: critic reset
            if (FLAGS.critic_reset_step > 0
                    and i == FLAGS.critic_reset_step
                    and not did_reset):
                fresh = SACLearnerV2.create(
                    FLAGS.seed + 7777,
                    env.observation_space,
                    env.action_space,
                    bootstrap_mask=FLAGS.bootstrap_mask,
                    independent_targets=FLAGS.independent_targets,
                    **kwargs)
                agent = agent.replace(
                    critic=fresh.critic,
                    target_critic=agent.target_critic.replace(
                        params=fresh.critic.params,
                        batch_stats=fresh.critic.batch_stats))
                did_reset = True
                print("[{:>7}] CRITIC RESET (seed={})".format(
                    i, FLAGS.seed + 7777), flush=True)

            total = FLAGS.batch_size * FLAGS.utd_ratio
            n_offline = int(total * FLAGS.offline_ratio)
            n_online = total - n_offline
            online_batch = replay_buffer.sample(n_online)
            offline_batch = ds.sample(n_offline)
            batch = combine(offline_batch, online_batch)
            if "antmaze" in FLAGS.env_name:
                batch["rewards"] = batch["rewards"] - 1

            agent, update_info = agent.update(
                batch, FLAGS.utd_ratio,
                do_actor_update=(i % FLAGS.actor_delay == 0))
            last_info = {k: float(v)
                         for k, v in update_info.items()}

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log(
                        {"training/{}".format(k): v}, step=i)

            if i % FLAGS.eval_interval == 0:
                eval_info = evaluate(
                    agent, eval_env,
                    num_episodes=FLAGS.eval_episodes,
                    save_video=FLAGS.save_video)
                success = eval_info.get("return", 0.0)

                try:
                    raw_env = eval_env
                    while hasattr(raw_env, "env"):
                        raw_env = raw_env.env
                    norm_score = float(
                        raw_env.get_normalized_score(success))
                except Exception:
                    norm_score = success * 100.0

                elapsed = (time.time() - t_start) / 60.0

                # Sharpness bundle every 50k steps: roughness (perturbation
                # variance), grad_norm (||grad_a Qbar||), grad_variation
                # (change in gradient under perturbation). Total cost ~3x
                # roughness-alone; still cheap relative to training.
                roughness = ""
                sharpness_single = ""
                smooth_gain = ""
                smooth_ratio = ""
                head_var = ""
                grad_var = ""
                single_head_grad_sharpness = ""
                ensemble_grad_sharpness = ""
                grad_smooth_gain = ""
                grad_smooth_ratio = ""
                grad_norm = ""
                grad_variation = ""
                if i % 50000 == 0:
                    try:
                        bundle = compute_sharpness_bundle(agent, diag_buf)
                        head_var = round(bundle["head_var"], 6)
                        grad_var = round(bundle["grad_var"], 6)
                        single_head_grad_sharpness = round(
                            bundle["single_head_grad_sharpness"], 6)
                        ensemble_grad_sharpness = round(
                            bundle["ensemble_grad_sharpness"], 6)
                        grad_smooth_gain = round(
                            bundle["grad_smooth_gain"], 6)
                        grad_smooth_ratio = round(
                            bundle["grad_smooth_ratio"], 6)
                        roughness = round(bundle["roughness"], 6)
                        sharpness_single = round(
                            bundle["sharpness_single"], 6)
                        smooth_gain = round(bundle["smooth_gain"], 6)
                        smooth_ratio = round(bundle["smooth_ratio"], 6)
                        grad_norm = round(bundle["grad_norm"], 6)
                        grad_variation = round(bundle["grad_variation"], 6)
                        wandb.log({
                            "diag/head_var": head_var,
                            "diag/grad_var": grad_var,
                            "diag/single_head_grad_sharpness":
                                single_head_grad_sharpness,
                            "diag/ensemble_grad_sharpness":
                                ensemble_grad_sharpness,
                            "diag/grad_smooth_gain": grad_smooth_gain,
                            "diag/grad_smooth_ratio": grad_smooth_ratio,
                            "diag/roughness": roughness,
                            "diag/sharpness_single": sharpness_single,
                            "diag/smooth_gain": smooth_gain,
                            "diag/smooth_ratio": smooth_ratio,
                            "diag/grad_norm": grad_norm,
                            "diag/grad_variation": grad_variation,
                        }, step=i)
                    except Exception as e:
                        print("[SHARPNESS {:>7}] failed: {}".format(i, e),
                              flush=True)

                if (i % (FLAGS.eval_interval * 4) == 0
                        or i < FLAGS.eval_interval * 3):
                    print(
                        "[EVAL {:>7}] score={:.1f} success={:.3f}"
                        " tag={} elapsed={:.1f}min"
                        " rough={} gain={} head_var={} grad_var={}"
                        " gradS={} |gradQ|={} dgradQ={}".format(
                            i, norm_score, success, tag, elapsed,
                            roughness, smooth_gain, head_var, grad_var,
                            ensemble_grad_sharpness,
                            grad_norm, grad_variation),
                        flush=True)

                for k, v in eval_info.items():
                    wandb.log(
                        {"evaluation/{}".format(k): v}, step=i)

                row = {
                    "step": i,
                    "success_rate": success,
                    "normalized_score": norm_score,
                    "elapsed_min": elapsed,
                    "tag": tag,
                    "critic_loss": last_info.get(
                        "critic_loss", 0.0),
                    "mean_q": last_info.get("q", 0.0),
                    "head_var": head_var,
                    "grad_var": grad_var,
                    "single_head_grad_sharpness":
                        single_head_grad_sharpness,
                    "ensemble_grad_sharpness": ensemble_grad_sharpness,
                    "grad_smooth_gain": grad_smooth_gain,
                    "grad_smooth_ratio": grad_smooth_ratio,
                    "roughness": roughness,
                    "sharpness_single": sharpness_single,
                    "smooth_gain": smooth_gain,
                    "smooth_ratio": smooth_ratio,
                    "grad_norm": grad_norm,
                    "grad_variation": grad_variation,
                }
                log_rows.append(row)
                with open(log_path, "w", newline="") as f:
                    w = csv.DictWriter(
                        f, fieldnames=log_rows[0].keys())
                    w.writeheader()
                    w.writerows(log_rows)

    if log_rows:
        scores = [r["normalized_score"] for r in log_rows]
        n_final = min(10, len(scores))
        summary = {
            "env": FLAGS.env_name,
            "seed": FLAGS.seed,
            "tag": tag,
            "nqs": nqs,
            "ln": ln,
            "spec_norm_coef": spec,
            "bootstrap_mask": FLAGS.bootstrap_mask,
            "independent_targets": FLAGS.independent_targets,
            "critic_reset_step": FLAGS.critic_reset_step,
            "final_score": float(np.mean(scores[-n_final:])),
            "peak_score": float(np.max(scores)),
            "peak_step": int(
                log_rows[int(np.argmax(scores))]["step"]),
            "wall_hours": round(
                (time.time() - t_start) / 3600, 2),
        }
        with open(os.path.join(
                log_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print("")
        print("FINAL: tag={} score={:.1f} peak={:.1f}".format(
            tag, summary["final_score"],
            summary["peak_score"]))


if __name__ == "__main__":
    app.run(main)
