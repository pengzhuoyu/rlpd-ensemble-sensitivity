"""train_diagnostic.py — pen-binary with per-head Q diagnostics."""
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
from diagnostic import setup_diag_buffer, run_diagnostic

FLAGS = flags.FLAGS
flags.DEFINE_string("project_name", "rlpd_diag", "wandb project.")
flags.DEFINE_string("env_name", "pen-binary-v0", "Environment.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 0, "Seed.")
flags.DEFINE_integer("eval_episodes", 20, "Eval episodes.")
flags.DEFINE_integer("log_interval", 1000, "Log interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("diag_interval", 5000, "Diagnostic interval.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("max_steps", 1000000, "Max steps.")
flags.DEFINE_integer("start_training", 5000, "Start training.")
flags.DEFINE_boolean("tqdm", True, "Progress bar.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_integer("utd_ratio", 20, "UTD ratio.")
flags.DEFINE_string("results_dir", "results", "Results dir.")
flags.DEFINE_boolean("bootstrap_mask", False, "A: bootstrap masks.")
flags.DEFINE_boolean("independent_targets", False, "B: per-head tgts.")
flags.DEFINE_integer("critic_reset_step", 0, "C: reset step.")
config_flags.DEFINE_config_file(
    "config", "configs/sac_config.py", "Config.", lock_config=False)


def combine(one_dict, other_dict):
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


def main(_):
    kwargs = dict(FLAGS.config)
    kwargs.pop("model_cls")
    nqs = kwargs.get("num_qs", 10)
    ln = kwargs.get("critic_layer_norm", True)
    min_qs = kwargs.get("num_min_qs", 1)
    drop = kwargs.get("critic_dropout_rate", None)
    drop_tag = "drop{}".format(drop) if drop else "nodrop"
    spec = kwargs.get("spec_norm_coef", None)
    spec_tag = "sn{}".format(spec) if spec is not None else "nosn"

    # Run name encodes every config so dropout/no-dropout runs don't collide.
    tag = "diag_nq{}_mq{}_{}_{}".format(nqs, min_qs, drop_tag, spec_tag)
    run_name = "{}_{}_s{}".format(FLAGS.env_name, tag, FLAGS.seed)
    log_dir = os.path.join(FLAGS.results_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    wandb.init(
        project=FLAGS.project_name,
        mode=os.environ.get("WANDB_MODE", "online"))
    wandb.config.update(FLAGS)

    print("")
    print("=" * 60)
    print("RLPD DIAGNOSTIC RUN")
    print("Env: {} | nq={} min_qs={} ln={} seed={}".format(
        FLAGS.env_name, nqs, min_qs, ln, FLAGS.seed))
    print("Dropout={} | SpectralNorm={}".format(drop, spec))
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

    diag_buf = setup_diag_buffer(ds, env)
    diag_path = os.path.join(log_dir, "diagnostic.csv")
    print("Diagnostic: 1000 samples, logging to {}".format(
        diag_path), flush=True)
    run_diagnostic(agent, diag_buf, 0, diag_path)

    log_rows = []
    eval_log_path = os.path.join(log_dir, "online_log.csv")
    t_start = time.time()
    last_info = {}
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
            observations=observation, actions=action,
            rewards=reward, masks=mask, dones=done,
            next_observations=next_observation))
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log(
                    {"training/{}".format(decode[k]): v}, step=i)

        if i >= FLAGS.start_training:
            total = FLAGS.batch_size * FLAGS.utd_ratio
            n_offline = int(total * FLAGS.offline_ratio)
            n_online = total - n_offline
            online_batch = replay_buffer.sample(n_online)
            offline_batch = ds.sample(n_offline)
            batch = combine(offline_batch, online_batch)
            if "antmaze" in FLAGS.env_name:
                batch["rewards"] = batch["rewards"] - 1
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)
            last_info = {k: float(v) for k, v in update_info.items()}
            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({"training/{}".format(k): v}, step=i)

        if i % FLAGS.diag_interval == 0 and i > 0:
            run_diagnostic(agent, diag_buf, i, diag_path)

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
            if i % (FLAGS.eval_interval * 4) == 0 or i < 15000:
                print("[EVAL {:>7}] score={:.1f} elapsed={:.1f}min".format(
                    i, norm_score, elapsed), flush=True)
            for k, v in eval_info.items():
                wandb.log({"evaluation/{}".format(k): v}, step=i)
            row = {"step": i, "success_rate": success,
                   "normalized_score": norm_score,
                   "elapsed_min": elapsed,
                   "critic_loss": last_info.get("critic_loss", 0.0),
                   "mean_q": last_info.get("q", 0.0)}
            log_rows.append(row)
            with open(eval_log_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
                w.writeheader()
                w.writerows(log_rows)

    if log_rows:
        scores = [r["normalized_score"] for r in log_rows]
        n_final = min(10, len(scores))
        summary = {"env": FLAGS.env_name, "seed": FLAGS.seed,
                   "nqs": nqs, "num_min_qs": min_qs,
                   "spec_norm_coef": spec,
                   "final_score": float(np.mean(scores[-n_final:])),
                   "peak_score": float(np.max(scores)),
                   "wall_hours": round((time.time() - t_start) / 3600, 2)}
        with open(os.path.join(log_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("")
        print("FINAL: nq={} mq={} score={:.1f} peak={:.1f}".format(
            nqs, min_qs, summary["final_score"], summary["peak_score"]))


if __name__ == "__main__":
    app.run(main)
