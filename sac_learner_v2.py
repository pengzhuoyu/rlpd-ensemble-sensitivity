"""SACLearner with bootstrap masks (A) and independent targets (B).
Changes from original sac_learner.py:
  - bootstrap_mask: Bernoulli(0.5) mask per (head, sample) in critic loss
  - independent_targets: each head uses its own target Q via critic.apply_fn
    with target_critic.params (which has num_qs heads stored)
"""
from functools import partial
from typing import Dict, Optional, Sequence, Tuple
import flax
import gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
from rlpd.agents.agent import Agent
from rlpd.agents.sac.temperature import Temperature
from rlpd.data.dataset import DatasetDict
from rlpd.distributions import TanhNormal
from rlpd.networks import (
    MLP, Ensemble, MLPResNetV2, StateActionValue, subsample_ensemble,
)


def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    return flax.core.FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))


class SACLearnerV2(Agent):
    critic: TrainState
    target_critic: TrainState
    temp: TrainState
    tau: float
    discount: float
    target_entropy: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)
    backup_entropy: bool = struct.field(pytree_node=False)
    bootstrap_mask: bool = struct.field(pytree_node=False)
    independent_targets: bool = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls, seed, observation_space, action_space,
        actor_lr=3e-4, critic_lr=3e-4, temp_lr=3e-4,
        hidden_dims=(256, 256), discount=0.99, tau=0.005,
        num_qs=2, num_min_qs=None,
        critic_dropout_rate=None, critic_weight_decay=None,
        critic_layer_norm=False, target_entropy=None,
        init_temperature=1.0, backup_entropy=True,
        use_pnorm=False, use_critic_resnet=False,
        bootstrap_mask=False, independent_targets=False,
    ):
        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()
        if target_entropy is None:
            target_entropy = -action_dim / 2
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_base_cls = partial(
            MLP, hidden_dims=hidden_dims, activate_final=True,
            use_pnorm=use_pnorm)
        actor_def = TanhNormal(actor_base_cls, action_dim)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply, params=actor_params,
            tx=optax.adam(learning_rate=actor_lr))

        if use_critic_resnet:
            critic_base_cls = partial(MLPResNetV2, num_blocks=1)
        else:
            critic_base_cls = partial(
                MLP, hidden_dims=hidden_dims, activate_final=True,
                dropout_rate=critic_dropout_rate,
                use_layer_norm=critic_layer_norm,
                use_pnorm=use_pnorm)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(
            critic_key, observations, actions)["params"]

        if critic_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn)
        else:
            tx = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params, tx=tx)

        target_critic_def = Ensemble(
            critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(
                lambda _: None, lambda _: None))

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply, params=temp_params,
            tx=optax.adam(learning_rate=temp_lr))

        return cls(
            rng=rng, actor=actor, critic=critic,
            target_critic=target_critic, temp=temp,
            target_entropy=target_entropy, tau=tau,
            discount=discount, num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
            bootstrap_mask=bootstrap_mask,
            independent_targets=independent_targets)

    def update_actor(self, batch):
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params):
            dist = self.actor.apply_fn(
                {"params": actor_params}, batch["observations"])
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"], actions, True,
                rngs={"dropout": key2})
            q = qs.mean(axis=0)
            actor_loss = (
                log_probs
                * self.temp.apply_fn({"params": self.temp.params})
                - q
            ).mean()
            return actor_loss, {
                "actor_loss": actor_loss,
                "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(
            actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)
        return self.replace(actor=actor, rng=rng), actor_info

    def update_temperature(self, entropy):
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (
                entropy - self.target_entropy).mean()
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss}

        grads, temp_info = jax.grad(
            temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)
        return self.replace(temp=temp), temp_info

    def update_critic(self, batch):
        dist = self.actor.apply_fn(
            {"params": self.actor.params},
            batch["next_observations"])
        rng = self.rng
        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)

        key, rng = jax.random.split(rng)

        if self.independent_targets:
            # B: per-head targets via critic.apply_fn + target params
            # critic.apply_fn expects num_qs heads of params
            # target_critic.params stores num_qs heads (EMA of critic)
            next_qs = self.critic.apply_fn(
                {"params": self.target_critic.params},
                batch["next_observations"],
                next_actions, True,
                rngs={"dropout": key})
            # next_qs: [num_qs, batch_size]
            target_q = (batch["rewards"]
                        + self.discount * batch["masks"] * next_qs)
            # target_q: [num_qs, batch_size]
        else:
            # Original: shared target via subsampled min
            target_params = subsample_ensemble(
                key, self.target_critic.params,
                self.num_min_qs, self.num_qs)
            key, rng = jax.random.split(rng)
            next_qs = self.target_critic.apply_fn(
                {"params": target_params},
                batch["next_observations"],
                next_actions, True,
                rngs={"dropout": key})
            next_q = next_qs.min(axis=0)
            target_q = (batch["rewards"]
                        + self.discount * batch["masks"] * next_q)
            # target_q: [batch_size]

        if self.backup_entropy:
            next_log_probs = dist.log_prob(next_actions)
            # [batch_size] broadcasts with both target_q shapes
            target_q = target_q - (
                self.discount * batch["masks"]
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs)

        key, rng = jax.random.split(rng)
        mask_key, rng = jax.random.split(rng)
        do_bootstrap = self.bootstrap_mask

        def critic_loss_fn(critic_params):
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"], True,
                rngs={"dropout": key})
            # qs: [num_qs, batch_size]
            # target_q: [batch_size] or [num_qs, batch_size]
            td_errors = (qs - target_q) ** 2

            if do_bootstrap:
                bmask = jax.random.bernoulli(
                    mask_key, p=0.5, shape=td_errors.shape)
                critic_loss = ((bmask * td_errors).sum()
                               / jnp.maximum(bmask.sum(), 1.0))
            else:
                critic_loss = td_errors.mean()

            return critic_loss, {
                "critic_loss": critic_loss,
                "q": qs.mean()}

        grads, info = jax.grad(
            critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)
        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau)
        target_critic = self.target_critic.replace(
            params=target_critic_params)
        return self.replace(
            critic=critic, target_critic=target_critic, rng=rng), info

    @partial(jax.jit, static_argnames=("utd_ratio", "do_actor_update"))
    def update(self, batch, utd_ratio, do_actor_update=True):
        new_agent = self
        for i in range(utd_ratio):
            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]
            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)
        if do_actor_update:
            new_agent, actor_info = new_agent.update_actor(mini_batch)
            new_agent, temp_info = new_agent.update_temperature(
                actor_info["entropy"])
        else:
            actor_info = {"actor_loss": jnp.float32(0),
                          "entropy": jnp.float32(0)}
            temp_info = {"temperature": jnp.float32(0),
                         "temperature_loss": jnp.float32(0)}
        return new_agent, {**actor_info, **critic_info, **temp_info}
