"""critic_spec_norm.py

Spectral-normalized critic for the paper's causal test.

We need an intervention on action-space sharpness that is mechanistically
different from dropout. Spectral normalization reparameterizes each Dense
layer's kernel so that its largest singular value is bounded, which
directly bounds the layer's contribution to the gradient norm of Q w.r.t.
its input. With per-layer bound c, the full-critic Lipschitz constant in
action space is bounded by c^L (L = number of layers).

This file defines a drop-in replacement for rlpd.networks.StateActionValue
that uses spec-normalized Dense layers. It is selected in sac_learner_v2.py
when `spec_norm_coef` is not None in the agent config.

The Ensemble wrapper in rlpd.networks.Ensemble is applied outside this
module (in sac_learner_v2.py), so this file only defines the single-critic
behavior. The Ensemble vmap handles the multi-head stacking.
"""

from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp


default_init = nn.initializers.xavier_uniform


class SpectralDense(nn.Module):
    """Dense layer with spectral-normalized kernel, scaled by coef.

    After the spectral-norm reparameterization, the layer's Lipschitz
    constant is bounded by coef. The bias is unaffected (biases don't
    contribute to the gradient w.r.t. inputs).

    flax.linen.SpectralNorm takes the wrapped layer and performs one step
    of power iteration per call to update the singular-value estimate.
    The estimate is stored in the 'batch_stats' collection, which is why
    sac_learner_v2.py carries batch_stats through its TrainState.
    """

    features: int
    coef: float = 1.0
    kernel_init: Callable = default_init()

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        update_stats: bool = True,
    ) -> jnp.ndarray:
        dense = nn.Dense(features=self.features, kernel_init=self.kernel_init)
        dense = nn.SpectralNorm(dense)
        y = dense(x, update_stats=update_stats)
        return self.coef * y


class StateActionValueSpecNorm(nn.Module):
    """Spectral-normalized drop-in for rlpd.networks.StateActionValue.

    Matches the call signature (observations, actions, training) and output
    shape of the upstream StateActionValue. The LayerNorm / activation /
    dropout pattern is preserved; only the Dense kernels are spec-normed.

    The final scalar Dense head is also spec-normed -- if we left it
    unnormalized, the network could scale its output arbitrarily regardless
    of what the backbone does, defeating the purpose of the regularization.
    """

    hidden_dims: Sequence[int]
    activations: Callable = nn.relu
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    spec_norm_coef: float = 1.0

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        training: bool = False,
        update_stats: bool = False,
    ) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)

        for size in self.hidden_dims:
            dense = SpectralDense(
                features=size,
                coef=self.spec_norm_coef)
            x = dense(x, update_stats=update_stats)
            if self.dropout_rate is not None and self.dropout_rate > 0.0:
                x = nn.Dropout(rate=self.dropout_rate)(
                    x, deterministic=not training)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activations(x)

        # Scalar head (also spec-normed).
        head = SpectralDense(
            features=1,
            coef=self.spec_norm_coef)
        q = head(x, update_stats=update_stats)
        return jnp.squeeze(q, -1)
