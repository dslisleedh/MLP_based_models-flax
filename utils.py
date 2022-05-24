import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Sequence


class Droppath(nn.Module):
    survival_prob: float
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic: Optional[bool] = None):
        deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)
        if self.survival_prob == 1.:
            return inputs
        elif self.survival_prob == 0.:
            return jnp.zeros_like(inputs)

        if deterministic:
            return inputs * self.survival_prob
        else:
            rng = self.make_rng('droppath')
            broadcast_shape = [inputs.shape[0] + [1 for _ in range(len(inputs.shape) - 1)]]
            epsilon = jax.random.bernoulli(key=rng,
                                           p=self.survival_prob,
                                           shape=broadcast_shape
                                           )
            return inputs * epsilon


