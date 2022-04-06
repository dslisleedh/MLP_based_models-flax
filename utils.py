import jax
import jax.numpy as jnp
import flax.linen as nn


class Droppath(nn.Module):
    survival_prob: float
    deterministic: bool

    @nn.compact
    def __call__(self, inputs):
        if self.deterministic:
            return inputs
        else:
            epsilon = jax.random.bernoulli(key=jax.random.PRNGKey(42),
                                           p=self.survival_prob,
                                           shape=[inputs.shape[0]] + [1 for _ in range(len(inputs.shape) - 1)]
                                           )
            return inputs * epsilon

