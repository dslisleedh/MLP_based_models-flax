from utils import Droppath
import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange


class SpatialShift(nn.Module):
    group: int

    @nn.compact
    def spatial_shift(self, x):
        b, h, w, c = x.shape
        x = jnp.split(x, 4, axis=-1)
        x[0] = x[0].at[:, :, 1:, :].set(x[0][:, :, :w - 1, :])
        x[1] = x[1].at[:, :, :w - 1, :].set(x[1][:, :, 1:, :])
        x[2] = x[2].at[:, 1:, :, :].set(x[2][:, :h - 1, :, :])
        x[3] = x[3].at[:, :h - 1, :, :].set(x[3][:, 1:, :, :])
        return jnp.concatenate(x, axis=-1)

    @nn.compact
    def __call__(self, x):
        groups = jnp.split(x, self.group, axis=-1)
        return jnp.concatenate([self.spatial_shift(g) for g in groups],
                               axis=-1
                               )


class MLP(nn.Module):
    c: int
    r: int
    shift: bool

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.c * self.r)(x)
        x = nn.gelu(x)
        if self.shift:
            x = SpatialShift(4)(x)
        x = nn.Dense(self.c)(x)
        x = nn.LayerNorm()(x)
        return x


class S2Block(nn.Module):
    c: int
    r: int
    survival_prob: float
    deterministic: bool

    @nn.compact
    def __call__(self, x):
        x = Droppath(self.survival_prob, self.deterministic)(MLP(self.c, self.r, True)(x)) + x
        x = Droppath(self.survival_prob, self.deterministic)(MLP(self.c, self.r, False)(x)) + x
        return x


class S2MLP(nn.Module):
    p: int
    c: int
    r: int
    n: int
    num_labels: int
    stochastic_depth: float
    is_training: bool

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.c,
                    kernel_size=(self.p, self.p),
                    strides=(self.p, self.p),
                    padding='VALID'
                    )
        x = rearrange(x, 'b h w c -> b (h w) c')
        survival_prob = jnp.linspace(0., self.stochastic_depth, num=self.n)
        for prob in survival_prob:
            x = S2Block(self.c, self.r, prob, not self.is_training)(x)
        x = jnp.mean(x, 1)
        x = nn.Dense(self.num_labels)(x)
        x = nn.softmax(x)
        return x
