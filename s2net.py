from utils import Droppath
import jax
import jax.numpy as jnp
import flax.linen as nn


class SpatialShift(nn.Module):
    group: int

    def spatial_shift(self, x):
        _, h, w, c = x.shape
        x = x.at[:, :, 1:, :c // 4].set(x[:, :, :w - 1, :c // 4]) \
            .at[:, :, :w - 1, c // 4:c // 2].set(x[:, :, 1:, c // 4:c // 2]) \
            .at[:, 1:, :, c // 2:c // 4 * 3].set(x[:, :h - 1, :, c // 2:c // 4 * 3]) \
            .at[:, :h - 1, :, c // 4 * 3:].set(x[:, 1:, :, c // 4 * 3:])
        return x

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
                    )(x)
        survival_prob = jnp.linspace(0., self.stochastic_depth, num=self.n)
        for prob in survival_prob:
            x = S2Block(self.c, self.r, prob, not self.is_training)(x)
        x = jnp.mean(x, [1, 2])
        x = nn.Dense(self.num_labels)(x)
        x = nn.softmax(x)
        return x