from utils import Droppath
from einops import rearrange
import jax
import jax.numpy as jnp
import flax.linen as nn


def spatial_shift1(x):
    _, h, w, c = x.shape
    x = x.at[:, :, 1:, :c // 4].set(x[:, :, :w - 1, :c // 4]) \
            .at[:, :, :w - 1, c // 4:c // 2].set(x[:, :, 1:, c // 4:c // 2]) \
            .at[:, 1:, :, c // 2:c // 4 * 3].set(x[:, :h - 1, :, c // 2:c // 4 * 3]) \
            .at[:, :h - 1, :, c // 4 * 3:].set(x[:, 1:, :, c // 4 * 3:])
    return x


def spatial_shift2(x):
    _, h, w, c = x.shape
    x = x.at[:, 1:, :, :c // 4].set(x[:, :h - 1, :, :c // 4]) \
            .at[:, :h - 1, :, c // 4:c // 2].set(x[:, 1:, :, c // 4:c // 2]) \
            .at[:, :, 1:, c // 2:c // 4 * 3].set(x[:, :, :w - 1, c // 2:c // 4 * 3]) \
            .at[:, :, :w - 1, c // 4 * 3:].set(x[:, :, 1:, c // 4 * 3:])
    return x


class SpatialShiftAttention(nn.Module):
    k: int = 3

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape
        x = nn.Dense(c * self.k)(x)
        x = x.at[:, :, :, :c].set(spatial_shift1(x[:, :, :, :c])) \
            .at[:, :, :, c:c * 2].set(spatial_shift2(x[:, :, :, c:c * 2]))
        x = rearrange(x, 'b h w (k c) -> b k (h w) c',
                      k=self.k
                      )
        a = jnp.sum(x, (1, 2))
        a_hat = nn.Dense(c * self.k, use_bias=False)(nn.gelu(nn.Dense(c, use_bias=False)(a)))
        a_hat = jnp.reshape(a_hat, (b, self.k, c))
        a_bar = nn.softmax(a_hat, axis=1)
        attention = jnp.expand_dims(a_bar, axis=-2)
        x = attention * x
        x = jnp.reshape(jnp.sum(x, axis=1), (b, h, w, c))
        x = nn.Dense(c)(x)
        return x


class CMMLP(nn.Module):
    n_filters: int
    expansion_rate: int = 3

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.n_filters * self.expansion_rate)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.n_filters)(x)
        return x


class S2Blockv2(nn.Module):
    n_filters: int
    survival_prob: float

    @nn.compact
    def __call__(self, x, deterministic: bool):
        y = Droppath(self.survival_prob)(SpatialShiftAttention()(nn.LayerNorm()(x)), deterministic) + x
        z = Droppath(self.survival_prob)(CMMLP(self.n_filters)(nn.LayerNorm()(y)), deterministic) + y
        return z


class S2MLPv2(nn.Module):
    n_classes: int
    is_training: bool = False
    patch_size = [7, 2]
    n_filters = [256, 512]
    n_blocks = [7, 17]
    stochastic_depth = .1

    @nn.compact
    def __call__(self, x):
        survival_prob = 1. - jnp.linspace(0., self.stochastic_depth, sum(self.n_blocks))
        for i in range(len(self.patch_size)):
            x = nn.Conv(self.n_filters[i],
                        kernel_size=(self.patch_size[i], self.patch_size[i]),
                        strides=self.patch_size[i],
                        padding='VALID',
                        use_bias=False
                        )(x)
            for k in range(self.n_blocks[i]):
                x = S2Blockv2(self.n_filters[i],
                              survival_prob[sum(self.n_filters[:i]) + k],
                              )(x, deterministic=not self.is_training)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.n_classes)(x)
        x = nn.softmax(x)
        return x
