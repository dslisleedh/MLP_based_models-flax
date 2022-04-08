from utils import Droppath
import jax
import jax.numpy as jnp
import flax.linen as nn


class SpatialShiftAttention(nn.Module):
    n_filters: int
    k: int = 3

    def spatial_shift1(self, x):
        _, h, w, c = x.shape
        x = x.at[:, :, 1:, :c // 4].set(x[:, :, :w - 1, :c // 4]) \
            .at[:, :, :w - 1, c // 4:c // 2].set(x[:, :, 1:, c // 4:c // 2]) \
            .at[:, 1:, :, c // 2:c // 4 * 3].set(x[:, :h - 1, :, c // 2:c // 4 * 3]) \
            .at[:, :h - 1, :, c // 4 * 3:].set(x[:, 1:, :, c // 4 * 3:])
        return x

    def spatial_shift2(self, x):
        _, h, w, c = x.shape
        x = x.at[:, 1:, :, :c // 4].set(x[:, :h - 1, :, :c // 4]) \
            .at[:, :h - 1, :, c // 4:c // 2].set(x[:, 1:, :, c // 4:c // 2]) \
            .at[:, :, 1:, c // 2:c // 4 * 3].set(x[:, :, :w - 1, c // 2:c // 4 * 3]) \
            .at[:, :, :w - 1, c // 4 * 3:].set(x[:, :, 1:, c // 4 * 3:])
        return x

    @nn.compact
    def __call__(self, x):
        b, h, w, _ = x.shape
        x = nn.Dense(self.n_filters * self.k)(x)
        x1 = self.spatial_shift1(x[:, :, :, :self.n_filters])
        x2 = self.spatial_shift2(x[:, :, :, self.n_filters:self.n_filters*2])
        x3 = x[:, :, :, self.n_filters*2:]
        x = jnp.reshape(jnp.stack([x1, x2, x3], axis=1), (b, self.k, -1, self.n_filters))
        a = jnp.sum(jnp.sum(x, 1), 1)
        a_hat = nn.Dense(self.n_filters*self.k,use_bias=False)(nn.gelu(nn.Dense(self.n_filters, use_bias=False)(a)))
        a_hat = jnp.reshape(a_hat, (b, self.k, self.n_filters))
        a_bar = nn.softmax(a_hat, axis=1)
        attention = jnp.expand_dims(a_bar, axis=-2)
        x = attention * x
        x = jnp.reshape(jnp.sum(x, axis=1), (b, h, w, self.n_filters))
        x = nn.Dense(self.n_filters)(x)
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
    deterministic: bool

    @nn.compact
    def __call__(self, x):
        y = Droppath(self.survival_prob, self.deterministic)(SpatialShiftAttention(self.n_filters)(nn.LayerNorm()(x))) + x
        z = Droppath(self.survival_prob, self.deterministic)(CMMLP(self.n_filters)(nn.LayerNorm()(y))) + y
        return z


class S2MLPv2(nn.Module):
    n_classes: int
    training: bool
    patch_size = [7, 2]
    n_filters = [256, 512]
    n_blocks = [7, 17]
    stochastic_depth = .1

    @nn.compact
    def __call__(self, x):
        survival_prob = jnp.linspace(0., self.stochastic_depth, sum(self.n_filters))
        for i in range(len(self.patch_size)):
            x = nn.Conv(self.n_filters[i],
                        kernel_size=(self.patch_size[i], self.patch_size[i]),
                        strides=self.patch_size[i],
                        padding='VALID',
                        use_bias=False
                        )(x)
            for k in range(self.n_blocks[i]):
                x = S2Blockv2(self.n_filters[i],
                              survival_prob[i * (self.n_blocks[0]) + k],
                              not self.training
                              )(x)
        x = jnp.mean(x, axis=[1, 2])
        x = nn.Dense(self.n_classes)(x)
        x = nn.softmax(x)
        return x
