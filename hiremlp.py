import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange
from utils import Droppath


def inner_region_rearrange(x, mode, pixel, n_pads):
    if mode == 'h':
        x = jnp.concatenate([x[:, -n_pads:, :, :], x], axis=1)
        return rearrange(x, 'b (p h) w c -> b h w (p c)',
                         p=pixel
                         )
    else:
        x = jnp.concatenate([x[:, :, -n_pads:, :], x], axis=2)
        return rearrange(x, 'b h (p w) c -> b h w (p c)',
                         p=pixel
                         )


def inner_region_restore(x, mode, pixel, n_pads):
    if mode == 'h':
        x = rearrange(x, 'b h w (p c) -> b (p h) w c',
                      p=pixel
                      )
        return x[:, n_pads:, :, :]
    else:
        x = rearrange(x, 'b h w (p c) -> b h (p w) c',
                      p=pixel
                      )
        return x[:, :, n_pads:, :]


'''
paper suggests to implement cross region rearrange by circular padding, but i used jnp.roll function for convenience
'''
def cross_region_rearrange(x, mode, s):
    if mode == 'h':
        return jnp.roll(x, s, 1)
    else:
        return jnp.roll(x, s, 2)


def cross_region_restore(x, mode, s):
    if mode == 'h':
        return jnp.roll(x, -s, 1)
    else:
        return jnp.roll(x, -s, 2)


class MLP(nn.Module):
    expansion_rate: int = 4

    @nn.compact
    def __call__(self, x):
        _, _, _, C = x.shape
        x = nn.Dense(C * self.expansion_rate)(x)
        x = nn.gelu(x)
        x = nn.Dense(C)(x)
        return x


class HireModule(nn.Module):
    deterministic: bool
    pixel_size: int
    s: int
    norm: str = 'batch'

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        pad_h = (self.pixel_size - H % self.pixel_size) % self.pixel_size
        pad_w = (self.pixel_size - W % self.pixel_size) % self.pixel_size

        # height direction
        h = cross_region_rearrange(x, 'h', self.s)
        h = inner_region_rearrange(h, 'h', self.pixel_size, pad_h)
        h = nn.Dense(C // 2,
                     use_bias=False
                     )(h)
        if self.norm == 'batch':
            h = nn.BatchNorm(self.deterministic)(h)
        elif self.norm == 'layer':
            h = nn.LayerNorm()(h)
        else:
            raise NotImplementedError('Batchnorm or Layernorm')
        h = nn.relu(h)
        h = nn.Dense(C * self.pixel_size)(h)
        h = inner_region_restore(h, 'h', self.pixel_size, pad_h)
        h = cross_region_restore(h, 'h', self.s)
        # width direction
        w = cross_region_rearrange(x, 'w', self.s)
        w = inner_region_rearrange(w, 'w', self.pixel_size, pad_w)
        w = nn.Dense(C // 2,
                     use_bias=False
                     )(w)
        if self.norm == 'batch':
            w = nn.BatchNorm(self.deterministic)(w)
        elif self.norm == 'layer':
            w = nn.LayerNorm()(w)
        else:
            raise NotImplementedError('Batchnorm or Layernorm')
        w = nn.relu(w)
        w = nn.Dense(C * self.pixel_size)(w)
        w = inner_region_restore(w, 'w', self.pixel_size, pad_w)
        w = cross_region_restore(w, 'w', self.s)
        # channel direction
        c = nn.Dense(C)(x)

        # split attention
        a = jnp.mean(h + w + c, axis=[1, 2])
        a = nn.Dense(C // 4)(a)
        a = nn.gelu(a)
        a = nn.Dense(C * 3)(a)
        a = jnp.expand_dims(nn.softmax(jnp.reshape(a, (B, C, 3)).transpose(2, 0, 1), axis=0), [2, 3])

        x = h * a[0] + w * a[1] + c * a[2]
        x = nn.Dense(C)(x)
        return x


class HireBlock(nn.Module):
    deterministic: bool
    pixel_size: int
    s: int
    survival_prob: float

    @nn.compact
    def __call__(self, x):
        residual = nn.BatchNorm(self.deterministic)(x)
        residual = HireModule(self.deterministic, self.pixel_size, self.s)(residual)
        x = Droppath(self.survival_prob, self.deterministic)(residual) + x
        residual = nn.BatchNorm(self.deterministic)(x)
        residual = MLP()(residual)
        x = Droppath(self.survival_prob, self.deterministic)(residual) + x
        return x

