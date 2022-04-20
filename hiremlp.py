import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange
from utils import Droppath


def inner_region_rearrange(x, mode, groups):
    if mode == 'h':
        return rearrange(x, 'b (g h) w c -> b g w (h c)',
                         g=groups
                         )
    else:
        return rearrange(x, 'b h (g w) c -> b g h (w c)',
                         g=groups
                         )


def inner_region_restore(x, mode, pixel_split_size):
    if mode == 'h':
        return rearrange(x, 'b g w (h c) -> b (g h) w c',
                         h=pixel_split_size
                         )
    else:
        return rearrange(x, 'b g h (w c) -> b h (g w) c',
                         w=pixel_split_size
                         )


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


class HireModule(nn.Module):
    deterministic: bool
    pixel_split_size: int
    s: int
    norm: str = 'batch'

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        g_H = H // self.pixel_split_size
        g_W = W // self.pixel_split_size

        # height direction
        h = cross_region_rearrange(x, 'h', self.s)
        h = inner_region_rearrange(h, 'h', g_H)
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
        h = nn.Dense(C * self.pixel_split_size)(h)
        h = inner_region_restore(h, 'h', self.pixel_split_size)
        h = cross_region_restore(h, 'h', self.s)
        # width direction
        w = cross_region_rearrange(x, 'w', self.s)
        w = inner_region_rearrange(w, 'w', g_W)
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
        w = nn.Dense(C * self.pixel_split_size)(w)
        w = inner_region_restore(w, 'w', self.pixel_split_size)
        w = cross_region_restore(w, 'w', self.s)
        # channel direction
        c = nn.Dense(C)(x)

        a = jnp.mean(h + w + c, axis=[1, 2])
        a = nn.Dense(C // 4)(a)
        a = nn.gelu(a)
        a = nn.Dense(C * 3)(a)
        a = jnp.expand_dims(nn.softmax(jnp.reshape(a, (B, C, 3)).transpose(2, 0, 1), axis=0), [2, 3])

        x = h * a[0] + w * a[1] + c * a[2]
        x = nn.Dense(C)(x)
        return x

