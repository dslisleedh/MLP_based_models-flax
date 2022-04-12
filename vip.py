import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange
from utils import Droppath


class MLP(nn.Module):
    n_filters: int
    deterministic: bool
    mlp_ratio: int = 3
    dropout_rate: float = 0.

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.n_filters * self.mlp_ratio)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=self.deterministic)(x)
        x = nn.Dense(self.n_filters)(x)
        x = nn.Dropout(self.dropout_rate, deterministic=self.deterministic)(x)
        return x


class WeightedPermutator(nn.Module):
    deterministic: bool
    n_segmentations: int = 8
    qkv_bias: bool = False
    attn_droprate: float = 0.
    proj_droprate: float = 0.


    @nn.compact
    def __call__(self, x):
        shape = x.shape
        h = rearrange(x, 'b h w (s c) -> b c w (s h)',
                      s=self.n_segmentations
                      )
        h = nn.Dense(shape[-1], use_bias=self.qkv_bias)(h)
        h = rearrange(h, 'b c w (s h) -> b h w (s c)',
                      s=self.n_segmentations
                      )

        w = rearrange(x, 'b h w (s c) -> b h c (s w)',
                      s=self.n_segmentations
                      )
        w = nn.Dense(shape[-1], use_bias=self.qkv_bias)(w)
        w = rearrange(w, 'b h c (s w) -> b h w (s c)',
                      s=self.n_segmentations
                      )
        c = nn.Dense(shape[-1], use_bias=self.qkv_bias)(x)

        a = jnp.mean(rearrange(h + w + c, 'b h w c -> b c (h w)'), axis=2)
        a = nn.gelu(nn.Dropout(self.attn_droprate, deterministic=self.deterministic)(nn.Dense(shape[-1] // 4)(a)))
        a = nn.Dropout(self.attn_droprate, deterministic=self.deterministic)(nn.Dense(shape[-1] * 3)(a))
        a = jnp.expand_dims(nn.softmax(a.reshape(shape[0], shape[-1], 3).transpose(2, 0, 1), axis=0), [2,3])

        x = h * a[0, :, :, :, :] + w * a[1, :, :, :, :] + c * a[2, :, :, :, :]
        x = nn.Dropout(self.proj_droprate, deterministic=self.deterministic)(nn.Dense(shape[-1])(x))
        return x

