import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange
from utils import Droppath


class MLP(nn.Module):
    n_filters: int
    mlp_ratio: int = 3

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.n_filters * self.mlp_ratio)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.n_filters)(x)
        return x


class WeightedPermutator(nn.Module):
    qkv_bias: bool = False

    @nn.compact
    def __call__(self, x):
        shape = x.shape
        h = rearrange(x, 'b h w (s c) -> b c w (s h)',
                      s=shape[-1] // shape[1]
                      )
        h = nn.Dense(shape[-1], use_bias=self.qkv_bias)(h)
        h = rearrange(h, 'b c w (s h) -> b h w (s c)',
                      s=shape[-1] // shape[1]
                      )

        w = rearrange(x, 'b h w (s c) -> b h c (s w)',
                      s=shape[-1] // shape[2]
                      )
        w = nn.Dense(shape[-1], use_bias=self.qkv_bias)(w)
        w = rearrange(w, 'b h c (s w) -> b h w (s c)',
                      s=shape[-1] // shape[2]
                      )
        c = nn.Dense(shape[-1], use_bias=self.qkv_bias)(x)

        a = jnp.mean(rearrange(h + w + c, 'b h w c -> b c (h w)'), axis=2)  # b c
        a = nn.gelu(nn.Dense(shape[-1] // 4)(a))  # b 1/4*c
        a = nn.Dense(shape[-1] * 3)(a)  # b 3*c
        a = jnp.expand_dims(nn.softmax(a.reshape(shape[0], shape[-1], 3).transpose(2, 0, 1), axis=0), [2, 3])  # k b 1(h) 1(w) c

        x = h * a[0, :, :, :, :] + w * a[1, :, :, :, :] + c * a[2, :, :, :, :]  # b h w c
        x = nn.Dense(shape[-1])(x)  # b h w c
        return x


class PermutationBlock(nn.Module):
    n_filters: int
    survival_prob: float

    @nn.compact
    def __call__(self, x, deterministic: bool):
        x = Droppath(self.survival_prob, self.deterministic)(WeightedPermutator()(nn.LayerNorm()(x)), deterministic) + x
        x = Droppath(self.survival_prob, self.deterministic)(MLP(self.n_filters)(nn.LayerNorm()(x)), deterministic) + x
        return x


class ViP(nn.Module):
    is_training: bool
    n_labels: int
    stochastic_depth: float
    n_filters = [256, 512]
    patch_size = [7, 2]
    n_layers = [7, 17]

    @nn.compact
    def __call__(self, x):
        survival_prob = 1. - jnp.linspace(0., self.stochastic_depth, sum(self.n_layers))
        for i in range(len(self.patch_size)):
            x = nn.Conv(self.n_filters[i],
                        kernel_size=(self.patch_size[i], self.patch_size[i]),
                        strides=(self.patch_size[i], self.patch_size[i]),
                        use_bias=False
                        )(x)
            for n in range(self.n_layers[i]):
                x = PermutationBlock(self.n_filters[i],
                                     survival_prob[sum(self.n_filters[:i]) + n]
                                     )(x, deterministic=not self.is_training)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.n_labels,
                     kernel_init=nn.initializers.zeros
                     )(x)
        x = nn.softmax(x)
        return x
