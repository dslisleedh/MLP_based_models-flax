import jax
import jax.numpy as jnp
import flax.linen as nn
import einops


class MLP(nn.Module):
    expansion_rate: int = 4
    act=nn.gelu

    @nn.compact
    def __call__(self, x):
        c = x.shape[-1]
        x = nn.Dense(c * self.expansion_rate)(x)
        x = self.act(x)
        x = nn.Dense(c)(x)
        return x


class MixerBlock(nn.Module):

    @nn.compact
    def __call__(self, x):
        x_res = nn.LayerNorm()(x)
        x_res = jnp.swapaxes(x_res, 1, 2)
        x_res = MLP()(x_res)
        x_res = jnp.swapaxes(x_res, 1, 2)
        x = x + x_res
        x_res = nn.LayerNorm()(x)
        return x + MLP()(x_res)


class MlpMixer(nn.Module):
    num_classes: int
    num_blocks: int
    patch_size: int
    n_filters: int

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape
        s = self.patch_size
        assert (h % s == 0) and (w % s == 0)
        feature_map = nn.Conv(
            self.n_filters, (s, s), strides=(s, s)
        )(x)
        feature_map = einops.rearrange(
            feature_map, 'b h_s w_s c -> b (h_s w_s) c'
        )
        for _ in range(self.num_blocks):
            feature_map = MixerBlock()(feature_map)
        feature_map = nn.LayerNorm()(feature_map)
        feature_map = jnp.mean(feature_map, axis=1)
        y = nn.Dense(
            self.num_classes, kernel_init=nn.initializers.zeros
        )(feature_map)
        return nn.softmax(y)
