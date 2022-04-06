import jax
import jax.numpy as jnp
import flax.linen as nn


class MLP(nn.Module):
    features: int
    expansion_rate: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features * self.expansion_rate)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.features)(x)
        return x


class MixerBlock(nn.Module):
    n_patches: int
    n_features: int

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm()(x)
        y = jnp.swapaxes(y, 1, 2)
        y = MLP(self.n_patches, 4)(y)
        y = jnp.swapaxes(y, 1, 2)
        x = x + y
        y = nn.LayerNorm()(x)
        return x + MLP(self.n_features, 4)(y)


class MlpMixer(nn.Module):
    num_classes: int
    num_blocks: int
    patch_size: int
    num_patches: int
    n_features: int

    @nn.compact
    def __call__(self, x):
        s = self.patch_size
        b, h, w, c = x.shape
        featuremap = nn.Conv(self.n_features,
                             (s, s),
                             strides=(s, s)
                             )(x)
        featuremap = jnp.reshape(featuremap,
                                 (b, -1, self.n_features)
                                 )
        for _ in range(self.num_blocks):
            featuremap = MixerBlock(self.num_patches, self.n_features)(featuremap)
        featuremap = nn.LayerNorm()(featuremap)
        featuremap = jnp.mean(featuremap, axis=1)
        y = nn.Dense(self.num_classes,
                     kernel_init=nn.initializers.zeros
                     )(featuremap)
        return y
