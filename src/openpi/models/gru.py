import flax.nnx as nnx
import jax
import jax.numpy as jnp

class GRUCell(nnx.Module):
    def __init__(self, channels: int, kernel_size: int = 3, *, rngs: nnx.Rngs, param_dtype: jnp.dtype = jnp.bfloat16
    ):