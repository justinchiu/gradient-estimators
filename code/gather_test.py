import jax.numpy as np
from jax.lax import gather, GatherDimensionNumbers

from numpy.random import choice, randn


shape = (5,4,3)

x = randn(*shape)

z = choice(shape[-1], size=shape[:-1])

y = x[
    np.arange(shape[0])[:,None],
    np.arange(shape[1]),
    z,
]

print("Correct answer")
print(y)

print("Wrong shapes")
print(gather(x, z[:,:,None], GatherDimensionNumbers((0,2,), (1,), (2,)), (1,1,1)).shape)
print(gather(x, z[:,:,None], GatherDimensionNumbers((0,1,), (2,), (2,)), (1,1,1)).shape)

print("Right shape and answer with gather:")
print(gather(
    x,
    z[:,:,None],
    GatherDimensionNumbers((2,), (0,1,), (2,)), (1,1,1)).squeeze(-1),
)

print("Totally wrong answer with incorrect indexing")
print(x[0, 0, z])
