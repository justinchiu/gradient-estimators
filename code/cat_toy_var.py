import jax.numpy as np
from jax import grad, jit, vmap
from jax import random

from jax.scipy.special import lse

import tensorflow_probability as tfp

seed = 1234

jtfp = tfp.experimental.substrates.jax

rng = random.PRNGKey(seed)

#rng, rng_input = random.split(rng)
#g0 = jtfp.distributions.Gumbel(rng_input, 0, 1)

# omg, make this a monad, seriously.
def sample_relaxed(rng, logits):
    rng, rng_input = random.split(rng)
    g0 = random.gumbel(rng_input, shape=logits.shape)
    g = logits + g0
    return rng, g

def sample_hard(rng, logits):
    rng, rng_input = random.split(rng)
    g0 = random.gumbel(rng_input, shape=logits.shape)
    g = logits + g0
    return rng, g



n_trials = 256 
N = 128

shape = (n_trials, N)

rng, rng_input = random.split(rng)
true_theta = random.normal(rng_input, shape=shape)

rng, rng_input = random.split(rng)
theta = random.normal(rng_input, shape=shape)

# estimate gradients


