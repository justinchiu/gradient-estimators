import jax
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random

from jax.scipy.special import logsumexp as lse

import tensorflow_probability as tfp

seed = 1234

jtfp = tfp.experimental.substrates.jax

rng = random.PRNGKey(seed)

#rng, rng_input = random.split(rng)
#g0 = jtfp.distributions.Gumbel(rng_input, 0, 1)

# omg, make this a monad, seriously.
def sample_relaxed(rng, logits, n=1):
    rng, rng_input = random.split(rng)
    g0 = random.gumbel(rng_input, shape=(n,) + logits.shape)
    g = logits + g0
    return rng, g

def sample_hard(rng, logits, n=0):
    rng, rng_input = random.split(rng)
    shape = logits.shape if n == 0 else (n,) + logits.shape
    g0 = random.gumbel(rng_input, shape=shape)
    g = logits + g0
    return rng, g.argmax(-1)

def f(params, z):
    # z is an index here
    emb, proj = params
    hid = emb[z]
    logits = hid.dot(proj)
    #return logits - lse(logits, -1, keepdims=True)
    return logits - lse(logits)

def f_relaxed(params, z):
    emb, proj = params
    import pdb; pdb.set_trace()
    hid = emb[z]
    return hid.dot(proj)

def init_params(Z, X):
    H = 127
    emb = random.normal(rng_input, shape=(Z, H))
    proj = random.normal(rng_input, shape=(H,X))
    return emb, proj

n_trials = 64 
Z = 128
X = 256
n_samples = 100

params = init_params(Z, X)

shape = (n_trials, Z)

rng, rng_input = random.split(rng)
true_theta = random.normal(rng_input, shape=shape)

rng, rng_input = random.split(rng)
theta = random.normal(rng_input, shape=shape)

# get data
rng, zs = sample_hard(rng, true_theta, n_samples)
rng, xs = sample_hard(rng, f(params, zs))

# estimate gradients
