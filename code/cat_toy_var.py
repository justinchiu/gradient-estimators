import jax
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random

from jax.scipy.special import logsumexp as lse

import tensorflow_probability as tfp

seed = 1234

# unused
jtfp = tfp.experimental.substrates.jax

rng = random.PRNGKey(seed)

# omg, make this a monad, seriously.
def sample_gumbel(rng, shape, n=0):
    rng, rng_input = random.split(rng)
    shape = shape if n == 0 else (n,) + shape
    return rng, random.gumbel(rng_input, shape=shape)

def sample_relaxed(logits, g0):
    g = logits + g0
    return g - lse(g, -1, keepdims=True)

def sample_hard(logits, g0):
    g = logits + g0
    return g.argmax(-1)

def f(params, z):
    # z is an index here
    emb, proj = params
    hid = emb[z]
    logits = hid.dot(proj)
    return logits - lse(logits, -1, keepdims=True)

def f_relaxed(params, z):
    # z is coefficients
    emb, proj = params
    hid = z @ emb
    logits = hid @ proj
    return logits - lse(logits, -1, keepdims=True)

def logp_x_z(theta, params, x, g):
    z = sample_hard(theta, g)
    return (f(params, z)[x]).sum()

def logp_x_z_relaxed(theta, params, x, g):
    z = sample_relaxed(theta, g)
    return f_relaxed(params, z)[x].sum()


def init_params(rng, Z, X):
    H = 127
    rng, rng_input = random.split(rng)
    emb = random.normal(rng_input, shape=(Z, H))
    rng, rng_input = random.split(rng)
    proj = random.normal(rng_input, shape=(H,X))
    return emb, proj

n_trials = 64 
Z = 128
X = 256
n_samples = 100

params = init_params(rng, Z, X)

shape = (n_trials, Z)

rng, rng_input = random.split(rng)
true_theta = random.normal(rng_input, shape=shape)

rng, rng_input = random.split(rng)
theta = random.normal(rng_input, shape=shape)

# get data
rng, g = sample_gumbel(rng, shape, n_samples)
zs = sample_hard(true_theta, g)
rng, g = sample_gumbel(rng, (n_trials, X), n_samples)
xs = sample_hard(f(params, zs), g)

# estimate gradients
rng, g = sample_gumbel(rng, shape, n_samples)
logp_x_z(theta, params, xs, g)
d = grad(logp_x_z)(theta, params, xs, g)
dr = grad(logp_x_z_relaxed)(theta, params, xs, g)

print(d.shape)
print(dr.shape)
