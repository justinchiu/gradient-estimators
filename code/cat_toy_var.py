
import numpy as onp

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

def f_relaxed(params, z, tau=1):
    # z is coefficients
    emb, proj = params
    hid = z @ emb
    logits = hid @ proj / tau
    return np.exp(logits - lse(logits, -1, keepdims=True))

def logp_x(theta, params, x):
    emb, proj = params
    S, N = x.shape
    Z, H = emb.shape

    # logits: Z x X
    logits = f(params, np.arange(Z))

    # fz: S x N x Z
    #fz = logits[np.arange(Z)[None,None],x[:,:,None]]
    fz = logits[:,x].transpose((1, 2, 0))
    probs = np.exp(theta)

    #print(fz.shape)
    #print(probs.shape)
    #print(probs.sum(-1))

    return (fz * probs).mean(0).sum()

def logp_x_z(theta, params, x, g):
    z = sample_hard(theta, g)
    Sz, Sx, N, Z = g.shape
    fz = f(params, np.arange(Z))[z, x]
    #fz = f(params, z)[np.arange(S)[:,None],np.arange(N)[None],x]
    logp_z = theta[np.arange(N), z]
    #print("fz.shape")
    #print(fz.shape)
    #print("logp_z.shape")
    #print(logp_z.shape)
    # first order surrogate
    return (fz * logp_z).mean(0).mean(0).sum()

def logp_x_z_relaxed(theta, params, x, g, tau=1):
    z = sample_relaxed(theta, g)
    Sz, Sx, N, Z = g.shape
    fz = f_relaxed(params, z, tau)
    fz = fz[:, np.arange(Sx)[:,None], np.arange(N), x]
    return fz.mean(0).mean(0).sum()

# straight through?
###

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

n_trials = 2
Z = 4
X = 13
sx = 128
sx = 4
sz = 3

params = init_params(rng, Z, X)

shape = (n_trials, Z)

rng, rng_input = random.split(rng)
true_theta = random.normal(rng_input, shape=shape)
true_theta = true_theta - lse(true_theta, -1, keepdims=True)

rng, rng_input = random.split(rng)
theta = random.normal(rng_input, shape=shape)
theta = theta - lse(theta, -1, keepdims=True)

# get data
rng, g = sample_gumbel(rng, shape, sx)
zs = sample_hard(true_theta, g)
rng, g = sample_gumbel(rng, (n_trials, X), sx)
xs = sample_hard(f(params, zs), g)

# estimate gradients
sample_shape = (sz, sx, n_trials, Z)
rng, g = sample_gumbel(rng, sample_shape)
#print("Sample shape")
#print(g.shape)
d = grad(logp_x)(theta, params, xs)
ds = grad(logp_x_z)(theta, params, xs, g)
dr = grad(logp_x_z_relaxed)(theta, params, xs, g, 1.2)

print("Raw gradients")
print(d)
print(ds)
print(dr)

import contextlib
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = onp.get_printoptions()
    onp.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        np.set_printoptions(**original)

print("Pretty gradients")
with printoptions(precision=5, suppress=True):
    print(d)
    print(ds)
    print(dr)

import pdb; pdb.set_trace()
