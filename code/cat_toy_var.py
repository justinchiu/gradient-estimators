
import matplotlib.pyplot as plt

import numpy as onp

import jax
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random

from jax.scipy.special import logsumexp as lse

from np_lib import printoptions

seed = 1234

# unused
import tensorflow_probability as tfp
jtfp = tfp.experimental.substrates.jax

rng = random.PRNGKey(seed)

# omg, make this a monad, seriously.
def sample_gumbel(rng, shape, n=0):
    rng, rng_input = random.split(rng)
    shape = shape if n == 0 else (n,) + shape
    return rng, random.gumbel(rng_input, shape=shape)

def sample_relaxed(logits, g0):
    g = logits + g0
    return np.exp(g - lse(g, -1, keepdims=True))

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
    return logits - lse(logits, -1, keepdims=True)

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

    return (fz * probs).mean(0).sum()

def logp_x_z(theta, params, x, g):
    z = sample_hard(theta, g)
    Sz, Sx, N, Z = g.shape
    fz = f(params, np.arange(Z))[z, x]
    #logp_z = theta[np.arange(N), z] # non repeated theta
    #lpz = theta[None].repeat(Sz, 0)[np.arange(Sz)[:,None,None], np.arange(N), z]
    logp_z = theta[np.arange(Sz)[:,None,None], 0, np.arange(N), z]

    """
    # dbg code
    fz0 = f(params, z)
    fz1 = fz0[:, np.arange(Sx)[:,None], np.arange(N), x]
    assert np.allclose(fz, fz1)
    """

    # first order surrogate
    return (fz * logp_z).sum()

def logp_x_z_relaxed(theta, params, x, g, tau=1):
    z = sample_relaxed(theta, g)
    Sz, Sx, N, Z = g.shape
    fz = f_relaxed(params, z, tau)
    fz = fz[:, np.arange(Sx)[:,None], np.arange(N), x]
    return fz.sum()

# straight through?
###

def init_params(rng, Z, X):
    H = 127
    rng, rng_input = random.split(rng)
    emb = random.normal(rng_input, shape=(Z, H))
    rng, rng_input = random.split(rng)
    proj = random.normal(rng_input, shape=(H,X))
    return emb, proj

n_trials = 2
Z = 4
X = 13
sx = 128
sx = 32
sz = 16

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

d_logp_x = jit(grad(logp_x))
d_logp_x_z = jit(grad(logp_x_z))
d_logp_x_z_relaxed = jit(grad(logp_x_z_relaxed))

d = d_logp_x(theta, params, xs)
ds = d_logp_x_z(theta[None, None].repeat(sz, 0).repeat(sx, 1), params, xs, g)
dp = d_logp_x_z_relaxed(theta[None, None].repeat(sz, 0).repeat(sx, 1), params, xs, g, 0.7)

def sample_var(g):
    g = g.mean(1)
    gc = g - g.mean(0)
    return np.einsum("zns,znt->znst", gc, gc)

def marg_var(S):
    return S.diagonal(axis1=1,axis2=2)

print("Preliminary gradient check")
# gradient: Sz x Sx x N x |theta|
with printoptions(precision=5, suppress=True):
    print("True theta")
    print(np.exp(true_theta))
    print("theta")
    print(np.exp(theta))

    print()
    print("True gradient")
    print(d)

    print()
    print("SF gradient")
    print("Mean")
    print(ds.mean(0).mean(0))
    Ss = sample_var(ds).sum(0) / (sz-1)
    print("Sample Cov")
    print(Ss)
    print("Marg var")
    print(marg_var(Ss))

    print()
    print("PW gradient")
    print("Mean")
    print(dp.mean(0).mean(0))
    Sp = sample_var(dp).sum(0) / (sz-1)
    print("Sample Cov")
    print(Sp)
    print("Marg var")
    print(marg_var(Sp))

print("We see that the SF gradients have means comparable to the true gradient, as expected.")
print("However the pathwise (PW) gradient estimators appears to be biased.")
print("Let's investigate the how different the PW estimator is from the true gradient.")
print()
print("Analysis 0: Mean-squared Error (MSE)")
print("Using a naive temperature of 0 results in quite a large MSE.")

def mse(g, g_true):
    # g_true: N x theta
    # g: Sz x Sx x N x theta
    g_hat = g.mean(0).mean(0)
    bias = g_true - g_hat
    var = sample_var(g).sum(0) / (g.shape[0] - 1)
    return np.einsum("ns,nt->nst", bias, bias) + var

mse_s = mse(ds, d)
mse_p = mse(dp, d)
print(f"MSE of SF: {mse_s}")
print(f"MSE of PW: {mse_p}")

print("Let's check the MSE of the PW estimator as we vary the temperature.")
for tau in onp.linspace(0, 2, 41):
    dp = d_logp_x_z_relaxed(theta[None, None].repeat(sz, 0).repeat(sx, 1), params, xs, g, tau)
    mse_p = mse(dp, d)
    print(f"tr(MSE) of PW at tau={tau}:")
    print(mse_p.trace(axis1=1,axis2=2))

print("Compared to tr(MSE) of SF:")
print(mse_s.trace(axis1=1, axis2=2))

print("TODO")
print("Let's optimize the MSE of the PW estimator wrt temperature")

print("Analysis 1: ")

import pdb; pdb.set_trace()
