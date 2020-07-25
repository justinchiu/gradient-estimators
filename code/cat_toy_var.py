
from itertools import permutations

import matplotlib.pyplot as plt

import numpy as onp

import jax
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
from jax import lax

from jax.scipy.special import logsumexp as lse

from np_lib import printoptions

import streamlit as st

seed = 1234

# unused
#import tensorflow_probability as tfp
#jtfp = tfp.experimental.substrates.jax

rng = random.PRNGKey(seed)

# omg, make this a monad, seriously.
def sample_gumbel(rng, shape, n=0):
    rng, rng_input = random.split(rng)
    shape = shape if n == 0 else (n,) + shape
    return rng, random.gumbel(rng_input, shape=shape)

def sample_relaxed(logits, g0, tau=1):
    g = (logits + g0) / tau
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

def f_relaxed(params, z):
    # z is coefficients
    emb, proj = params
    hid = z @ emb
    logits = hid @ proj
    return logits - lse(logits, -1, keepdims=True)

def f_relaxed_subset(params, z, idxs):
    # z is coefficients
    emb, proj = params
    hid = np.einsum("abnz,abnzh->abnh", z, emb[idxs])
    logits = hid @ proj
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
    logp_z = theta[np.arange(Sz)[:,None,None], np.arange(Sx)[:,None], np.arange(N), z]

    # first order surrogate
    return (fz * logp_z).sum()

def logp_x_z_relaxed(theta, params, x, g, tau=1):
    z = sample_relaxed(theta, g, tau)
    Sz, Sx, N, Z = g.shape
    fz = f_relaxed(params, z)
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
expanded_theta = theta[None, None].repeat(sz, 0).repeat(sx, 1)

# get data
rng, g = sample_gumbel(rng, shape, sx)
zs = sample_hard(true_theta, g)
rng, g = sample_gumbel(rng, (n_trials, X), sx)
xs = sample_hard(f(params, zs), g)

# estimate gradients
sample_shape = (sz, sx, n_trials, Z)
rng, g = sample_gumbel(rng, sample_shape)
#st.write("Sample shape")
#st.write(g.shape)

d_logp_x = jit(grad(logp_x))
d_logp_x_z = jit(grad(logp_x_z))
d_logp_x_z_relaxed = jit(grad(logp_x_z_relaxed))

d = d_logp_x(theta, params, xs)
ds = d_logp_x_z(expanded_theta, params, xs, g)
dp = d_logp_x_z_relaxed(expanded_theta, params, xs, g, 0.5)

def sample_var(g):
    g = g.mean(1)
    gc = g - g.mean(0)
    return np.einsum("zns,znt->znst", gc, gc)

def marg_var(S):
    return S.diagonal(axis1=1,axis2=2)

st.markdown("## Preliminary gradient check")
# gradient: Sz x Sx x N x |theta|
with printoptions(precision=3, suppress=True):
    st.write("True theta")
    st.write(np.exp(true_theta))
    st.write("theta")
    st.write(np.exp(theta))

    st.write()
    st.write("True gradient")
    st.write(d)

    st.write()
    st.write("SF gradient")
    st.write("Mean")
    st.write(ds.mean(0).mean(0))
    Ss = sample_var(ds).sum(0) / (sz-1)
    st.write("Sample Cov")
    st.write(Ss)
    st.write("Marg var")
    st.write(marg_var(Ss))

    st.write()
    st.write("PW gradient")
    st.write("Mean")
    st.write(dp.mean(0).mean(0))
    Sp = sample_var(dp).sum(0) / (sz-1)
    st.write("Sample Cov")
    st.write(Sp)
    st.write("Marg var")
    st.write(marg_var(Sp))

st.write("We see that the SF gradients have means comparable to the true gradient, as expected.")
st.write("However the pathwise (PW) gradient estimators appears to be biased.")
st.write("Let's investigate the how different the PW estimator is from the true gradient.")
st.write("Analysis 0: Mean-squared Error (MSE)")
st.write("Using a temperature of 0.5 results in quite a large MSE.")

def mse(g, g_true):
    # g_true: N x theta
    # g: Sz x Sx x N x theta
    g_hat = g.mean(0).mean(0)
    bias = g_true - g_hat
    var = sample_var(g).sum(0) / (g.shape[0] - 1)
    return np.einsum("ns,nt->nst", bias, bias) + var

mse_s = mse(ds, d)
mse_p = mse(dp, d)
with printoptions(precision=3, suppress=True):
    st.write(f"MSE of SF: {mse_s}")
    st.write(f"MSE of PW: {mse_p}")

st.write("Let's check the MSE of the PW estimator as we vary the temperature.")
for tau in onp.linspace(0, 1, 21):
    dp = d_logp_x_z_relaxed(expanded_theta, params, xs, g, tau)
    mse_p = mse(dp, d)
    with printoptions(precision=3, suppress=True):
        st.write(f"tr(MSE) of PW at tau={tau}:")
        st.write(mse_p.trace(axis1=1,axis2=2))

st.write("Compared to tr(MSE) of SF:")
with printoptions(precision=3, suppress=True):
    st.write(mse_s.trace(axis1=1, axis2=2))

st.write("TODO")
st.write("Let's optimize the MSE of the PW estimator wrt temperature")

st.write("Analysis 1: ")
st.write("PASS")

#import pdb; pdb.set_trace()

# Subset relaxation

def sample_relaxed_subset_old(logits, g0, K=1, tau=1):
    Sz, Sx, N, Z = g0.shape
    g = logits + g0
    # get bottom Z-K to set to -inf.
    # TODO: check negative values are ok, otherwise can translate
    _, I = lax.top_k(-g, Z-K)
    # sugar for jax.ops.index_update
    g_masked = g.at[
        np.arange(Sz)[:,None,None,None],
        np.arange(Sx)[:,None,None],
        np.arange(N)[:,None],
        I,
    ].set(float("-inf"))
    # need to return top k indices to compute subset prob
    _, I = lax.top_k(g, K)
    return np.exp(g_masked - lse(g_masked, -1, keepdims=True)), I
    # TODO: grad failure here? masking [x] and float -inf [x]
     
def sample_relaxed_subset(logits, g0, K=1, tau=1):
    Sz, Sx, N, Z = g0.shape
    g = logits + g0
    # get bottom Z-K to set to -inf.
    # TODO: check negative values are ok, otherwise can translate
    # IS TOPK NOT DIFFERENTIABLE?
    #g_masked, I = lax.top_k(g, K)
    g_masked, I = lax.top_k(g, K)
    #import pdb; pdb.set_trace()
    """
    g_masked = g[
        np.arange(Sz)[:,None,None,None],
        np.arange(Sx)[:,None,None],
        np.arange(N)[:,None],
        I,
    ]
    import pdb; pdb.set_trace()
    """
    # works without topk
    return np.exp(g_masked - lse(g_masked, -1, keepdims=True)), I
    # TODO: grad failure here? masking [x] and float -inf [x]

jsample_relaxed_subset = jit(sample_relaxed_subset, static_argnums=(2,3))
# Note: expansion done to prevent reduction of gradient in order to compute cov
rzs, zs = sample_relaxed_subset(expanded_theta, g, 2, 0.7)


def log_cumsum_exp(x, dim=0):
    pass


# adapted from https://github.com/wouterkool/estimating-gradients-without-replacement/blob/master/bernoulli/gumbel.py
def log1mexp(x):
    # Computes log(1-exp(-|x|))
    # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    x = -np.abs(x)
    #return x
    return np.where(x > -0.693, np.log(-np.expm1(x)), np.log1p(-np.exp(x)))
    # TODO: grad failure here? where [x]

def all_perms(K):
    return np.array(list(permutations(range(K))))

def logp_subset(theta, zs):
    # last dimension of the zs indicates the selected elements
    # sparse index representation
    #
    # Wouter et al use the Gumbel representation to compute p(Sk) in
    # exponential time rather than factorial.
    # We do it in factorial time.
    Sz, Sx, N, K = zs.shape
    # Is there a better syntax for gather
    logp_z = theta[
        np.arange(Sz)[:,None,None,None],
        np.arange(Sx)[:,None,None],
        np.arange(N)[:,None],
        zs,
    ]

    # get denominator orderings
    perms = all_perms(K)
    logp = logp_z[..., perms]


    # cumlogsumexp would be more stable? but there are only two elements here...
    # sum_i p(b_i)
    a = logp.max(-1, keepdims=True)
    p = np.exp(logp - a)
    sbi = a + np.log(p.cumsum(-1) - p)

    logp_bs = logp.sum(-1) - log1mexp(sbi).sum(-1)
    # TODO: grad failure here? cumsum [HERE]
    logp_b = lse(logp_bs, -1)
    return logp_b
# /adaptation

logp_zs = logp_subset(expanded_theta, zs)

def logp_x_z_relaxed_subset(theta, params, x, g, K=1, tau=1):
    rzs, zs = jsample_relaxed_subset(theta, g, K, tau)
    Sz, Sx, N, Z = g.shape
    fz = f_relaxed_subset(params, rzs, zs)
    #fz = f_relaxed(params, rzs)
    fxz = fz[:, np.arange(Sx)[:,None], np.arange(N), x]
    logp_b = logp_subset(theta, zs)
    #return (lax.stop_gradient(fxz) * logp_b + fxz).sum()
   
    #rzso, zso = sample_relaxed_subset_old(theta, g, K, tau)
    #fz_old = f_relaxed(params, rzso)
    tz = sample_relaxed(theta, g, tau)
    tfz = f_relaxed(params, tz)
    tfxz = tfz[:, np.arange(Sx)[:,None], np.arange(N), x]

    return (fxz).sum()
    #return (tfxz).sum()
    # check correctness of below
    #return (np.exp(logp_b) * fz).sum()

ok = logp_x_z_relaxed_subset(expanded_theta, params, xs, g, 2, 0.5)

d_logp_x_z_relaxed_subset = jit(grad(logp_x_z_relaxed_subset), static_argnums=(4, 5))

dps = d_logp_x_z_relaxed_subset(expanded_theta, params, xs, g, 2, 0.5)
dps1 = d_logp_x_z_relaxed_subset(expanded_theta, params, xs, g, 1, 0.5)
print(dps1)
import pdb; pdb.set_trace()

with printoptions(precision=3, suppress=True):
    st.write("True theta")
    st.write(np.exp(true_theta))
    st.write("theta")
    st.write(np.exp(theta))

    st.write()
    st.write("True gradient")
    st.write(d)

    st.write()
    st.write("SF gradient")
    st.write("Mean")
    st.write(ds.mean(0).mean(0))
    Ss = sample_var(ds).sum(0) / (sz-1)
    st.write("Sample Cov")
    st.write(Ss)
    st.write("Marg var")
    st.write(marg_var(Ss))

    st.write()
    st.write("PW gradient")
    st.write("Mean")
    st.write(dp.mean(0).mean(0))
    Sp = sample_var(dp).sum(0) / (sz-1)
    st.write("Sample Cov")
    st.write(Sp)
    st.write("Marg var")
    st.write(marg_var(Sp))

    st.write()
    st.write("SPW gradient")
    st.write("Mean")
    st.write(dps.mean(0).mean(0))
    Sp = sample_var(dps).sum(0) / (sz-1)
    st.write("Sample Cov")
    st.write(Sp)
    st.write("Marg var")
    st.write(marg_var(Sp))
