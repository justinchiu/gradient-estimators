
from itertools import permutations

import jax.numpy as np
from jax import random
from jax import lax

from jax.scipy.special import logsumexp as lse

from utils import normalize

# Sampling
 
def sample_gumbel(rng, shape, n=0):
    rng, rng_input = random.split(rng)
    shape = shape if n == 0 else (n,) + shape
    return rng, random.gumbel(rng_input, shape=shape)

def sample_relaxed(logits, g0, tau=1):
    g = (logits + g0) / tau
    return np.exp(normalize(g))

def sample_hard(logits, g0):
    g = logits + g0
    return g.argmax(-1)
     
def sample_relaxed_subset(logits, g0, K=1, tau=1):
    Sz, Sx, N, Z = g0.shape
    g = logits + g0
    g_masked, I = lax.top_k(g, K)
    # works without topk
    return np.exp(normalize(g_masked)), I

def sample_relaxed_subset(logits, g0, K=1, tau=1):
    Sz, Sx, N, Z = g0.shape
    g = logits + g0
    g_masked, I = lax.top_k(g, K)
    # works without topk
    return np.exp(normalize(g_masked)), I
 
def sample_relaxed_part(logits, g0, K=1, tau=1):
    Sz, Sx, N, Z = g0.shape
    C = Z // K
    g = logits + g0
    # logits sorted in ascending order
    # sort by logits not perturbed logits
    # so operation is deterministic
    I_sorted = logits.argsort(-1)
    g_sorted = g[
        np.arange(Sz)[:,None,None,None],
        np.arange(Sx)[:,None,None],
        np.arange(N)[:,None],
        I_sorted,
    ]
    Ik = I_sorted.reshape(Sz, Sx, N, C, K)
    gk = g_sorted.reshape(Sz, Sx, N, C, K)
    gk_agg = lse(gk, -1)
    Ik_agg = gk_agg.argmax(-1)

    g_out = gk[
        np.arange(Sz)[:,None,None,None,None],
        np.arange(Sx)[:,None,None,None],
        np.arange(N)[:,None,None],
        Ik_agg[:,:,:,None,None],
        :,
    ][:,:,:,0,0]
    I_out = Ik[
        np.arange(Sz)[:,None,None,None,None],
        np.arange(Sx)[:,None,None,None],
        np.arange(N)[:,None,None],
        Ik_agg[:,:,:,None,None],
        :,
    ][:,:,:,0,0]
    # works without topk
    return np.exp(normalize(g_out)), I_out
 
# Models

def f(params, z):
    # z is an index here
    emb, proj = params
    hid = emb[z]
    logits = hid.dot(proj)
    return normalize(logits)

def f_relaxed(params, z):
    # z is coefficients
    emb, proj = params
    hid = z @ emb
    logits = hid @ proj
    return normalize(logits)

def f_relaxed_subset(params, z, idxs):
    # z is coefficients
    emb, proj = params
    hid = np.einsum("abnz,abnzh->abnh", z, emb[idxs])
    logits = hid @ proj
    return normalize(logits)

def logp_x(theta, params, x):
    theta = normalize(theta)

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
    theta = normalize(theta)
    z = sample_hard(theta, g)
    Sz, Sx, N, Z = g.shape
    fz = f(params, np.arange(Z))[z, x]
    #logp_z = theta[np.arange(N), z] # non repeated theta
    #lpz = theta[None].repeat(Sz, 0)[np.arange(Sz)[:,None,None], np.arange(N), z]
    logp_z = theta[np.arange(Sz)[:,None,None], np.arange(Sx)[:,None], np.arange(N), z]

    # first order surrogate
    return (fz * logp_z).sum()

def logp_x_z_relaxed(theta, params, x, g, tau=1):
    theta = normalize(theta)
    z = sample_relaxed(theta, g, tau)
    Sz, Sx, N, Z = g.shape
    fz = f_relaxed(params, z)
    fz = fz[:, np.arange(Sx)[:,None], np.arange(N), x]
    return fz.sum()


## PROPER SUBSET SAMPLING for subset relaxation
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

def logp_unordered_subset(theta, zs):
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
    #a = logp.max(-1, keepdims=True)
    #p = np.exp(logp - a)
    #sbi0 = a + np.log(p.cumsum(-1) - p)

    # slow implementation, the above seems wrong
    sbis = [np.log(np.zeros(logp[..., 0].shape))]
    for i in range(K-1):
        sbis.append(np.logaddexp(sbis[-1], logp[..., i]))
    sbi = np.stack(sbis, -1)

    logp_bs = logp.sum(-1) - log1mexp(sbi).sum(-1)
    logp_b = lse(logp_bs, -1)
    return logp_b
# /adaptation

def logp_ordered_subset(theta, zs):
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

    sbis = [np.log(np.zeros(logp_z[..., 0].shape))]
    for i in range(K-1):
        sbis.append(np.logaddexp(sbis[-1], logp_z[..., i]))
    sbi = np.stack(sbis, -1)

    logp_b = logp_z.sum(-1) - log1mexp(sbi).sum(-1)
    return logp_b

def logp_x_z_relaxed_subset(theta, params, x, g, K=1, tau=1):
    theta = normalize(theta)
    rzs, zs = sample_relaxed_subset(theta, g, K, tau)
    Sz, Sx, N, Z = g.shape
    fz = f_relaxed_subset(params, rzs, zs)
    fxz = fz[:, np.arange(Sx)[:,None], np.arange(N), x]
    logp_b = logp_unordered_subset(theta, zs)
    #logp_b = logp_ordered_subset(theta, zs)
    return (lax.stop_gradient(fxz) * logp_b + fxz).sum()

def logp_x_z_relaxed_part(theta, params, x, g, K=1, tau=1):
    theta = normalize(theta)
    rzs, zs = sample_relaxed_part(theta, g, K, tau)
    Sz, Sx, N, Z = g.shape
    fz = f_relaxed_subset(params, rzs, zs)
    fxz = fz[:, np.arange(Sx)[:,None], np.arange(N), x]
    logp_b = lse(theta[
        np.arange(Sz)[:,None,None,None],
        np.arange(Sx)[:,None,None],
        np.arange(N)[:,None],
        zs,
    ], -1)
    return (lax.stop_gradient(fxz) * logp_b + fxz).sum()



