
import numpy as onp

import jax
import jax.numpy as np
from jax import random

from jax.scipy.special import logsumexp as lse

def normalize(x):
    return x - lse(x, -1, keepdims=True)

def mse(g, g_true):
    # g_true: N x theta
    # g: Sz x Sx x N x theta
    g_hat = g.mean(0).mean(0)
    bias = g_true - g_hat
    var = sample_var(g).sum(0) / (g.shape[0] - 1)
    return np.einsum("ns,nt->nst", bias, bias) + var

def sample_var(g):
    g = g.mean(1)
    gc = g - g.mean(0)
    return np.einsum("zns,znt->znst", gc, gc)

def marg_var(S):
    return S.diagonal(axis1=1,axis2=2)
