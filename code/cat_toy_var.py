
import matplotlib.pyplot as plt

import numpy as onp

import jax
import jax.numpy as np
from jax import value_and_grad, grad, jit 
from jax import random
from jax import lax

from utils import normalize, mse, sample_var, marg_var
from models import *
from np_lib import printoptions

import streamlit as st

seed = 1234

# unused
#import tensorflow_probability as tfp
#jtfp = tfp.experimental.substrates.jax

rng = random.PRNGKey(seed)

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
Z = 64
X = 256
sx = 32
sz = 16

params = init_params(rng, Z, X)

shape = (n_trials, Z)

rng, rng_input = random.split(rng)
true_theta = normalize(random.normal(rng_input, shape=shape) * 3)

rng, rng_input = random.split(rng)
theta = normalize(random.normal(rng_input, shape=shape))
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

# Subset relaxation

d_logp_x_z_relaxed_subset = jit(value_and_grad(logp_x_z_relaxed_subset), static_argnums=(4, 5))

pss, dpss = zip(*[
    d_logp_x_z_relaxed_subset(expanded_theta, params, xs, g, s, 0.5)
    for s in range(1, 5)
])

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

    for s, dps in enumerate(dpss):
        st.write()
        st.write(f"SPW gradient {s+1}")
        st.write("Mean")
        st.write(dps.mean(0).mean(0))
        Sp = sample_var(dps).sum(0) / (sz-1)
        st.write("Sample Cov")
        st.write(Sp)
        st.write("Marg var")
        st.write(marg_var(Sp))

st.write("MSE Analysis")
mse_s = mse(ds, d)
mse_p = mse(dp, d)
with printoptions(precision=3, suppress=True):
    st.write(f"tr(MSE) of SF: {mse_s.trace(axis1=1, axis2=2)}")
    st.write(f"tr(MSE) of PW: {mse_p.trace(axis1=1, axis2=2)}")
    for s, dps in enumerate(dpss):
        mse_ps = mse(dps, d)
        st.write(f"tr(MSE) of SPW{s+1}: {mse_ps.trace(axis1=1, axis2=2)}")

st.markdown("## Relaxed parts")

rzs, zs = sample_relaxed_part(expanded_theta, g, 2, 0.5)
out_relaxed = logp_x_z_relaxed(expanded_theta, params, xs, g, 0.5)
out = logp_x_z_relaxed_part(expanded_theta, params, xs, g, Z, 0.5)
out = logp_x_z_relaxed_part(expanded_theta, params, xs, g, Z // 2, 0.5)

d_logp_x_z_relaxed_part = jit(value_and_grad(logp_x_z_relaxed_part), static_argnums=(4, 5))

parts = [1, 4, 16, 32, 64 ]
pss, dpss = zip(*[
    d_logp_x_z_relaxed_part(expanded_theta, params, xs, g, s, 0.5)
    for s in parts
])

with printoptions(precision=3, suppress=True):
    for s, dps in enumerate(dpss):
        st.write()
        st.write(f"Subset Part Pathwise (SPPW) gradient {parts[s]}")
        st.write("Mean")
        st.write(dps.mean(0).mean(0))
        Sp = sample_var(dps).sum(0) / (sz-1)
        st.write("Sample Cov")
        st.write(Sp)
        st.write("Marg var")
        st.write(marg_var(Sp))

    for s, dps in enumerate(dpss):
        st.write(f"SPPW{parts[s]}")
        mean = dps.mean(0).mean(0)
        st.write(f"sum(bias^2) = {((mean - d) ** 2).sum(-1)}")
        Sp = sample_var(dps).sum(0) / (sz-1)
        st.write(f"sum(var) = {marg_var(Sp).sum(-1)}")
        mse_ps = mse(dps, d)
        st.write(f"tr(MSE) = {mse_ps.trace(axis1=1, axis2=2)}")

