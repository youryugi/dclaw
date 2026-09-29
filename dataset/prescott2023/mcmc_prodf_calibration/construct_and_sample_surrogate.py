#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:51:32 2022

@author: krbarnhart
"""

from scipy.interpolate import LinearNDInterpolator

import pandas as pd
import numpy as np
import emcee


# set up functions needed for MCMC
# this follows the example provided by emcee documentation. 

def maximum_likelihood_objective_function(theta):
    chi, tau_y = theta

    S = interpolator(chi.flat, tau_y.flat)  # objective function

    Sprime = (ND + NPR) * np.log(2*np.pi) - np.log(determinant_of_weight_matrix) + S
    # Hill and Tiedeman Eq 3.3
    
    return np.atleast_1d(Sprime)


def log_likelihood(theta):
    ll = -0.5*maximum_likelihood_objective_function(theta)
    
    if ll.size == 1:
        return ll[0]
    else:
        return ll


def log_prior(theta):
    chi, tau_y = theta
    chi_low = 0.0 > chi.flat
    chi_hi = chi_max < chi.flat
    tau_low = 0.0 > tau_y.flat
    tau_hi = tau_y_max < tau_y.flat
    lp  = np.zeros(chi_low.size)
    lp[chi_low] = -np.inf
    lp[chi_hi] = -np.inf
    lp[tau_low] = -np.inf
    lp[tau_hi] = -np.inf
    
    if lp.size == 1:
        return lp[0]
    else:
        return lp


def log_probability(theta):
    lp = log_prior(theta)
    ll = log_likelihood(theta)
    lpb = lp + ll 
    
    return lpb


# Read in simulation results
df = pd.read_csv("results_training.csv", header=None, names=["chi", "tau_y", "similarity"])
    
# limits of chi and tau_y (this is the permissible space for the MCMC)
chi_min = df.chi.min()
chi_max = df.chi.max()
tau_y_min = df.tau_y.min()
tau_y_max = df.tau_y.max()

# variables for determining the number of degrees of freedom.
ND = 3 # Data (consider each of TP, FP, FN as an independent observation)
NPR = 0 # number of priors
NP = 2 # Number of parameters. 

        
ndf = ND + NPR - NP
determinant_of_weight_matrix = 1.


labels = [r"$\chi$", r"$\tau_y$"]

# Generate posterior for a factor of 20. This value was found to match the probabilities
# expected from the expert opinion solicitation.

factor = 20

df['one_minus_TS'] = (1-df.similarity)/2
df.sort_values("one_minus_TS", inplace=True)

df["Sb"] = factor * df.one_minus_TS


# Make interpolator for model response surface
interpolator = LinearNDInterpolator((df.chi, df.tau_y), df.Sb, fill_value = factor) 
# using simple linear interpolation because there are so many observations and surface is smooth.

# choose the nwalker best simulation results as the starting locations. 
nwalkers = 32
pos = np.vstack((df.chi.iloc[:nwalkers], df.tau_y.iloc[:nwalkers])).T
_, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=())
sampler.run_mcmc(pos, 20000, progress=True);

#https://emcee.readthedocs.io/en/stable/tutorials/line/
tau = sampler.get_autocorr_time()
print("autocorrelation: ", tau)
discard = 5 * int(np.mean(tau))
thin = int(0.5 * np.mean(tau))
print("discard: ", discard)
print("thin:", thin)
# thin based on autocorrelation time. 
flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
print(flat_samples.shape)


posterior_df = pd.DataFrame(dict(chi=flat_samples[:, 0],
                                    tau_y=flat_samples[:, 1]))

# you should be able to sample from this (uniformly) and use that for 
# prediction confidence intervals. 
posterior_df.to_csv("posterior_factor{}.csv".format(factor))
