"""User options file for running the RunProDF class with samples taken from a Monte Carlo Markov Chain calibration of the ProDF parameter space for the purpose of testing the calibration."""

import numpy as np

workdir = "../"  # change this to where you want model output written to
ftopo = "../input_data/thomas_preevent_5mresample_fill_meters.txt"
fdep = "../input_data/montecito_inundation_5mresample.txt"
fparam= "../input_data/temp_input_flowmobilityparams.txt"

# dictionary of debris flow start points + volumes, [UTMN,UTME,vol]
volinputdict = {
    "Montecito": np.asarray([[3815127,256284,179000],[3815100,256878,52000]]),  # Cold Springs & Hot Springs (Montecito Creek)
    "Oak_SanYsidro": np.asarray([[3815054,257898,10000],[3815013,259074,297000]]),  # Oak Creek, San Ysidro Creek
    "BuenaVista_Romero": np.asarray([[3814978,260110,41000],[3814906,262038,100000]]),  # Buena Vista Creek, Romero Creek
}

# dictionary of debris flow mapped deposit values
depvalsdict = {
    "Montecito": 1,
    "Oak_SanYsidro": 2,
    "BuenaVista_Romero": 3,
}

# Testing watershed(s)
watersheds = ['Montecito']

# Name of the MCMC calibration results CSV file
csv_results = "../mcmc_prodf_calibration/posterior_factor20_trimmed.csv"

# Specify number of ProDF samples to take (i.e. # of ProDF simulations)
number_of_samples = 10000

# Write simulated depth maps to file
sim_h5fname = "depths.h5"