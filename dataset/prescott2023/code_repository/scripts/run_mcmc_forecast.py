# Run a forecast of debris flow inundation using the ProDF debris flow
# inundation model and an ensemble of atmospheric models that forecast future
# precipitation rates
# ProDF parameters are sampled from a distribution created by Katy Barnhart of
# the USGS with Monte Carlo Markov Chain software
# Volume samples for the forecast are sampled in [0.1, 10.0] * V, where V is
# the most likely estimator of debris flow volume

import argparse
import numpy as np
from Debrisflows.Forecaster import Forecaster
from Debrisflows.RunProDF import RunProDF
from Debrisflows.Samplers import LogUniformSampler, CSVProDFSampler
from Debrisflows.Settings import Settings
from Debrisflows.WRFensemble import WRFensemble


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_file", help="User settings file")   
    args = parser.parse_args()

    options = Settings.read(args.settings_file)

    # Build the ProDF parameter sampler
    prodf_sampler = CSVProDFSampler(options["csv_results"])

    # Build the volume sampler
    vol_sampler = LogUniformSampler(0.1, 10.0)

    # Create WRF ensemble and RunProDF object instances
    wrf = WRFensemble(options["wrf_file"])
    sim = RunProDF(**options)

    # Create forecaster and set the samplers
    forecaster = Forecaster(wrf, sim)
    forecaster.set_prodf_sampler(prodf_sampler)
    forecaster.set_volume_sampler(vol_sampler)

    # Run the forecast experiment with n samples taken per ensemble member
    inun = forecaster.forecast(options["n_samples_per_ensemble"], h5fname="forecast_depths.h5")

    np.save(options["workdir"] + "inundation.npy", inun)
    np.save(options["workdir"] + "volume_samples.npy", forecaster.vol_samples)
    np.save(options["workdir"] + "prodf_samples.npy", forecaster.prodf_samples)
    
