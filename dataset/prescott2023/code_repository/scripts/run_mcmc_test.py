# Test the Markov Chain Monte Carlo calibrated posterior distribution on the test watershed

__author__ = "Alexander Prescott"
__credits__ = ["Brady Gales", "Alexander Prescott","Kwang-Sung Jun","Luke McGuire"]
__status__ = "Development"

import argparse
import h5py
import numpy as np
from Debrisflows.RunProDF import RunProDF
from Debrisflows.Settings import Settings
from Debrisflows.Samplers import CSVProDFSampler


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_file", help="User settings file")   
    args = parser.parse_args()

    options = Settings.read(args.settings_file)

    # Load and create a sampler from the MCMC results CSV file
    prodf_sampler = CSVProDFSampler(options["csv_results"])

    # Get the N number of ProDF parameter samples and save to file
    prodf_samples = prodf_sampler.get_samples(options["number_of_samples"])
    np.savetxt(options["fparam"], prodf_samples.T, delimiter=",")

    # Create a RunProDF instance
    sim = RunProDF(**options)

    # Simulate with specified keyword arguments
    sim_kwargs = {}
    sim_kwargs["h5fname"] = options["sim_h5fname"]  # required
    if "sim_rand_vol_frac" in options.keys():
        sim_kwargs["rand_vol_frac"] = options["sim_rand_vol_frac"]
    if "sim_seed" in options.keys():
        sim_kwargs["seed"] = options["sim_seed"]

    sim.simulate(**sim_kwargs)

    if "results_h5fname" in options.keys():
        sim.write_results_to_HDF5("/".join([sim.workdir,options["results_h5fname"]]))
    else:
        sim.write_results_to_HDF5()
    
    # Compute and save uncertainty-rated inundation
    inun = np.zeros((sim.ny, sim.nx))
    with h5py.File(options["sim_h5fname"], "r") as h5f:
        dset = h5f["data"]
        ntrials = dset.shape[0]
        for j in range(ntrials):
            depth = dset[j, ...]
            inun += np.float64(depth >= sim.thresh)    
    inun /= np.float64(ntrials)
    np.save(options["workdir"] + "/inundation.npy", inun)
