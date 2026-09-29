# Run ProDF simulations

__author__ = "Alexander Prescott"
__credits__ = ["Brady Gales", "Alexander Prescott","Kwang-Sung Jun","Luke McGuire"]
__status__ = "Development"

import argparse
from Debrisflows.RunProDF import RunProDF
from Debrisflows.Settings import Settings


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("settings_file", help="User settings file")   
    args = parser.parse_args()

    options = Settings.read(args.settings_file)

    if "discretize" not in options.keys():
        options["discretize"] = False

    # Create a RunProDF instance
    
    sim = RunProDF(**options)

    if options["discretize"]:
        expected_keys = ["discretize_n", "discretize_range1", "discretize_range2"]
        if not all (k in options for k in expected_keys):
            raise KeyError(f"create_discretization requires all of these keyword arguments: {expected_keys}")

        # Discretize the parameter space
        sim.create_discretization(options["discretize_n"], FR=options["discretize_range1"], YS=options["discretize_range2"])

    # Simulate with specified keyword arguments
    sim_kwargs = {}
    if "sim_h5fname" in options.keys():
        sim_kwargs["h5fname"] = options["sim_h5fname"]
    if "sim_rand_vol_frac" in options.keys():
        sim_kwargs["rand_vol_frac"] = options["sim_rand_vol_frac"]
    if "sim_seed" in options.keys():
        sim_kwargs["seed"] = options["sim_seed"]

    sim.simulate(**sim_kwargs)

    if "results_h5fname" in options.keys():
        sim.write_results_to_HDF5("/".join([sim.workdir,options["results_h5fname"]]))
    else:
        sim.write_results_to_HDF5()
    
