Scripts are run from the command line with Python and simulation options are passed from the command line via a user options file. The user options file specifies the details of the desired simulation that the given script reads-in and interprets.

For example, running ProDF simulations over a discretized grid on the training basins (Oak, San Ysidro, Buena Vista, Romero Creeks) is done with:

    python code_repository/scripts/run_simulation.py options_files/user_options_training_grid.py

To run the forecast:

    python code_repository/scripts/run_mcmc_forecast.py options_files/user_options_forecast.py