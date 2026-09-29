"""Forecaster class."""

import h5py
import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import pawn

class Forecaster():
    """A class to generate uncertainty-rated debris-flow inundation forecasts using ProDF and a WRF atmospheric model ensemble.
    
    Parameters
        ----------
        wrf_ensemble : WRFensemble
            object holding the WRF ensemble data
        runprodf_simulator : RunProDF
            object for running the ProDF simulations
            
    """
    def __init__(self, wrf_ensemble, runprodf_simulator):
        self.wrf = wrf_ensemble
        self.runprodf = runprodf_simulator

    def compute_inundated_area(self, depths_h5fname):
        """Computes the inundated area from an HDF5 file containing the simulated depths from a forecaster run.

        Parameters
        ----------
        depths_h5fname : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        inun_area = np.zeros(self.vol_samples.shape[0:-1])
        with h5py.File(depths_h5fname, "r") as h5f:
            for idx, x in np.ndenumerate(inun_area):
                data = h5f["depths"][(*idx, ...)]
                a = np.sum(data >= self.runprodf.thresh, dtype=np.float64)
                a *= (self.runprodf.dx ** 2)
                inun_area[idx] = a
        
        return inun_area

    def compute_mean_depth(self, depths_h5fname):
        """_summary_

        Parameters
        ----------
        depths_h5fname : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        mean_depth = np.zeros((self.runprodf.ny, self.runprodf.nx), dtype=np.float64)
        n_inundations = np.zeros((self.runprodf.ny, self.runprodf.nx), dtype=np.int64)
        with h5py.File(depths_h5fname, "r") as h5f:
            fshape = h5f["depths"].shape
            for j in range(fshape[0]):
                for k in range(fshape[1]):
                    data = h5f["depths"][j, k, ...]
                    inun = data >= self.runprodf.thresh
                    mean_depth[inun] += data[inun]
                    n_inundations[inun] += 1
        mean_depth[n_inundations > 0] /= n_inundations[n_inundations > 0]

        return mean_depth, n_inundations

    def load_forecast_results(self, directory):
        """Load forecast results from NumPy files.

        Parameters
        ----------
        directory : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        inun = np.load(directory + "/inundation.npy")
        self.vol_samples = np.load(directory + "/volume_samples.npy")
        self.prodf_samples = np.load(directory + "/prodf_samples.npy")
        return inun

    def forecast(self, n_samples_per_ensemble, h5fname=None):
        """_summary_

        Parameters
        ----------
        n_samples_per_ensemble : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        # Currently designed for a UniformSampler for the vol_sampler, expected to give samples as a multiplicative fraction of the actual volume samples

        n_ens = self.wrf.n_ens
        inun = np.zeros(self.runprodf.x.shape, dtype=np.float64)
        n_sims = np.int64(n_ens * n_samples_per_ensemble)
        n_start_pts = self.runprodf.volumeinput.shape[0]

        # arrays to save inputs and results
        self.vol_samples = np.zeros((n_ens, n_samples_per_ensemble, n_start_pts))
        self.prodf_samples = np.zeros((n_ens, n_samples_per_ensemble, 2))

        # create HDF5 file for output simulated depth maps
        if h5fname is not None:
            h5f = h5py.File(self.runprodf.workdir + "/" + h5fname, "w")
            dset = h5f.create_dataset(
                "depths",
                shape=(n_ens, n_samples_per_ensemble, self.runprodf.ny, self.runprodf.nx),
                dtype=np.float64,
                fillvalue=0.0,
                compression="gzip",
                compression_opts=9,
                chunks=(1, 1, self.runprodf.ny, self.runprodf.nx),
                )

        for j in range(n_ens):
            print(f"Ensemble member {j+1} of {n_ens} ")
            # Pull the samples for the simulations
            vol_scalar_samples = self.vol_sampler.get_samples(n_samples_per_ensemble)
            prodf_samples = self.prodf_sampler.get_samples(n_samples_per_ensemble)
            
            # Get the df volumes predicted by the jth ensemble member, and
            vol_input = np.copy(self.runprodf.volumeinput)
            vol_input[:, -1] = self.wrf.data[j, :, 3]
            # ...mask out the start points whose prob of df event is less than the threshold
            b = self.wrf._compute_binary_response[j, :]
            vol_input = vol_input[b==1, :]


            for k in range(n_samples_per_ensemble):
                
                prodf_params = prodf_samples[:, k]
                np.savetxt(self.runprodf.fparam, prodf_params, fmt='%.4f', delimiter=',')
                self.runprodf.prepareinputs()

                # Do this second because runprodf.prepareinputs() also calls prepare_vol_input(), we need to override it
                vols = np.copy(vol_input)
                vols[:, -1] *= vol_scalar_samples[k]
                self.runprodf.prepare_vol_input(vols)

                # save this iterations parameters; could be moved out of loop
                self.vol_samples[j, k, b==1] = vols[:, -1]
                self.prodf_samples[j, k, :] = prodf_params

                # Run ProDF
                self.runprodf.simulate(h5fname="results.h5")

                # Load final depth results and add to the inundation array
                with h5py.File(self.runprodf.workdir+"/results.h5","r") as h5f:
                    data = h5f["data"][0, ...]
                inun += (data >= self.runprodf.thresh).astype(np.int64)

                if h5fname is not None:
                    dset[j, k, :, :] = data
        
        inun /= n_sims
        
        if h5fname is not None:
            h5f.close()

        return inun

    def pawn_analyze(self, inundated_area, **pawn_kwargs):
        """PAWN Sensitivity analysis on Forecaster output using RunProDF inputs.

        PAWN sensitivity analysis from Pianosi & Wagener (2018), implemented in the SALib package.

        The analyzed model is X -> Y, where:
            X is a numpy array of the model parameters for every simulation (chi, tau, volume)
            Y is the model output (expected inundated area, though other metrics could be used)

        Parameters
        ----------
        inundated_area : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        problem = {
            'num_vars': 4,
            'names': ['chi', 'tau', 'vol', 'dummy'],
            'bounds': [[0.0, np.Inf],
                       [0.0, np.Inf],
                       [0.0, np.Inf],
                       [0.0, np.Inf]]
        }
        X = np.stack([
            self.prodf_samples[:,:,0].ravel(),
            self.prodf_samples[:,:,1].ravel(),
            self.vol_samples.sum(axis=-1).ravel(),
            np.random.uniform(0.0, 100.0, size=self.prodf_samples[:,:,0].size)]
            ).T
        Y = inundated_area
        Si = pawn.analyze(problem, X, Y, **pawn_kwargs)

        return Si

    def plot_inundatedarea_panel(self, inundated_area, **fig_kwargs):
        """Plot a 3-axis figure with inundated area on the Y axis, and volume, chi, tau on the x axes.

        Parameters
        ----------
        inundated_area : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        fig,ax = plt.subplots(1,3, **fig_kwargs)
        h0 = ax[0].scatter(self.vol_samples.sum(axis=-1), inundated_area, s=1)
        h1 = ax[1].scatter(self.prodf_samples[:,:,0].ravel(), inundated_area, c=np.log(self.vol_samples.sum(axis=-1)), s=1)
        h2 = ax[2].scatter(self.prodf_samples[:,:,1].ravel(), inundated_area, c=np.log(self.vol_samples.sum(axis=-1)), s=1)
        
        ax[0].set_yscale('log')
        ax[0].set_xscale('log')
        ax[0].set_xlabel("Summed volume samples")
        ax[0].set_ylabel("Total inundated area")
        ax[0].grid(visible=True, which="both")
        ax[1].set_yscale('log')
        ax[1].set_xlabel("chi")
        ax[1].grid(visible=True, which="both")
        ax[2].set_yscale('log')
        ax[2].set_xlabel("tau")
        ax[2].grid(visible=True, which="both")

        return fig

    def save_forecast_results(self, inun):
        """Save forecast results to NumPy files.

        Parameters
        ----------
        inun : _type_
            _description_
        """
        np.save(self.runprodf.workdir + "/inundation.npy", inun)
        np.save(self.runprodf.workdir + "/volume_samples.npy", self.vol_samples)
        np.save(self.runprodf.workdir + "/prodf_samples.npy", self.prodf_samples)

    def set_volume_sampler(self, sampler):
        """Set the input volume sampler.

        Parameters
        ----------
        sampler : Sampler subclass
            a subclass of Sampler that can sample from the input volume space
        """
        self.vol_sampler = sampler
    
    def set_prodf_sampler(self, sampler):
        """Set the ProDF parameter space sampler.

        Parameters
        ----------
        sampler : Sampler subclass
            a subclass of Sampler that can sample from the ProDF parameter space
        """
        self.prodf_sampler = sampler
