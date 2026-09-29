import numpy as np
from operator import itemgetter
import subprocess
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os


class RunProDF:
    """A class to run the ProDF simulation."""

    def __init__(
        self,
        volinputdict=None,
        depvalsdict=None,
        watersheds=None,
        workdir="./",
        ftopo="thomas_preevent_5mresample_fill_meters.txt",
        fdep="montecito_inundation_5mresample.txt",
        fparam="ThomasFireInputs.txt",
        thresh=0.1,
        **kwargs,
    ):
        """Initialize the RunProDF simulation.

        Parameters
        ----------

        volinputdict : dict, optional
            all watershed keys with volume start points ordered as [UTM_N,UTM_E,volume], by default None
        depvalsdict : dict, optional
            all watershed keys with corresponding values as given in fdep, by default None
        watersheds : list of strings, optional
            selected watershed keys to simulate, by default None
        workdir : str, optional
            path to work directory for all I/O files, by default "./"
        ftopo : str, optional
            path to input topography DEM formatted as ESRI ASCII raster, by default "thomas_preevent_5mresample_fill_meters.txt"
        fdep : str, optional
            path to mapped deposits formatted as ESRI ASCII, by default "montecito_inundation_5mresample.txt"
        fparam : str, optional
            path to input parameter sets as rows, 1st column=chi and 2nd column=tauy, by default "ThomasFireInputs.txt"
        thresh : float, optional
            threshold value to classify cells in simulation as inundated, by default 0.1
        """
        # Dictionary with watershed keys and volume start point values as numpy arrays
        if volinputdict is None:
            vol = {}
            vol['Montecito'] = np.asarray([[3815127,256284,179000],[3815100,256878,52000]])  # Cold Springs & Hot Springs (Montecito Creek) 
            vol['Oak_SanYsidro'] = np.asarray([[3815054,257898,10000],[3815013,259074,297000]])  # Oak Creek, San Ysidro Creek
            vol['BuenaVista_Romero'] = np.asarray([[3814978,260110,41000],[3814906,262038,100000]])  # Buena Vista Creek, Romero Creek
            self.volinputdict = vol
        else:
            self.volinputdict = volinputdict
        # Dictionary that maps values in the deposits input (fdep) to watersheds
        if depvalsdict is None:
            depvals = {}
            depvals['Montecito'] = 1
            depvals['Oak_SanYsidro'] = 2
            depvals['BuenaVista_Romero'] = 3
            self.depvalsdict = depvals
        else:
            self.depvalsdict = depvalsdict
        set1 = self.volinputdict.keys()
        set2 = self.depvalsdict.keys()
        if len(set1 ^ set2) > 0:
            raise KeyError("input dictionary keys do not match")
        # Watersheds to use in simulations
        if watersheds is None:
            watersheds = [wshed for wshed in self.volinputdict.keys()]
        self.workdir = workdir  # work_dir for ProDF I/O and RunProDF output
        self.ftopo = ftopo  # input topo Esri ASCII file
        self.fdep = fdep  # mapped debris flow deposits Esri ASCII file
        self.fparam = fparam  # input list of parameters
        self.thresh = thresh  # threshold value for simulated inundation

        # Fixed Parameters
        self.rho=2000  # [kg/m^3]  density from Iverson (1997)
        self.gravity=9.81  # [m^2/s]
        self.c1=0.135  # empirical constant from Rickenmann (1999)
        self.c2=0.78  # empirical constant from Rickenmann (1999)
        self.mu=300  # [Pa s] viscosity from Iverson (1997) (not needed but C code will expect it)

        # Read topo and grid info
        self.x, self.y, self.z, self.hdr = self.arcgridread(self.ftopo)
        # Load data from header dictionary, note that all input arcgridread() files must be on same rectilinear grid
        self.dx = self.hdr['cellsize']
        self.nx = self.hdr['ncols']
        self.ny = self.hdr['nrows']
        
        # Prepare simulation with selected watersheds
        self.select_watersheds(watersheds)

        self._pspace1 = None
        self._pspace2 = None

    def arcgridread(self, filename):
        """Read 2D data from an Esri ASCII formatted raster file.

        Parameters
        ----------
        filename : str
            path to file

        Returns
        -------
        x : NumPy ndarray
            UTM East coordinates at every raster node
        y : NumPy ndarray
            UTM North coordinates at every raster node
        z : Numpy ndarray
            data read from filename at every raster node
        hdr : dictionary
            key-value pairs read from the header of filename


        Raises
        ------
        Exception
            If data read from filename is not two dimensional.
        Exception
            If data shape does not match description in its header.
        """        
        # First, load the header information and assign values to keys in the hdr dictionary
        hdr = {"ncols":"","nrows":"","xllcorner":"","yllcorner":"","cellsize":"","nodata_value":""}
        with open(filename) as f:
            #print("reading arcgrid file: " + filename)
            for i in range(len(hdr)):
                key, val = next(f).split()
                hdr[key.lower()] = np.float64(val)

        # Convert hdr ints to NumPy types
        hdr['ncols'] = np.int64(hdr['ncols'])
        hdr['nrows'] = np.int64(hdr['nrows'])
        # Load data into a Numpy array
        z = np.loadtxt(filename, skiprows=len(hdr))
        if z.shape != (hdr['nrows'], hdr['ncols']):
            raise Exception("arcgridread error: "+filename+" header nrows x ncols inconsistent with its data")
        z[z == hdr['nodata_value']] = np.NaN
        
        # Make x, y
        input_delta_x = hdr['cellsize']
        input_delta_y = -input_delta_x
        input_lon_w = hdr['xllcorner'] + 0.5*hdr['cellsize']
        input_lon_e = input_lon_w + input_delta_x*(hdr['ncols']-1)
        input_lat_s = hdr['yllcorner'] + 0.5*hdr['cellsize']
        input_lat_n = input_lat_s + (hdr['nrows']-1)*hdr['cellsize']
        xx = np.arange(input_lon_w, input_lon_e+input_delta_x, input_delta_x)
        yy = np.arange(input_lat_n, input_lat_s+input_delta_y, input_delta_y)
        x, y = np.meshgrid(xx, yy)

        return x, y, z, hdr

    def create_discretization(self, m, FR=(15, 50), YS=(50, 500)):
        """Discretize the 2D simulation parameter space.

        Parameters
        ----------
        m : int
            number of nodes per dimension
        """
        self.pspace1 = np.linspace(FR[0], FR[1], m)  # 1D mesh in FR dimension
        self.pspace2 = np.linspace(YS[0], YS[1], m)  # 1D mesh in YS dimension
        pgrid1, pgrid2 = np.meshgrid(self.pspace1, self.pspace2)  # meshgrid to get all discretized nodes
        pgridlist = np.stack((pgrid1, pgrid2), -1).reshape(-1, 2)  # turn meshgrid results into one array of shape (m*m, 2)
        np.savetxt(self.workdir + "/ThomasFireInputs.txt", pgridlist, fmt='%.4f', delimiter=',')  # write parameters to file for simulation
        self.prepareinputs()  # reads in  ThomasFireInputs.txt in a structured way; duplicates some work but it's minor

    def load_results_from_HDF5(self, fname):
        """Load grid and simulation data/metrics from an HDF5 file.

        Loads data from an HDF5 file created with self.write_results_to_HDF5().

        Parameters
        ----------
        fname : str
            File to read
        """
        with h5py.File(fname, "r") as h5f:
            # Load the grid data
            grp1 = h5f["grid_data"]
            self.x = grp1["x"][()]
            self.y = grp1["y"][()]
            self.z = grp1["z"][()]
            self.dx = grp1["dx"][()]
            self.nx = grp1["nx"][()]
            self.ny = grp1["ny"][()]
            self.deposits = grp1["deposits"][()]

            # Load the simulation data
            grp2 = h5f["simulation_data"]
            wsheds = grp2.attrs["watersheds"].split()
            self.watersheds = [i for i in wsheds if i in self.volinputdict]
            self.rho = grp2["fixedparams"][0]
            self.gravity = grp2["fixedparams"][1]
            self.c1 = grp2["fixedparams"][2]
            self.c2 = grp2["fixedparams"][3]
            self.mu = grp2["fixedparams"][4]
            self.simparams = grp2["params"][()]
            self.volumeinput = grp2["volumeinput"][()]
            self.xstart = grp2["xstart"][()]
            self.ystart = grp2["ystart"][()]
            self.alpha = grp2["alpha"][()]
            self.beta = grp2["beta"][()]
            self.gamma = grp2["gamma"][()]
            self.omega = grp2["omega"][()]
            self.inundatedarea = grp2["inundatedarea"][()]
            if "pspace1" in grp2.keys():
                self.pspace1 = grp2["pspace1"][()]
            if "pspace2" in grp2.keys():
                self.pspace2 = grp2["pspace2"][()]

    @staticmethod
    def neg_log_likelihood(simmed, mapped, eps=1.0e-7):
        """Computes the negative log likelihood of simulated inundation map with 2D mapped deposits.
        
        If simmed has more than 2 dimensions, they must be to the left of the 2D spatial dimensions.
        That is, if mapped.shape = (800,1265), simmed must have shape like (n1,...,nn,800,1265).
        This allows the function to accept and return variably sized arrays.

        Parameters
        ----------
        simmed : _type_
            _description_
        mapped : _type_
            _description_
        eps : float
            a small positive number to prevent log(0)

        Returns
        -------
        _type_
            _description_
        """
        nll = -mapped*np.log((simmed + eps) / (1 + 2 * eps))
        nll -= np.logical_not(mapped) * np.log((np.logical_not(simmed) + eps) / (1 + 2 * eps))
        nll = nll.sum(axis=(-2,-1))  # negative indices to allow additional dimensions
        return nll

    def plot_discretesampling_metric(self, ax, plotvec=None, **kwargs):
        """Simple method to plot a metric defined on the 2D parameter discritization.
        """
        m = self.pspace1.size
        if plotvec is None:
            plotarr = self.prob.reshape((m,m))
        else:
            plotarr = plotvec.reshape((m,m))

        out = ax.pcolor(self.pspace1, self.pspace2, plotarr, **kwargs)
        
        return out

    def plot_inun(self, ax, *args, xlabel="UTM East", ylabel="UTM North", title="Simulated Inundation", **kwargs):
        """Simple method to plot the simulated inundation.

        Parameters
        ----------
        inun : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        plt1 = ax.pcolor(*args, **kwargs)
        ax.set_title(title, fontsize=20)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return plt1

    def plot_inun_default(self, inun, **pcolor_kwargs):
        
        fig = plt.figure(figsize=(10.0,10.0))
        ax = fig.add_subplot(111, aspect="equal")
        h1 = self.plot_inun(ax, self.x, self.y, inun, **pcolor_kwargs)
        fig.colorbar(mappable=h1, fraction=0.03, pad=0.04)

        return fig, ax, h1

    def plot_mappeddeposits(self, plotarr=None, title="Mapped Deposits"):
        """Plot mapped watershed deposits and debris flow start points.

        Returns
        -------
        plt.figure object
            handle for the plot figure
        """
        if plotarr is None:
            plotarr = np.int64(self.deposits)
        fig1 = plt.figure(figsize=(10.0,10.0))
        ax1 = fig1.add_subplot(111, aspect='equal')
        plt1 = ax1.pcolor(self.x, self.y, plotarr)
        ax1.scatter(self.x[0,self.xstart], self.y[self.ystart,0], c='w')
        ax1.set_title(title, fontsize=20)
        ax1.set_xlabel("UTM East")
        ax1.set_ylabel("UTM North")
        # Custom legend for mapped deposits, https://stackoverflow.com/questions/25482876
        colors = [plt1.cmap(plt1.norm(value)) for value in np.unique(plotarr[plotarr>0])]
        names = self.watersheds
        patches = [mpatches.Patch(color=colors[i], label=names[i]) for i in range(len(names))]
        patches.append(Line2D([0], [0], marker='o', color='w', markeredgecolor='k', label='Debris flow start points'))
        ax1.legend(handles=patches,bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)

        return fig1

    def plot_mapcompare(self, inun, ax=None, thresh=0.5, title="Model Performance", **pcolor_kwargs):
        """Plot comparison of simulated inundation and mapped deposits.

        Parameters
        ----------
        inun : _type_
            _description_
        thresh : float, optional
            _description_, by default 0.5

        Returns
        -------
        _type_
            _description_
        """
        if inun.shape != (self.ny, self.nx):
            raise ValueError("Simulated inundation array must be 2D with shape (self.ny, self.nx).")

        simmed = (inun >= thresh).astype(np.int64)
        mapped = (self.deposits > 0).astype(np.int64)
        alpha, beta, gamma, omega = self.similarity_index(simmed, mapped)
        alldepositcompare = (simmed & mapped) + 2*(np.logical_not(simmed) & mapped) + 3*(simmed & np.logical_not(mapped))
        
        plotarr2 = alldepositcompare
        if ax is None:
            fig3 = plt.figure(figsize=(10.0,10.0))
            ax2 = fig3.add_subplot(111, aspect='equal')
        else:
            fig3 = None
            ax2 = ax
        plt2 = ax2.pcolor(self.x, self.y, plotarr2, **pcolor_kwargs)
        ax2.set_title(title, fontsize=20)
        ax2.set_xlabel('UTM East')
        ax2.set_ylabel('UTM North')
        # Custom legend for mapped deposits, https://stackoverflow.com/questions/25482876
        colors = [plt2.cmap(plt2.norm(value)) for value in np.unique(plotarr2)]
        names = ['True Negative','True Positive','False Negative','False Positive']
        patches = [mpatches.Patch(color=colors[i], label=names[i]) for i in range(len(names))]
        ax2leg = ax2.legend(handles=patches,bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
        # Add text box, https://matplotlib.org/3.3.4/gallery/recipes/placing_text_boxes.html
        txtstr = '\n'.join((f"similarity score", str(omega)))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        txtbox = ax2.text(260000, 3.812e6, txtstr, fontsize=14, verticalalignment='top', bbox=props)

        if fig3 is None:
            return plt2

        return fig3

    def plot_similarityindex(self, ax, xlabel="Flow Resistance Coefficient", ylabel = "Yield Strength", title=None, best_pts=True, **kwargs):
        """Plots the similarity index from discretized ProDF parameter simulations.

        Parameters
        ----------
        ax : _type_
            _description_
        xlabel : str, optional
            _description_, by default "Flow Resistance Coefficient"
        ylabel : str, optional
            _description_, by default "Yield Strength"

        Returns
        -------
        _type_
            _description_
        """
        m = self.pspace1.size
        z = self.omega.reshape((m,m))
        obj = self.plot_discretesampling_metric(ax, plotvec=z, **kwargs)

        if best_pts:
            p1, p2 = np.meshgrid(self.pspace1, self.pspace2)
            idx = np.argmax(z)
            ax.scatter(p1.ravel()[idx],p2.ravel()[idx],s=100,c='k')
            print(f"Max Similarity Index  of {z.ravel()[idx]} from optimal set:")
            print(p1.ravel()[idx],p2.ravel()[idx])

        if title is None:
            title = ' '.join(['Similarity Index of',*self.watersheds])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        return obj

    def prepareinputs(self):
        """Generates input text files for ProDF simulation."""
        # Write input topo to file
        topo = np.copy(self.z)  # copy so we don't change the stored NaNs
        topo[np.isnan(topo)] = self.hdr['nodata_value']
        topoin = topo.ravel()
        topoin[np.isnan(topoin)] = self.hdr['nodata_value']
        np.savetxt(self.workdir + "/topoin.txt", topoin, fmt='%.3f')

        # Prepare 2D volume array; C order, array indices as [y, x]
        self.prepare_vol_input()

        # Read parameter sets
        if os.path.isfile(self.fparam):
            self.simparams = np.loadtxt(self.fparam, delimiter=',')
            self.ntrials, self.nparams = self.simparams.shape
        else:
            raise ValueError(f"input parameter file {self.fparam} not found.")

    def prepare_simulation(self, h5fname, rand_vol_frac, seed):
        
        h5f = None
        rng = None

        # Metrics arrays
        self.alpha = np.zeros((self.ntrials,))  # overlap
        self.beta = np.zeros((self.ntrials,))  # underestimation
        self.gamma = np.zeros((self.ntrials,))  # overestimation
        self.omega = np.zeros((self.ntrials,))  # similarity score
        self.inundatedarea = np.zeros((self.ntrials,))  # total simulated inundated area

        # Create HDF5 file for results if a filename is given
        if h5fname is not None:
            h5fpath = "/".join([self.workdir, h5fname])
            h5f = h5py.File(h5fpath, "w")

            # For large datasets, intelligent chunking is req'd for efficient processing
            # h5py will auto-chunk when compression is turned on unless we specify
            # Auto-chunking splits along every dimension, generally making for inefficient post-processing
            # For us, dataset slices will only ever be taken along the first axis
            # Thus we want the 2nd/3rd axes to be stored contiguously
            # Thus chunks should be of shape (nchnk, ny, nx), where 1 <= nchnk <= ntrials
            # If I recall correctly, nchnk doesn't need to perfectly divide ntrials but cannot be larger than it
            maxchnk = 2**18  # number of np.float32s to get 1 MiB size
            chnk1 = np.int64(maxchnk / (self.ny * self.nx))
            chnk1 = np.max([chnk1, np.int64(1)])
            chnk1 = np.min([chnk1, self.ntrials])
            # in effect, this auto-chunking will always make chkn1=1 for the Montecito study because input raster dimensions
            # are 800*1265 ==> 3.9 MiB 
            # From h5py docs, "It’s recommended to keep the total size of your chunks between 10 KiB and 1 MiB, larger for larger datasets."
            # Thus maybe chunks of 3.9 MiB in a dataset of many GBs is better

            h5f.create_dataset(
                "data",
                shape=(self.ntrials, self.ny, self.nx),
                dtype=np.float32,
                fillvalue=np.nan,
                compression="gzip",
                compression_opts=9,
                chunks=(chnk1, self.ny, self.nx)
            )
            print("Writing simulation depth results to HDF5 file: "+h5fpath, flush=True)

        if rand_vol_frac is not None:
            # Initialize the NumPy random number generator
            rng = np.random.default_rng(seed)
        
        return h5f, rng

    def prepare_vol_input(self, volinput=None):
        if volinput is None:
            volinput = self.volumeinput
        volume = np.zeros((self.ny,self.nx))
        numstartpts = volinput.shape[0]
        utm_start = volinput[:, 0:2]
        xdiff = np.tile(self.x[0,:], (numstartpts,1)).T - np.tile(utm_start[:,1],(1,1))
        self.xstart = np.argmin(np.abs(xdiff), axis=0)
        ydiff = np.tile(self.y[:,0], (numstartpts,1)).T - np.tile(utm_start[:,0],(1,1))
        self.ystart = np.argmin(np.abs(ydiff), axis=0)
        volume[self.ystart, self.xstart] = volinput[:, 2]
        # Write volume to input file
        np.savetxt(self.workdir + "/volumein.txt", volume.ravel(), fmt='%.1f')

    def process_sim_metrics(self, inun, i):
        """Compute the metrics associated with the input simulated inundation map.

        This function called within iterative loop of self.simulate()

        Parameters
        ----------
        inun : _type_
            _description_
        i : int
            index of the current iterate
        """
        simmed = np.asarray((inun>=self.thresh), dtype=np.int64)
        mapped = np.asarray((self.deposits>0), dtype=np.int64)
        alpha, beta, gamma, omega = self.similarity_index(simmed, mapped)
        self.alpha[i] = alpha
        self.beta[i] = beta
        self.gamma[i] = gamma
        self.omega[i] = omega
        # Total simulated inundated area
        self.inundatedarea[i] = simmed.sum()*self.dx*self.dx

    def rand_vol(self, rng, frac=0.3):
        
        if (np.abs(frac) > 1.0) or (np.abs(frac) < 0.0):
            raise ValueError("frac must be between 0 and 1")
        volin = np.copy(self.volumeinput)
        a = -1.0 * frac * volin[:,-1]
        b = frac * volin[:,-1]
        volin[:, -1] += rng.uniform(a, b, volin.shape[0])

        return volin

    def select_watersheds(self, watersheds):
        """Mask the input volume data and mapped deposits to selected watersheds.

        Calls self.prepareinputs() as well.

        Parameters
        ----------
        watersheds : list of strings
            The selected watersheds for the simulation
        """
        # Store the selected watersheds
        self.watersheds = watersheds

        # Assign flow volume to starting point(s)
        # Watershed input volumes stored as values in a dictionary, must be Numpy arrays
        # Numpy arrays have number of rows equal to the number of starting points for each watershed
        # Must have 3 columns ordered as: UTM North, UTM East, Volume [m^3]

        # Store volume start pointsin a stacked array
        vollist = itemgetter(*self.watersheds)(self.volinputdict)
        self.volumeinput = np.vstack([i for i in vollist]).astype(np.float64)

        # Next, mask the mapped deposits
        # Read-in mapped deposits
        self.deposits = self.arcgridread(self.fdep)[2]
        self.deposits[np.isnan(self.deposits)] = 0

        # Remove unneeded mapped deposits
        deplist = itemgetter(*self.watersheds)(self.depvalsdict)  # pulls the values from depvals for the keys in watersheds
        self.deposits[~np.in1d(self.deposits.ravel(),np.asarray(deplist)).reshape(self.deposits.shape)] = 0.0  # this line of magic searches for values in deposits that don't match a value drawn from the previous line, sets them to zero

        # Trim upstream upstream extent of mapped deposits to coincide with DF starting points
        # !!Note: move to pre-processing
        for wshed in self.watersheds:
            ys = self.volinputdict.get(wshed)
            if ys.ndim==1:
                ymin = ys[0]
            else:
                ymin = np.min(ys[:,0])
            self.deposits[(self.deposits==self.depvalsdict[wshed]) & (self.y>ymin)]=0

        # Read parameter sets, generate input text files for ProDF C code
        self.prepareinputs()

    def simulate(self, h5fname=None, rand_vol_frac=None, seed=None):
        """Runs the ProDF model.

        Parameters
        ----------
        h5fname : str, optional
            provide a file name ending with .h5 to write depths to HDF5 file, by default None
            this file will be written to the path: "/".join([self.workdir, h5fname])

        """

        # Prepare for simulation
        h5f, rng = self.prepare_simulation(h5fname, rand_vol_frac, seed)

        # Run simulations
        cname = "ProDF_22Sep2022.c"
        cpath = "/".join([os.path.dirname(os.path.abspath(__file__)), cname])
        exepath = "/".join([self.workdir, os.path.splitext(cname)[0]]) + ".exe"
        subprocess.run(["gcc","-o",exepath,cpath,"-lm","-O","-w"])
        for i in range(self.ntrials):
            print('Trial Number: ' + str(i+1) + ' of ' + str(self.ntrials), flush=True)
            
            # Set up input parameters and write to file
            chi=self.simparams[i,0]  # Empirical resistance coefficient
            tauy=self.simparams[i,1]  # [Pa] yield strength from Iverson (2003)
            startinfo = np.asarray([self.dx, self.nx, self.ny, self.hdr["nodata_value"], self.c1, self.c2, self.gravity, self.rho, self.mu, tauy, chi])
            np.savetxt(self.workdir + "/startinfo.txt", startinfo, fmt="%.6f")

            # Update volume input, optional
            if rng is not None:
                rand_vol = self.rand_vol(rng, frac=rand_vol_frac)  # update volume inputs stochastically in uniform range [(1-frac)*V0,(1+frac)*V0]
                self.prepare_vol_input(rand_vol)
                # print(f"Initial volumes: {self.volumeinput}")
                # print(f"Random volumes: {rand_vol}")

            # Run simulation
            subprocess.run([exepath, self.workdir])

            # Process results and metrics
            finaldepth = np.loadtxt(self.workdir + "/depthoutfinal.txt")
            self.process_sim_metrics(finaldepth, i)

            # Write HDF5 file, optional
            if h5f is not None:
                h5f["data"][i, :, :] = finaldepth

        print('Finished simulation of '+str(self.ntrials)+' parameter sets.', flush=True)

        # Close HDF5 file, if opened; always remember to close the HDF5 file!
        if h5f is not None:
            h5f.close()

        # Save simulation results summary
        summary = np.zeros((self.ntrials, 7))
        summary[:, 0:2] = self.simparams
        summary[:, 2] = self.alpha
        summary[:, 3] = self.beta
        summary[:, 4] = self.gamma
        summary[:, 5] = self.omega
        summary[:, 6] = self.inundatedarea
        np.savetxt(self.workdir + '/summary.txt', summary, fmt='%.8f', delimiter=',')

    @staticmethod
    def similarity_index(simmed, mapped):
        """Compute the similarity index of simulated inundation map with mapped deposits.

        If simmed has more than 2 dimensions, they must be to the left of the 2D spatial dimensions.
        That is, if mapped.shape = (800,1265), simmed must have shape like (n1,...,nn,800,1265).
        This allows the function to accept and return variably sized arrays.

        Parameters
        ----------
        simmed : _type_
            _description_
        mapped : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        overlap = np.sum(simmed & mapped, axis=(-2,-1))  # True Positives
        underestimation = np.sum(np.logical_not(simmed) & mapped, axis=(-2,-1))  # False Negatives
        overestimation = np.sum(simmed & np.logical_not(mapped), axis=(-2,-1))  # False Positives
        total = overlap + underestimation + overestimation
        a = overlap / total  # True Positive fraction
        b = underestimation / total  # False Negative fraction
        g = overestimation / total  # False Positive fraction
        o = a - b - g  # Similarity index

        return a, b, g, o

    def write_results_to_HDF5(self, h5fpath=None):
        """Write grid and simulation data/metrics to an HDF5 file. 

        Parameters
        ----------
        h5fpath : str, optional
            Path to the file to write, by default written to: self.workdir + 'results.h5'
        """
        if h5fpath is None:
            h5fpath = self.workdir + "/results.h5"

        with h5py.File(h5fpath, "w") as h5f:
            # Write the grid data
            grp1 = h5f.create_group('grid_data')
            grp1.create_dataset('x', data=self.x)  # (ny, nx)
            grp1.create_dataset('y', data=self.y)  # (ny, nx)
            grp1.create_dataset('z', data=self.z)  # (ny, nx)
            grp1.create_dataset("dx", data=self.dx)  # np.float64
            grp1.create_dataset("nx", data=self.nx)  # np.int64
            grp1.create_dataset("ny", data=self.ny)  # np.int64
            grp1.create_dataset('deposits', data=self.deposits)  # (ny, nx), dtype=np.int64
            
            # Write the simulation data
            grp2 = h5f.create_group('simulation_data')
            wshedstr = "The simulated watersheds are: " + " ".join(self.watersheds)
            fixedparams=np.asarray([self.rho, self.gravity, self.c1, self.c2, self.mu])
            grp2.attrs["watersheds"] = wshedstr
            grp2.create_dataset('fixedparams', data=fixedparams)
            grp2.create_dataset('params', data=self.simparams)  # (ntrials, 2)
            grp2.create_dataset('volumeinput', data=self.volumeinput)  # length varies
            grp2.create_dataset('xstart', data=self.xstart)  # length varies
            grp2.create_dataset('ystart', data=self.ystart)  # length varies
            grp2.create_dataset('alpha', data=self.alpha)  # (ntrials,)
            grp2.create_dataset('beta', data=self.beta)  # (ntrials,)
            grp2.create_dataset('gamma', data=self.gamma)  # (ntrials,)
            grp2.create_dataset('omega', data=self.omega)  # (ntrials,)
            grp2.create_dataset('inundatedarea', data=self.inundatedarea)  # (ntrials,)
            if self.pspace1 is not None:
                grp2.create_dataset("pspace1", data=self.pspace1)
            if self.pspace2 is not None:
                grp2.create_dataset("pspace2", data=self.pspace2)

    # Getter and Setters for protected variables

    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("alpha must be NumPy nd-array")
        if values.ndim != 1:
            raise ValueError("alpha must be 1D")
        self._alpha = values
    
    @property
    def beta(self):
        return self._beta
    
    @beta.setter
    def beta(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("beta must be NumPy nd-array")
        if values.ndim != 1:
            raise ValueError("beta must be 1D")
        self._beta = values

    @property
    def c1(self):
        return self._c1
    
    @c1.setter
    def c1(self, values):
        try:
            values = np.float64(values)
        except TypeError:
            print("c1 must be castable to NumPy float64")
            raise
        if values.size > 1:
            raise ValueError("c1 must be scalar")
        if values < 0.0:
            raise ValueError("c1 must be non-negative")
        self._c1 = values

    @property
    def c2(self):
        return self._c2
    
    @c2.setter
    def c2(self, values):
        try:
            values = np.float64(values)
        except TypeError:
            print("c2 must be castable to NumPy float64")
            raise
        if values.size > 1:
            raise ValueError("c2 must be scalar")
        if values < 0.0:
            raise ValueError("c2 must be non-negative")
        self._c2 = values

    @property
    def deposits(self):
        return self._deposits

    @deposits.setter
    def deposits(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("deposits must be NumPy nd-array")
        if values.shape != self.z.shape:
            raise ValueError("deposits must be defined on same grid as topography")
        self._deposits = values

    @property
    def depvalsdict(self):
        return self._depvalsdict

    @depvalsdict.setter
    def depvalsdict(self, values):
        if not isinstance(values, dict):
            raise TypeError("depvalsdict must be a Python dictionary")
        self._depvalsdict = values

    @property
    def dx(self):
        return self._dx
    
    @dx.setter
    def dx(self, values):
        try:
            values = np.float64(values)
        except TypeError:
            print("dx must be castable to NumPy float64")
            raise
        if values.size > 1:
            raise ValueError("dx must be scalar")
        if values < 0.0:
            raise ValueError("dx must be non-negative")
        self._dx = values

    @property
    def gamma(self):
        return self._gamma
    
    @gamma.setter
    def gamma(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("gamma must be NumPy nd-array")
        if values.ndim != 1:
            raise ValueError("gamma must be 1D")
        self._gamma = values

    @property
    def gravity(self):
        return self._gravity
    
    @gravity.setter
    def gravity(self, values):
        try:
            values = np.float64(values)
        except TypeError:
            print("gravity must be castable to NumPy float64")
            raise
        if values.size > 1:
            raise ValueError("gravity must be scalar")
        if values < 0.0:
            raise ValueError("gravity must be non-negative")
        self._gravity = values

    @property
    def hdr(self):
        return self._hdr

    @hdr.setter
    def hdr(self, values):
        if not isinstance(values, dict):
            raise TypeError("hdr must be a Python dictionary")
        self._hdr = values

    @property
    def inundatedarea(self):
        return self._inundatedarea
    
    @inundatedarea.setter
    def inundatedarea(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("inundatedarea must be NumPy nd-array")
        if values.ndim != 1:
            raise ValueError("inundatedarea must be 1D")
        self._inundatedarea = values

    @property
    def mu(self):
        return self._mu
    
    @mu.setter
    def mu(self, values):
        try:
            values = np.float64(values)
        except TypeError:
            print("mu must be castable to NumPy float64")
            raise
        if values.size > 1:
            raise ValueError("mu must be scalar")
        if values < 0.0:
            raise ValueError("mu must be non-negative")
        self._mu = values

    @property
    def nparams(self):
        return self._nparams
    
    @nparams.setter
    def nparams(self, values):
        try:
            values = np.int64(values)
        except TypeError:
            print("nparams must be castable to NumPy int64")
            raise
        if values.size > 1:
            raise ValueError("nparams must be scalar")
        if values < 0:
            raise ValueError("nparams must be non-negative")
        self._nparams = values

    @property
    def ntrials(self):
        return self._ntrials
    
    @ntrials.setter
    def ntrials(self, values):
        try:
            values = np.int64(values)
        except TypeError:
            print("ntrials must be castable to NumPy int64")
            raise
        if values.size > 1:
            raise ValueError("ntrials must be scalar")
        if values < 0:
            raise ValueError("ntrials must be non-negative")
        self._ntrials = values

    @property
    def nx(self):
        return self._nx
    
    @nx.setter
    def nx(self, values):
        try:
            values = np.int64(values)
        except TypeError:
            print("nx must be castable to NumPy int64")
            raise
        if values.size > 1:
            raise ValueError("nx must be scalar")
        if values < 1:
            raise ValueError("nx must be positive")
        self._nx = values
    
    @property
    def ny(self):
        return self._ny
    
    @ny.setter
    def ny(self, values):
        try:
            values = np.int64(values)
        except TypeError:
            print("ny must be castable to NumPy int64")
            raise
        if values.size > 1:
            raise ValueError("ny must be scalar")
        if values < 1:
            raise ValueError("ny must be non-negative")
        self._ny = values

    @property
    def omega(self):
        return self._omega
    
    @omega.setter
    def omega(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("omega must be NumPy nd-array")
        if values.ndim != 1:
            raise ValueError("omega must be 1D")
        self._omega = values

    @property
    def pspace1(self):
        return self._pspace1
    
    @pspace1.setter
    def pspace1(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("pspace1 must be NumPy nd-array")
        if values.ndim != 1:
            raise ValueError("pspace1 must be 1D")
        self._pspace1 = values
    
    @property
    def pspace2(self):
        return self._pspace2
    
    @pspace2.setter
    def pspace2(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("pspace2 must be NumPy nd-array")
        if values.ndim != 1:
            raise ValueError("pspace2 must be 1D")
        self._pspace2 = values

    @property
    def rho(self):
        return self._rho
    
    @rho.setter
    def rho(self, values):
        try:
            values = np.float64(values)
        except TypeError:
            print("rho must be castable to NumPy float64")
            raise
        if values.size > 1:
            raise ValueError("rho must be scalar")
        if values < 0.0:
            raise ValueError("rho must be non-negative")
        self._rho = values

    @property
    def simparams(self):
        return self._simparams
    
    @simparams.setter
    def simparams(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("simparams must be NumPy nd-array")
        if values.ndim == 1:
            values = np.tile(values, (1,1))
        if (values.ndim != 2) or (values.shape[1] != 2):
            raise ValueError("simparams must have exactly 2 columns, chi and tauy, and rows = # of simulations")
        self._simparams = values.astype(np.float64)

    @property
    def thresh(self):
        return self._thresh
    
    @thresh.setter
    def thresh(self, values):
        try:
            values = np.float64(values)
        except TypeError:
            print("thresh must be castable to NumPy float64")
            raise
        if values.size > 1:
            raise ValueError("thresh must be scalar")
        self._thresh = values

    @property
    def volinputdict(self):
        return self._volinputdict
    
    @volinputdict.setter
    def volinputdict(self, values):
        if not isinstance(values, dict):
            raise TypeError("volinputdict must be a Python dictionary")
        self._volinputdict = values

    @property
    def volumeinput(self):
        return self._volumeinput

    @volumeinput.setter
    def volumeinput(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("volumeinput must be NumPy nd-array")
        if values.ndim != 2:
            raise ValueError("volumeinput must be 2D")
        if values.shape[1] != 3:
            raise ValueError("volumeinput must have 3 columns: UTM_n, UTM_e, volume")
        self._volumeinput = values.astype(np.float64)

    @property
    def watersheds(self):
        return self._watersheds
    
    @watersheds.setter
    def watersheds(self, values):
        if isinstance(values, str):
            values = [values]
        if isinstance(values, (np.ndarray, tuple, list)):
            pass
        else:
            raise TypeError("watersheds must be NumPy ndarray, tuple, list, or str")
        self._watersheds = values

    @property
    def workdir(self):
        return self._workdir
    
    @workdir.setter
    def workdir(self, values):
        if not isinstance(values, str):
            raise TypeError("path to work directory must be a string")
        self._workdir = values

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("x must be NumPy nd-array")
        if values.ndim != 2:
            raise ValueError("x must be 2D")
        self._x = values
    
    @property
    def xstart(self):
        return self._xstart

    @xstart.setter
    def xstart(self, values):
        acceptabletypes = (np.ndarray, int, float, np.integer, np.floating)
        if isinstance(values, acceptabletypes):
            values = np.int64(values)
        else:
            raise TypeError("xstart must be NumPy ndarray or scalar, castable to np.int64 ")
        self._xstart = values
    
    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("y must be NumPy nd-array")
        if values.ndim != 2:
            raise ValueError("y must be 2D")
        self._y = values

    @property
    def ystart(self):
        return self._ystart

    @ystart.setter
    def ystart(self, values):
        acceptabletypes = (np.ndarray, int, float, np.integer, np.floating)
        if isinstance(values, acceptabletypes):
            values = np.int64(values)
        else:
            raise TypeError("ystart must be NumPy ndarray or scalar, castable to np.int64 ")
        self._ystart = values

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, values):
        if not isinstance(values, np.ndarray):
            raise TypeError("z must be NumPy nd-array")
        if values.ndim != 2:
            raise ValueError("z must be 2D")
        self._z = values