"""WRFensemble class"""

import numpy as np
import pandas as pd


class WRFensemble():
    """A class to hold results from WRF atmospheric model ensemble.
    
    This class loads the ensemble results from an Excel table. The table must be formatted
    as follows:
     - each row contains all the results from 1 unique ensemble member
     - there are 5 columns for each debris flow starting location (i.e. volume input point)
     - the 5 columns repeat for every volume input point
     - these columns are: UTM North coordinate, UTM East coordinate, peak I15 rainfall rate
       averaged over the upstream drainage area, predicted debris flow volume, and 
       probability of debris flow occurrence


    Parameters
        ----------
        fname : str
            path to the input Excel table
        df_prob_thresh : float, optional
            threshold probability for debris flow occurrence, by default 0.5

    """

    def __init__(self, fname, df_prob_thresh=0.5):
        self.df_prob_thresh=df_prob_thresh
        self.parse_input(fname)

    @property
    def _compute_binary_response(self):
        """Returns debris flow occurrence binary response.

        Returns
        -------
        ndarray
            binary response for every ensemble member and every start point, dtype=np.int64
        """

        probs = self.data[..., 4]  # probability of df event lives in the 5th column
        return (probs >= self.df_prob_thresh).astype(np.int64)

    def parse_input(self, fname):
        """Read the WRF ensemble member data from an excel file

        Expected format:
            rows: ensemble members
            columns: 5 columns per df start point arranged as
                UTM N, UTM E, peak I15 rain rate, predicted debris flow volume, and
                probability of occurrence

        Parameters
        ----------
        fname : str
            path to input excel file
        """

        # Read in excel table with pandas
        # expected format: each row is 1 WRF ensemble member's results, with 5 columns for every debris flow starting point:
        #      UTM N, UTM E, peak I15 rain rate, predicted debris flow volume, probability of occurrence
        sheet = pd.read_excel(fname)
        data = np.array(sheet)

        # Next reshape into 3D array with dimensions corresponding to:
        #      (number of ensemble members, number of start points, 5)
        self.n_ens = data.shape[0]
        self.n_pts = np.int64(data.shape[1] / 5)
        self.data = data.reshape(self.n_ens, self.n_pts, 5)
