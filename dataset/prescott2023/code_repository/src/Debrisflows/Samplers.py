"""Sampler class"""

import numpy as np
from scipy import stats


class Sampler():
    """A base class for different sampler types. Not meant to be directly instantiated.

    Three required methods for subclasses are __init__, get_samples, get_pdf
    
    """
    def __init__(self):
        raise NotImplementedError
    
    def get_samples(self, n):
        raise NotImplementedError
    
    def get_pdf(self, samples):
        raise NotImplementedError


class UniformSampler(Sampler):
    """A Sampler subclass to sample from a uniform distribution.

    Parameters
    ----------
    loc : float
        loc parameter of the scipy.stats.uniform class
        Lower bound of the support of the distribution
    scale : float
        scale parameter of the scipy.stats.uniform class
        Width of the support of the distribution, i.e. the upper bound = loc + scale
    random_state : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional
        If not None, seeds the random state for reproducibility, by default None
    """
    
    def __init__(self, loc, scale, random_state=None):
        self.loc = loc
        self.scale = scale
        self.random_state = random_state

    def get_samples(self, n):
        """Get random samples from the sampler.

        Parameters
        ----------
        n : int
            Number of samples

        Returns
        -------
        ndarray
            samples
        """
        return stats.uniform.rvs(loc=self.loc, scale=self.scale, size=n, random_state=self.random_state)
    
    def get_pdf(self, x):
        """Get the sampler's probability density function evaluated at the input locations.

        Parameters
        ----------
        x : ndarray, float-like
            input locations

        Returns
        -------
        ndarray
            pdf evaluations
        """
        return stats.uniform.pdf(x, loc=self.loc, scale=self.scale)


class BestProDFSampler(Sampler):
    """A Sampler subclass to sample ProDF parameters directly from a StochasticVolume simulation result.

    Parameters
        ----------
        p1 : _type_
            _description_
        p2 : _type_
            _description_
        best_idx : _type_
            _description_
        random_state : _type_, optional
            _description_, by default None
    """

    def __init__(self, p1, p2, best_idx, random_state=None):
        
        if p1.size != p2.size:
            raise ValueError("p1 and p2 must be NumPy ndarrays of the same size")

        self.p1 = p1.ravel()
        self.p2 = p2.ravel()
        self.best_idx = best_idx.ravel()
        self.n_input = p1.size
        self.random_state = random_state

    def get_samples(self, n):
        """Get random samples from the sampler.

        Parameters
        ----------
        n : int
            Number of samples

        Returns
        -------
        ndarray
            samples
        """
        ridx = stats.randint.rvs(low=0, high=self.n_input, size=n, random_state=self.random_state)
        idx = self.best_idx[ridx].astype(np.int64)
        return np.array([self.p1[idx], self.p2[idx]])

    def get_pdf(self, x):
        """Get the sampler's probability density function evaluated at the input locations.

        Parameters
        ----------
        x : ndarray, float-like
            input locations

        Returns
        -------
        ndarray
            pdf evaluations
        """
        p = np.float64(1.0 / self.n_input)
        return np.full(x.size, fill_value=p)


class CSVProDFSampler(Sampler):
    """A Sampler subclass to load a ProDF-parameter sampled distribution from CSV file.
    
    Parameters
    ----------
    csv_filename : string
        path to the CSV file that holds the ProDF flow mobility parameter samples
        expected to have two columns, ordered as (chi, tau_y)
    random_state : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional
        If not None, seeds the random state for reproducibility, by default None
    """

    def __init__(self, csv_filename, random_state=None):
        self.csv_filename = csv_filename
        try:
            d = np.genfromtxt(csv_filename, delimiter=",")
        except:
            print("error loading the sampler csv file")
        self.chi = d[:, 0]
        self.tau = d[:, 1]
        self.n_input = self.chi.size
        self.random_state = random_state
        
    def get_samples(self, n):
        """Get random samples from the sampler.

        Parameters
        ----------
        n : int
            Number of samples

        Returns
        -------
        ndarray
            samples
        """
        ridx = stats.randint.rvs(low=0, high=self.n_input, size=n, random_state=self.random_state)
        return np.array([self.chi[ridx], self.tau[ridx]])

    def get_pdf(self, x):
        """Get the sampler's probability density function evaluated at the input locations.

        Parameters
        ----------
        x : ndarray, float-like
            input locations

        Returns
        -------
        ndarray
            pdf evaluations
        """
        p = np.float64(1.0 / self.n_input)
        return np.full(x.size, fill_value=p)


class KDESampler(Sampler):
    """A class to sample from a scipy kernel density estimator.

    Parameters
        ----------
        kernel : SciPy stats gaussian_kde instance
            The kernel density estimator to sample from.
    """
    def __init__(self, kernel):
        if not isinstance(kernel, stats.gaussian_kde):
            raise TypeError("KDESampler only works with a scipy.stats.gaussian_kde instance")
        self.kde = kernel
    
    def get_samples(self, n):
        """Get random samples from the sampler.

        Parameters
        ----------
        n : int
            Number of samples

        Returns
        -------
        ndarray
            samples
        """
        return self.kde.resample(n)
    
    def get_pdf(self, x):
        """Get the sampler's probability density function evaluated at the input locations.

        Parameters
        ----------
        x : ndarray, float-like
            input locations

        Returns
        -------
        ndarray
            pdf evaluations
        """
        return self.kde(x)


class LogUniformSampler(Sampler):
    """A Sampler subclass to sample from a log-uniform distribution.

    Parameters
    ----------
    a : float
        shape parameter 'a' of the scipy.stats.loguniform class
        Lower bound of the support of the distribution
    b : float
        shape parameter 'b' of the scipy.stats.uniform class
        Upper bound of the support of the distribution
    random_state : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional
        If not None, seeds the random state for reproducibility, by default None
    """
    
    def __init__(self, a, b, random_state=None):
        self.a = a
        self.b = b
        self.random_state = random_state

    def get_samples(self, n):
        """Get random samples from the sampler.

        Parameters
        ----------
        n : int
            Number of samples

        Returns
        -------
        ndarray
            samples
        """
        return stats.loguniform.rvs(a=self.a, b=self.b, size=n, random_state=self.random_state)
    
    def get_pdf(self, x):
        """Get the sampler's probability density function evaluated at the input locations.

        Parameters
        ----------
        x : ndarray, float-like
            input locations

        Returns
        -------
        ndarray
            pdf evaluations
        """
        return stats.loguniform.pdf(x, a=self.a, b=self.b)
