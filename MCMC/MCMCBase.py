import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal
from tqdm import tqdm

'''
MCMC Base class, contains likelihood calculators and generic tools.
Contains abstract MCMC class for multiclass implementations
'''


class MCMCBase:
    def __init__(self) -> None:
        # To keep track of accepted stuff
        self._accepted_step_arr: 'np.array(np.array(float))' = np.array([], dtype=object)
        self._total_steps_curr: int = 0
        self._total_steps_accepted: int = 0  # Keeps track of whenever we change place

        # To keep track of steps
        self._current_step = None
        self._proposed_step = None

        # Likelihood stuff
        self._current_llh = 999.0
        self._proposed_llh = 999.0

        # Variables relating to prior distributions
        # For brevity let's assume every prior we deal with is gaussian
        self._prior_nominal_arr = np.array([], dtype=float)
        self._prior_covariance = np.array([], dtype=object)

    # Setup nominal values for priors
    @property
    def prior_nominal_arr(self) -> 'np.array(float)':
        return self._prior_nominal_arr

    @prior_nominal_arr.setter
    def prior_nominal_arr(self, new_prior: 'np.array(dtype=float)') -> None:
        self._prior_nominal_arr = new_prior

    # Setup standard deviations for gaussian priors
    @property
    def prior_covariance(self) -> 'np.array(float)':
        return self._prior_covariance

    @prior_covariance.setter
    def prior_sigmas(self, new_covariance: 'np.array(dtype=object)') -> None:
        self._prior_covariance = new_covariance

    @property
    def acceptance_rate(self) -> float:
        return self._total_steps_curr / self._total_steps_accepted

    @property
    def total_steps(self) -> int:
        return self._total_steps_curr

    @property
    def total_steps_accepted(self) -> int:
        return self._total_steps_accepted

    @property
    def accepted_step_arr(self) -> 'np.array(dtype=object)':
        return self._accepted_step_arr

    def calculate_llh(self, eval_point: 'np.array(float)') -> float:
        # Let's do our LLH calculation
        return multivariate_normal(self._prior_nominal_arr, self._prior_covariance).logpdf(eval_point)

    @property
    def current_step(self) -> 'np.array(dtype=float)':
        return self._current_step

    @current_step.setter
    def current_step(self, new_step: 'np.array(dtype=float)') -> None:
        self._current_step = new_step

    @property
    def current_llh(self) -> float:
        return self._current_llh

    @current_llh.setter
    def current_llh(self, new_llh: float):
        self._current_llh = new_llh

    def save_chain(self, outfile: str) -> None:
        """
        Saves current chain to outfile
        :param outfile: File to save
        :return:
        """
        if outfile[-4:] != '.npy':
            outfile += '.npy'
        np.save(outfile, self._accepted_step_arr)

    def load_chain_from_file(self, infile: object) -> None:
        """
        Loads in previous chain from file
        :param infile: Input file
        :return:
        """
        self._accepted_step_arr = np.load(infile)


class GenericMCMC(MCMCBase, ABC):
    """
    Class with generic functions for use with mcmc
    -> Inherits from MCMC base class
    -
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, n_steps: int) -> None:
        """
        Run the MCMC chain
        :param n_steps: Number of Steps to run the chain for
        :return:
        """
        pass

    @abstractmethod
    def accept_step(self) -> bool:
        """
        Accept/Reject criterion
        :return: Whether step is accepted/rejected
        """
        pass

    @abstractmethod
    def do_mcmc_step(self) -> None:
        """
        Perform step using MCMC algorithm
        :return:
        """
        pass

    def __call__(self, n_steps: int):
        self._proposed_step = self._current_step
        for _ in tqdm(range(n_steps)):
            self.do_mcmc_step()
            if self.accept_step():
                self._current_step = self._proposed_step
                self._current_llh = self._proposed_llh
                self._total_steps_accepted += 1

            self._accepted_step_arr = np.append(self._accepted_step_arr, self._current_step)
            self._total_steps_curr += 1

        print(f"Done with an acceptance rate of {self.acceptance_rate}")
