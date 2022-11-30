"""
Implementation of Metropolis-Hastings
"""
from MCMC.MCMCBase import GenericMCMC
import numpy as np

class metropolis_hastings(GenericMCMC):

    def __init__(self):
        super().__init__()
        self._step_sizes: 'np.array(float)' = None

    @property
    def step_sizes(self) -> np.array(float):
        return self._step_sizes

    @step_sizes.setter
    def step_sizes(self, proposed_sizes: 'np.array(float)') -> None:
        self._step_sizes = proposed_sizes

    def accept_step(self) -> bool:
        """
        Do we accept the step for metropolis criterion?
        :return: Whether step is accepted
        """
        acceptance_lim = np.random.randint(0, 1)

        # Get LLH for proposed step
        self._proposed_llh = self.calculate_llh(self._proposed_step)

        # Get ratio of likelihoods
        likelihood_ratio = min(np.exp(self._proposed_llh - self._current_llh), 1.0)

        return likelihood_ratio > acceptance_lim

    def do_mcmc_step(self) -> None:
        # Does steps for MCMC
        proposed_shift = np.random.multivariate_normal(self._current_step, np.diag(self._step_sizes))
        self._proposed_step = self._current_step + proposed_shift

        # Here we'd likely add a re-weight, so we pick the right thing

        if self.accept_step():
            self._current_step = self._proposed_step

        self._total_steps_curr += 1

    def __call__(self, n_steps: int):
        super().__call__(n_steps)

    def __str__(self):
        return f"Metropolis Hastings MCMC"

