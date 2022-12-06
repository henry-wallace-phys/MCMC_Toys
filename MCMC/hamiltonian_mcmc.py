"""
Implementation of Hamiltonian MCMC
"""
from MCMC.MCMCBase import GenericMCMC
import numpy as np
import numdifftools as nd


class hamiltonian_mcmc(GenericMCMC):

    def __init__(self) -> None:
        super().__init__()
        self._time_step: int = 0
        self._leapfrog_steps: int = 0
        self._fulfilled_nuts: bool = False

        self._current_momentum: 'np.array(float)' = None
        self._current_hamiltonian: float = -999

    @property
    def time_step(self) -> int:
        return self._time_step

    @time_step.setter
    def time_step(self, new_time_step: int) -> None:
        """
        Set time step (epsilon) for each leapfrog
        :param new_time_step:
        :return: None
        """
        self._time_step = new_time_step

    @property
    def leapfrog_steps(self) -> int:
        return self._leapfrog_steps

    @leapfrog_steps.setter
    def leapfrog_steps(self, new_leapfrog_steps: int) -> None:
        """
        Sets number of leapfrog steps (L)
        :int new_leapfrog_steps:
        :return: None
        """
        self._leapfrog_steps = new_leapfrog_steps

    def randomise_momentum(self) -> None:
        """
        Randomises momentum vector
        :return: None
        """
        self._current_momentum = np.random.multivariate_normal(np.zeros(len(self._prior_nominal_arr)),
                                                               np.diag(np.ones(len(self._prior_nominal_arr))))

    def do_momentum_step(self) -> None:
        """
        Calculates gradient of potential at given point
        :return:
        """
        grad: float = -1 * nd.Gradient(self.calculate_llh)(self._proposed_step)
        self._current_momentum -= self._time_step * 0.5 * grad

    def do_leapfrog_step(self) -> None:
        """
        Performs single leap frog step
        :return: None
        """
        self.do_momentum_step()
        self._proposed_step += self._time_step * self._current_momentum
        self.do_momentum_step()

    def calculate_hamiltonian(self, momentum_vec: 'np.array(dtype=float)', position_vec: 'np.array(float)') -> float:
        """
        Gives us the Hamiltonian for the system
        :return: Total Hamiltonian
        """
        kinetic_energy: float = 0.5 * np.dot(momentum_vec, momentum_vec)
        potential_energy: float = self.calculate_llh(position_vec)
        return kinetic_energy + potential_energy

    def accept_step(self) -> bool:
        """
        Accept current step
        :return: whether step is accepted
        """
        acceptance_lim = np.random.randint(0, 1)

        prop_hamiltonian: float = self.calculate_hamiltonian(self._current_momentum, self._proposed_step)
        alpha = min(1.0, np.exp(-prop_hamiltonian + self._current_hamiltonian))

        return alpha > acceptance_lim

    def do_mcmc_step(self) -> None:

        self.randomise_momentum()
        self._current_hamiltonian = self.calculate_hamiltonian(self._current_momentum, self._current_step)

        for _ in range(self._leapfrog_steps):
            self.do_leapfrog_step()

    def __call__(self, n_steps: int) -> None:
        super().__call__(n_steps)

    def __str__(self):
        return f"Hamiltonian MCMC using time step of {self._time_step} and {self._leapfrog_steps} leapfrog steps"

