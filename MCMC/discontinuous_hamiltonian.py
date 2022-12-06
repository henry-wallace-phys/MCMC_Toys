from hamiltonian_mcmc import hamiltonian_mcmc
import numpy as np
import multiprocessing as mp
from numdifftools import partial_derivative


class discontinuous_hamiltonian(hamiltonian_mcmc):
    def __init__(self) -> None:
        super().__init__()
        self._disc_array = np.array()  # Array of discontinuous indices
        self._threshold: int = 0.00001

        # For getting discontinuous momenta and positions in parallel
        self._disc_mom_array = np.array()
        self._disc_step_array = np.array()

    # What do we want?
    # Well
    # Leap frog now does the following
    # 1. For each point, check if leapfrog is continuous (we shall assume continuity <-> differentiability for now)
    # 2. Do momentum + param steps for continuous params
    # 3. Update disc params
    # 4. Do momentum step for cts params

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, new_thresh: float):
        assert (new_thresh, float)
        self._threshold = new_thresh

    def update_disc_params(self, index: int) -> None:
        array_above = self._current_step
        array_below = self._current_step

        array_above[index] += np.finfo(float).eps
        array_below[index] -= np.finfo(float).eps

        llh_above = self.calculate_llh(array_above)
        llh_below = self.calculate_llh(array_below)

        assym_cond = np.abs(llh_above - llh_below) / np.abs(llh_above + llh_below) > self._threshold

        self._disc_array[index] = assym_cond

    def disc_params_processor(self):
        self._disc_array = np.empty(self._current_step.size())
        pool = mp.pool()
        for i_param in range(self._disc_array.size):
            pool.apply_async(self.update_disc_params, args=(i_param,))
        pool.close()
        pool.join()

    # Functions to calculate the derivative of our potential and update the momentum
    # for continuous parameters
    def continuous_gradient_calculator(self, index: int) -> 'int, float':
        grad_func = partial_derivative(self.calculate_llh(), index)
        grad_val = grad_func(self._proposed_step)
        return index, grad_val

    def update_continuous_mom(self, index, gradient: float) -> None:
        self._current_momentum[index] += self._time_step * 0.5 * gradient

    def continuous_mom_processor(self):
        continuous_params = np.where(self._disc_array == False)
        pool = mp.pool()
        for i_param in continuous_params:
            pool.apply_async(self.continuous_gradient_calculator, args=(i_param,), callback=self.update_continuous_mom)
        pool.close()
        pool.join()

    # Functions for updating the position of continuous params
    def update_position(self, index: int):
        self._proposed_step[index] += self._time_step * self._current_momentum * 0.5

    def position_processor(self):
        continuous_params = np.where(self._disc_array == False)
        pool = np.pool()
        for i_param in continuous_params:
            pool.apply_async(self.update_position, args=(i_param,))
        pool.close()
        pool.join()

    def discontinuous_integrator(self, index) -> 'int, float, float':
        """
        Does discontinuous step for each param
        :param index: param index
        :return: changed momentum for index and changed position
        """
        proposed_step_next = self._proposed_step
        proposed_step_next[index] += self._time_step * np.sign(self._current_momentum[index])

        current_llh = self.calculate_llh(self._proposed_step)
        prop_llh = self.calculate_llh(proposed_step_next)

        delta_likelihood = prop_llh - current_llh

        if self._current_momentum[index] > delta_likelihood:
            momentum_change = self._current_momentum[index] - np.sign(self._current_momentum[index]) * delta_likelihood
            pos_change = self._proposed_step[index] + self._time_step * np.sign(self._current_momentum[index])
        else:
            momentum_change = self._current_momentum[index]
            pos_change = -self._proposed_step[index]

        return index, momentum_change, pos_change

    def update_proposal_params(self, index: int, momentum_change: float, pos_change: float):
        self._disc_cov_array[index] = pos_change
        self._disc_mom_array[index] = momentum_change

    def discontinuous_integrator_processor(self):
        discontinuous_params = np.where(self._disc_array)
        self._disc_mom_array = self._current_momentum
        self._disc_step_array = self._current_step
        pool = mp.pool()
        for i_param in discontinuous_params:
            pool.apply_async(self.discontinuous_integrator, args=(i_param,), callback=self.update_proposal_params)
        pool.close()
        pool.join()

        self._current_momentum = self._disc_mom_array
        self._current_step = self._disc_step_array

    def do_leapfrog_step(self) -> None:
        self.continuous_mom_processor()
        self.position_processor()
        self.discontinuous_integrator_processor()
        self.position_processor()
        self.continuous_mom_processor()

    def __str__(self):
        return f"Discontinuous Hamiltonian MCMC using time step of {self._time_step} and {self._leapfrog_steps} leapfrog steps"
