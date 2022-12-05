from hamiltonian_mcmc import hamiltonian_mcmc
import numpy as np
import multiprocessing as mp


class discontinuous_hamiltonian(hamiltonian_mcmc):
    def __init__(self) -> None:
        super().__init__()

    def do_leapfrog_step(self) -> None:
        """
        Overrides leapfrog steps for Hamiltonian MCMC
        Accounts for discontinuities by giving a momentum condition
        :return: None
        """
        self.do_momentum_step()


    def disc_coord_integrator(self, proposed_step_temp: 'np.array(float)', curr_momentum_tmp: 'np.array(float)',
                              step_index: int) -> 'list(int, float, float)':
        """
        Performs HMCMC step for discontinuous elements
        :param curr_momentum_tmp:
        :param proposed_step_temp: Temporary holder for propsoed step
        :param step_index: Index
        :return: index, change to momentum, change to o
        """
        proposed_step_init = proposed_step_temp
        proposed_step_temp[step_index] += self._time_step * np.sign(curr_momentum_tmp[step_index])

        # Work out difference in likelihoods between two points
        delta_likelihood = self.calculate_llh(proposed_step_temp) - self.calculate_llh(proposed_step_init)

        # Pass over the discontinuity
        if abs(self._current_momentum[step_index] > delta_likelihood):
            momentum_change = curr_momentum_tmp[step_index] - \
                              np.sign(curr_momentum_tmp[step_index]) * delta_likelihood

            return [step_index, proposed_step_temp[step_index], momentum_change]

        # Reflect off the discontinuity
        return [step_index, proposed_step_temp[step_index], -1 * curr_momentum_tmp[step_index]]

    def coord_callback_func(self, update_params: list(int, float, float)) -> None:
        # Updates current step and momentum
        index: int = update_params[0]
        step_change: float = update_params[1]
        mom_change: float = update_params[2]

        self._current_step[index] = step_change
        self._current_momentum[index] = mom_change

    def discontinuity_update_mp(self) -> None:
        """
        Updates params with discontinuities using multiprocessing
        For now we assume all params are discontinuous since it just embeds on a smooth space
        :return: None
        """
        # Potentially might be worth keeping track of what re-weighting makes discontinuous in MaCh3
        pool = mp.pool()
        for i_param in range(len(self._current_step)):
            pool.apply_async(self.disc_coord_integrator, args=(i_param, ), callback=self.coord_update())
        pool.close()
        pool.join()

