from MCMC import hamiltonian_mcmc
import numpy as np
import multiprocessing as mp


class discontinuous_hamiltonian(hamiltonian_mcmc):
    def __init__(self) -> None:
        super().__init__()
        # For getting discontinuous momenta and positions in parallel
        self._disc_step_array: 'np.array(float)' = None
        self._cont_step_array: 'np.array(float' = None
        self._continuous_params = 0

    # What do we want?
    # Well
    # Leap frog now does the following
    # 1. For each point, check if leapfrog is continuous (we shall assume continuity <-> differentiability for now)
    # 2. Do momentum + param steps for continuous params
    # 3. Update disc params
    # 4. Do momentum step for cts params

    @property
    def continuous_params(self) -> 'np.array(float)':
        return self._cont_step_array
    
    @continuous_params.setter
    def continuous_params(self,new_cont) -> None:
        self._cont_step_array = new_cont
        self._continuous_params = self._cont_step_array.size

    @property
    def discontinuous_params(self) -> 'np.array(float)':
        return self._disc_step_array

    @discontinuous_params.setter
    def discontinuous_params(self, new_disc_pars: 'np.array(float)') -> None:
        self._disc_step_array = new_disc_pars
    

    def do_discontinuous_step(self) -> None:
        for index in range(self._continuous_params, self._proposed_step.size):
            prop_step = self._proposed_step
            prop_step[index] += self._time_step * np.sign(self._current_momentum[index])

            new_llh = self.calculate_llh(prop_step)

            delta = new_llh - self._proposed_llh

            if(np.abs(self._current_momentum[index])>delta):
                self._proposed_step = prop_step
                self._current_momentum[index] -= np.sign(self._current_momentum[index])*delta
                self._proposed_llh = new_llh
            else:
                self._current_momentum[index] *= -1

    def do_leapfrog_step(self) -> None:
        #Slightly Different Leapfrog
        self._proposed_step = self._current_step
        self.do_momentum_step(self._continuous_params)
        self._proposed_step[:self._continuous_params] += 0.5*self._time_step*self._current_momentum[:self._continuous_params]
        self._proposed_llh = self.calculate_llh(self._proposed_step)
        self.do_discontinuous_step()
        self._proposed_step[:self._continuous_params] += 0.5*self._time_step*self._current_momentum[:self._continuous_params]
        self.do_momentum_step(self._continuous_params)

    def __call__(self, n_steps: int) -> None:
        self._current_step = np.append(self._cont_step_array, self._disc_step_array)
        return super().__call__(n_steps)