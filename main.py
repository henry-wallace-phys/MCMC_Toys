from MCMC import hamiltonian_mcmc, metropolis_hastings, discontinuous_hamiltonian
import numpy as np

if __name__ == "__main__":
    N_STEPS = 1000
    SPACE_DIM = 10

    NOMINAL = np.zeros(SPACE_DIM)
    COV_ROOT = np.random.randn(SPACE_DIM, SPACE_DIM)

    COV = np.dot(COV_ROOT, COV_ROOT.T)

    mh_algo = metropolis_hastings
    mh_algo.step_sizes=np.ones(SPACE_DIM)*0.1

    h_mcmc = discontinuous_hamiltonian()
    h_mcmc.time_step = 0.1
    h_mcmc.leapfrog_steps = 4
    h_mcmc.continuous_params = np.ones(SPACE_DIM-5)
    h_mcmc.discontinuous_params = np.zeros(5)


    mcmc_arr = [h_mcmc]

    for mcmc_algo in mcmc_arr:
        algo_name = str(mcmc_algo)
        mcmc_algo.current_step = NOMINAL
        mcmc_algo.prior_nominal_arr = NOMINAL
        mcmc_algo.prior_sigmas = COV

        mcmc_algo.current_llh = mcmc_algo.calculate_llh(mcmc_algo.current_step)

        mcmc_algo(N_STEPS)

        mcmc_algo.save_chain(algo_name)
