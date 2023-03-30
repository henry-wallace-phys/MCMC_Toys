"""
Abstract Base class for multi-modal metropolis hastings
Inherits most properties from mcmc_base but has additional proposal kernels for local/jumping steps
"""

from MCMC import generic_mcmc
from abc import ABC, abstractmethod

class multimodal_mcmc_base(generic_mcmc, ABC):
    def __init__(self) -> None:
        super().__init__()


    @abstractmethod
    def propose_local_step(self) -> None:
        pass

    @abstractmethod
    def propose_jump_step(self) -> None:
        pass

    @abstractmethod
    def get_local_likelihood(self) -> None:
        pass

    @abstractmethod
    def get_jump_likelihood(self) -> None:
        pass

    @abstractmethod
    def do_jump_step(self) -> bool:
        pass

    def propose_step(self) -> None:
        if(self.do_jump_step):
            self.propose_jump_step()
        else:
            self.propose_local_step()
        

