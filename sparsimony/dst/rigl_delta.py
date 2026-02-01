from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from sparsimony.dst.rigl import RigL
from sparsimony.parametrization.dfsb import DFSB_RigL
from sparsimony.schedulers.base import BaseScheduler
from sparsimony.distributions.base import BaseDistribution

class RigLDelta(RigL):
    """
    RigL-Delta: A Dynamic Sparse Training method that combines:
    1. Dense Forward Pass (from SET-Delta/DFSB)
    2. Gradient-based Growth (from RigL)
    
    It keeps the forward pass dense, but pruning and growing happen on the mask.
    The backward pass is sparse (masked gradients flow to weights), but we also
    accumulate dense gradients to inform the growth step.
    """
    def __init__(
        self,
        scheduler: BaseScheduler,
        distribution: BaseDistribution,
        optimizer: torch.optim.Optimizer,
        defaults: Optional[Dict[str, Any]] = None,
        sparsity: float = 0.5,
        grown_weights_init: float = 0.0,
        init_method: Optional[str] = None,
        low_mem_mode: bool = False,
        *args,
        **kwargs,
    ):
        if defaults is None:
            defaults = dict(parametrization=DFSB_RigL)
            
        super().__init__(
            scheduler=scheduler,
            distribution=distribution,
            optimizer=optimizer,
            defaults=defaults,
            sparsity=sparsity,
            grown_weights_init=grown_weights_init,
            init_method=init_method,
            low_mem_mode=low_mem_mode,
            *args,
            **kwargs,
        )
