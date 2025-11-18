from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier

from sparsimony.parametrization.dfsb import dfsb
from sparsimony.distributions.base import BaseDistribution
from sparsimony.schedulers.base import BaseScheduler
from sparsimony.utils import get_mask, get_original_tensor
from sparsimony.dst.base import DSTMixin, GlobalPruningDataHelper
from sparsimony.mask_calculators import (
    UnstructuredPruner,
    UnstructuredGrower,
    MagnitudeScorer,
    RandomScorer,
)
from .set import SET


class SET_Delta(SET):
    def __init__(
        self,
        scheduler: BaseScheduler,
        distribution: BaseDistribution,
        optimizer: torch.optim.Optimizer,
        defaults: Optional[Dict[str, Any]] = None,
        sparsity: float = 0.5,
        grown_weights_init: float = 0.0,
        init_method: Optional[str] = None,
        *args,
        **kwargs,
    ):
        self.scheduler = scheduler
        self.distribution = distribution
        self.sparsity = sparsity
        self.grown_weights_init = grown_weights_init
        self.init_method = init_method
        if defaults is None:
            defaults = dict(parametrization=dfsb)

        super().__init__(optimizer=optimizer, defaults=defaults, init_method = init_method, scheduler = scheduler, distribution = distribution, *args, **kwargs)


    def grow_mask(self, sparsity, mask, original_weights, *args, **kwargs):
        old_mask = torch.clone(mask)
        # Grow new weights
        new_mask = self.grower.calculate_mask(sparsity, mask, *args, **kwargs)
        # Assign newly grown weights to self.grown_weights_init
        
        # Overwrite old mask
        mask.data = new_mask.data