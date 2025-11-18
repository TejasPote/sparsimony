import torch
import torch.nn as nn
from torch import autograd
from typing import Tuple, Any

import deepspeed

from sparsimony.parametrization.fake_sparsity import FakeSparsity



class dfsb(FakeSparsity):
    
    def __name__(self):
        return "dfsb"

    def __init__(self, mask):
        super().__init__(mask)
        self.register_full_backward_hook(_grad_bhook)

    def forward(self, x):
        return x

   
def _grad_bhook(self, grad_input, grad_output): 
    if len(grad_input)>1:
        raise ValueError("long input")
    return (grad_output[0]*self.mask,)

    