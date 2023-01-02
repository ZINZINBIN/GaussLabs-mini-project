# RevIN algorithm
import torch
import torch.nn as nn
from typing import Literal, Optional

class RevIN(nn.Module):
    def __init__(self, num_features : int, eps : float = 1e-5, affine : bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x : torch.Tensor, mode : Literal["norm", "denorm"]):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x
    
    def _init_params(self):
        self.affine_W = nn.Parameter(torch.ones(self.num_features))
        self.affine_b = nn.Parameter(torch.zeros(self.num_features))
    
    def _get_statistics(self, x : torch.Tensor):
        ndim = x.ndim
        dim2reduce = tuple(range(1,ndim-1))
        self.mean = torch.mean(x, dim = dim2reduce, keepdim = True).detach()
        self.stdev = torch.sqrt(torch.var(x,dim=dim2reduce, keepdim = True, unbiased = False) + self.eps).detach()
    
    def _normalize(self, x : torch.Tensor):
        x = x - self.mean
        x = x / self.stdev

        if self.affine:
            x = x * self.affine_W
            x = x + self.affine_b
        return x
    
    def _denormalize(self, x : torch.Tensor):
        if self.affine:
            x = x - self.affine_b
            x = x / (self.affine_W + self.eps*self.eps)
        
        x = x * self.stdev
        x = x + self.mean
        return x