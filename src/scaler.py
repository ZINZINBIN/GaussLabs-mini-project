import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.nn import BatchNorm1d, LayerNorm
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from src.RevIN import RevIN
from typing import Literal, Union, Optional

class MinMax(nn.Module):
    def __init__(self, num_features : int, eps : float = 1e-12):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.max_value = None

    def forward(self, x : torch.Tensor, mode : Literal["norm", "denorm"]):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x
    
    def _get_statistics(self, x : torch.Tensor):
        value, _ = torch.max(x, dim = 1, keepdim = True)
        bias = torch.ones_like(value).to(value.device) * self.eps
        self.max_value = torch.max(value, bias).detach()
        
    def _normalize(self, x : torch.Tensor):
        x = x / self.max_value
        return x
    
    def _denormalize(self, x : torch.Tensor):
        x = x * self.max_value
        return x
    
# class InstanceNorm(nn.Module):
#     def __init__(self, num_features : int, eps : float = 1e-5, momentum : float = 0.1, affine : bool = True, track_running_stats : bool = True):
#         super().__init__()
#         self.eps = eps
#         self.num_features = num_features
#         self.In = nn.InstanceNorm1d(num_features, eps, momentum, affine, track_running_stats)
#         self.mu_b = self.In.running_mean.view(1,self.num_features,1)
#         self.sig_b = self.In.running_var.view(1,self.num_features,1)
        
#     def forward(self, x : torch.Tensor, mode : Literal["norm", "denorm"]):
#         if mode == "norm":
#             x = x.permute(0,2,1)
#             x = self.In(x)
#             self._get_statistics(x)
#             x = x.permute(0,2,1)
#         elif mode == "denorm":
#             x = x.permute(0,2,1)
#             x = self._denormalize(x)
#             x = x.permute(0,2,1)
#         else:
#             raise NotImplementedError
#         return x
    
#     def _get_statistics(self, x : torch.Tensor):
#         self.mu_b = self.In.running_mean.view(1,self.num_features,1)
#         self.sig_b = self.In.running_var.view(1,self.num_features,1)
        
#     def _denormalize(self, x : torch.Tensor):
    
#         if self.In.affine:
#             w = self.In.weight.view(1,self.num_features, 1)
#             b = self.In.bias.view(1,self.num_features, 1)
#             x = x - b
#             x = x / (w + self.eps*self.eps)
            
#         x = x * self.sig_b + self.mu_b
        
#         return x
    
class InstanceNorm(nn.Module):
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
        dim2reduce = (1,)
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
        
class BatchNorm(nn.Module):
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
        dim2reduce = (0,1)
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

# class LayerNorm(nn.Module):
#     def __init__(self, num_features : int, eps : float=1e-10, gamma:bool=True, beta:bool=True):
#         super(LayerNorm, self).__init__()
#         normal_shape = (num_features, )
            
#         self.normal_shape = torch.Size(normal_shape)
#         self.eps = eps
#         if gamma:
#             self.gamma = nn.Parameter(torch.Tensor(*normal_shape))
#         else:
#             self.register_parameter('gamma', None)
#         if beta:
#             self.beta = nn.Parameter(torch.Tensor(*normal_shape))
#         else:
#             self.register_parameter('beta', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         if self.gamma is not None:
#             self.gamma.data.fill_(1)
#         if self.beta is not None:
#             self.beta.data.zero_()

#     def forward(self, x:torch.Tensor, mode : Literal['norm', 'denorm']):
#         if mode == "norm":
#             x = self._normalize(x)
#         elif mode == "denorm":
#             x = self._denormalize(x)
#         else:
#             raise NotImplementedError
#         return x
    
#     def extra_repr(self):
#         return 'normal_shape={}, gamma={}, beta={}, epsilon={}'.format(
#             self.normal_shape, self.gamma is not None, self.beta is not None, self.epsilon,
#         )
        
#     def _normalize(self, x : torch.Tensor):
#         mean = x.mean(dim=(1,2), keepdim=True)
#         var = ((x - mean) ** 2).mean(dim=(1,2), keepdim=True)
#         std = (var + self.eps).sqrt()
        
#         self.mean= mean
#         self.std = std
        
#         y = (x - mean) / std
#         if self.gamma is not None:
#             y *= self.gamma
#         if self.beta is not None:
#             y += self.beta
#         return y
    
#     def _denormalize(self, x : torch.Tensor):
#         if self.beta is not None:
#             x -= self.beta
#         if self.gamma is not None:
#             x /= (self.gamma + self.eps * self.eps)
#         y = (x + self.mean) * self.std
        
#         return y

class LayerNorm(nn.Module):
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
        dim2reduce = (1,2)
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
        
class Wrapper(nn.Module):
    def __init__(self, model : nn.Module, scaler_type : Literal['normal', 'MinMax', 'BN', 'LN', 'IN','RevIN'], **kwargs):
        super().__init__()
        
        self.scaler_type = scaler_type
        self.model = model
        
        if scaler_type == 'MinMax':
            scaler = MinMax(**kwargs)
        elif scaler_type == 'BN':
            scaler = BatchNorm(**kwargs)
        elif scaler_type == 'LN':
            scaler = LayerNorm(**kwargs)
        elif scaler_type == 'IN':
            scaler = InstanceNorm(**kwargs)
        elif scaler_type == 'RevIN':
            scaler = RevIN(**kwargs)
        else:
            scaler = None
            
        self.scaler = scaler

    def forward(self, x : torch.Tensor, target : Optional[torch.Tensor] = None, target_len : Optional[torch.Tensor] = None, teacher_forcing_ratio : Optional[float] = None):
        
        if target is not None and self.scaler and target_len is not None: 
            x = self.scaler(x, 'norm')
            x = self.model(x, target, target_len, teacher_forcing_ratio)
            x = self.scaler(x, 'denorm')
            return x
        
        elif target is not None and self.scaler and target_len is None:
            x = self.scaler(x, 'norm')
            x = self.model(x, target)
            x = self.scaler(x, 'denorm')
            return x
        
        elif target is None and self.scaler:
            x = self.scaler(x, 'norm')
            x = self.model(x)
            x = self.scaler(x, 'denorm')
            return x
        
        elif target is not None and not self.scaler and target_len is not None:
            x = self.model(x, target, target_len, teacher_forcing_ratio)
            return x
        
        elif target is not None and not self.scaler and target_len is None:
            x = self.model(x, target)
            return x
        
        else:
            x = self.model(x)
            return x
        
    def predict(self, x : torch.Tensor, target_len : Optional[int] = None):
        
        with torch.no_grad():
            
            if self.scaler:
                x = self.scaler(x, 'norm')

            x = self.model.predict(x, target_len)
            
            if self.scaler:
                output = self.scaler(x, 'denorm')
            else:
                output = x
            
            return output