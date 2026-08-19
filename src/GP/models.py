from __future__ import annotations
import torch
import gpytorch
from typing import Tuple

from .components import get_mean, get_kernel, get_likelihood
from .config import GPConfig

class ExactGP(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        mean_module: gpytorch.means.Mean,
        covar_module: gpytorch.kernels.Kernel
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def initialize_composite_kernels(
    covar_module: gpytorch.kernels.Kernel,
    train_x: torch.Tensor,
    train_y: torch.Tensor
):
    for module in covar_module.modules():
        if isinstance(module, gpytorch.kernels.SpectralMixtureKernel):
            module.to(dtype=train_x.dtype, device=train_x.device)
            
            if module.active_dims is not None:
                active_indices = list(module.active_dims)
                x_sliced = train_x[:, active_indices]
            else:
                x_sliced = train_x
                
            try:
                module.initialize_from_data(x_sliced, train_y)
            except Exception as e:
                pass

def build_model(
    config: GPConfig, 
    train_x: torch.Tensor, 
    train_y: torch.Tensor
) -> Tuple[gpytorch.models.GP, gpytorch.likelihoods.Likelihood]:

    if train_x.dim() == 1:
        train_x = train_x.unsqueeze(-1)
        
    input_dim = train_x.size(-1)

    likelihood = get_likelihood(config.model, train_y)
    mean_module = get_mean(config.model, input_dim)
    covar_module = get_kernel(config.model.kernel, input_dim)

    initialize_composite_kernels(covar_module, train_x, train_y)

    likelihood = likelihood.to(dtype=train_x.dtype, device=train_x.device)
    mean_module = mean_module.to(dtype=train_x.dtype, device=train_x.device)
    covar_module = covar_module.to(dtype=train_x.dtype, device=train_x.device)

    model_gp = ExactGP(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        mean_module=mean_module,
        covar_module=covar_module
    )
    
    return (model_gp, likelihood)