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

def build_model(
    config: GPConfig, 
    train_x: torch.Tensor, 
    train_y: torch.Tensor
) -> Tuple[gpytorch.models.GP, gpytorch.likelihoods.Likelihood]:
    
    input_dim = train_x.size(-1) if train_x.dim() > 1 else 1

    likelihood = get_likelihood(config.model)
    mean_module = get_mean(config.model, input_dim)
    covar_module = get_kernel(config.model.kernel, input_dim)

    modelGP = ExactGP(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        mean_module=mean_module,
        covar_module=covar_module
        )
    
    return (modelGP, likelihood)