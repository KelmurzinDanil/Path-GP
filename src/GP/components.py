import torch
import gpytorch

from gpytorch.means import Mean, ConstantMean, LinearMean, ZeroMean
from gpytorch.kernels import Kernel, RBFKernel, MaternKernel, LinearKernel, ScaleKernel
from gpytorch.likelihoods import Likelihood, GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood

from typing import Optional, Any, Iterable
from .config import ModelConfig, KernelConfig, TrainingConfig, KernelType, MeanType


def get_mean(config: ModelConfig, input_dim: Optional[int] = None) -> gpytorch.means.Mean:
    mean_type = config.mean_type

    if isinstance(mean_type, gpytorch.means.Mean):
        return mean_type
    
    if isinstance(mean_type, type) and issubclass(mean_type, gpytorch.means.Mean):
        return mean_type()

    if mean_type == "zero":
        return gpytorch.means.ZeroMean()
    elif mean_type == "constant":
        return gpytorch.means.ConstantMean()
    elif mean_type == "linear":
        if input_dim is None:
            raise ValueError("input_dim is required for LinearMean")
        return gpytorch.means.LinearMean(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown mean type: {mean_type}")
    
def get_kernel(config: KernelConfig, input_dim: Optional[int] = None) -> gpytorch.kernels.Kernel:
    k_type = config.type
    ard_dims = input_dim if config.ard else None
    base_kernel = None
    
    if isinstance(k_type, gpytorch.kernels.Kernel):
        base_kernel = k_type
    
    if isinstance(k_type, type) and issubclass(k_type, gpytorch.kernels.Kernel):
        if ard_dims != None:
            base_kernel = k_type(ard_num_dims=ard_dims)
        else:
            base_kernel = k_type()
    
    if k_type == "rbf":
        base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims)
    elif k_type == "matern_32":
        base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_dims)
    elif k_type == "matern_52":
        base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_dims)
    elif k_type == "linear":
        base_kernel = gpytorch.kernels.LinearKernel(ard_num_dims=ard_dims)
    elif k_type == "rq":
        base_kernel = gpytorch.kernels.RQKernel(ard_num_dims=ard_dims)
    elif k_type == "periodic":
        base_kernel = gpytorch.kernels.PeriodicKernel(ard_num_dims=ard_dims)
    elif k_type == "cosine":
        base_kernel = gpytorch.kernels.CosineKernel(ard_num_dims=ard_dims)
    elif k_type == "polynomial":
        base_kernel = gpytorch.kernels.PolynomialKernel(power=2, ard_num_dims=ard_dims)
    else:
        raise ValueError(f"Unknown k_type: {k_type}")

    if config.scale_kernel:
        return gpytorch.kernels.ScaleKernel(base_kernel)
    return base_kernel

def get_likelihood(config: ModelConfig) -> gpytorch.likelihoods.Likelihood:
    lh_config = config.likelihood
    lh_type = lh_config.type
    kwargs = lh_config.extra_kwargs

    if isinstance(lh_type, gpytorch.likelihoods.Likelihood):
        return lh_type

    if isinstance(lh_type, type) and issubclass(lh_type, gpytorch.likelihoods.Likelihood):
        return lh_type(**kwargs)

    if lh_type == "gaussian":
        return gpytorch.likelihoods.GaussianLikelihood(**kwargs)
    elif lh_type == "student_t":
        return gpytorch.likelihoods.StudentTLikelihood(**kwargs)
    elif lh_type == "bernoulli":
        return gpytorch.likelihoods.BernoulliLikelihood(**kwargs)
    else:
        raise ValueError(f"Unknown likelihood type: {lh_type}")

def get_loss_fn(
    model: gpytorch.models.GP, 
    likelihood: gpytorch.likelihoods.Likelihood, 
    training_config: TrainingConfig,
    num_data: Optional[int] = None  
) -> gpytorch.mlls.MarginalLogLikelihood:
    
    loss_type = training_config.loss_type
    
    if isinstance(loss_type, gpytorch.mlls.MarginalLogLikelihood):
        return loss_type
        
    if isinstance(loss_type, type) and issubclass(loss_type, gpytorch.mlls.MarginalLogLikelihood):
        if num_data is not None:
            return loss_type(likelihood, model, num_data=num_data)
        return loss_type(likelihood, model)

    if loss_type == "mll":
        return gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    elif loss_type == "loo":
        return gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood, model)
    elif loss_type == "elbo":
        if num_data is None:
            raise ValueError("num_data is required for VariationalELBO loss")
        return gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
def get_optimizer(
    model_parameters: Iterable[torch.nn.Parameter], 
    config: TrainingConfig
) -> torch.optim.Optimizer:
    
    opt_type = config.optimizer
    lr = config.lr

    if isinstance(opt_type, torch.optim.Optimizer):
        return opt_type

    if isinstance(opt_type, type) and issubclass(opt_type, torch.optim.Optimizer):
        return opt_type(model_parameters, lr=lr)

    if opt_type == "adam":
        return torch.optim.Adam(model_parameters, lr=lr)
    elif opt_type == "sgd":
        return torch.optim.SGD(model_parameters, lr=lr)
    elif opt_type == "lbfgs":
        return torch.optim.LBFGS(model_parameters, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")
