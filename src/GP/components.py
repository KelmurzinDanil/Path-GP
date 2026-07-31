import torch
import gpytorch

from gpytorch.means import Mean, ConstantMean, LinearMean, ZeroMean
from gpytorch.kernels import Kernel, RBFKernel, MaternKernel, LinearKernel, ScaleKernel
from gpytorch.likelihoods import Likelihood, GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood

from typing import Optional, Any, Iterable
from .config import ModelConfig, KernelConfig, TrainingConfig, KernelType, MeanType


class CauchyKernel(gpytorch.kernels.Kernel):
    is_stationary = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter(
            name="raw_lengthscale",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        self.register_parameter(
            name="raw_alpha",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )
        self.register_constraint("raw_lengthscale", gpytorch.constraints.Positive())
        self.register_constraint("raw_alpha", gpytorch.constraints.Positive())

    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value):
        self._set_value("raw_lengthscale", value)

    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        self._set_value("raw_alpha", value)

    def forward(self, x1, x2, diag=False, **params):
        dist = self.covar_dist(x1, x2, square_dist=True, diag=diag, **params)
        return (1.0 + dist / (self.lengthscale ** 2)).pow(-self.alpha)
    
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
    else:
        raise ValueError(f"Unknown mean type: {mean_type}")
    
def get_kernel(config: KernelConfig, input_dim: Optional[int] = None) -> gpytorch.kernels.Kernel:
    k_type = config.type
    ard_dims = input_dim if config.ard else None
    base_kernel = None
    
    if isinstance(k_type, gpytorch.kernels.Kernel):
        if getattr(k_type, "ard_num_dims", None) == ard_dims or not config.ard:
            base_kernel = k_type
        else:
            kernel_cls = type(k_type)
            base_kernel = kernel_cls(ard_num_dims=ard_dims) if config.ard else kernel_cls()

    elif isinstance(k_type, type) and issubclass(k_type, gpytorch.kernels.Kernel):
        base_kernel = k_type(ard_num_dims=ard_dims) if config.ard else k_type()
    
    elif k_type == "rbf":
        base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims)
    elif k_type == "matern_32":
        base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_dims)
    elif k_type == "matern_52":
        base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_dims)
    elif k_type == "rq":
        base_kernel = gpytorch.kernels.RQKernel(ard_num_dims=ard_dims)
    elif k_type == "periodic":
        base_kernel = gpytorch.kernels.PeriodicKernel(ard_num_dims=ard_dims)
    elif k_type == "cosine":
        base_kernel = gpytorch.kernels.CosineKernel(ard_num_dims=ard_dims)
    elif k_type == "spectral_mixture":
        num_mixtures = config.lengthscale_kernel if config.lengthscale_kernel is not None else 4
        num_dims_val = input_dim if input_dim is not None else 1
        base_kernel = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=num_mixtures, 
            num_dims=num_dims_val
        )
    elif k_type == "cauchy":
        base_kernel = CauchyKernel(ard_num_dims=ard_dims)
    else:
        raise ValueError(f"Unknown k_type: {k_type}")

    if config.use_lengthscale_prior and base_kernel is not None and hasattr(base_kernel, "register_prior"):
        conc, rate = config.lengthscale_prior_params
        base_kernel.register_prior(
            "lengthscale_prior",
            gpytorch.priors.GammaPrior(concentration=conc, rate=rate),
            "lengthscale"
        )
        
    if config.scale_kernel and not isinstance(base_kernel, gpytorch.kernels.SpectralMixtureKernel):
        return gpytorch.kernels.ScaleKernel(base_kernel)
    return base_kernel

def get_likelihood(config: ModelConfig, train_y: torch.Tensor = None) -> gpytorch.likelihoods.Likelihood:
    lh_config = config.likelihood
    kwargs = lh_config.extra_kwargs
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood(**kwargs)
    
    if train_y is not None:
        y_var = float(torch.var(train_y).item())
        if y_var < 1e-8: 
            y_var = 1.0
            
        min_noise = 1e-6
        max_noise = max(0.05 * y_var, 1e-2)
        
        likelihood.noise_covar.register_constraint(
            "raw_noise", 
            gpytorch.constraints.Interval(min_noise, max_noise)
        )
        likelihood.noise = torch.tensor(min(1e-3, 0.001 * y_var), dtype=train_y.dtype)
    else:
        likelihood.noise_covar.register_constraint(
            "raw_noise", 
            gpytorch.constraints.Interval(1e-6, 0.05)
        )

    return likelihood

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
