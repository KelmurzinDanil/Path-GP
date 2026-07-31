import torch
import gpytorch
from typing import List, Tuple, Optional

from .config import GPConfig
from .models import build_model
from .trainer import GPTrainer

class GPRegressionPipeline:
    def __init__(self, config: GPConfig):
        self.config = config
        self.trainer = GPTrainer(config)
        self.model: Optional[gpytorch.models.GP] = None
        self.likelihood: Optional[gpytorch.likelihoods.Likelihood] = None
        self.scale_x_factor: Optional[torch.Tensor] = None
        self.scale_y_factor: Optional[torch.Tensor] = None

    def _compute_scale_factors(self, train_x: torch.Tensor, train_y: torch.Tensor):
        if not getattr(self.config, "scaling", None) or not self.config.scaling.enabled:
            self.scale_x_factor = torch.ones(train_x.size(-1), dtype=train_x.dtype, device=train_x.device)
            self.scale_y_factor = torch.tensor(1.0, dtype=train_y.dtype, device=train_y.device)
            return

        self.scale_x_factor = torch.ones(train_x.size(-1), dtype=train_x.dtype, device=train_x.device)

        sy = torch.std(train_y)
        self.scale_y_factor = torch.tensor(1.0, dtype=train_y.dtype, device=train_y.device) if torch.abs(sy) < 1e-12 else sy

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor, trial = None) -> List[float]:

        seed = getattr(self.config, "seed", 42)

        train_x = train_x.double()
        train_y = train_y.double()

        self._compute_scale_factors(train_x, train_y)

        scaled_train_x = train_x / self.scale_x_factor
        scaled_train_y = train_y / self.scale_y_factor

        self.model, self.likelihood = build_model(self.config, scaled_train_x, scaled_train_y)

        loss_history = self.trainer.fit(self.model, self.likelihood, scaled_train_x, scaled_train_y, trial=trial)
        return loss_history
    
    def predict(self, test_x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        if self.model is None or self.likelihood is None:
            raise RuntimeError("Model is not fitted yet. Call .fit() before predicting.")

        test_x = test_x.double()
        scaled_test_x = test_x / self.scale_x_factor

        self.model.eval()
        self.likelihood.eval()

        jitter_val = getattr(self.config.training, "cholesky_jitter", 1e-3)

        with torch.no_grad(), \
             gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.cholesky_jitter(jitter_val):
            f_dist = self.model(scaled_test_x)
            pred_dist = self.likelihood(f_dist)

            unscaled_mean = pred_dist.mean * self.scale_y_factor
            
            lazy_cov = getattr(pred_dist, "lazy_covariance_matrix", None)
            if lazy_cov is not None:
                unscaled_covar = lazy_cov * (self.scale_y_factor ** 2)
            else:
                unscaled_covar = pred_dist.covariance_matrix * (self.scale_y_factor ** 2)

            return gpytorch.distributions.MultivariateNormal(unscaled_mean, unscaled_covar)