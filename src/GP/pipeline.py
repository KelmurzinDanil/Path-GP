import torch
import gpytorch
from typing import List, Tuple, Optional

# Импортируем наши модули
from .config import GPConfig
from .models import build_model
from .trainer import GPTrainer

class GPRegressionPipeline:
    def __init__(self, config: GPConfig):
        self.config = config
        self.trainer = GPTrainer(config)
        self.model:  Optional[gpytorch.models.GP] = None
        self.likelihood: Optional[gpytorch.likelihoods.Likelihood] = None

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor) -> List[float]:
        train_x = train_x.float()
        train_y = train_y.float()

        self.model, self.likelihood = build_model(self.config, train_x, train_y)

        loss_history = self.trainer.fit(self.model, self.likelihood, train_x, train_y)

        return loss_history
    
    def predict(self, test_x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        if self.model is None or self.likelihood is None:
            raise RuntimeError("Model is not fitted yet. Call .fit() before predicting.")

        test_x = test_x.float()

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_dist = self.model(test_x)
            predictions = self.likelihood(f_dist)

        return predictions
