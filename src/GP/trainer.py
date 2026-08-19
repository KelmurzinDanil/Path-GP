import torch
import gpytorch
import optuna
from typing import List

from .components import get_optimizer, get_loss_fn
from .config import GPConfig

class GPTrainer:
    def __init__(self, config: GPConfig):
        self.config = config

    def fit(
        self,
        model: gpytorch.models.GP,
        likelihood: gpytorch.likelihoods.Likelihood,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        trial = None
    ) -> List[float]:
        model.train()
        likelihood.train()

        params = list(set(model.parameters()).union(set(likelihood.parameters())))
        opt_type = self.config.training.optimizer
        if isinstance(opt_type, str) and opt_type.lower() == "lbfgs":
            optimizer = torch.optim.LBFGS(
                params,
                lr=self.config.training.lr,
                line_search_fn="strong_wolfe"
            )
        else:
            optimizer = get_optimizer(params, self.config.training)

        loss_fn = get_loss_fn(model, likelihood, self.config.training, num_data=train_x.size(0))

        loss_history = []
        epochs = self.config.training.epochs

        patience = self.config.training.early_stopping_patience or 15
        best_loss = float('inf')
        patience_counter = 0

        loss_modifier = self.config.training.loss_modifier
        jitter_val = getattr(self.config.training, "cholesky_jitter", 1e-3)
        fast_solves = getattr(self.config.training, "fast_solves", False)
        fast_log_prob = getattr(self.config.training, "fast_log_prob", False)

        with gpytorch.settings.cholesky_jitter(jitter_val), \
             gpytorch.settings.fast_computations(solves=fast_solves, log_prob=fast_log_prob):
            
            for epoch in range(epochs):
                try:
                    if loss_modifier is not None:
                        x_input = train_x.clone().detach().requires_grad_(True)
                    else:
                        x_input = train_x
                    if self.config.training.optimizer == "lbfgs":
                        def closure():
                            optimizer.zero_grad()
                            output = model(x_input)
                            loss = -loss_fn(output, train_y)
                            
                            if loss_modifier is not None:
                                loss = loss_modifier(model, likelihood, output, x_input, train_y, loss)
                                
                            loss.backward()
                            return loss
                        
                        optimizer.step(closure)

                        with torch.no_grad():
                            output = model(train_x)
                            loss = -loss_fn(output, train_y)

                    else:
                        optimizer.zero_grad()
                        output = model(x_input)
                        loss = -loss_fn(output, train_y)

                        if loss_modifier is not None:
                            loss = loss_modifier(model, likelihood, output, x_input, train_y, loss)
                            
                        loss.backward()
                        optimizer.step()

                except (gpytorch.utils.errors.NanError, RuntimeError) as e:
                    if self.config.training.verbose:
                            print(f"\n[Предупреждение] Обучение остановлено из-за численной нестабильности на эпохе {epoch+1}")
                    break
                
                current_loss = loss.item()
                loss_history.append(current_loss)

                if trial is not None:
                        trial.report(current_loss, step=epoch)
                        
                        if trial.should_prune():
                            if self.config.training.verbose:
                                print(f"\n[Optuna Pruning] Попытка прервана на эпохе {epoch + 1}")
                            raise optuna.TrialPruned()
                        
                if self.config.training.verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                    noise_val = likelihood.noise.item() if hasattr(likelihood, "noise") else 0.0
                    print(
                        f"Epoch {epoch + 1:3d}/{self.config.training.epochs} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Noise: {noise_val:.3f}"
                    )
                    try:
                        from .utils import get_quick_kernel_info
                        k_info = get_quick_kernel_info(model.covar_module)
                        if k_info:
                            print(f"  Params: {k_info}")
                    except Exception:
                        pass
                        
                if current_loss < best_loss - 1e-4:
                        best_loss = current_loss
                        patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if self.config.training.verbose:
                        print(f"\n[Early Stopping] Обучение завершено на эпохе {epoch + 1}, лосс стабилизировался.")
                    break
            
        return loss_history
