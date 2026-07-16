import pandas as pd
import numpy as np
import torch
import sympy as sp
from abc import ABC, abstractmethod
from scipy.sparse.csgraph import connected_components
from itertools import combinations
from symbolic.sm_context import SymbolicRegressionContext, DimensionalityEvaluator
from function_analysis.py_brute_force import BruteForceRunner
from get_pi_complex import DimensionalError, PhysicalDimension, PhysicalRegistry
from typing import List, Optional, Tuple, Dict


class BaseSimplifier(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def try_simplify(self, gp_model, dataset: pd.DataFrame):
        pass
    
    def _get_dimension_compatibility_matrix(self, context: 'SymbolicRegressionContext') -> np.ndarray:
        featert_names = context.get_active_features()
        n = len(featert_names)
        mask = np.zeros((n, n), dtype=int)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                dim_i = context.registry.get_dim(featert_names[i])
                dim_j = context.registry.get_dim(featert_names[j])
                if dim_i == dim_j:
                    mask[i, j] = 1
        return mask


    def _get_best_points_and_threshold(self, gp_model, context: 'SymbolicRegressionContext',
                                       num_test_points: int, pool_size: int, k_sigma: float):
        dataset = context.df
        target_name = context.target_name
        feature_names = [col for col in dataset.columns if col != target_name]
        n_features = len(feature_names)
        
        X_tensor = torch.tensor(dataset[feature_names].values, dtype=torch.float32)
        n_samples = X_tensor.shape[0]

        n_pool = min(pool_size, n_samples)
        pool_indices = np.random.choice(n_samples, n_pool, replace=False)
        pool_points = X_tensor[pool_indices]

        gp_model.model.eval()
        gp_model.likelihood.eval()
        with torch.no_grad():
            predictions = gp_model.predict(pool_points)
            variance = predictions.variance.numpy()
            mean_vals = predictions.mean.numpy()
        
        n_points = min(num_test_points, n_samples)
        best_indices = np.argsort(variance)[:n_points]
        test_points = pool_points[best_indices]

        sigma_avg = np.sqrt(np.mean(variance[best_indices]))
        y_avg = np.mean(np.abs(mean_vals[best_indices]))
        y_avg = max(y_avg, 1e-9)

        
        base_threshold = k_sigma * (sigma_avg/y_avg)
        base_threshold = max(base_threshold, 1e-4)

        return test_points, base_threshold, sigma_avg, y_avg, feature_names
    
    def _compute_pathwise_z_score(self, gp_model, context: 'SymbolicRegressionContext', 
                                  num_test_points: int, pool_size: int, k_sigma: float,
                                  num_samples: int, is_symmetry: bool, evaluation_fn, dim_mask: np.ndarray = None) -> tuple:
        dataset = context.df
        test_points, _, sigma_avg, _, feature_names = self._get_best_points_and_threshold(
            gp_model, context, num_test_points, pool_size, k_sigma
        )
        n_features = len(feature_names)

        X_train = gp_model.model.train_inputs[0]
        Y_train = gp_model.model.train_targets

        gp_model.model.eval()
        with torch.no_grad():
            K_train = gp_model.model.covar_module(X_train).evaluate()
            noise = gp_model.model.likelihood.noise
            K_noise = K_train + noise * torch.eye(X_train.size(0))
            K_noise_inv = torch.linalg.inv(K_noise)        

        sampled_errors = torch.zeros((num_samples, len(test_points), n_features, n_features))
        lengthscale = gp_model.model.covar_module.base_kernel.lengthscale.detach().squeeze()
        D = 100      

        for s_idx in range(num_samples):
            W = torch.randn(D, n_features) / lengthscale
            b = torch.rand(D) * 2 * np.pi
            a = torch.randn(D) * np.sqrt(2.0 / D)
            
            def f_0(x):
                return torch.sum(a * torch.cos(x @ W.T + b))

            with torch.no_grad():
                f0_train = torch.tensor([f_0(x) for x in X_train])
                residual = Y_train - f0_train
                w = K_noise_inv @ residual

            def sample_path(x):
                K_x_train = gp_model.model.covar_module(x.unsqueeze(0), X_train).evaluate().squeeze(0)
                return f_0(x) + torch.dot(K_x_train, w)

            for pt_idx, pt in enumerate(test_points):
                sampled_errors[s_idx, pt_idx] = evaluation_fn(sample_path, pt, n_features)

        mean_E = torch.mean(sampled_errors, dim=0)
        std_E = torch.std(sampled_errors, dim=0)
        
        Z_score_points = torch.abs(mean_E) / (std_E + 1e-9)
        avg_Z_score = torch.mean(Z_score_points, dim=0)

        if is_symmetry:
            adj_matrix = (avg_Z_score < 3.0).numpy().astype(int)
        else:
            adj_matrix = (avg_Z_score > 3.0).numpy().astype(int)
        np.fill_diagonal(adj_matrix, 0)

        if dim_mask is not None:
            adj_matrix = adj_matrix * dim_mask

        return avg_Z_score, adj_matrix, sigma_avg, feature_names


    def _evaluate_graph_and_split(self, matrix: torch.Tensor, threshold: float, 
                                  sigma_avg: float, feature_names: list,  adj_mat,
                                  matrix_name: str = "Матрица результатов",
                                  is_symmetry: bool = False) -> tuple:
        adj_matrix = adj_mat
        
        n_components, labels = connected_components(adj_matrix, directed=False)

        groups = {}
        for idx, comp_label in enumerate(labels):
            groups.setdefault(comp_label, []).append(feature_names[idx])
        
        separated_groups = list(groups.values())
        
        print("\n" + "="*50)
        print(f"{self.name}")
        print(f"1. {matrix_name}:")
        print(np.round(matrix.detach().numpy(), 5))
        print(f"2. Средний шум GP (sigma_avg): {sigma_avg:.6f}")
        print(f"3. Рассчитанный порог: {threshold:.6f}")
        print("4. Матрица смежности графа:")
        print(adj_matrix)
        print(f"5. Результат графа: Компонентов = {n_components}, Ярлыки = {labels}")
        print("="*50 + "\n")

        if is_symmetry:
            valid_groups = [g for g in separated_groups if len(g) >= 2]
            if len(valid_groups) > 0:
                return True, valid_groups
            else:
                return False, None
        else:
            if n_components > 1:
                return True, separated_groups
            else:
                return False, None

class AdditiveSeparabilitySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=10, pool_size=100, num_samples=30):
        super().__init__("Additive Separability")
        self.k_sigma = k_sigma             
        self.num_test_points = num_test_points  
        self.pool_size = pool_size         
        self.num_samples = num_samples

    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext'):
        n_features = len(context.get_active_features())
        if n_features < 2:
            return False, None
        
        def eval_fn(sample_path, pt, n_features):
            return torch.autograd.functional.hessian(sample_path, pt)

        avg_Z_score, adj_matrix, sigma_avg, feature_names = self._compute_pathwise_z_score(
            gp_model, context, self.num_test_points, self.pool_size, self.k_sigma, self.num_samples,
              is_symmetry = False, evaluation_fn = eval_fn
        )

        return self._evaluate_graph_and_split(
            matrix=avg_Z_score,
            threshold=3.0,
            sigma_avg=sigma_avg,
            feature_names=feature_names,
            adj_mat=adj_matrix,
            matrix_name="Матрица Z-значений Гессиана"
        )
        

class MultiplicationSeparabilitySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=10, pool_size=100, num_samples=30):
        super().__init__("Multiplicative Separability")
        self.k_sigma = k_sigma             
        self.num_test_points = num_test_points  
        self.pool_size = pool_size         
        self.num_samples = num_samples

    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext'):
        n_features = len(context.get_active_features())
        if n_features < 2:
            return False, None

        # E = y * H_ij - G_i * G_j
        def eval_fn(sample_path, pt, n_features):
            y = sample_path(pt)
            
            pt_temp = pt.clone().requires_grad_(True)
            y_temp = sample_path(pt_temp)
            G = torch.autograd.grad(y_temp, pt_temp, create_graph=True)[0]
            
            H = torch.autograd.functional.hessian(sample_path, pt)
            
            # G_i * G_j
            G_outer = torch.outer(G, G)
            return y * H - G_outer

        avg_Z_score, adj_matrix, sigma_avg, feature_names = self._compute_pathwise_z_score(
            gp_model, context, self.num_test_points, self.pool_size, self.k_sigma, self.num_samples,
              is_symmetry = False, evaluation_fn = eval_fn
        )

        return self._evaluate_graph_and_split(
            matrix=avg_Z_score,
            threshold=3.0,
            sigma_avg=sigma_avg,
            feature_names=feature_names,
            adj_mat=adj_matrix,
            matrix_name="Матрица Z-значений мультипликативных ошибок"
        )
    
class GeneralAdditiveSeparabilitySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=10, pool_size=100, num_samples=30, bf_max_length=6):
        super().__init__("General Additive Separability")
        self.k_sigma = k_sigma             
        self.num_test_points = num_test_points  
        self.pool_size = pool_size  
        self.num_samples = num_samples
        self.bf_max_length = bf_max_length

    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext'):
        feature_names = context.get_active_features()
        n_features = len(feature_names)

        if n_features < 2:
            return False, None
        

        def eval_fn(sample_path, pt, n_features):
            # V = H_ii / G_i
            def compute_V(x):
                x_temp = x.clone().requires_grad_(True)
                y = sample_path(x_temp)
                G = torch.autograd.grad(y, x_temp, create_graph=True)[0]

                H = torch.autograd.functional.hessian(sample_path, x, create_graph=True)
                H_diag = torch.diag(H)

                return H_diag / (G + 1e-9)

            J = torch.autograd.functional.jacobian(compute_V, pt)
            E = J - J.T

            return E

        avg_Z_score, adj_matrix, sigma_avg, feature_names = self._compute_pathwise_z_score(
            gp_model, context, self.num_test_points, self.pool_size, self.k_sigma, self.num_samples,
              is_symmetry = False, evaluation_fn = eval_fn
        )

        success, separated_groups = self._evaluate_graph_and_split(
            matrix=avg_Z_score,
            threshold=3.0,
            sigma_avg=sigma_avg,
            feature_names=feature_names,
            adj_mat=adj_matrix,
            matrix_name="Средний Гессиан логарифма отношений"
        )

        if not success or len(separated_groups) < 2:
            return False, None
        
        group_A = separated_groups[0]
        group_B = []
        for g in separated_groups[1:]:
            group_B.extend(g)

        num_points = 100
        pts_a = torch.zeros((num_points, n_features), dtype=torch.float32)
        pts_b = torch.zeros((num_points, n_features), dtype=torch.float32)

        for col_idx, col_name in enumerate(feature_names):
            col_vals = context.df[col_name].values
            if col_name in group_A:
                grid_vals = np.random.uniform(np.min(col_vals), np.max(col_vals), num_points)
                pts_a[:, col_idx] = torch.tensor(grid_vals, dtype=torch.float32)
                pts_b[:, col_idx] = torch.tensor(grid_vals, dtype=torch.float32)
            else:
                median_val = context.df[col_name].median()
                pts_a[:, col_idx] = torch.full((num_points,), median_val, dtype=torch.float32)
                pts_b[:, col_idx] = torch.full((num_points,), median_val * 1.5 + 1e-5, dtype=torch.float32)

        gp_model.model.eval()
        gp_model.likelihood.eval()
        with torch.no_grad():
            y1 = gp_model.predict(pts_a).mean.numpy()
            y2 = gp_model.predict(pts_b).mean.numpy()

        registry_slice = PhysicalRegistry()
        target_dim_vector = context.registry.get_dim(context.target_name).vector
        registry_slice.register("y1", target_dim_vector)
        registry_slice.register("target", target_dim_vector)

        df_slice = pd.DataFrame({"y1": y1, "target": y2})
        slice_context = SymbolicRegressionContext(
            df=df_slice,
            registry=registry_slice,
            target_name="target",
            symbolic_mapping={"y1": sp.Symbol("y1")},
            target_expr=sp.Symbol("target")
        )

        runner = BruteForceRunner(max_lenght=self.bf_max_length)
        T_expr, _ = runner.run(slice_context, optimize_constants=True)

        if T_expr is None:
            g_inv_expr = sp.Symbol("target")
            trans_type = "linear"
        else:
            g_inv_expr, trans_type = self._identify_inverse_g(T_expr, sp.Symbol("y1"))

        registry_temp = PhysicalRegistry()
        registry_temp.register("target", context.registry.get_dim(context.target_name).vector)
        try:
            DimensionalityEvaluator.evaluate(g_inv_expr, registry_temp)
        except DimensionalError:
            print(f"[GAS] Упрощение отклонено: трансформация {trans_type} физически несовместима с размерностью таргета.")
            return False, None

        print(f"[GAS] Обнаружена обобщенная аддитивность. Функция сдвига T(y1): {T_expr}")
        print(f"[GAS] Восстановленное обратное преобразование g^-1(y): {g_inv_expr} ({trans_type})")

        return True, (separated_groups, g_inv_expr, trans_type)

                
    def _identify_inverse_g(self, y1: np.ndarray, y2: np.ndarray) -> Tuple[sp.Expr, str]:
        y1_safe = np.abs(y1) + 1e-9
        y2_safe = np.abs(y2) + 1e-9

        y1_clip = np.clip(y1, -0.99, 0.99)
        y2_clip = np.clip(y2, -0.99, 0.99)

        candidates = {
            "linear": {
                "g_inv": lambda y: y,
                "expr": sp.Symbol("target"),
                "forward_name": "linear"
            },
            "log": {
                "g_inv": lambda y: np.log(np.abs(y) + 1e-9),
                "expr": sp.log(sp.Abs(sp.Symbol("target")) + 1e-9),
                "forward_name": "log"
            },
            "sqrt": {
                "g_inv": lambda y: np.sqrt(np.abs(y)),
                "expr": sp.sqrt(sp.Abs(sp.Symbol("target"))),
                "forward_name": "pow_2"  # g(z) = z^2
            },
            "square": {
                "g_inv": lambda y: y**2,
                "expr": sp.Symbol("target")**2,
                "forward_name": "pow_half"  # g(z) = z^0.5
            },
            "arcsin": {
                "g_inv": lambda y: np.arcsin(np.clip(y, -0.99, 0.99)),
                "expr": sp.asin(sp.Symbol("target")),
                "forward_name": "sin"
            },
            "arctan": {
                "g_inv": lambda y: np.arctan(y),
                "expr": sp.atan(sp.Symbol("target")),
                "forward_name": "tan"
            }
        }

        best_cand_name = "linear"
        min_std = float('inf')

        for name, cand in candidates.items():
            try:
                z1 = cand["g_inv"](y1)
                z2 = cand["g_inv"](y2)
                
                diff = z2 - z1
                std_val = np.std(diff)
                
                if np.isnan(std_val) or np.isinf(std_val):
                    continue

                if std_val < min_std:
                    min_std = std_val
                    best_cand_name = name
            except Exception:
                continue

        chosen = candidates[best_cand_name]

        if min_std > 0.15:
            return sp.Symbol("target"), "linear"

        return chosen["expr"], chosen["forward_name"]
        

        
class AdditionSymmetrySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=10, pool_size=100, num_samples=30):
        super().__init__("Addition Symmetry")
        self.k_sigma = k_sigma          
        self.num_test_points = num_test_points
        self.pool_size = pool_size
        self.num_samples = num_samples

    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext'):
        n_features = len(context.get_active_features())
        if n_features < 2:
            return False, None
        
        dim_mask = self._get_dimension_compatibility_matrix(context)
        if np.sum(dim_mask) == 0:
            return False, None
        
        def eval_fn(sample_path, pt, n_features):
            pt_temp = pt.clone().requires_grad_(True)
            y_temp = sample_path(pt_temp)

            G = torch.autograd.grad(y_temp, pt_temp)[0]
            E = G.unsqueeze(1) - G.unsqueeze(0)
            return E

        avg_Z_score, adj_matrix, sigma_avg, feature_names = self._compute_pathwise_z_score(
            gp_model, context, self.num_test_points, self.pool_size, self.k_sigma, self.num_samples,
            is_symmetry=True, evaluation_fn=eval_fn
        )

        return self._evaluate_graph_and_split(
            matrix=avg_Z_score,
            threshold=3.0,
            sigma_avg=sigma_avg,
            feature_names=feature_names,
            adj_mat=adj_matrix,
            is_symmetry=True,
            matrix_name="Матрица попарных ошибок сложения"
        )
    
class TranslationalSymmetrySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=10, pool_size=100, num_samples = 30):
        super().__init__("Translational Symmetry")
        self.k_sigma = k_sigma          
        self.num_test_points = num_test_points
        self.pool_size = pool_size
        self.num_samples = num_samples

    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext'):
        n_features = len(context.get_active_features())
        if n_features < 2:
            return False, None
        
        dim_mask = self._get_dimension_compatibility_matrix(context)
        if np.sum(dim_mask) == 0:
            return False, None
        
        def eval_fn(sample_path, pt, n_features):
            pt_temp = pt.clone().requires_grad_(True)
            y_temp = sample_path(pt_temp)

            G = torch.autograd.grad(y_temp, pt_temp)[0]
            E = G.unsqueeze(1) + G.unsqueeze(0)

            return E

        avg_Z_score, adj_matrix, sigma_avg, feature_names = self._compute_pathwise_z_score(
            gp_model, context, self.num_test_points, self.pool_size, self.k_sigma, self.num_samples,
              is_symmetry = True, evaluation_fn = eval_fn
        )

        return self._evaluate_graph_and_split(
            matrix=avg_Z_score,
            threshold=3.0,
            sigma_avg=sigma_avg,
            feature_names=feature_names,
            adj_mat=adj_matrix,
            matrix_name="Матрица попарных ошибок сдвига",
            is_symmetry=True 
        )



class LargeScaleSymmetrySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=10, pool_size=100, num_samples = 30):
        super().__init__("LargeScale Symmetry")
        self.k_sigma = k_sigma          
        self.num_test_points = num_test_points
        self.pool_size = pool_size
        self.num_samples = num_samples

    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext'):
        n_features = len(context.get_active_features())
        if n_features < 2:
            return False, None
        
        def eval_fn(sample_path, pt, n_features):
            pt_temp = pt.clone().requires_grad_(True)
            y_temp = sample_path(pt_temp)

            G = torch.autograd.grad(y_temp, pt_temp)[0]
            V = G * pt_temp
            E = V.unsqueeze(1) + V.unsqueeze(0)

            return E



        avg_Z_score, adj_matrix, sigma_avg, feature_names = self._compute_pathwise_z_score(
            gp_model, context, self.num_test_points, self.pool_size, self.k_sigma, self.num_samples,
                is_symmetry = True, evaluation_fn = eval_fn
        )

        return self._evaluate_graph_and_split(
            matrix=avg_Z_score,
            threshold=3.0,
            sigma_avg=sigma_avg,
            feature_names=feature_names,
            adj_mat=adj_matrix,
            matrix_name="Матрица попарных ошибок отношения",
            is_symmetry=True 
        )
        
class MultiplySymmetrySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=10, pool_size=100, num_samples = 30):
        super().__init__("Multiply Symmetry")
        self.k_sigma = k_sigma          
        self.num_test_points = num_test_points
        self.pool_size = pool_size
        self.num_samples = num_samples

    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext'):
        n_features = len(context.get_active_features())
        if n_features < 2:
            return False, None
        
        def eval_fn(sample_path, pt, n_features):
            pt_temp = pt.clone().requires_grad_(True)
            y_temp = sample_path(pt_temp)

            G = torch.autograd.grad(y_temp, pt_temp)[0]
            V = G * pt_temp
            E = V.unsqueeze(1) - V.unsqueeze(0)

            return E


        avg_Z_score, adj_matrix, sigma_avg, feature_names = self._compute_pathwise_z_score(
            gp_model, context, self.num_test_points, self.pool_size, self.k_sigma, self.num_samples,
                is_symmetry = True, evaluation_fn = eval_fn
        )

        return self._evaluate_graph_and_split(
            matrix=avg_Z_score,
            threshold=3.0,
            sigma_avg=sigma_avg,
            feature_names=feature_names,
            adj_mat=adj_matrix,
            matrix_name="Матрица попарных ошибок произведения",
            is_symmetry=True 
        )
        
class GeneralizedSymmetrySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=10, num_samples_m=100, max_group_size=3, pool_size=100):
        super().__init__("Generalized Symmetry")
        self.k_sigma = k_sigma             
        self.num_test_points = num_test_points  
        self.num_samples_m = num_samples_m      
        self.max_group_size = max_group_size    
        self.pool_size = pool_size   

    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext'):
        feature_names = context.get_active_features()
        n_features = len(feature_names)

        if n_features < 2:
            return False, None
        
        test_points, base_threshold, sigma_avg, y_avg, feature_names = self._get_best_points_and_threshold(
            gp_model, context, self.num_test_points, self.pool_size, self.k_sigma
        )
        dynamic_threshold = base_threshold

        X_tensor = torch.tensor(context.df[feature_names].values, dtype=torch.float32)
        n_samples = X_tensor.shape[0]

        best_error = 1.0
        best_group = None

        gp_model.model.eval()
        gp_model.likelihood.eval()

        for group_size in range(2, min(self.max_group_size + 1, n_features)):
            for subset in combinations(range(n_features), group_size):
                remaining = [idx for idx in range(n_features) if idx not in subset]
                test_errors = []

                for pt in test_points:
                    x_prime_0 = pt[list(subset)]

                    sample_indices = np.random.choice(n_samples, self.num_samples_m, replace=False)
                    samples_x_double_prime = X_tensor[sample_indices][:, remaining]

                    pts = torch.zeros((self.num_samples_m, n_features))
                    pts[:, subset] = x_prime_0
                    pts[:, remaining] = samples_x_double_prime

                    pts.requires_grad_(True)

                    
                    y = gp_model.likelihood(gp_model.model(pts)).mean

                    grads = torch.autograd.grad(y.sum(), pts)[0]
                    grads_prime = grads[:, subset]

                    norms = torch.norm(grads_prime, dim=1, keepdim=True) + 1e-9
                    v = grads_prime / norms

                    V = (v.t() @ v) / self.num_samples_m
                    eigenvalues = torch.linalg.eigvalsh(V)
                    max_eigenvalue = eigenvalues[-1] 

                    err = 1.0 - max_eigenvalue.item()
                    test_errors.append(err)

                avg_err = np.mean(test_errors)
                if avg_err < best_error:
                    best_error = avg_err
                    best_group = [feature_names[i] for i in subset]

        print("\n" + "="*50)
        print("Generalized Symmetry")
        print(f"1. Лучшая группа: {best_group}")
        print(f"2. Средний шум GP (sigma_avg): {sigma_avg:.6f}")
        print(f"3. Ошибка лучшей группы (1 - lambda_max): {best_error:.6f}")
        print(f"4. Рассчитанный порог: {dynamic_threshold:.6f}")
        print("="*50 + "\n")

        if best_error < dynamic_threshold:
            return True, best_group
        else:
            return False, None

class CompositionalitySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=10, pool_size=100, bf_max_lenght=6):
        super().__init__("Compositionality")
        self.k_sigma = k_sigma
        self.num_test_points = num_test_points
        self.pool_size = pool_size
        self.bf_max_lenght = bf_max_lenght

    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext'):
        n_features = len(context.get_active_features())
        if n_features < 2:
            return False, None

        test_points, base_threshold, sigma_avg, y_avg, feature_names = self._get_best_points_and_threshold(
            gp_model, context.df, self.num_test_points, self.pool_size, self.k_sigma
        )
        dynamic_threshold = base_threshold

        test_points.requires_grad_(True)
        gp_model.model.eval()
        gp_model.likelihood.eval()

        y = gp_model.likelihood(gp_model.model(test_points)).mean
        grads_target = torch.autograd.grad(y.sum(), test_points)[0]

        symbols = [sp.Symbol(name) for name in feature_names]
        
        best_error = 1.0
        best_candidate = None
        best_pair = None

        test_points_np = test_points.detach().numpy()

        for idx1, idx2 in combinations(range(n_features), 2):
            col_name1, col_name2 = feature_names[idx1], feature_names[idx2]
            
            grads_target_pair = grads_target[:, [idx1, idx2]]
            norms_target_pair = torch.norm(grads_target_pair, dim=1, keepdim=True) + 1e-9
            v_target_pair = grads_target_pair / norms_target_pair

            num_grid_points = 100
            col1_vals = context.df[col_name1].values
            col2_vals = context.df[col_name2].values

            grid_x1 = np.random.uniform(np.min(col1_vals), np.max(col1_vals), num_grid_points)
            grid_x2 = np.random.uniform(np.min(col2_vals), np.max(col2_vals), num_grid_points)

            pts_slice = torch.zeros((num_grid_points, n_features), dtype=torch.float32)

            for col_idx, col_name in enumerate(feature_names):
                if col_idx == idx1:
                    pts_slice[:, col_idx] = torch.tensor(grid_x1, dtype=torch.float32)
                elif col_idx == idx2:
                    pts_slice[:, col_idx] = torch.tensor(grid_x2, dtype=torch.float32)
                else:
                    median_val = context.df[col_name].median()
                    pts_slice[:, col_idx] = torch.full((num_grid_points,), median_val, dtype=torch.float32)
            
            with torch.no_grad():
                predictions = gp_model.predict(pts_slice)
                Y_slice = predictions.mean.numpy()

            registry_slice = PhysicalRegistry()
            registry_slice.register("target", context.registry.get_dim(context.target_name).vector)
            registry_slice.register(col_name1, context.registry.get_dim(col_name1).vector)
            registry_slice.register(col_name2, context.registry.get_dim(col_name2).vector)

            df_slice = pd.DataFrame({
                col_name1: grid_x1,
                col_name2: grid_x2,
                "target": Y_slice
            })

            mapping_slice = {
                col_name1: sp.Symbol(col_name1),
                col_name2: sp.Symbol(col_name2)
            }

            slice_context = SymbolicRegressionContext(
                df=df_slice,
                registry=registry_slice,
                target_name="target",
                symbolic_mapping=mapping_slice,
                target_expr=sp.Symbol("target")
            )

            runner = BruteForceRunner(max_length=self.bf_max_lenght)
            h_expr, _ = runner.run(slice_context, optimize_constants=False)

            
            if h_expr is None:
                print("Не нашел h_expr")
                continue
            
            s1, s2 = sp.Symbol(col_name1), sp.Symbol(col_name2)
            grad_h_sym = [sp.diff(h_expr, s1), sp.diff(h_expr, s2)]

            test_points_np = test_points.detach().numpy()
            args = [test_points_np[:, idx1], test_points_np[:, idx2]]

            try:
                grad_arrays = []
                for g_expr in grad_h_sym:
                    f_g = sp.lambdify([s1,s2], g_expr, 'numpy')
                    vals = f_g(*args)
                    if isinstance(vals, (int, float, np.integer, np.floating)):
                        vals = np.full(len(test_points), vals)
                    grad_arrays.append(vals)
                grads_cand = np.stack(grad_arrays, axis=1)

                grads_cand = np.stack(grad_arrays, axis=1)
                grads_cand_tensor = torch.tensor(grads_cand, dtype=torch.float32)

                norms_cand = torch.norm(grads_cand_tensor, dim=1, keepdim=True) + 1e-9
                v_cand = grads_cand_tensor / norms_cand

                cos_sim = torch.sum(v_target_pair*v_cand, dim=1)
                error = torch.mean(1.0 - cos_sim**2).item()
                
                if error < best_error:
                    best_error = error
                    best_candidate = h_expr
                    best_pair = (col_name1, col_name2)
            except (ZeroDivisionError, ValueError, TypeError, OverflowError):
                continue

        print("\n" + "="*50)
        print("Compositionality (Brute-Force)")
        print(f"1. Лучшая пара переменных: {best_pair}")
        print(f"2. Лучшая найденная связь h(x): {best_candidate}")
        print(f"3. Средний шум GP (sigma_avg): {sigma_avg:.6f}")
        print(f"4. Ошибка косинуса (1 - cos^2): {best_error:.6f}")
        print(f"5. Рассчитанный порог: {dynamic_threshold:.6f}")
        print("="*50 + "\n")

        if best_error < dynamic_threshold:
            return True, best_candidate
        else:
            return False, None