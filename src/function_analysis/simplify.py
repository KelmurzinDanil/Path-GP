import pandas as pd
import numpy as np
import torch
import sympy as sp
import gpytorch
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
    def try_simplify(self, gp_model, dataset: pd.DataFrame, k_sigma: float = None):
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
                                       num_test_points: Optional[int] = None, pool_size: Optional[int] = None, k_sigma: float = 3.0):
        X_train = gp_model.model.train_inputs[0]
        test_points = X_train

        gp_model.model.eval()
        gp_model.likelihood.eval()
        with torch.no_grad():
            f_dist = gp_model.model(test_points)
            predictions = gp_model.likelihood(f_dist)
            variance = predictions.variance.cpu().numpy()
            mean_vals = predictions.mean.cpu().numpy()

        sigma_avg = np.sqrt(np.mean(variance))
        y_avg = np.mean(np.abs(mean_vals))
        y_avg = max(y_avg, 1e-9)

        base_threshold = k_sigma * (sigma_avg / y_avg)
        base_threshold = max(base_threshold, 1e-4)

        feature_names = context.get_active_features()
        return test_points, base_threshold, sigma_avg, y_avg, feature_names
    
    def _compute_pathwise_nrmse(self, gp_model, context: 'SymbolicRegressionContext', 
                              num_test_points: Optional[int], pool_size: Optional[int], k_sigma: float,
                              num_samples: int = 30, is_symmetry: bool = False, evaluation_fn = None, dim_mask: np.ndarray = None) -> tuple:

        test_points, base_threshold, sigma_avg, y_avg, feature_names = self._get_best_points_and_threshold(
            gp_model, context, num_test_points, pool_size, k_sigma
        )
        n_features = len(feature_names)

        gp_model.model.eval()
        gp_model.likelihood.eval()


        def gp_mean_fn(x):
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return gp_model.model(x).mean.squeeze()

        n_pts = len(test_points)
        residuals = torch.zeros((n_pts, n_features, n_features), dtype=torch.float64)
        signals = torch.zeros((n_pts, n_features, n_features), dtype=torch.float64)


        with gpytorch.settings.fast_computations(solves=False, covar_root_decomposition=False, log_prob=False), \
             gpytorch.settings.fast_pred_var(False):
             
            for pt_idx, pt in enumerate(test_points):
                pt_req = pt.clone().detach().requires_grad_(True)
                res, sig = evaluation_fn(gp_mean_fn, pt_req, n_features)
                residuals[pt_idx] = res.detach()
                signals[pt_idx] = sig.detach()

        rms_E = torch.sqrt(torch.mean(residuals ** 2, dim=0))
        rms_S = torch.sqrt(torch.mean(signals ** 2, dim=0))

        nrmse_matrix = rms_E / (rms_S + 1e-9)

        base_k = k_sigma if k_sigma is not None else 3.0
        nrmse_threshold = 0.05 * (base_k / 3.0) if base_k >= 0.5 else base_k

        nrmse_matrix_np = nrmse_matrix.cpu().detach().numpy()
        base_k = k_sigma if k_sigma is not None else 3.0

        if not is_symmetry:
            default_threshold = 0.12 * (base_k / 3.0)
            off_diag_vals = nrmse_matrix_np[~np.eye(n_features, dtype=bool)]
            
            if len(off_diag_vals) > 1:
                sorted_vals = np.sort(np.unique(off_diag_vals))
                gaps = np.diff(sorted_vals)
                
                if len(gaps) > 0:
                    max_gap_idx = np.argmax(gaps)
                    max_gap = gaps[max_gap_idx]
                    
                    lower_val = sorted_vals[max_gap_idx]    
                    upper_val = sorted_vals[max_gap_idx + 1] 
                    

                    if lower_val < 0.20 and max_gap > 0.1:
                        nrmse_threshold = (lower_val + upper_val) / 2.0
                    else:
                        nrmse_threshold = default_threshold
                else:
                    nrmse_threshold = default_threshold
            else:
                nrmse_threshold = default_threshold
                
            adj_matrix = (nrmse_matrix_np >= nrmse_threshold).astype(int)

        else:
            max_allowed_sym_error = 0.02
            calculated_sym_thresh = 0.05 * (base_k / 3.0)
            
            nrmse_threshold = min(max_allowed_sym_error, calculated_sym_thresh)
            adj_matrix = (nrmse_matrix_np < nrmse_threshold).astype(int)

        np.fill_diagonal(adj_matrix, 0)

        np.fill_diagonal(adj_matrix, 0)

        if dim_mask is not None:
            adj_matrix = adj_matrix * dim_mask

        return nrmse_matrix, adj_matrix, nrmse_threshold, feature_names


    def _evaluate_graph_and_split(self, matrix: torch.Tensor, threshold: float, 
                                  sigma_avg: float, feature_names: list,  adj_mat,
                                  matrix_name: str = "Матрица результатов",
                                  is_symmetry: bool = False) -> tuple:
        adj_matrix = adj_mat
        
        n_components, labels = connected_components(adj_matrix, directed=False)

        groups = {}
        for idx, comp_label in enumerate(labels):
            groups.setdefault(comp_label, []).append(feature_names[idx])
        
        separated_groups = []
        for comp_label, group_vars in groups.items():
            if is_symmetry and len(group_vars) > 2:
                is_clique = True
                for var1 in group_vars:
                    for var2 in group_vars:
                        if var1 != var2:
                            i = feature_names.index(var1)
                            j = feature_names.index(var2)
                            if adj_matrix[i, j] == 0:
                                is_clique = False
                                break
                    if not is_clique:
                        break
                
                if is_clique:
                    separated_groups.append(group_vars)
                else:
                    for var in group_vars:
                        separated_groups.append([var])
            else:
                separated_groups.append(group_vars)
        
        print("\n" + "="*50)
        print(f"{self.name} (NRMSE)")
        print(f"1. {matrix_name}:")
        print(np.round(matrix.cpu().detach().numpy(), 5))
        print(f"2. Порог прохождения NRMSE: {threshold:.6f}")
        print("3. Матрица смежности графа:")
        print(adj_matrix)
        print(f"4. Результат графа: Компонентов = {len(separated_groups)}, Ярлыки = {labels}")
        print("="*50 + "\n")

        valid_groups = [g for g in separated_groups if len(g) >= 2]
        
        if is_symmetry:
            if len(valid_groups) > 0:
                return True, valid_groups
            else:
                return False, None
        else:
            if len(separated_groups) > 1:
                return True, separated_groups
            else:
                return False, None

class AdditiveSeparabilitySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=None, pool_size=None, num_samples=30):
        super().__init__("Additive Separability")
        self.k_sigma = k_sigma             
        self.num_test_points = num_test_points  
        self.pool_size = pool_size         
        self.num_samples = num_samples

    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext', k_sigma: float = None):
        effective_k_sigma = k_sigma if k_sigma is not None else self.k_sigma
        n_features = len(context.get_active_features())
        if n_features < 2:
            return False, None
        
        def eval_fn(sample_path, pt, n_features):
            H = torch.autograd.functional.hessian(sample_path, pt)
            diag_H = torch.diag(H)
            S = torch.sqrt(diag_H.unsqueeze(1)**2 + diag_H.unsqueeze(0)**2)
            return H, S

        nrmse_matrix, adj_matrix, threshold, feature_names = self._compute_pathwise_nrmse(
            gp_model, context, self.num_test_points, self.pool_size, effective_k_sigma, self.num_samples,
            is_symmetry=False, evaluation_fn=eval_fn
        )

        return self._evaluate_graph_and_split(
            matrix=nrmse_matrix,
            threshold=threshold,
            sigma_avg=0.0,
            feature_names=feature_names,
            adj_mat=adj_matrix,
            matrix_name="Матрица RFF NRMSE Гессиана"
        )
        

class MultiplicationSeparabilitySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=None, pool_size=None, num_samples=30):
        super().__init__("Multiplicative Separability")
        self.k_sigma = k_sigma             
        self.num_test_points = num_test_points  
        self.pool_size = pool_size         
        self.num_samples = num_samples

    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext', k_sigma: float = None):
        effective_k_sigma = k_sigma if k_sigma is not None else self.k_sigma
        n_features = len(context.get_active_features())
        if n_features < 2:
            return False, None

        def eval_fn(sample_path, pt, n_features):
            y = sample_path(pt)
            
            pt_temp = pt.clone().requires_grad_(True)
            y_temp = sample_path(pt_temp)
            G = torch.autograd.grad(y_temp, pt_temp, create_graph=True)[0]
            
            H = torch.autograd.functional.hessian(sample_path, pt)
            G_outer = torch.outer(G, G)

            E = y * H - G_outer
            S = torch.abs(G_outer)
            return E, S

        nrmse_matrix, adj_matrix, threshold, feature_names = self._compute_pathwise_nrmse(
            gp_model, context, self.num_test_points, self.pool_size, effective_k_sigma, self.num_samples,
            is_symmetry=False, evaluation_fn=eval_fn
        )

        return self._evaluate_graph_and_split(
            matrix=nrmse_matrix,
            threshold=threshold,
            sigma_avg=0.0,
            feature_names=feature_names,
            adj_mat=adj_matrix,
            matrix_name="Матрица RFF NRMSE мультипликативных ошибок"
        )

class GeneralAdditiveSeparabilitySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=None, pool_size=None, num_samples=30, bf_max_length=6,
                 optimize_constants=True, allowed_constants=[1.0, 2.0],
                 allowed_ops=["add", "sub", "mul", "div", "sin", "cos", "exp", "log"]):
        super().__init__("General Additive Separability")
        self.k_sigma = k_sigma             
        self.num_test_points = num_test_points  
        self.pool_size = pool_size  
        self.num_samples = num_samples

    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext', k_sigma: float = None):
        effective_k_sigma = k_sigma if k_sigma is not None else self.k_sigma
        feature_names = context.get_active_features()
        n_features = len(feature_names)

        if n_features < 2:
            return False, None

        X_train = gp_model.model.train_inputs[0]
        pts = X_train.clone().detach().requires_grad_(True)
        n_pts = pts.shape[0]

        candidates = {
            "linear": {
                "g_inv": lambda y: y,
                "expr": sp.Symbol("target"),
                "forward_name": "linear"
            },
            "log": {
                "g_inv": lambda y: torch.log(torch.abs(y) + 1e-9),
                "expr": sp.log(sp.Abs(sp.Symbol("target")) + 1e-9),
                "forward_name": "log"
            },
            "sqrt": {
                "g_inv": lambda y: torch.sqrt(torch.abs(y)),
                "expr": sp.sqrt(sp.Abs(sp.Symbol("target"))),
                "forward_name": "pow_2"
            },
            "square": {
                "g_inv": lambda y: y**2,
                "expr": sp.Symbol("target")**2,
                "forward_name": "pow_half"
            },
            "arctan": {
                "g_inv": lambda y: torch.atan(y),
                "expr": sp.atan(sp.Symbol("target")),
                "forward_name": "tan"
            },
            "arcsin": {
                "g_inv": lambda y: torch.asin(y),
                "expr": sp.asin(sp.Symbol("target")),
                "forward_name": "sin"
            }
        }

        best_cand_name = None
        best_groups = None
        best_nrmse_score = float('inf')

        gp_model.model.eval()
        gp_model.likelihood.eval()

        for cand_name, cand in candidates.items():
            try:
                with torch.no_grad(), gpytorch.settings.fast_computations(solves=False, covar_root_decomposition=False):
                    y_pred = gp_model.model(pts).mean

                if cand_name == "arcsin" and torch.max(torch.abs(y_pred)) > 0.99:
                    continue

                E = torch.zeros((n_pts, n_features, n_features), dtype=torch.float64)
                S = torch.zeros((n_pts, n_features, n_features), dtype=torch.float64)

                with gpytorch.settings.fast_computations(solves=False, covar_root_decomposition=False):
                    for pt_idx in range(n_pts):
                        pt_single = pts[pt_idx:pt_idx+1]
                        
                        def get_z(x):
                            return cand["g_inv"](gp_model.model(x).mean.squeeze())

                        H_z = torch.autograd.functional.hessian(get_z, pt_single).squeeze()
                        if n_features == 1:
                            H_z = H_z.unsqueeze(0).unsqueeze(0)

                        E[pt_idx] = H_z
                        diag_Hz = torch.diag(H_z)
                        S[pt_idx] = torch.sqrt(diag_Hz.unsqueeze(1)**2 + diag_Hz.unsqueeze(0)**2)

                rms_E = torch.sqrt(torch.mean(E ** 2, dim=0))
                rms_S = torch.sqrt(torch.mean(S ** 2, dim=0))
                nrmse_matrix = (rms_E / (rms_S + 1e-9)).detach().cpu().numpy()
                np.fill_diagonal(nrmse_matrix, 0)

                best_cand_nrmse = float('inf')
                best_cand_groups = None

                for threshold in np.linspace(0.02, 0.15, 14):
                    adj_matrix = (nrmse_matrix < threshold).astype(int)
                    np.fill_diagonal(adj_matrix, 0)

                    n_comp, labels = connected_components(adj_matrix, directed=False)

                    if n_comp > 1:
                        groups_dict = {}
                        for idx, comp_label in enumerate(labels):
                            groups_dict.setdefault(comp_label, []).append(feature_names[idx])
                        
                        current_groups = list(groups_dict.values())
                        
                        off_block_mask = (adj_matrix == 1) & (~np.eye(n_features, dtype=bool))
                        score = np.mean(nrmse_matrix[off_block_mask]) if np.any(off_block_mask) else 1.0

                        if score < best_cand_nrmse:
                            best_cand_nrmse = score
                            best_cand_groups = current_groups

                if best_cand_groups is not None and best_cand_nrmse < best_nrmse_score:
                    best_nrmse_score = best_cand_nrmse
                    best_cand_name = cand_name
                    best_groups = best_cand_groups

            except Exception:
                continue

        if best_cand_name is None or best_groups is None or best_nrmse_score > 0.08:
            print("[GAS] Обобщенная аддитивность не обнаружена.")
            return False, None

        chosen = candidates[best_cand_name]
        g_inv_expr = chosen["expr"]
        trans_type = chosen["forward_name"]

        registry_temp = PhysicalRegistry()
        registry_temp.register("target", context.registry.get_dim(context.target_name).vector)
        try:
            DimensionalityEvaluator.evaluate(g_inv_expr, registry_temp)
        except DimensionalError:
            print(f"[GAS] Упрощение отклонено: трансформация {trans_type} физически несовместима с размерностью таргета.")
            return False, None

        print(f"\n" + "="*50)
        print("General Additive Separability (RFF NRMSE Block-Hessian)")
        print(f"1. Найдено g^-1(y)      : {g_inv_expr} ({trans_type})")
        print(f"2. NRMSE между блоками  : {best_nrmse_score:.6f}")
        print(f"3. Разделение на группы : {best_groups}")
        print("="*50 + "\n")

        return True, (best_groups, g_inv_expr, trans_type)
        

        
class AdditionSymmetrySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=None, pool_size=None, num_samples=30):
        super().__init__("Addition Symmetry")
        self.k_sigma = k_sigma          
        self.num_test_points = num_test_points
        self.pool_size = pool_size
        self.num_samples = num_samples

    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext', k_sigma: float = None):
        effective_k_sigma = k_sigma if k_sigma is not None else self.k_sigma
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
            G_sq = G**2
            S = torch.sqrt(G_sq.unsqueeze(1) + G_sq.unsqueeze(0))
            return E, S

        nrmse_matrix, adj_matrix, threshold, feature_names = self._compute_pathwise_nrmse(
            gp_model, context, self.num_test_points, self.pool_size, effective_k_sigma, self.num_samples,
            is_symmetry=True, evaluation_fn=eval_fn, dim_mask=dim_mask
        )

        return self._evaluate_graph_and_split(
            matrix=nrmse_matrix,
            threshold=threshold,
            sigma_avg=0.0,
            feature_names=feature_names,
            adj_mat=adj_matrix,
            is_symmetry=True,
            matrix_name="Матрица RFF NRMSE сложения"
        )

    
class TranslationalSymmetrySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=None, pool_size=None, num_samples=30):
        super().__init__("Translational Symmetry")
        self.k_sigma = k_sigma          
        self.num_test_points = num_test_points
        self.pool_size = pool_size
        self.num_samples = num_samples

    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext', k_sigma: float = None):
        effective_k_sigma = k_sigma if k_sigma is not None else self.k_sigma
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
            G_sq = G**2
            S = torch.sqrt(G_sq.unsqueeze(1) + G_sq.unsqueeze(0))
            return E, S

        nrmse_matrix, adj_matrix, threshold, feature_names = self._compute_pathwise_nrmse(
            gp_model, context, self.num_test_points, self.pool_size, effective_k_sigma, self.num_samples,
            is_symmetry=True, evaluation_fn=eval_fn, dim_mask=dim_mask
        )

        return self._evaluate_graph_and_split(
            matrix=nrmse_matrix,
            threshold=threshold,
            sigma_avg=0.0,
            feature_names=feature_names,
            adj_mat=adj_matrix,
            matrix_name="Матрица RFF NRMSE сдвига",
            is_symmetry=True 
        )

class LargeScaleSymmetrySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=None, pool_size=None, num_samples=30):
        super().__init__("LargeScale Symmetry")
        self.k_sigma = k_sigma          
        self.num_test_points = num_test_points
        self.pool_size = pool_size
        self.num_samples = num_samples

    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext', k_sigma: float = None):
        effective_k_sigma = k_sigma if k_sigma is not None else self.k_sigma
        n_features = len(context.get_active_features())
        if n_features < 2:
            return False, None
        
        def eval_fn(sample_path, pt, n_features):
            pt_temp = pt.clone().requires_grad_(True)
            y_temp = sample_path(pt_temp)

            G = torch.autograd.grad(y_temp, pt_temp)[0]
            V = G * pt_temp
            E = V.unsqueeze(1) + V.unsqueeze(0)
            V_sq = V**2
            S = torch.sqrt(V_sq.unsqueeze(1) + V_sq.unsqueeze(0))
            return E, S

        nrmse_matrix, adj_matrix, threshold, feature_names = self._compute_pathwise_nrmse(
            gp_model, context, self.num_test_points, self.pool_size, effective_k_sigma, self.num_samples,
            is_symmetry=True, evaluation_fn=eval_fn
        )

        return self._evaluate_graph_and_split(
            matrix=nrmse_matrix,
            threshold=threshold,
            sigma_avg=0.0,
            feature_names=feature_names,
            adj_mat=adj_matrix,
            matrix_name="Матрица RFF NRMSE отношения",
            is_symmetry=True 
        )
        
class MultiplySymmetrySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=None, pool_size=None, num_samples=30):
        super().__init__("Multiply Symmetry")
        self.k_sigma = k_sigma          
        self.num_test_points = num_test_points
        self.pool_size = pool_size
        self.num_samples = num_samples

    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext', k_sigma: float = None):
        effective_k_sigma = k_sigma if k_sigma is not None else self.k_sigma
        n_features = len(context.get_active_features())
        if n_features < 2:
            return False, None
        
        def eval_fn(sample_path, pt, n_features):
            pt_temp = pt.clone().requires_grad_(True)
            y_temp = sample_path(pt_temp)

            G = torch.autograd.grad(y_temp, pt_temp)[0]
            V = G * pt_temp
            E = V.unsqueeze(1) - V.unsqueeze(0)
            V_sq = V**2
            S = torch.sqrt(V_sq.unsqueeze(1) + V_sq.unsqueeze(0))
            return E, S

        nrmse_matrix, adj_matrix, threshold, feature_names = self._compute_pathwise_nrmse(
            gp_model, context, self.num_test_points, self.pool_size, effective_k_sigma, self.num_samples,
            is_symmetry=True, evaluation_fn=eval_fn
        )

        return self._evaluate_graph_and_split(
            matrix=nrmse_matrix,
            threshold=threshold,
            sigma_avg=0.0,
            feature_names=feature_names,
            adj_mat=adj_matrix,
            matrix_name="Матрица RFF NRMSE произведения",
            is_symmetry=True 
        )
        
class GeneralizedSymmetrySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=None, num_samples_m=200, max_group_size=3, pool_size=None):
        super().__init__("Generalized Symmetry")
        self.k_sigma = k_sigma             
        self.num_test_points = num_test_points  
        self.num_samples_m = num_samples_m      
        self.max_group_size = max_group_size    
        self.pool_size = pool_size   

    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext', k_sigma: float = None):
        effective_k_sigma = k_sigma if k_sigma is not None else self.k_sigma
        feature_names = context.get_active_features()
        n_features = len(feature_names)

        if n_features < 2:
            return False, None
        
        X_tensor = torch.tensor(context.df[feature_names].values, dtype=torch.float64)
        n_eval = X_tensor.shape[0]
        pts_eval = X_tensor.clone().detach().requires_grad_(True)

        best_error = 1.0
        best_group = None

        gp_model.model.eval()
        gp_model.likelihood.eval()

        for subset in combinations(range(n_features), 2):
            with gpytorch.settings.cholesky_jitter(1e-3):
                y = gp_model.likelihood(gp_model.model(pts_eval)).mean

            grads = torch.autograd.grad(y.sum(), pts_eval)[0]
            grads_prime = grads[:, list(subset)]

            norms = torch.norm(grads_prime, dim=1, keepdim=True) + 1e-9
            v = grads_prime / norms

            V = (v.t() @ v) / n_eval
            eigenvalues = torch.linalg.eigvalsh(V)
            max_eigenvalue = eigenvalues[-1] 

            err = 1.0 - max_eigenvalue.item()

            if err < best_error:
                best_error = err
                best_group = [feature_names[i] for i in subset]

        strict_threshold = 0.008

        print("\n" + "="*50)
        print("Generalized Symmetry (Full Dataset Evaluation)")
        print(f"1. Лучшая группа                       : {best_group}")
        print(f"2. Ошибка коллинеарности (1-lambda_max): {best_error:.6f}")
        print(f"3. Порог прохождения                   : {strict_threshold:.6f}")
        print("="*50 + "\n")

        if best_group is not None and best_error < strict_threshold: 
            return True, best_group
        else:
            return False, None
        
class CompositionalitySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=None, pool_size=None, bf_max_length=6, optimize_constants=True, allowed_constants=[1.0, 2.0],
                 allowed_ops=["add", "sub", "mul", "div", "sin", "cos", "exp", "log"], k_best=50):
        super().__init__("Compositionality")
        self.k_sigma = k_sigma
        self.num_test_points = num_test_points
        self.pool_size = pool_size
        self.bf_max_length = bf_max_length
        self.optimize_constants = optimize_constants
        self.allowed_constants = allowed_constants
        self.allowed_ops = allowed_ops
        self.k_best = k_best

    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext', k_sigma: float = None):
        feature_names = context.get_active_features()
        n_features = len(feature_names)
        if n_features < 2:
            return False, None

        X_train = gp_model.model.train_inputs[0]
        test_pts = X_train.clone().detach().requires_grad_(True)

        gp_model.model.eval()
        gp_model.likelihood.eval()

        with gpytorch.settings.fast_computations(solves=False, covar_root_decomposition=False, log_prob=False), \
             gpytorch.settings.fast_pred_var(False):
            y_pred = gp_model.model(test_pts).mean
            grads_target = torch.autograd.grad(y_pred.sum(), test_pts)[0]

        best_error = float('inf')
        best_candidate = None
        best_pair = None

        for idx1, idx2 in combinations(range(n_features), 2):
            col_name1, col_name2 = feature_names[idx1], feature_names[idx2]
            s1, s2 = sp.Symbol(col_name1), sp.Symbol(col_name2)
            hypotheses = [
                s1 + s2, s1 - s2, s1 * s2, s1 / (s2 + 1e-9), 
                s1**2 + s2**2, s1**2 - s2**2, sp.sqrt(sp.Abs(s1) + 1e-9) * s2
            ]

            test_pts_np = test_pts.detach().cpu().numpy()
            args = [test_pts_np[:, idx1], test_pts_np[:, idx2]]
            
            grad_y_1 = grads_target[:, idx1].cpu().numpy()
            grad_y_2 = grads_target[:, idx2].cpu().numpy()

            for h_expr in hypotheses:
                try:
                    DimensionalityEvaluator.evaluate(h_expr, context.registry)
                except DimensionalError:
                    continue
                    
                grad_h_sym = [sp.diff(h_expr, s1), sp.diff(h_expr, s2)]

                try:
                    grad_arrays = []
                    for g_expr in grad_h_sym:
                        f_g = sp.lambdify([s1, s2], g_expr, 'numpy')
                        vals = f_g(*args)
                        if np.isscalar(vals):
                            vals = np.full(len(test_pts), vals)
                        grad_arrays.append(vals)

                    grad_h_1 = grad_arrays[0]
                    grad_h_2 = grad_arrays[1]

                    cross_product = grad_y_1 * grad_h_2 - grad_y_2 * grad_h_1
                    
                    magnitude = np.abs(grad_y_1 * grad_h_2) + np.abs(grad_y_2 * grad_h_1) + 1e-9
                    relative_error = np.mean(np.abs(cross_product) / magnitude)

                    if relative_error < best_error:
                        best_error = relative_error
                        best_candidate = h_expr
                        best_pair = (col_name1, col_name2)
                        
                except Exception:
                    continue

        strict_threshold = 0.05

        print("\n" + "="*50)
        print("Compositionality (Wedge Product Evaluation)")
        print(f"1. Лучшая пара переменных                  : {best_pair}")
        print(f"2. Лучшая найденная связь h(x)             : {best_candidate}")
        print(f"3. Ошибка коллинеарности (Relative Wedge)  : {best_error:.6f}")
        print(f"4. Порог прохождения                       : {strict_threshold:.6f}")
        print("="*50 + "\n")

        if best_candidate is not None and best_error < strict_threshold:
            return True, (best_candidate, best_pair)
        else:
            return False, None