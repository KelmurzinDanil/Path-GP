import pandas as pd
import numpy as np
import torch
import sympy as sp
from abc import ABC, abstractmethod
from scipy.sparse.csgraph import connected_components
from itertools import combinations


class BaseSimplifier(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def try_simplify(self, gp_model, dataset: pd.DataFrame):
        pass
    
    def _get_best_points_and_threshold(self, gp_model, dataset: pd.DataFrame, 
                                       num_test_points: int, pool_size: int, k_sigma: float):
        
        feature_names = [col for col in dataset.columns if col != 'target']
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
        
        n_points = min(num_test_points, n_samples)
        best_indices = np.argsort(variance)[:n_points]
        test_points = pool_points[best_indices]

        sigma_avg = np.sqrt(np.mean(variance[best_indices]))
        y_avg = np.max(np.mean(np.abs(predictions[best_indices])), 1e-9)

        
        base_threshold = k_sigma * (sigma_avg/y_avg)
        base_threshold = max(base_threshold, 1e-4)

        return test_points, base_threshold, sigma_avg, y_avg, feature_names
    
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
    def __init__(self, k_sigma=3.0, num_test_points=10, pool_size=100):
        super().__init__("Additive Separability")
        self.k_sigma = k_sigma             
        self.num_test_points = num_test_points  
        self.pool_size = pool_size         

    def try_simplify(self, gp_model, dataset: pd.DataFrame):
        n_features = len(dataset.columns) - 1

        if n_features < 2:
            return False, None
        
        test_points, base_threshold, sigma_avg, feature_names = self._get_best_points_and_threshold(
            gp_model, dataset, self.num_test_points, self.pool_size, self.k_sigma
        )
        dynamic_threshold = base_threshold

        def predict_mean(x):
            gp_model.model.eval()
            gp_model.likelihood.eval()
            return gp_model.likelihood(gp_model.model(x.unsqueeze(0))).mean.squeeze()

        avg_hessian = torch.zeros((n_features, n_features))

        for pt in test_points:
            pt.requires_grad_(True)
            H = torch.autograd.functional.hessian(predict_mean, pt)
            avg_hessian += torch.abs(H)

        avg_hessian /= len(test_points)
        adj_matrix = (avg_hessian > dynamic_threshold).numpy().astype(int)
        np.fill_diagonal(adj_matrix, 0)

        return self._evaluate_graph_and_split(
            matrix=avg_hessian,
            threshold=dynamic_threshold,
            sigma_avg=sigma_avg,
            feature_names=feature_names,
            adj_mat=adj_matrix,
            matrix_name="Средний Гессиан"
        )
        

class MultiplicationSeparabilitySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=10, pool_size=100):
        super().__init__("Multiplicative Separability")
        self.k_sigma = k_sigma             
        self.num_test_points = num_test_points  
        self.pool_size = pool_size  

    def try_simplify(self, gp_model, dataset: pd.DataFrame):
        n_features = len(dataset.columns) - 1

        if n_features < 2:
            return False, None
        
        test_points, base_threshold, sigma_avg, feature_names = self._get_best_points_and_threshold(
            gp_model, dataset, self.num_test_points, self.pool_size, self.k_sigma
        )

        dynamic_threshold = base_threshold

        mult_errors = torch.zeros((n_features, n_features))
        
        gp_model.model.eval()
        gp_model.likelihood.eval()
        
        for pt in test_points:
            pt_temp = pt.clone().requires_grad_(True)
            
            y = gp_model.likelihood(gp_model.model(pt_temp.unsqueeze(0))).mean.squeeze()
            
            grads = torch.autograd.grad(y, pt_temp, create_graph=True)[0]
            
            for i in range(n_features):
                gi = grads[i]
                hessian_row = torch.autograd.grad(gi, pt_temp, retain_graph=True)[0]
                
                for j in range(i + 1, n_features):
                    gj = grads[j]
                    hij = hessian_row[j] 
                    
                    term1 = y * hij
                    term2 = gi * gj
                    
                    denom = torch.abs(term1) + torch.abs(term2) + 1e-9
                    err = torch.abs(term1 - term2) / denom
                    
                    mult_errors[i, j] += err
                    mult_errors[j, i] += err

        mult_errors /= len(test_points)
        
        adj_matrix = (mult_errors > dynamic_threshold).numpy().astype(int)
        np.fill_diagonal(adj_matrix, 0)

        return self._evaluate_graph_and_split(
            matrix=mult_errors,
            threshold=dynamic_threshold,
            sigma_avg=sigma_avg,
            feature_names=feature_names,
            adj_mat=adj_matrix,
            matrix_name="Матрица ошибок умножения"
        )

class GeneralAdditiveSeparabilitySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=10, pool_size=100):
        super().__init__("General Additive Separability")
        self.k_sigma = k_sigma             
        self.num_test_points = num_test_points  
        self.pool_size = pool_size  

    def try_simplify(self, gp_model, dataset: pd.DataFrame):
        n_features = len(dataset.columns) - 1

        if n_features < 2:
            return False, None
        
        test_points, base_threshold, sigma_avg, feature_names = self._get_best_points_and_threshold(
            gp_model, dataset, self.num_test_points, self.pool_size, self.k_sigma
        )

        dynamic_threshold = base_threshold

        def log_gradient_ratio(x):
            x_temp = x.clone().requires_grad_(True)

            gp_model.model.eval()
            gp_model.likelihood.eval()
            y = gp_model.likelihood(gp_model.model(x_temp.unsqueeze(0))).mean.squeeze()

            grads = torch.autograd.grad(y, x_temp, create_graph=True)[0]

            df_dx = grads[0] # производная по x
            df_dy = grads[1] # производная по y

            return torch.log(torch.abs(df_dx / (df_dy + 1e-9)) + 1e-9)
        

        avg_hessian = torch.zeros((n_features, n_features))
        # gp_model.eval()
        
        for pt in test_points:
            pt.requires_grad_(True)

            H = torch.autograd.functional.hessian(log_gradient_ratio, pt)

            avg_hessian += torch.abs(H)

        avg_hessian /= len(test_points)
        adj_matrix = (avg_hessian > dynamic_threshold).numpy().astype(int)
        np.fill_diagonal(adj_matrix, 0)

        return self._evaluate_graph_and_split(
            matrix=avg_hessian,
            threshold=dynamic_threshold,
            sigma_avg=sigma_avg,
            feature_names=feature_names,
            adj_mat=adj_matrix,
            matrix_name="Средний Гессиан логарифма отношений"
        )
        

class TranslationalSymmetrySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=10, pool_size=100):
        super().__init__("Translational Symmetry")
        self.k_sigma = k_sigma          
        self.num_test_points = num_test_points
        self.pool_size = pool_size

    def try_simplify(self, gp_model, dataset: pd.DataFrame):
        n_features = len(dataset.columns) - 1
        if n_features < 2:
            return False, None
        
        test_points, base_threshold, sigma_avg, feature_names = self._get_best_points_and_threshold(
            gp_model, dataset, self.num_test_points, self.pool_size, self.k_sigma
        )
        dynamic_threshold = base_threshold

        sym_errors = torch.zeros((n_features, n_features))
        
        for pt in test_points:
            pt_temp = pt.clone().requires_grad_(True)
            gp_model.model.eval()
            gp_model.likelihood.eval()
            y = gp_model.likelihood(gp_model.model(pt_temp.unsqueeze(0))).mean.squeeze()

            grads = torch.autograd.grad(y, pt_temp)[0]

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    gi = grads[i]
                    gj = grads[j]

                    denom = torch.abs(gi) + torch.abs(gj) + 1e-9
                    err = torch.abs(gi + gj) / denom

                    sym_errors[i, j] += err
                    sym_errors[j, i] += err

        sym_errors /= len(test_points)
        adj_matrix = (sym_errors < dynamic_threshold).numpy().astype(int)
        np.fill_diagonal(adj_matrix, 0)

        return self._evaluate_graph_and_split(
            matrix=sym_errors,
            threshold=dynamic_threshold,
            sigma_avg=sigma_avg,
            feature_names=feature_names,
            adj_mat=adj_matrix,
            matrix_name="Матрица попарных ошибок сдвига",
            is_symmetry=True 
        )



class LargeScaleSymmetrySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=10, pool_size=100):
        super().__init__("LargeScale Symmetry")
        self.k_sigma = k_sigma          
        self.num_test_points = num_test_points
        self.pool_size = pool_size

    def try_simplify(self, gp_model, dataset: pd.DataFrame):
        n_features = len(dataset.columns) - 1
        if n_features < 2:
            return False, None
        
        test_points, base_threshold, sigma_avg, feature_names = self._get_best_points_and_threshold(
            gp_model, dataset, self.num_test_points, self.pool_size, self.k_sigma
        )
        dynamic_threshold = base_threshold

        sym_errors = torch.zeros((n_features, n_features))
        
        for pt in test_points:
            pt_temp = pt.clone().requires_grad_(True)
            gp_model.model.eval()
            gp_model.likelihood.eval()
            y = gp_model.likelihood(gp_model.model(pt_temp.unsqueeze(0))).mean.squeeze()
            grads = torch.autograd.grad(y, pt_temp)[0]

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    xi = pt_temp[i]
                    xj = pt_temp[j]
                    gi = grads[i]
                    gj = grads[j]

                    denom = torch.abs(xi * gi) + torch.abs(xj * gj) + 1e-9
                    err = torch.abs(xi * gi + xj * gj) / denom

                    sym_errors[i, j] += err
                    sym_errors[j, i] += err

        sym_errors /= len(test_points)
        adj_matrix = (sym_errors < dynamic_threshold).numpy().astype(int)
        np.fill_diagonal(adj_matrix, 0)

        return self._evaluate_graph_and_split(
            matrix=sym_errors,
            threshold=dynamic_threshold,
            sigma_avg=sigma_avg,
            feature_names=feature_names,
            adj_mat=adj_matrix,
            matrix_name="Матрица попарных ошибок отношения",
            is_symmetry=True 
        )
        
class MultiplySymmetrySimplifier(BaseSimplifier):
    def __init__(self, k_sigma=3.0, num_test_points=10, pool_size=100):
        super().__init__("Multiply Symmetry")
        self.k_sigma = k_sigma          
        self.num_test_points = num_test_points
        self.pool_size = pool_size

    def try_simplify(self, gp_model, dataset: pd.DataFrame):
        n_features = len(dataset.columns) - 1
        if n_features < 2:
            return False, None
        
        test_points, base_threshold, sigma_avg, feature_names = self._get_best_points_and_threshold(
            gp_model, dataset, self.num_test_points, self.pool_size, self.k_sigma
        )
        dynamic_threshold = base_threshold

        sym_errors = torch.zeros((n_features, n_features))
        
        for pt in test_points:
            pt_temp = pt.clone().requires_grad_(True)
            gp_model.model.eval()
            gp_model.likelihood.eval()
            y = gp_model.likelihood(gp_model.model(pt_temp.unsqueeze(0))).mean.squeeze()

            grads = torch.autograd.grad(y, pt_temp)[0]

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    xi = pt_temp[i]
                    xj = pt_temp[j]
                    gi = grads[i]
                    gj = grads[j]

                    denom = torch.abs(xi * gi) + torch.abs(xj * gj) + 1e-9
                    err = torch.abs(xi * gi - xj * gj) / denom

                    sym_errors[i, j] += err
                    sym_errors[j, i] += err

        sym_errors /= len(test_points)
        adj_matrix = (sym_errors < dynamic_threshold).numpy().astype(int)
        np.fill_diagonal(adj_matrix, 0)

        return self._evaluate_graph_and_split(
            matrix=sym_errors,
            threshold=dynamic_threshold,
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

    def try_simplify(self, gp_model, dataset: pd.DataFrame):
        feature_names = [col for col in dataset.columns if col != 'target']
        n_features = len(feature_names)

        if n_features < 2:
            return False, None
        
        test_points, base_threshold, sigma_avg, feature_names = self._get_best_points_and_threshold(
            gp_model, dataset, self.num_test_points, self.pool_size, self.k_sigma
        )
        dynamic_threshold = base_threshold

        X_tensor = torch.tensor(dataset[feature_names].values, dtype=torch.float32)
        n_samples = X_tensor.shape[0]

        best_error = 1.0
        best_group = None

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
                    gp_model.model.eval()
                    gp_model.likelihood.eval()
                    
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
    def __init__(self, k_sigma=3.0, num_test_points=10, pool_size=100):
        super().__init__("Compositionality")
        self.k_sigma = k_sigma          
        self.num_test_points = num_test_points
        self.pool_size = pool_size

    def _generate_candidate_expressions(self, feature_names):
        symbols = [sp.Symbol(name) for name in feature_names]
        candidates = []

        for s1, s2 in combinations(symbols, 2):
            candidates.extend([
                s1 + s2,
                s1 - s2,
                s1 * s2,
                s1 / s2,
                s1**2 + s2**2,
                s1**2 - s2**2,
                s1 / (s2**2 + 1e-9),
                s2 / (s1**2 + 1e-9),
                
                sp.sin(s1 - s2),
                sp.sin(s1 + s2),
                sp.cos(s1 - s2),
                sp.cos(s1 + s2),

                s1 * sp.exp(s2),
                s2 * sp.exp(s1),
                s1 * sp.exp(-s2),
                s2 * sp.exp(-s1),
                
                sp.sin(s1) * sp.exp(s2),
                sp.sin(s2) * sp.exp(s1),
                sp.sin(s1 - s2) * sp.exp(s1), # или exp(s2)
                
                sp.log(s1 + 1e-9) - sp.log(s2 + 1e-9),
                sp.log(s1 + 1e-9) + sp.log(s2 + 1e-9),
            ])

        return symbols, candidates

    def try_simplify(self, gp_model, dataset: pd.DataFrame):
        n_features = len(dataset.columns) - 1

        if n_features < 2:
            return False, None

        test_points, base_threshold, sigma_avg, feature_names = self._get_best_points_and_threshold(
            gp_model, dataset, self.num_test_points, self.pool_size, self.k_sigma
        )
        dynamic_threshold = base_threshold

        test_points.requires_grad_(True)
        gp_model.model.eval()
        gp_model.likelihood.eval()

        y = gp_model.likelihood(gp_model.model(test_points)).mean

        grads_target = torch.autograd.grad(y.sum(), test_points)[0]
        norms_target = torch.norm(grads_target, dim=1, keepdim=True) + 1e-9
        v_target = grads_target / norms_target

        symbols, candidates = self._generate_candidate_expressions(feature_names)

        best_error = 1.0
        best_candidate = None

        test_points_np = test_points.detach().numpy()
        args = [test_points_np[:, i] for i in range(n_features)]
        
        for h_expr in candidates:
            grad_h_sym = [sp.diff(h_expr, sym) for sym in symbols]
            grad_arrays = []
            
            for g_expr in grad_h_sym:
                f_g = sp.lambdify(symbols, g_expr, 'numpy')
                vals = f_g(*args)
                if isinstance(vals, (int, float, np.integer, np.floating)):
                    vals = np.full(len(test_points), vals)
                grad_arrays.append(vals)
            
            grads_cand = np.stack(grad_arrays, axis=1)
            grads_cand_tensor = torch.tensor(grads_cand, dtype=torch.float32)

            norms_cand = torch.norm(grads_cand_tensor, dim=1, keepdim=True) + 1e-9
            v_cand = grads_cand_tensor / norms_cand

            cos_sim = torch.sum(v_target * v_cand, dim=1)
            error = torch.mean(1.0 - cos_sim**2).item()

            if error < best_error:
                best_error = error
                best_candidate = h_expr

        print("\n" + "="*50)
        print("Compositionality")
        print(f"1. Лучший кандидат h(x): {best_candidate}")
        print(f"2. Средний шум GP (sigma_avg): {sigma_avg:.6f}")
        print(f"3. Ошибка косинуса (1 - cos^2): {best_error:.6f}")
        print(f"4. Рассчитанный порог: {dynamic_threshold:.6f}")
        print("="*50 + "\n")

        if best_error < dynamic_threshold:
            return True, best_candidate
        else:
            return False, None

