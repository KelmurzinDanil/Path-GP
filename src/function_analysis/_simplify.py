import copy
import pandas as pd
import numpy as np
import sympy as sp
import torch
from abc import ABC, abstractmethod
from scipy.sparse.csgraph import connected_components
from itertools import combinations
from scipy.stats import rankdata
from typing import List, Optional, Tuple, Any
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

from symbolic.sm_context import SymbolicRegressionContext, DimensionalityEvaluator
from get_pi_complex import DimensionalError, PhysicalRegistry
from GP.utils import train_gp_model
from function_analysis.py_brute_force import BruteForceRunner, rpn_to_sympy
from config import PipelineConfig
from logger import setup_logger

logger = setup_logger("Simplify")

class BaseSimplifier(ABC):
    def __init__(self, name: str, config: PipelineConfig):
        self.name = name
        self.pipeline_config = config
        self.verbose = config.verbose

    @abstractmethod
    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext', k_sigma: float = None):
        pass

class BaseGPFriedmanSimplifier(BaseSimplifier):
    def __init__(self, name: str, config: PipelineConfig, mode: str = "add"):
        super().__init__(name, config)
        self.mode = mode 
        self.config = config.friedman
        self.gp_config = config.gp_config

    def _predict_gp_pipeline(self, gp_pipeline, X_np: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(X_np, dtype=torch.float64)
        gp_pipeline.model.eval()
        gp_pipeline.likelihood.eval()
        with torch.no_grad():
            return gp_pipeline.predict(x_tensor).mean.cpu().numpy()

    def _compute_friedman_h_gp(self, gp_pipeline, X_np: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_vars = len(feature_names)
        H2_add_matrix = np.zeros((n_vars, n_vars))
        H2_mult_matrix = np.zeros((n_vars, n_vars))
        C_matrix = np.zeros((n_vars, n_vars))
        
        np.random.seed(42)
        X_bg = X_np[np.random.choice(len(X_np), min(len(X_np), self.config.bg_samples), replace=False)].copy()
        pd_null = np.mean(self._predict_gp_pipeline(gp_pipeline, X_bg))

        pd_1d, grids_1d = {}, {}
        for i in range(n_vars):
            grid = np.linspace(np.percentile(X_np[:, i], 2), np.percentile(X_np[:, i], 98), self.config.grid_density)
            grids_1d[i] = grid
            pd_vals = np.zeros(self.config.grid_density)
            for g_idx, val in enumerate(grid):
                X_temp = X_bg.copy()
                X_temp[:, i] = val
                pd_vals[g_idx] = np.mean(self._predict_gp_pipeline(gp_pipeline, X_temp))
            pd_1d[i] = pd_vals

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                grid_i, grid_j = grids_1d[i], grids_1d[j]
                std_i, std_j = np.std(X_bg[:, i]) + 1e-9, np.std(X_bg[:, j]) + 1e-9
                knn = NearestNeighbors(n_neighbors=1).fit(np.column_stack([X_bg[:, i] / std_i, X_bg[:, j] / std_j]))

                pd_2d = np.zeros((self.config.grid_density, self.config.grid_density))
                pd_i_grid = np.zeros((self.config.grid_density, self.config.grid_density))
                pd_j_grid = np.zeros((self.config.grid_density, self.config.grid_density))
                valid_mask = np.zeros((self.config.grid_density, self.config.grid_density), dtype=bool)

                for u_idx, val_i in enumerate(grid_i):
                    for v_idx, val_j in enumerate(grid_j):
                        dist, _ = knn.kneighbors(np.array([[val_i / std_i, val_j / std_j]]))
                        if dist[0][0] <= self.config.max_allowed_distance:
                            valid_mask[u_idx, v_idx] = True
                            X_temp = X_bg.copy()
                            X_temp[:, i], X_temp[:, j] = val_i, val_j
                            pd_2d[u_idx, v_idx] = np.mean(self._predict_gp_pipeline(gp_pipeline, X_temp))
                            pd_i_grid[u_idx, v_idx] = pd_1d[i][u_idx]
                            pd_j_grid[u_idx, v_idx] = pd_1d[j][v_idx]

                if np.sum(valid_mask) == 0:
                    continue

                # 1. H2 Аддитивный
                M_add = pd_2d - pd_i_grid - pd_j_grid + pd_null
                denom_add = np.sum((pd_2d[valid_mask] - pd_null)**2) + 1e-9
                h2_add = np.sum(M_add[valid_mask]**2) / denom_add

                # 2. Аналитический поиск константы C (Формула: Y_ij = C * X_ij)
                X_ij = M_add
                Y_ij = pd_2d * pd_null - pd_i_grid * pd_j_grid
                
                var_x = np.sum(X_ij[valid_mask] ** 2)

                if var_x < 1e-12:
                    c_est = 0.0
                else:
                    c_est = np.sum(
                        X_ij[valid_mask] * Y_ij[valid_mask]
                    ) / var_x

                # 3. H2 Мультипликативный (со сдвигом на C)
                F_prime = pd_2d - c_est
                pd_i_prime = pd_i_grid - c_est
                pd_j_prime = pd_j_grid - c_est
                pd_null_prime = pd_null - c_est
                
                denom_mult_val = pd_null_prime
                if abs(denom_mult_val) < 1e-7:
                    denom_mult_val = 1e-7 * np.sign(denom_mult_val + 1e-12)
                    
                M_mult = F_prime - (pd_i_prime * pd_j_prime) / denom_mult_val
                denom_mult = np.sum((F_prime[valid_mask] - pd_null_prime)**2) + 1e-9
                h2_mult = np.sum(M_mult[valid_mask]**2) / denom_mult

                H2_add_matrix[i, j] = H2_add_matrix[j, i] = h2_add
                H2_mult_matrix[i, j] = H2_mult_matrix[j, i] = h2_mult
                C_matrix[i, j] = C_matrix[j, i] = c_est

        return H2_add_matrix, H2_mult_matrix, C_matrix

    def _fit_surrogate_and_get_h2(self, X_mat: np.ndarray, Y_val: np.ndarray, feature_names: List[str], gp_model):
        cfg = self.gp_config if self.gp_config is not None else getattr(gp_model, "config")
        surrogate_pipeline, _ = train_gp_model(torch.tensor(X_mat, dtype=torch.float64), torch.tensor(Y_val, dtype=torch.float64), cfg)
        return self._compute_friedman_h_gp(surrogate_pipeline, X_mat, feature_names)

    def _compute_inter_group_score(self, H2_mat: np.ndarray, groups: List[List[str]], feature_names: List[str]) -> float:
        feat_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        inter_values = []
        for g1_idx in range(len(groups)):
            for g2_idx in range(g1_idx + 1, len(groups)):
                for f1 in groups[g1_idx]:
                    for f2 in groups[g2_idx]:
                        inter_values.append(H2_mat[feat_to_idx[f1], feat_to_idx[f2]])
        return float(np.median(inter_values)) if inter_values else 0.0

    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext', k_sigma: float = None) -> Tuple[bool, Optional[Any]]:
        feature_names = context.get_active_features()
        if len(feature_names) < 2: return False, None

        X_mat = context.df[feature_names].values
        Y_val = context.df[context.target_name].values
        
        if self.verbose: 
            logger.info(f"[{self.name}] Запуск анализа H2 (Линейный с аналитическим C) для {feature_names}...")

        H2_add_mat, H2_mult_mat, C_mat = self._fit_surrogate_and_get_h2(X_mat, Y_val, feature_names, gp_model)

        primary_mat = H2_add_mat if self.mode == "add" else H2_mult_mat
        
        adj_matrix = np.zeros((len(feature_names), len(feature_names)), dtype=int)
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                if primary_mat[i, j] >= self.config.h2_threshold:
                    adj_matrix[i, j] = adj_matrix[j, i] = 1

        n_components, labels = connected_components(adj_matrix, directed=False)
        groups = list({label: [feature_names[i] for i, l in enumerate(labels) if l == label] for label in set(labels)}.values())

        if self.verbose:
            df_h2 = pd.DataFrame(primary_mat, index=feature_names, columns=feature_names)
            logger.info(
                f"[{self.name}] Матрица H2 ({self.mode}):\n"
                f"{df_h2.to_string(float_format=lambda x: f'{x:.5f}')}\n"
                f"Выявленные блоки (порог {self.config.h2_threshold}): {groups}"
            )

        if len(groups) <= 1: 
            if self.verbose:
                logger.info(f"[{self.name}] ОТКЛОНЕНО: Не найдено независимых блоков (все переменные связаны).")
            return False, None

        score_add = self._compute_inter_group_score(H2_add_mat, groups, feature_names)
        score_mult = self._compute_inter_group_score(H2_mult_mat, groups, feature_names)

        detected_operator = "+" if score_add < score_mult else "*"
        target_operator = "+" if self.mode == "add" else "*"

        if self.verbose:
            logger.info(
                f"[{self.name}] Сравнение топологии для блоков {groups}:\n"
                f"  -> Межблочный H2 (add):  {score_add:.5f}\n"
                f"  -> Межблочный H2 (mult): {score_mult:.5f}\n"
                f"  -> Выявленный оператор:  '{detected_operator}'"
            )

        if detected_operator != target_operator:
            if self.verbose: logger.info(f"[{self.name}] ОТКЛОНЕНО: Ищем '{target_operator}', но топология показывает '{detected_operator}'.")
            return False, None
            
        if self.mode == "mult":
            if self.verbose:
                logger.info(
                    f"[{self.name}] ПОДТВЕРЖДЕНО: "
                    f"мультипликативная структура {groups} "
                    f"передается на functional isolation. "
                    f"Внешний сдвиг C не используется."
                )

            return True, groups

        if self.verbose:
            logger.info(f"[{self.name}] ПОДТВЕРЖДЕНО: Гипотеза {groups} передается на проверку лосса.")

        return True, groups

class AdditiveSeparabilitySimplifier(BaseGPFriedmanSimplifier):
    def __init__(self, config: PipelineConfig):
        super().__init__("Additive Separability", config, mode="add")


class MultiplicationSeparabilitySimplifier(BaseGPFriedmanSimplifier):
    def __init__(self, config: PipelineConfig):
        super().__init__("Multiplicative Separability", config, mode="mult")


class GeneralAdditiveSeparabilitySimplifier(BaseGPFriedmanSimplifier):
    def __init__(self, config: PipelineConfig):
        super().__init__("General Additive Separability", config, mode="add")

    def _extract_rank_groups(self, H2_mat: np.ndarray, feature_names: List[str]) -> Tuple[List[List[str]], float]:
        n_vars = len(feature_names)
        if n_vars < 2:
            return [[f] for f in feature_names], 0.0

        upper_vals = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                upper_vals.append(H2_mat[i, j])
        upper_vals = np.array(upper_vals)

        if n_vars == 2:
            if upper_vals[0] < self.config.h2_threshold:
                return [[feature_names[0]], [feature_names[1]]], float(upper_vals[0])
            else:
                return [feature_names], float(upper_vals[0])

        if np.max(upper_vals) < self.config.h2_threshold:
            return [[f] for f in feature_names], float(np.max(upper_vals))

        sorted_vals = np.sort(upper_vals)
        gaps = np.diff(sorted_vals)

        if len(gaps) == 0:
            effective_threshold = self.config.h2_threshold
        else:
            max_gap_idx = int(np.argmax(gaps))
            max_gap = gaps[max_gap_idx]
            val_low = sorted_vals[max_gap_idx]
            val_high = sorted_vals[max_gap_idx + 1]

            if max_gap >= 0.003 or (val_high / (val_low + 1e-6) >= 1.8 and val_high >= self.config.h2_threshold):
                effective_threshold = (val_low + val_high) / 2.0
            else:
                effective_threshold = self.config.h2_threshold

        adj_matrix = np.zeros((n_vars, n_vars), dtype=int)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if H2_mat[i, j] >= effective_threshold:
                    adj_matrix[i, j] = adj_matrix[j, i] = 1

        n_components, labels = connected_components(adj_matrix, directed=False)
        groups = list({label: [feature_names[i] for i, l in enumerate(labels) if l == label] for label in set(labels)}.values())

        return groups, effective_threshold

    def try_simplify(self, gp_model, context: 'SymbolicRegressionContext', k_sigma: float = None) -> Tuple[bool, Optional[Tuple[List[List[str]], Any, Any, str]]]:
        feature_names = context.get_active_features()
        if len(feature_names) < 2: 
            return False, None

        if self.verbose:
            logger.info(f"[{self.name}] Запуск рангового анализа H2 для переменных {feature_names}...")

        X_np, Y_np = context.df[feature_names].values, context.df[context.target_name].values
        Y_ranks_norm = (rankdata(Y_np) - np.mean(rankdata(Y_np))) / (np.std(rankdata(Y_np)) + 1e-9)
        cfg = self.gp_config if self.gp_config is not None else getattr(gp_model, "config")
        
        try:
            rank_gp, _ = train_gp_model(torch.tensor(X_np, dtype=torch.float64), torch.tensor(Y_ranks_norm, dtype=torch.float64), cfg)
            H2_mat_ranks, _, _ = self._compute_friedman_h_gp(rank_gp, X_np, feature_names)
        except Exception as e:
            if self.verbose: 
                logger.error(f"[{self.name}] Ошибка обучения рангового суррогата: {e}", exc_info=True)
            return False, None

        best_groups, effective_threshold = self._extract_rank_groups(H2_mat_ranks, feature_names)

        if self.verbose:
            df_h2_rank = pd.DataFrame(H2_mat_ranks, index=feature_names, columns=feature_names)
            logger.info(
                f"[{self.name}] Матрица рангового H2:\n{df_h2_rank.to_string(float_format=lambda x: f'{x:.5f}')}\n"
                f"Выявленные блоки (адаптивный порог {effective_threshold:.5f}): {best_groups}"
            )
        
        if len(best_groups) <= 1:
            if self.verbose:
                logger.info(f"[{self.name}] ОТКЛОНЕНО: Ранговый суррогат не нашел независимых блоков.")
            return False, None

        if self.verbose: 
            logger.info(f"[{self.name}] Топология выявила блоки {best_groups}. Генерация кандидатов g^-1(y) через C++ BRF...")

        bf_cfg = copy.deepcopy(self.pipeline_config.brute_force)
        bf_cfg.max_length = max(self.config.bf_max_length, 6)
        bf_cfg.k_best = 100
        bf_cfg.optimize_constants = False
        bf_cfg.allowed_constants = [1.0, 2.0]
        bf_cfg.allowed_ops = ["add", "sub", "mul", "div", "pow", "sqrt", "log", "exp"]
        
        registry_temp = PhysicalRegistry()
        registry_temp.register(context.target_name, context.registry.get_dim(context.target_name).vector)

        valid_cands_dicts = BruteForceRunner(bf_cfg).get_valid_candidates(
            context=SymbolicRegressionContext(pd.DataFrame({context.target_name: Y_np}), registry_temp, context.target_name),
            X_np=Y_np.reshape(-1, 1), 
            Y_np=Y_np,
            active_features=[context.target_name]
        )

        if not valid_cands_dicts:
            if self.verbose:
                logger.info(f"[{self.name}] ОТКЛОНЕНО: C++ BruteForce не вернул кандидатов.")
            return False, None

        valid_cands = []
        seen_simplified_exprs = set()
        target_sym = sp.Symbol(context.target_name, positive=True, real=True)
        Z_sym = sp.Symbol("Z", positive=True, real=True)
        var_y = np.var(Y_np) + 1e-12

        for cand in valid_cands_dicts:
            raw_expr = cand['expr_mapped']
            if not raw_expr.free_symbols:
                continue

            g_inv_expr = sp.simplify(sp.nsimplify(raw_expr, rational=True))
            expr_key = str(g_inv_expr)

            if expr_key in seen_simplified_exprs:
                continue
            seen_simplified_exprs.add(expr_key)

            try:
                sols = sp.solve(g_inv_expr - Z_sym, target_sym)
                if not sols:
                    sols = sp.solve(g_inv_expr - Z_sym, sp.Symbol(context.target_name))
                if not sols: 
                    continue

                fn_ginv = sp.lambdify(target_sym, g_inv_expr, 'numpy')
                z_test = fn_ginv(Y_np)
                if np.isnan(z_test).any() or np.isinf(z_test).any() or np.std(z_test) < 1e-9:
                    continue

                for sol in sols:
                    try:
                        g_fwd_candidate = sol.subs(Z_sym, target_sym)
                        fn_gfwd = sp.lambdify(target_sym, g_fwd_candidate, 'numpy')
                        y_reconstructed = fn_gfwd(z_test)

                        if np.isnan(y_reconstructed).any() or np.isinf(y_reconstructed).any():
                            continue

                        rel_recon_err = np.mean((y_reconstructed - Y_np) ** 2) / var_y
                        if rel_recon_err > 1e-4:
                            continue

                        valid_cands.append({
                            'g_inv': g_inv_expr,
                            'g_forward': g_fwd_candidate,
                            'fn': fn_ginv,
                            'name': str(g_inv_expr),
                            'complexity': cand['raw_complexity']
                        })
                        break
                    except Exception:
                        continue

            except Exception:
                continue 

        if not valid_cands: 
            if self.verbose: 
                logger.info(f"[{self.name}] ОТКЛОНЕНО: C++ BruteForce не нашел аналитически обратимых функций g^-1(y).")
            return False, None

        if self.verbose:
            logger.info(f"[{self.name}] Отобрано {len(valid_cands)} уникальных обратимых форм. Оценка остаточного H²...")

        best_cand, best_off_block_h2 = None, float("inf")
        off_h2_raw = self._compute_inter_group_score(H2_mat_ranks, best_groups, feature_names)

        valid_cands.sort(key=lambda c: c.get('complexity', len(str(c['g_inv']))))

        for cand in valid_cands:
            try:
                Z_cand = cand['fn'](Y_np)
                if np.isnan(Z_cand).any() or np.isinf(Z_cand).any() or np.std(Z_cand) < 1e-9: 
                    continue
                H2_mat_cand, _, _ = self._fit_surrogate_and_get_h2(X_np, Z_cand, feature_names, gp_model)
                off_h2 = self._compute_inter_group_score(H2_mat_cand, best_groups, feature_names)
                
                if self.verbose:
                    logger.info(f"  -> Проверка Z = {cand['name']:<25} | Остаточный H²: {off_h2:.6f}")

                if off_h2 < best_off_block_h2:
                    best_off_block_h2, best_cand = off_h2, cand

                if best_off_block_h2 < 1e-4:
                    if self.verbose:
                        logger.info(f"[{self.name}] Найдена идеальная обратная функция '{best_cand['name']}' (H² = {best_off_block_h2:.6f}). Завершаем перебор досрочно!")
                    break

            except Exception: 
                continue

        if best_cand is None: 
            if self.verbose: 
                logger.info(f"[{self.name}] ОТКЛОНЕНО: Все кандидаты выдали NaN/Inf при преобразовании таргета.")
            return False, None
            
        contrast = (off_h2_raw + 1e-6) / (best_off_block_h2 + 1e-6)

        if best_off_block_h2 > self.config.h2_threshold and contrast < self.config.min_contrast:
            if self.verbose: 
                logger.info(
                    f"[{self.name}] ОТКЛОНЕНО: Лучший кандидат '{best_cand['name']}' не прошел пороги.\n"
                    f"  -> Остаточный H2: {best_off_block_h2:.4f} (Порог: {self.config.h2_threshold:.4f})\n"
                    f"  -> Контраст: {contrast:.2f}x (Порог: {self.config.min_contrast}x)"
                )
            return False, None

        if self.verbose:
            logger.info(
                f"[{self.name}] УСПЕШНАЯ ДЕКОМПОЗИЦИЯ! Блоки: {best_groups}\n"
                f"g^-1(y): {best_cand['g_inv']} | g(z): {best_cand['g_forward']}\n"
                f"Остаточный H^2: {best_off_block_h2:.6f} (Контраст: {contrast:.2f}x)"
            )

        return True, (best_groups, best_cand['g_forward'], best_cand['g_inv'], best_cand['name'])

class CompositionalitySimplifier(BaseSimplifier):
    def __init__(self, config: PipelineConfig):
        super().__init__("Compositionality", config)
        self.config = config.compositionality

    def _find_h_via_bruteforce(self, flat_i: np.ndarray, flat_j: np.ndarray, pdp_surface: np.ndarray, context, col_i: str, col_j: str) -> tuple:
        bf_cfg = copy.deepcopy(self.pipeline_config.brute_force)
        bf_cfg.max_length = self.config.bf_max_length
        bf_cfg.optimize_constants = False
        bf_cfg.allowed_constants = []
        bf_cfg.allowed_ops = ["add", "sub", "mul", "div", "pow", "sqrt"]
        
        cands = BruteForceRunner(bf_cfg).get_valid_candidates(
            context, 
            np.column_stack([flat_i, flat_j]), 
            pdp_surface, 
            [col_i, col_j],
            require_all_vars=True 
        )
        
        s1, s2 = sp.Symbol(col_i), sp.Symbol(col_j)
        best_h, best_spearman = None, -1.0

        for cand in cands:
            mapped_expr = cand['expr_mapped']
            f_h = sp.lambdify([s1, s2], mapped_expr, 'numpy')
            try:
                h_vals = f_h(flat_i, flat_j)
                if np.isscalar(h_vals): h_vals = np.full(len(flat_i), h_vals)
                if np.std(h_vals) < 1e-9: continue
                corr = abs(spearmanr(h_vals, pdp_surface)[0])
                if corr > best_spearman:
                    best_spearman, best_h = corr, mapped_expr
            except Exception: continue
                
        return best_h, best_spearman

    def try_simplify(self, gp_model, context, k_sigma: float = None):
        feature_names = context.get_active_features()
        if len(feature_names) < 2: return False, None

        X_bg = context.df[feature_names].values
        np.random.seed(42)
        X_slices = X_bg[np.random.choice(len(X_bg), min(len(X_bg), self.config.n_slices), replace=False)].copy()

        gp_model.model.eval()
        gp_model.likelihood.eval()

        candidate_pairs = []
        for i, j in combinations(range(len(feature_names)), 2):
            grid_i = np.linspace(np.percentile(X_bg[:, i], 5), np.percentile(X_bg[:, i], 95), self.config.grid_density)
            grid_j = np.linspace(np.percentile(X_bg[:, j], 5), np.percentile(X_bg[:, j], 95), self.config.grid_density)
            mesh_i, mesh_j = np.meshgrid(grid_i, grid_j)
            flat_i, flat_j = mesh_i.ravel(), mesh_j.ravel()
            
            surfaces = []
            with torch.no_grad():
                for m_idx in range(len(X_slices)):
                    X_eval = np.tile(X_slices[m_idx], (len(flat_i), 1))
                    X_eval[:, i], X_eval[:, j] = flat_i, flat_j
                    surfaces.append(gp_model.predict(torch.tensor(X_eval, dtype=torch.float64)).mean.cpu().numpy())
            
            mean_surf = np.mean(surfaces, axis=0)
            if np.ptp(mean_surf) < 1e-9: continue 

            corr_matrix = np.zeros((len(surfaces), len(surfaces)))
            for a in range(len(surfaces)):
                for b in range(a + 1, len(surfaces)):
                    corr_matrix[a, b] = abs(spearmanr(surfaces[a], surfaces[b])[0])
            
            median_corr = np.median(corr_matrix[np.triu_indices(len(surfaces), k=1)])
            if median_corr > self.config.slice_spearman_threshold:
                candidate_pairs.append({'pair': (feature_names[i], feature_names[j]), 'slice_spearman': median_corr, 'mean_surf': mean_surf, 'flat_i': flat_i, 'flat_j': flat_j})

        if not candidate_pairs: return False, None

        best_h, best_pair, best_template_score, best_slice_stability = None, None, -1.0, -1.0
        for cand in candidate_pairs:
            h_expr, spearman_score = self._find_h_via_bruteforce(cand['flat_i'], cand['flat_j'], cand['mean_surf'], context, cand['pair'][0], cand['pair'][1])
            if spearman_score > best_template_score:
                best_template_score, best_h, best_pair, best_slice_stability = spearman_score, h_expr, cand['pair'], cand['slice_spearman']

        if best_template_score < self.config.template_spearman_threshold:
            if self.verbose and best_template_score > 0:
                logger.info(f"[{self.name}] Отклонено: Лучшее совпадение C++ '{best_h}' для {best_pair} дало Spearman={best_template_score:.4f} (Порог: {self.config.template_spearman_threshold})")
            return False, None

        if self.verbose:
            logger.info(f"[{self.name}] УСПЕХ: C++ BRF НАШЕЛ ФИЗИЧЕСКУЮ СИММЕТРИЮ!\n"
                        f"Переменные: {best_pair} | Стабильность срезов: {best_slice_stability:.5f}\n"
                        f"h(x): {best_h} (Совпадение: {best_template_score:.5f})")

        return True, (best_h, best_pair)