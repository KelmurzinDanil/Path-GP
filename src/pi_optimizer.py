import itertools
from math import gcd
from functools import reduce
from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np
import scipy.linalg as la
import sympy as sp
import dcor


def compute_dcor(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0

    res = dcor.distance_correlation(x, y)
    return float(np.nan_to_num(res, nan=0.0))


class STEQuantizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_val, step=0.5):
        return torch.round(input_val / step) * step

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def ste_quantize(x, step=0.5):
    return STEQuantizerFunction.apply(x, step)


class STEAnchorOptimizer(nn.Module):
    def __init__(self, c_particular: np.ndarray, V: np.ndarray, l1_lambda: float = 0.08):
        super().__init__()
        self.c_p = torch.tensor(c_particular, dtype=torch.float32).reshape(-1, 1)
        self.V = torch.tensor(V, dtype=torch.float32)
        self.l1_lambda = l1_lambda

        k = V.shape[1]
        self.beta = nn.Parameter(torch.zeros(k, 1, dtype=torch.float32))

    def forward(self, step: float = 1.0):
        beta_quantized = ste_quantize(self.beta, step=step)
        c_exact = self.c_p + self.V @ beta_quantized
        return c_exact

    def fit(self, log_X: np.ndarray, log_Y: np.ndarray, epochs: int = 300, lr: float = 0.1) -> sp.Matrix:
        X_tensor = torch.tensor(log_X, dtype=torch.float32)
        Y_tensor = torch.tensor(log_Y, dtype=torch.float32).reshape(-1, 1)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for _ in range(epochs):
            optimizer.zero_grad()

            c_exact = self.forward(step=1.0)
            log_anchor = X_tensor @ c_exact
            log_pi_y = Y_tensor - log_anchor

            var_loss = torch.var(log_pi_y)
            l1_loss = self.l1_lambda * torch.sum(torch.abs(c_exact))
            total_loss = var_loss + l1_loss

            total_loss.backward()
            optimizer.step()

        final_c = self.forward(step=1.0).detach().numpy().flatten()
        sympy_elems = [sp.nsimplify(np.round(val, 2), rational=True) for val in final_c]
        return sp.Matrix(sympy_elems)


class RationalProjector:
    @staticmethod
    def simplify_vector(vec: np.ndarray) -> sp.Matrix:
        vec_flat = np.array(vec, dtype=float).flatten()
        vec_clean = np.round(vec_flat, 3)

        sympy_elems = [sp.nsimplify(val, rational=True) for val in vec_clean]

        denominators = [e.q for e in sympy_elems if isinstance(e, sp.Rational)]
        if denominators:
            lcm = np.lcm.reduce(denominators)
            if lcm <= 2:
                sympy_elems = [e * lcm for e in sympy_elems]

        return sp.Matrix(sympy_elems)


class SequentialPiOptimizer:
    def __init__(
        self,
        max_integer_search: int = 2,
        max_redundancy_dcor: float = 0.50,
        sparsity_penalty: float = 0.01,
        min_quality_score: float = 0.05
    ):
        self.max_int = max_integer_search
        self.max_redundancy = max_redundancy_dcor

    def optimize(
        self,
        log_X: np.ndarray,
        target_dimless: np.ndarray,
        nullspace_matrix: np.ndarray,
        c_anchor_vector: np.ndarray = None
    ) -> List[sp.Matrix]:
        N, n_features = log_X.shape
        k = nullspace_matrix.shape[1]

        if k == 0:
            return []

        covered_indices = set()
        if c_anchor_vector is not None:
            c_flat = np.array(c_anchor_vector, dtype=float).flatten()
            covered_indices.update(np.where(np.abs(c_flat) > 1e-3)[0])

        range_coeffs = list(range(-self.max_int, self.max_int + 1))
        raw_combinations = itertools.product(range_coeffs, repeat=k)

        valid_a_vectors = []
        seen_directions = set()

        for a_tuple in raw_combinations:
            if all(c == 0 for c in a_tuple):
                continue

            g = reduce(gcd, [abs(c) for c in a_tuple if c != 0])
            a_norm = tuple(c // g for c in a_tuple)

            if a_norm not in seen_directions:
                seen_directions.add(a_norm)
                valid_a_vectors.append(np.array(a_norm, dtype=float))

        candidates = []
        for a_vec in valid_a_vectors:
            power_vec = np.round(nullspace_matrix @ a_vec, 3)
            u_log = log_X @ power_vec
            pi_values = np.exp(u_log)

            active_indices = set(np.where(np.abs(power_vec) > 1e-3)[0])
            n_active_vars = len(active_indices)
            l1_norm = np.sum(np.abs(power_vec))

            candidates.append({
                "power_vec": power_vec,
                "u_log": u_log,
                "pi_values": pi_values,
                "active_indices": active_indices,
                "n_active_vars": n_active_vars,
                "l1_norm": l1_norm
            })

        selected_features = []
        selected_pi_data = []

        current_residual_log = np.log(np.where(target_dimless <= 0, 1e-9, target_dimless))

        all_features = set(range(n_features))

        while len(selected_features) < k:
            missing_features = all_features - covered_indices

            current_residual_vals = np.exp(current_residual_log)

            best_cand = None
            best_effective_score = -np.inf

            for cand in candidates:
                if any(np.allclose(cand["power_vec"], prev) for prev in selected_features):
                    continue

                is_redundant = False
                for prev_pi in selected_pi_data:
                    inter_dcor = compute_dcor(cand["pi_values"], prev_pi)
                    if inter_dcor > self.max_redundancy:
                        is_redundant = True
                        break

                if is_redundant:
                    continue

                res_dcor = compute_dcor(cand["pi_values"], current_residual_vals)

                purity_penalty = (cand["n_active_vars"] ** 2) * cand["l1_norm"]
                purity_score = res_dcor / purity_penalty

                if len(missing_features) > 0:
                    has_missing_var = bool(cand["active_indices"] & missing_features)
                    bonus = 10.0 if has_missing_var else 0.0
                else:
                    bonus = 0.0

                effective_score = purity_score + bonus

                if effective_score > best_effective_score:
                    best_effective_score = effective_score
                    best_cand = cand

            if best_cand is None:
                break

            selected_features.append(best_cand["power_vec"])
            selected_pi_data.append(best_cand["pi_values"])
            covered_indices.update(best_cand["active_indices"])

            poly_coeff = np.polyfit(best_cand["u_log"], current_residual_log, deg=2)
            trend = np.polyval(poly_coeff, best_cand["u_log"])
            current_residual_log = current_residual_log - trend

        sympy_results = []
        for vec in selected_features:
            sympy_matrix = RationalProjector.simplify_vector(vec)
            sympy_results.append(sympy_matrix)

        return sympy_results