import sympy as sp
import numpy as np
import pandas as pd
from symbolic.sm_context import SymbolicRegressionContext
from logger import setup_logger

logger = setup_logger("Evaluation")

def resolve_expression(expr: sp.Expr, mapping: dict) -> sp.Expr:
    curr_expr = expr
    for _ in range(10):  
        sub_dict = {}
        for sym in curr_expr.free_symbols:
            if sym.name in mapping:
                val = mapping[sym.name]
                sub_dict[sym] = val if isinstance(val, sp.Expr) else sp.sympify(val)
            elif str(sym) in mapping:
                val = mapping[str(sym)]
                sub_dict[sym] = val if isinstance(val, sp.Expr) else sp.sympify(val)
        if not sub_dict:
            break
        curr_expr = curr_expr.subs(sub_dict)
    return curr_expr


def evaluate_accuracy(expr: sp.Expr, df: pd.DataFrame, y_true: np.ndarray) -> dict:
    free_symbols = sorted(list(expr.free_symbols), key=lambda s: s.name)
    
    if not free_symbols:
        y_pred = np.full_like(y_true, float(expr))
    else:
        missing_cols = [s.name for s in free_symbols if s.name not in df.columns]
        if missing_cols:
            raise KeyError(f"Переменные {missing_cols} из выражения отсутствуют в датасете.")
            
        f_compiled = sp.lambdify(free_symbols, expr, 'numpy')
        args = [df[s.name].values for s in free_symbols]
        try:
            y_pred = f_compiled(*args)
            if np.isscalar(y_pred):
                y_pred = np.full_like(y_true, y_pred)
            if np.iscomplexobj(y_pred):
                y_pred = np.real(y_pred)
        except Exception:
            return {"mse": float('inf'), "rmse": float('inf'), "mae": float('inf'), 
                    "r2": float('-inf'), "mre": float('inf'), "mdre": float('inf'), "nan_pct": 100.0}

    invalid_count = np.sum(~np.isfinite(y_pred))
    total_count = len(y_true)
    nan_pct = (invalid_count / total_count) * 100.0

    if invalid_count > 0:
        return {
            "mse": float('inf'), "rmse": float('inf'), "mae": float('inf'), 
            "r2": float('-inf'), "mre": float('inf'), "mdre": float('inf'),
            "nan_pct": nan_pct
        }

    yt = y_true
    yp = y_pred

    denom = np.where(np.abs(yt) < 1e-9, 1e-9, yt)
    relative_errors = (np.abs(yt - yp) / np.abs(denom)) * 100.0

    mse = float(np.mean((yt - yp) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(yt - yp)))
    mre = float(np.mean(relative_errors))       
    mdre = float(np.median(relative_errors))    
    
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))

    return {
        "mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mre": mre, "mdre": mdre, "nan_pct": 0.0
    }

def display_pipeline_results(final_context: SymbolicRegressionContext, initial_df: pd.DataFrame, original_target_name: str):
    gp_formula_pi = final_context.target_expr
    f_dimless_physical = sp.simplify(resolve_expression(gp_formula_pi, final_context.symbolic_mapping))
    current_target = final_context.target_name

    y_true_working = final_context.df[current_target].values
    working_metrics = evaluate_accuracy(gp_formula_pi, final_context.df, y_true_working)

    y_true_phys = initial_df[original_target_name].values
    anchor_values = y_true_phys / (final_context.df[current_target].values + 1e-19)
    
    phys_metrics = None
    try:
        free_syms = sorted(list(f_dimless_physical.free_symbols), key=lambda s: s.name)
        f_compiled = sp.lambdify(free_syms, f_dimless_physical, 'numpy')
        args = [initial_df[s.name].values for s in free_syms]
        y_pred_working = f_compiled(*args)
        
        if np.isscalar(y_pred_working):
            y_pred_working = np.full_like(y_true_phys, y_pred_working)
            
        y_pred_phys = y_pred_working * anchor_values
        phys_metrics = evaluate_accuracy(
            sp.Symbol("target_dummy"), 
            pd.DataFrame({"target_dummy": y_pred_phys}), 
            y_true_phys
        )
    except Exception as e:
        logger.warning(f"Не удалось вычислить физическую точность: {e}")

    # Формируем итоговый красивый лог
    log_msg = "\n" + "="*60 + "\n"
    log_msg += "ВЫВОД ИТОГОВЫХ РЕЗУЛЬТАТОВ ПАЙПЛАЙНА\n"
    log_msg += "="*60 + "\n\n"
    
    log_msg += "1. Найдена базовая зависимость (в рабочем пространстве):\n"
    log_msg += f"   target_working = {gp_formula_pi}\n\n"

    log_msg += "2. Соотношения комплексов и исходных переменных:\n"
    for pi_name, original_expr in final_context.symbolic_mapping.items():
        log_msg += f"   {pi_name} = {original_expr}\n"
    
    log_msg += "\n3. Итоговое уравнение через физические переменные:\n"
    log_msg += f"   f_final = {f_dimless_physical}\n\n"

    log_msg += "-"*60 + "\n"
    log_msg += f"[Внутренняя модель рабочего пространства ({current_target})]:\n"
    log_msg += f"   MSE: {working_metrics['mse']:.6e} | R²: {working_metrics['r2']:.6f}\n"
    log_msg += f"   MRE: {working_metrics['mre']:.4f}%   | MdRE: {working_metrics['mdre']:.4f}%\n"

    if phys_metrics:
        log_msg += f"\n[Итоговая физическая модель ({original_target_name})]:\n"
        log_msg += f"   MSE: {phys_metrics['mse']:.6e} | R²: {phys_metrics['r2']:.6f}\n"
        log_msg += f"   MRE: {phys_metrics['mre']:.4f}%   | MdRE: {phys_metrics['mdre']:.4f}%\n"

    log_msg += "="*60
    logger.info(log_msg)