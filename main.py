import sympy as sp
import numpy as np
import pandas as pd
import warnings

try:
    from linear_operator.utils.warnings import NumericalWarning
    warnings.simplefilter("ignore", NumericalWarning)
except ImportError:
    pass

from generate.generate_dataset import (
    load_4_tooth_ar_br_dr_Fr_1,
    generate_complex_gravity_relativity_dataset,
    generate_complex_thermo_wave_dataset,
    generate_test_1_trans_addsep,
    generate_test_2_scale_multsep,
    generate_test_3_dim_addition,
    generate_test_4_mul_trans,
    generate_test_5_genadd_scale,
    generate_test_6_gen_comp
)
from GP.config import GPConfig, ModelConfig, KernelConfig, TrainingConfig, ScalingConfig, LikelihoodConfig
from GP.physics_losses import PhysicsLossFactory
from symbolic.sm_context import SymbolicRegressionContext
from symbolic.pipeline_step import (
    GPHyperparameterTuningStep,
    BaselineGPStep,
    SymmetryPreprocessingStep,
    DimensionalAnalysisStep,
    GPSimplificationStep
)

import random
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Seed] Установлен единый генератор случайных чисел: seed={seed}")
set_seed(42)

def resolve_expression(expr: sp.Expr, mapping: dict) -> sp.Expr:
    """Подставляет символьные выражения из маппинга рекурсивно."""
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
    """Вычисляет MSE, RMSE, MAE, R², MRE (%) и MdRE (%)."""
    free_symbols = sorted(list(expr.free_symbols), key=lambda s: s.name)
    
    if not free_symbols:
        y_pred = np.full_like(y_true, float(expr))
    else:
        missing_cols = [s.name for s in free_symbols if s.name not in df.columns]
        if missing_cols:
            raise KeyError(f"Переменные {missing_cols} из выражения отсутствуют в датасете.")
            
        f_compiled = sp.lambdify(free_symbols, expr, 'numpy')
        args = [df[s.name].values for s in free_symbols]
        y_pred = f_compiled(*args)
        
        if np.isscalar(y_pred):
            y_pred = np.full_like(y_true, y_pred)

    valid_mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if not np.any(valid_mask):
        return {
            "mse": float('inf'), "rmse": float('inf'), "mae": float('inf'), 
            "r2": float('-inf'), "mre": float('inf'), "mdre": float('inf')
        }

    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]

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
        "mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mre": mre, "mdre": mdre
    }


def display_pipeline_results(final_context: SymbolicRegressionContext, initial_df: pd.DataFrame, original_target_name: str):
    """Выводит итоговые полученные формулы и статистику точности."""
    print("\n" + "="*60)
    print("ВЫВОД ИТОГОВЫХ РЕЗУЛЬТАТОВ ПАЙПЛАЙНА И ОЦЕНКА ТОЧНОСТИ")
    print("="*60)

    gp_formula_pi = final_context.target_expr
    print(f"\n1. Найдена безразмерная зависимость (в терминах Пи-групп):")
    print(f"   target_dimensionless = {gp_formula_pi}")

    print(f"\n2. Соотношения Пи-комплексов и исходных переменных:")
    for pi_name, original_expr in final_context.symbolic_mapping.items():
        print(f"   {pi_name} = {original_expr}")
    
    f_dimless_physical = sp.simplify(resolve_expression(gp_formula_pi, final_context.symbolic_mapping))

    print(f"\n3. Итоговое безразмерное уравнение через физические переменные:")
    print(f"   f_dimless = {f_dimless_physical}")

    print("\n" + "-"*60)
    print("4. ОЦЕНКА ТОЧНОСТИ НА ДАННЫХ:")
    print("-"*60)

    if "target" in final_context.df.columns:
        y_true_dimless = final_context.df["target"].values
        dimless_metrics = evaluate_accuracy(gp_formula_pi, final_context.df, y_true_dimless)
        
        print("\n[Безразмерная модель (target_dimensionless)]:")
        print(f"   MSE       : {dimless_metrics['mse']:.6e}")
        print(f"   RMSE      : {dimless_metrics['rmse']:.6e}")
        print(f"   MAE       : {dimless_metrics['mae']:.6e}")
        print(f"   R²        : {dimless_metrics['r2']:.6f}")
        print(f"   Средняя относительная ошибка (MRE) : {dimless_metrics['mre']:.4f}%")
        print(f"   Медианная отн. ошибка (Median RE)  : {dimless_metrics['mdre']:.4f}%")

    y_true_phys = initial_df[original_target_name].values
    anchor_values = y_true_phys / (final_context.df["target"].values + 1e-19)
    
    try:
        free_syms = sorted(list(f_dimless_physical.free_symbols), key=lambda s: s.name)
        f_compiled = sp.lambdify(free_syms, f_dimless_physical, 'numpy')
        args = [initial_df[s.name].values for s in free_syms]
        y_pred_dimless = f_compiled(*args)
        
        if np.isscalar(y_pred_dimless):
            y_pred_dimless = np.full_like(y_true_phys, y_pred_dimless)
            
        y_pred_phys = y_pred_dimless * anchor_values

        phys_metrics = evaluate_accuracy(
            sp.Symbol("target"), 
            pd.DataFrame({"target": y_pred_phys}), 
            y_true_phys
        )

        print(f"\n[Итоговая физическая модель ({original_target_name})]:")
        print(f"   MSE       : {phys_metrics['mse']:.6e}")
        print(f"   RMSE      : {phys_metrics['rmse']:.6e}")
        print(f"   MAE       : {phys_metrics['mae']:.6e}")
        print(f"   R²        : {phys_metrics['r2']:.6f}")
        print(f"   Средняя относительная ошибка (MRE) : {phys_metrics['mre']:.4f}%")
        print(f"   Медианная отн. ошибка (Median RE)  : {phys_metrics['mdre']:.4f}%")
    except Exception as e:
        print(f"\n[Предупреждение] Не удалось вычислить физическую точность: {e}")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":

    df, reg, target_name = generate_test_1_trans_addsep(n_samples=500, noise_std=0.0)
    # df, reg, target_name = generate_test_2_scale_multsep(n_samples=500, noise_std=0.0)
    # df, reg, target_name = generate_test_3_dim_addition(n_samples=500, noise_std=0.0)
    # df, reg, target_name = generate_test_4_mul_trans(n_samples=500, noise_std=0.0)
    # df, reg, target_name = generate_test_5_genadd_scale(n_samples=500, noise_std=0.0)
    # df, reg, target_name = generate_test_6_gen_comp(n_samples=500, noise_std=0.0)
    # df, reg, target_name = load_4_tooth_ar_br_dr_Fr_1()
    # df, reg, target_name = generate_complex_gravity_relativity_dataset(n_samples=400)
    # df, reg, target_name = generate_complex_thermo_wave_dataset(n_samples=400)

    # Инициализация контекста символьной регрессии
    context = SymbolicRegressionContext.create_initial_context(
        df=df,
        registry=reg,
        target_name=target_name
    )


    gp_config = GPConfig(
        scaling=ScalingConfig(
            enabled=True   
        ),
        model=ModelConfig(
            mean_type="constant",  
            kernel=KernelConfig(
                type="rq",   # "rbf", "matern_32", "matern_52", "rq", "periodic", "cosine", "spectral_mixture", "cauchy"
                scale_kernel=True,
                use_lengthscale_prior = True, 
                ard=True
            ),
            likelihood=LikelihoodConfig(extra_kwargs={})
        ),
        training=TrainingConfig(
            lr=0.1,
            epochs=61,
            early_stopping_patience=200,
            optimizer="lbfgs",
            loss_type="mll",
            verbose=True,
            cholesky_jitter=1e-2,
            fast_solves=False,
            fast_log_prob=False,
            loss_modifier=None     
        ),
        seed=42
    )

    pipeline = [
        # GPHyperparameterTuningStep(
        #     gp_config=gp_config,
        #     cache_path="gp_best_params.json",
        #     force_tune=False,      
        #     n_trials=40,           
        #     subsample_size=200,    
        #     gamma=0.002,           
        #     verbose=True,
        #     fixed_optimizer="adam",     
        #     fixed_kernel_type=None,     
        #     fixed_mean_type=None,       
        #     fixed_loss_type=None,       
        #     fixed_lr=None,              
        #     loss_modifier=None
        # ),

        BaselineGPStep(
            gp_config=gp_config,
            verbose=True
        ),

        # SymmetryPreprocessingStep(
        #     gp_config=gp_config,
        #     verbose=True,
        #     optimize_constants=False,       
        #     allowed_constants=[1.0, 2.0],  
        #     allowed_ops=["add", "sub", "mul", "div", "sin", "cos", "exp", "log"],
        #     active_simplifiers=[
        #         "translational",  # (x1 - x2)
        #         "addition",       # (x1 + x2)
        #         "largescale",     # (x1 / x2)
        #         "multiply",       # (x1 * x2)
        #         "generalized",    # h(x1, x2)
        #         "compositionality"
        #     ],
        #     k_best=50             
        # ),

        # DimensionalAnalysisStep(
        #     verbose=True
        # ),


        GPSimplificationStep(
            gp_config=gp_config,
            max_depth=7,                    
            local_bf_max_length=5,           
            final_bf_max_length=6,           
            base_k_sigma=3.0,               
            k_sigma_multipliers=[1.0, 2.0], 
            loss_degradation_tolerance=0.25, 
            verbose=True,
            optimize_constants=False,         
            allowed_constants=[1.0, 2.0],    
            allowed_ops=["add", "sub", "mul", "div", "pow", "sqrt", "log", "exp", "sin"], 
            k_best=50                        
        )
    ]

    print("\nЗапуск пайплайна символьной регрессии...")
    for step in pipeline:
        context = step.transform(context)

    display_pipeline_results(context, initial_df=df, original_target_name=target_name)