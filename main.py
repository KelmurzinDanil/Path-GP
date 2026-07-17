from generate.generate_dataset import *
from GP.config import GPConfig, ModelConfig, KernelConfig, TrainingConfig
from GP.pipeline import GPRegressionPipeline
from symbolic.sm_context import SymbolicRegressionContext, DimensionalityEvaluator
from symbolic.pipeline_step import DimensionalAnalysisStep, GPSimplificationStep, SymmetryPreprocessingStep
from get_pi_complex import PhysicalRegistry

import sympy as sp
import numpy as np
import pandas as pd


import warnings
try:
    from linear_operator.utils.warnings import NumericalWarning
    warnings.simplefilter("ignore", NumericalWarning)
except ImportError:
    pass

def display_pipeline_results(final_context: SymbolicRegressionContext, original_target_name: str):
    print("\n" + "="*60)
    print("ВЫВОД ИТОГОВЫХ РЕЗУЛЬТАТОВ ПАЙПЛАЙНА")
    print("="*60)

    gp_formula = final_context.target_expr
    print(f"\n1. Безразмерная зависимость (в терминах Пи-групп):")
    print(f"   target_dimensionless = {gp_formula}")

    print(f"\n2. Раскрытие Пи-комплексов через исходные физические переменные:")
    for pi_name, original_expr in final_context.symbolic_mapping.items():
        print(f"   {pi_name} = {original_expr}")
    
    resolved_inner_formula = gp_formula.subs(final_context.symbolic_mapping)
    original_target_symbol = sp.Symbol(original_target_name)

    print(f"\n3. Итоговое физическое уравнение:")
    full_physics_expr = sp.simplify(resolved_inner_formula)
    print(f"   {original_target_name} = {full_physics_expr} · [Размерный Якорь]")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    print("Генерация физических данных...")
    n_samples = 500
    
    G_vals = np.random.uniform(0.5, 2.0, n_samples)
    m1_vals = np.random.uniform(1.0, 10.0, n_samples)
    m2_vals = np.random.uniform(1.0, 10.0, n_samples)
    x1_vals = np.random.uniform(-5.0, -1.0, n_samples)
    x2_vals = np.random.uniform(1.0, 6.0, n_samples)
    y1_vals = np.random.uniform(-5.0, -1.0, n_samples)
    y2_vals = np.random.uniform(1.0, 6.0, n_samples)
    
    r_squared = (x2_vals - x1_vals)**2 + (y2_vals - y1_vals)**2
    F_vals = (G_vals * m1_vals * m2_vals) / r_squared

    raw_data = {
        "F": F_vals, 
        "G": G_vals, 
        "m1": m1_vals, 
        "m2": m2_vals, 
        "x1": x1_vals, 
        "x2": x2_vals, 
        "y1": y1_vals, 
        "y2": y2_vals
    }
    df_physical = pd.DataFrame(raw_data)

    original_registry = PhysicalRegistry()
    original_registry.register("F",  [1, 1, -2])   # Сила
    original_registry.register("G",  [-1, 3, -2])  # Гравитационная постоянная
    original_registry.register("m1", [1, 0, 0])    # Масса 1
    original_registry.register("m2", [1, 0, 0])    # Масса 2
    original_registry.register("x1", [0, 1, 0])    # Координаты
    original_registry.register("x2", [0, 1, 0])
    original_registry.register("y1", [0, 1, 0])
    original_registry.register("y2", [0, 1, 0])

    context = SymbolicRegressionContext.create_initial_context(
        df=df_physical,
        registry=original_registry,
        target_name="F"
    )

    gp_config = GPConfig(
        model=ModelConfig(
            mean_type="constant",
            kernel=KernelConfig(
                type="rbf",    
                scale_kernel=True,   
                ard=True             
            )
        ),
        training=TrainingConfig(
            lr=0.05,                 
            epochs=2000,  
            early_stopping_patience=15,          
            optimizer="lbfgs",
            loss_type="mll",
            verbose=False     
        ),
    )

    pipeline = [
        SymmetryPreprocessingStep(
            gp_config=gp_config, 
            verbose=True,
            optimize_constants=False,       
            allowed_constants=[1.0, 2.0],    
            allowed_ops=["add", "sub", "mul", "div"], 
            active_simplifiers=["translational", "addition", "largescale", "multiply", "generalized"]
        ),
        
        DimensionalAnalysisStep(verbose=True),
        
        GPSimplificationStep(
            gp_config=gp_config, 
            verbose=True,
            optimize_constants=False,
            allowed_constants=[1.0, 2.0],
            allowed_ops=["add", "sub", "mul", "div", "pow"]
        )
    ]

    print("\nЗапуск пайплайна обработки...")
    for step in pipeline:
        context = step.transform(context)

    display_pipeline_results(context, original_target_name="F")