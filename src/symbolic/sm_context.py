import copy
from typing import Dict, List, Tuple
import pandas as pd
import sympy as sp
from get_pi_complex import PhysicalDimension, PhysicalRegistry, DimensionalError

class SymbolicRegressionContext:
    def __init__(
            self,
            df: pd.DataFrame,
            registry,
            target_name: str,
            symbolic_mapping: Dict[str, sp.Expr] = None,
            target_expr: sp.Expr = None
    ):
        self.df = df.copy()
        self.registry = copy.deepcopy(registry)
        self.target_name = target_name

        if symbolic_mapping is None:
            self.symbolic_mapping = {
                col: sp.Symbol(col)
                for col in self.df.columns
                if col != self.target_name
            }
        else:
            self.symbolic_mapping = symbolic_mapping.copy()

        if target_expr is None:
            self.target_expr = sp.Symbol(self.target_name)
        else:
            self.target_expr = target_expr

        self._validate_consistency()

    @classmethod
    def create_initial_context(
        cls, 
        df: pd.DataFrame, 
        registry, 
        target_name: str
    ) -> 'SymbolicRegressionContext':
        return cls(df=df, registry=registry, target_name=target_name)
    
    def _validate_consistency(self):
        df_cols = set(self.df.columns)

        if self.target_name not in df_cols:
            raise ValueError(
                f"Целевая переменная '{self.target_name}' отсутствует в DataFrame."
            )
        
        active_features = df_cols - {self.target_name}
        mapping_features = set(self.symbolic_mapping.keys())

        if active_features != mapping_features:
            raise ValueError(
                f"Несоответствие между признаками в DataFrame и в символьном маппинге.\n"
                f"Признаки DF: {active_features}\n"
                f"Признаки маппинга: {mapping_features}"
            )
        
    def register_mutation(
        self, 
        old_cols: List[str], 
        new_col_name: str, 
        h_expr: sp.Expr, 
        new_dim
    ):
        sub_dict = {col: self.symbolic_mapping[col] for col in old_cols}
        resolved_expr = h_expr.subs(sub_dict)

        for col in old_cols:
            self.symbolic_mapping.pop(col)
            if col in self.registry.variables:
                self.registry.variables.pop(col)

        self.symbolic_mapping[new_col_name] = resolved_expr  
        self.registry.register(new_col_name, new_dim.vector)

    def get_original_expression(self, col_name: str) -> sp.Expr:
        if col_name == self.target_name:
            return self.target_expr
        if col_name not in self.symbolic_mapping:
            raise KeyError(f"Переменная '{col_name}' не найдена в символьном маппинге.")
        return self.symbolic_mapping[col_name]
    

    def copy(self) -> 'SymbolicRegressionContext':
        return SymbolicRegressionContext(
            df=self.df,
            registry=self.registry,
            target_name=self.target_name,
            symbolic_mapping=self.symbolic_mapping,
            target_expr=self.target_expr
        )
    
    def get_active_features(self) -> List[str]:
        return [col for col in self.df.columns if col != self.target_name]
    
    def __repr__(self) -> str:
        features = self.get_active_features()
        mapping_str = "\n".join(
            f"  {col} -> {self.symbolic_mapping[col]}" for col in features
        )
        return (
            f"SymbolicRegressionContext(\n"
            f"  Текущих признаков: {len(features)} {features},\n"
            f"  Целевая переменная: '{self.target_name}',\n"
            f"  Символьные связи:\n{mapping_str}\n"
            f")"
        )

class DimensionalityEvaluator:

    @staticmethod
    def evaluate(expr: sp.Expr, registry: 'PhysicalRegistry') -> 'PhysicalDimension':
        if not registry.variables:
            raise ValueError("Реестр переменных пуст. Невозможно определить базис размерностей.")
        sample_dim = list(registry.variables.values())[0].dimension

        if isinstance(expr, sp.Symbol):
            if expr.name in registry.variables:
                return registry.get_dim(expr.name)
            raise DimensionalError(
                f"Переменная '{expr.name}' отсутствует в физическом реестре. "
                f"Невозможно определить её размерность."
            )
        
        if expr.is_Number or isinstance(expr, (sp.NumberSymbol, sp.Integer, sp.Float, sp.Rational)):
            return PhysicalDimension.dimensionless_like(sample_dim)
        
        if isinstance(expr, sp.Add):
            args = expr.args
            
            ref_dim = None
            for arg in args:
                dim_next = DimensionalityEvaluator.evaluate(arg, registry)
                
                if ref_dim is None:
                    ref_dim = dim_next
                else:
                    try:
                        ref_dim = ref_dim + dim_next
                    except DimensionalError as e:
                        raise DimensionalError(
                            f"Ошибка при сложении в выражении [{expr}]:\n"
                            f"  Конфликт размерностей: {ref_dim} и {dim_next}."
                        ) from e
                        
            if ref_dim is None:
                return PhysicalDimension.dimensionless_like(sample_dim)
                
            return ref_dim
        if isinstance(expr, sp.Mul):
            args = expr.args
            current_dim = DimensionalityEvaluator.evaluate(args[0], registry)

            for arg in args[1:]:
                dim_next = DimensionalityEvaluator.evaluate(arg, registry)
                current_dim = current_dim * dim_next

            return current_dim
        
        if isinstance(expr, sp.Pow):
            base, exponent = expr.args
            dim_base = DimensionalityEvaluator.evaluate(base, registry)
            dim_exponent = DimensionalityEvaluator.evaluate(exponent, registry)

            if not dim_exponent.is_dimensionless():
                raise DimensionalError(
                    f"Нарушение размерности в [{expr}]: "
                    f"показатель степени '{exponent}' должен быть безразмерным, но имеет размерность {dim_exponent}."
                )
            
            if exponent.is_Number:
                power = float(exponent)
                return dim_base ** power
            else:
                if not dim_base.is_dimensionless():
                    raise DimensionalError(
                        f"Нарушение размерности в [{expr}]: "
                        f"база '{base}' имеет размерность {dim_base}, но возводится в переменную степень '{exponent}'. "
                        f"База обязана быть безразмерной для переменных показателей степеней."
                    )
                return dim_base
            
        if isinstance(expr, sp.Function):
            func_name = expr.func.__name__
            
            if len(expr.args) != 1:
                raise NotImplementedError(
                    f"Функция '{func_name}' имеет более одного аргумента. "
                    f"Оценка многоаргументных функций не реализована."
                )
                
            arg = expr.args[0]
            dim_arg = DimensionalityEvaluator.evaluate(arg, registry)
            
            if not dim_arg.is_dimensionless():
                raise DimensionalError(
                    f"Нарушение размерности в [{expr}]: "
                    f"аргумент функции {func_name}({arg}) должен быть безразмерным, но имеет размерность {dim_arg}."
                )
                
            return PhysicalDimension.dimensionless_like(sample_dim)
        
        raise NotImplementedError(
            f"Оценка размерности для типа узла {type(expr)} ({expr}) пока не поддерживается."
        )
