import numpy as np
import sympy as sp
import itertools


class DimensionalError(ValueError):
    """Исключение, выбрасываемое при нарушении законов физической размерности."""
    pass

class PhysicalDimension:
    def __init__(self, vector):
        self.vector = np.array(vector, dtype=float)

    @classmethod
    def dimensionless_like(cls, other):
        """Создает безразмерный вектор той же длины, что и у 'other'."""
        return cls(np.zeros_like(other.vector))

    def __add__(self, other):
        if not np.allclose(self.vector, other.vector):
            raise DimensionalError(
                f"Несовместимые размерности для сложения: {self} и {other}"
            )

        return PhysicalDimension(self.vector)

    def __sub__(self, other):
        if not np.allclose(self.vector, other.vector):
            raise DimensionalError(
                f"Несовместимые размерности для вычитания: {self} и {other}"
            )
        return PhysicalDimension(self.vector)

    def __mul__(self, other):
        return PhysicalDimension(self.vector + other.vector)

    def __truediv__(self, other):
        return PhysicalDimension(self.vector - other.vector)

    def __pow__(self, power):
        return PhysicalDimension(self.vector*power)

    def __eq__(self, other):
        return np.allclose(self.vector, other.vector)

    def is_dimensionless(self):
        return np.allclose(self.vector, 0)

    def __repr__(self):
        components = ", ".join([f"dim_{i}:{val:.0f}" for i, val in enumerate(self.vector)])
        return f"Dim[{components}]"

def sqrt(dim: PhysicalDimension) -> PhysicalDimension:
    return dim ** 0.5

def sin(dim: PhysicalDimension) -> PhysicalDimension:
    if not dim.is_dimensionless():
        raise DimensionalError(f"Аргумент sin должен быть безразмерным, а не {dim}")
    return PhysicalDimension.dimensionless_like(dim)

def cos(dim: PhysicalDimension) -> PhysicalDimension:
    if not dim.is_dimensionless():
        raise DimensionalError(f"Аргумент cos должен быть безразмерным, а не {dim}")
    return PhysicalDimension.dimensionless_like(dim)

def exp(dim: PhysicalDimension) -> PhysicalDimension:
    if not dim.is_dimensionless():
        raise DimensionalError(f"Аргумент exp должен быть безразмерным, а не {dim}")
    return PhysicalDimension.dimensionless_like(dim)

class PhysicalVariable:
    def __init__(self, name_col: str, physicalDimension: PhysicalDimension):
        self.name = name_col
        self.dimension = physicalDimension

    def __repr__(self):
        return f"{self.name} - {self.dimension}"

class PhysicalRegistry():
    def __init__(self):
        self.variables = {}

    def register(self, name: str, vector: list):
        dim = PhysicalDimension(vector=vector)
        self.variables[name] = PhysicalVariable(name, dim)

    def get_dim(self, name: str) -> PhysicalDimension:
        """Быстрый поиск размерности по имени колонки"""
        return self.variables[name].dimension

    def get_all_variables(self) -> list[str]:
        """Возвращает список всех зарегистрированных колонок"""
        return list(self.variables.keys())

    def find_columns_by_dim(self, target_dim: PhysicalDimension) -> list[str]:
        """Находит все колонки в датасете, у которых размерность совпадает с target_dim"""
        return [
            var.name for var in self.variables.values()
            if var.dimension == target_dim
        ]

    def build_matrix_numpy(self, names):
        columns = [self.get_dim(name).vector for name in names]
        return np.column_stack(columns)

    def build_matrix_sympy(self, names: list[str]) -> sp.Matrix:
        columns = [sp.Matrix(self.get_dim(name).vector) for name in names]
        return sp.Matrix.hstack(*columns)

    def vector_to_formula(self, vector: sp.Matrix, names: list[str]) -> sp.Expr:
        expr = sp.Integer(1)
        for name, power in zip(names, vector):
            if power != 0:
                clean_power = sp.nsimplify(power)
                expr *= sp.Symbol(name) ** clean_power
        return expr
    
    def format_expr_inline(self, expr: sp.Expr) -> str:
        """Форматирует выражение SymPy в красивую однострочную математическую строку"""
        s = str(expr)

        s = s.replace("**", "^")
        s = s.replace("*", " · ")
        
        return s

    def display_solutions(self, target_name: str, c_particular: sp.Matrix, nullspace_vectors: list[sp.Matrix]):
        names = [name for name in self.get_all_variables() if name != target_name]
        
        print("=== Результаты анализа размерностей ===")
        anchor_expr = self.vector_to_formula(c_particular, names)
        
        anchor_str = self.format_expr_inline(anchor_expr)
        print(f"\nРазмерный якорь: [{target_name}]_anchor = {anchor_str}")

        if nullspace_vectors:
            print("\nБезразмерные комплексы (Пи-группы):")
            for i, vec in enumerate(nullspace_vectors, start=1):
                pi_expr = self.vector_to_formula(vec, names)
                pi_str = self.format_expr_inline(pi_expr)
                print(f"  Pi_{i} = {pi_str}")
        else:
            print("\nБезразмерные комплексы не найдены.")

    def transform_dataset(self, data: dict, target_name: str, 
                          c_particular: sp.Matrix, nullspace_vectors: list[sp.Matrix]) -> dict:
        
        names = [name for name in self.get_all_variables() if name != target_name]
        new_dataset = {}

        def evaluate_vector(vector):

            first_col = np.array(data[names[0]], dtype=float)
            result = np.ones_like(first_col)

            for name, power in zip(names, vector):
                if power != 0:
                    p = float(power)
                    result *= np.array(data[name], dtype=float) ** p
            return result
        
        for i, vec in enumerate(nullspace_vectors, start=1):
            new_dataset[f"Pi_{i}"] = evaluate_vector(vec)

        anchor_values = evaluate_vector(c_particular)
        target_values = np.array(data[target_name], dtype=float)
        new_target_name = f"{target_name}_dimensionless"

        new_dataset[new_target_name] = np.divide(target_values, anchor_values)

        return new_dataset

    def find_all_basic_solutions_sympy(self, target_name: str):
        names = [name for name in self.get_all_variables() if name != target_name]
        n = len(names)
        a_Q = sp.Matrix(self.get_dim(target_name).vector)
        D = self.build_matrix_sympy(names)

        augmented_matrix = D.row_join(a_Q)

        rref_matrix, pivots = augmented_matrix.rref()

        if n in pivots:
            print(f"Размерность '{target_name}' не может быть составлена из базовых величин.")
            return None, []
        
        print("Размерность A:", D.shape)
        print("Ведущие столбцы (pivots):", pivots)
        print("RREF матрица:\n", rref_matrix)
        
        valid_pivots = [p for p in pivots if p < n]
        rank = len(valid_pivots)

        free_vars = [j for j in range(n) if j not in valid_pivots]

        c_particular = sp.Matrix.zeros(n, 1)
        for i, p_col in enumerate(valid_pivots):
            c_particular[p_col] = rref_matrix[i, n]
        
        nullspace_vectors = []
        for free_idx in free_vars:
            v = sp.Matrix.zeros(n, 1)
            v[free_idx] = 1 

            for i, p_col in enumerate(valid_pivots):
                v[p_col] = -rref_matrix[i, free_idx]
                
            nullspace_vectors.append(v)
            
        return c_particular, nullspace_vectors
