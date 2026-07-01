from get_pi_complex import (PhysicalRegistry,)
from itertools import combinations
import numpy as np
import pandas as pd

def generate_center_mass_formula(n_samples: int)-> pd.DataFrame:
    """Центр масс двух тел: R = (m1*r1 + m2*r2) / (m1 + m2)"""
    reg = PhysicalRegistry()
    reg.register("R",  [0, 1, 0])  # Таргет (Длина)
    reg.register("m1", [1, 0, 0])  # Масса 1
    reg.register("r1", [0, 1, 0])  # Координата 1
    reg.register("m2", [1, 0, 0])  # Масса 2
    reg.register("r2", [0, 1, 0])  # Координата 2

    m1 = np.random.uniform(1.0, 10.0, n_samples)
    m2 = np.random.uniform(1.0, 10.0, n_samples)
    r1 = np.random.uniform(0.5, 5.0, n_samples)
    r2 = np.random.uniform(0.5, 5.0, n_samples)

    R = (m1*r1 + m2*r2) / (m1 + m2)

    raw_data = {"R": R, "m1": m1, "r1": r1, "m2": m2, "r2": r2}
    
    c_part, nullspace = reg.find_all_basic_solutions_sympy("R")
    reg.display_solutions("R", c_part, nullspace)
    
    df_raw = pd.DataFrame(reg.transform_dataset(raw_data, "R", c_part, nullspace))
    
    df_raw.rename(columns={"R_dimensionless": "target"}, inplace=True)
    
    return df_raw, reg, "R", c_part, nullspace

def generate_relativistic_momentum_formula(n_samples: int) -> pd.DataFrame:
    """Релятивистский импульс: p = m0 * v / sqrt(1 - v^2 / c^2)"""
    reg = PhysicalRegistry()
    reg.register("p",  [1, 1, -1]) # Таргет (Импульс)
    reg.register("m0", [1, 0, 0])  # Масса покоя
    reg.register("v",  [0, 1, -1]) # Скорость частицы
    reg.register("c",  [0, 1, -1]) # Скорость света

    m0 = np.random.uniform(0.1, 5.0, n_samples)
    c = np.random.uniform(3.0, 5.0, n_samples) 
    v = np.random.uniform(0.1, 0.9, n_samples) * c 
    
    p = (m0 * v) / np.sqrt(1.0 - (v**2 / c**2))

    raw_data = {"p": p, "m0": m0, "v": v, "c": c}
    
    c_part, nullspace = reg.find_all_basic_solutions_sympy("p")
    reg.display_solutions("p", c_part, nullspace)

    df_raw = pd.DataFrame(reg.transform_dataset(raw_data, "p", c_part, nullspace))
    
    df_raw.rename(columns={"p_dimensionless": "target"}, inplace=True)

    return df_raw, reg, "p", c_part, nullspace

def generate_gravitational_attraction_in_2D_formula(n_samples: int) -> pd.DataFrame:
    """Сила тяжести в 2D: F = G * m1 * m2 / ((x2-x1)^2 + (y2-y1)^2)"""
    reg = PhysicalRegistry()
    reg.register("F",  [1, 1, -2])  # Таргет (Сила)
    reg.register("G",  [-1, 3, -2]) # Гравитационная постоянная
    reg.register("m1", [1, 0, 0])   # Масса 1
    reg.register("m2", [1, 0, 0])   # Масса 2
    reg.register("x1", [0, 1, 0])   # Координаты
    reg.register("x2", [0, 1, 0])
    reg.register("y1", [0, 1, 0])
    reg.register("y2", [0, 1, 0])

    G = np.random.uniform(0.5, 2.0, n_samples)
    m1 = np.random.uniform(1.0, 10.0, n_samples)
    m2 = np.random.uniform(1.0, 10.0, n_samples)
    x1 = np.random.uniform(-5.0, -1.0, n_samples)
    x2 = np.random.uniform(1.0, 6.0, n_samples)
    y1 = np.random.uniform(-5.0, -1.0, n_samples)
    y2 = np.random.uniform(1.0, 6.0, n_samples)
    
    r_squared = (x2 - x1)**2 + (y2 - y1)**2
    F = (G * m1 * m2) / r_squared

    raw_data = {"F": F, "G": G, "m1": m1, "m2": m2, "x1": x1, "x2": x2, "y1": y1, "y2": y2}
    
    c_part, nullspace = reg.find_all_basic_solutions_sympy("F")
    reg.display_solutions("F", c_part, nullspace)

    df_raw = pd.DataFrame(reg.transform_dataset(raw_data, "F", c_part, nullspace))
    
    df_raw.rename(columns={"F_dimensionless": "target"}, inplace=True)

    return df_raw, reg, "F", c_part, nullspace

def generate_harmonic_oscillator_energy_formula(n_samples: int) -> pd.DataFrame:
    """Средняя-Сложная: Энергия осциллятора: E = 0.5 * m * (w^2 + w0^2) * x^2"""
    reg = PhysicalRegistry()
    reg.register("E",      [1, 2, -2]) # Таргет (Энергия)
    reg.register("m",      [1, 0, 0])  # Масса
    reg.register("omega",  [0, 0, -1]) # Вынужденная частота
    reg.register("omega0", [0, 0, -1]) # Собственная частота
    reg.register("x",      [0, 1, 0])  # Смещение

    m = np.random.uniform(0.5, 5.0, n_samples)
    omega = np.random.uniform(1.0, 10.0, n_samples)
    omega0 = np.random.uniform(1.0, 10.0, n_samples)
    x = np.random.uniform(0.1, 3.0, n_samples)
    
    E = 0.5 * m * (omega**2 + omega0**2) * (x**2)

    raw_data = {"E": E, "m": m, "omega": omega, "omega0": omega0, "x": x}
    
    c_part, nullspace = reg.find_all_basic_solutions_sympy("E")
    reg.display_solutions("E", c_part, nullspace)

    df_raw = pd.DataFrame(reg.transform_dataset(raw_data, "E", c_part, nullspace))
    df_raw.rename(columns={"E_dimensionless": "target"}, inplace=True)
    return df_raw, reg, "E", c_part, nullspace

def generate_boltzmann_density_formula(n_samples: int) -> pd.DataFrame:
    """Сложная: Плотность газа Больцмана: n = n0 * exp(- m * g * x / E_th)"""
    reg = PhysicalRegistry()
    reg.register("n",    [0, -3, 0]) # Таргет (Концентрация)
    reg.register("n0",   [0, -3, 0]) # Начальная концентрация
    reg.register("m",    [1, 0, 0])  # Масса молекулы
    reg.register("g",    [0, 1, -2]) # Ускорение силы тяжести
    reg.register("x",    [0, 1, 0])  # Высота
    reg.register("Eth",  [1, 2, -2]) # Тепловая энергия (kT)

    n0 = np.random.uniform(10.0, 100.0, n_samples)
    
    m = np.random.uniform(1.0, 2.0, n_samples)
    g = np.random.uniform(9.8, 10.0, n_samples)
    x = np.random.uniform(0.1, 2.0, n_samples)
    Eth = np.random.uniform(10.0, 30.0, n_samples) 
    
    n = n0 * np.exp(- (m * g * x) / Eth)

    raw_data = {"n": n, "n0": n0, "m": m, "g": g, "x": x, "Eth": Eth}
    
    c_part, nullspace = reg.find_all_basic_solutions_sympy("n")
    reg.display_solutions("n", c_part, nullspace)

    df_raw = pd.DataFrame(reg.transform_dataset(raw_data, "n", c_part, nullspace))
    df_raw.rename(columns={"n_dimensionless": "target"}, inplace=True)
    return df_raw, reg, "n", c_part, nullspace

def generate_boltzmann_density_formula_with_noise(n_samples: int, noise_level: float = 0.02) -> pd.DataFrame:
    """Сложная: Плотность газа Больцмана: n = n0 * exp(- m * g * x / E_th)"""
    reg = PhysicalRegistry() 
    reg.register("n", [0, -3, 0])     # Таргет (Концентрация)
    reg.register("n0", [0, -3, 0])    # Начальная концентрация 
    reg.register("m", [1, 0, 0])      # Масса молекулы 
    reg.register("g", [0, 1, -2])     # Ускорение силы тяжести 
    reg.register("x", [0, 1, 0])      # Высота 
    reg.register("Eth", [1, 2, -2])   # Тепловая энергия (kT)

    n0 = np.random.uniform(10.0, 100.0, n_samples)

    m = np.random.uniform(1.0, 2.0, n_samples)
    g = np.random.uniform(9.8, 10.0, n_samples)
    x = np.random.uniform(0.1, 2.0, n_samples)
    Eth = np.random.uniform(10.0, 30.0, n_samples) 

    n = n0 * np.exp(- (m * g * x) / Eth)

    if noise_level > 0:
        noise = np.random.normal(0, noise_level, n_samples)
        n = n * (1 + noise)
        n = np.clip(n, 1e-9, None)

    raw_data = {"n": n, "n0": n0, "m": m, "g": g, "x": x, "Eth": Eth}

    c_part, nullspace = reg.find_all_basic_solutions_sympy("n")
    reg.display_solutions("n", c_part, nullspace)

    df_raw = pd.DataFrame(reg.transform_dataset(raw_data, "n", c_part, nullspace))
    df_raw.rename(columns={"n_dimensionless": "target"}, inplace=True)
    return df_raw, reg, "n", c_part, nullspace

if __name__ == "__main__":
    print("\n--- ГЕНЕРАЦИЯ ДАТАСЕТОВ ДЛЯ ФИЗИЧЕСКИХ ЭКСПЕРИМЕНТОВ ---\n")
    
    df_1 = generate_center_mass_formula(500)
    print(f"1. Центр масс. Колонки: {df_1.columns.tolist()}\n")

    df_2 = generate_relativistic_momentum_formula(500)
    print(f"2. Релятивистский импульс. Колонки: {df_2.columns.tolist()}\n")

    df_3 = generate_harmonic_oscillator_energy_formula(500)
    print(f"3. Энергия осциллятора. Колонки: {df_3.columns.tolist()}\n")

    df_4 = generate_boltzmann_density_formula(500)
    print(f"4. Плотность Больцмана. Колонки: {df_4.columns.tolist()}\n")

    df_5 = generate_gravitational_attraction_in_2D_formula(500)
    print(f"5. Сила тяжести 2D. Колонки: {df_5.columns.tolist()}\n")