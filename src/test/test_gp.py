import torch
import matplotlib.pyplot as plt

from GP.config import GPConfig, ModelConfig, KernelConfig, TrainingConfig
from GP.pipeline import GPRegressionPipeline

# 1. Генерируем синтетические данные
torch.manual_seed(42)
train_x = torch.linspace(0, 10, 15)
# Обучающая функция: sin(x) + случайный шум
train_y = torch.sin(train_x) + torch.randn(train_x.size()) * 0.2

test_x = torch.linspace(0, 10, 50)
test_y_true = torch.sin(test_x) # Истинные значения для проверки

# 2. Создаем конфигурацию эксперимента
config = GPConfig(
    model=ModelConfig(
        mean_type="constant",
        kernel=KernelConfig(type="matern_52", scale_kernel=True, ard=False)
    ),
    training=TrainingConfig(
        lr=0.1,
        epochs=150,
        optimizer="adam",
        verbose=True
    )
)

# 3. Инициализируем и обучаем пайплайн
pipeline = GPRegressionPipeline(config)
print("--- Начало обучения ---")
loss_history = pipeline.fit(train_x, train_y)
print("--- Обучение завершено ---")

# 4. Делаем предсказание
predictions = pipeline.predict(test_x)
mean = predictions.mean
lower, upper = predictions.confidence_region()

# 5. Визуализируем результаты
plt.figure(figsize=(10, 6))
plt.plot(train_x.numpy(), train_y.numpy(), 'kx', label='Данные обучения')
plt.plot(test_x.numpy(), test_y_true.numpy(), 'g--', label='Истинная функция $f(x) = \sin(x)$')
plt.plot(test_x.numpy(), mean.numpy(), 'b', label='Предсказание GP (Mean)')
plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.2, color='blue', label='Доверительный интервал 95% ($2\sigma$)')
plt.legend()
plt.title("Тест работы шаблона Гауссовского процесса")
plt.grid(True)
plt.show()