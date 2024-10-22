from data import X, X_interactions, Y, Y_mean
import numpy as np
from variance import calculate_variances
from cochran_test import cochran_test
from regression import calculate_coefficients
from student_test import student_test
from fisher_test import adequacy_dispersion, fisher_test
from print_utils import print_experiment_table, print_extended_experiment_table, format_regression_results, format_fisher_test_results
from scipy import stats  # Для расчёта критического значения Фишера

# Шаг 0: Создание новой матрицы X с учетом взаимодействий
# Первая модель: без взаимодействий, вторая - с взаимодействиями

# Матрица для первой модели (без взаимодействий)
X_first = X
# Матрица для второй модели (с взаимодействиями)
X_full = np.hstack([X, X_interactions])

# Шаг 1: Вычисление дисперсий
variances, s0_sq = calculate_variances(Y)

# Шаг 2: Проверка критерия Кохрена (одинаково для обеих моделей)
Gp, Gt = cochran_test(variances, Y)

# Шаг 3: Вычисление коэффициентов регрессии
# Первая модель
coefficients_first = calculate_coefficients(X_first, Y_mean)
# Вторая модель (с взаимодействиями)
coefficients_full = calculate_coefficients(X_full, Y_mean)

# Шаг 4: Проверка значимости по критерию Стьюдента
S_bi_first = s0_sq ** 0.5
N = len(X_first)  # Количество опытов
m = len(Y[0])  # Количество повторных измерений
t_values_first = student_test(coefficients_first, S_bi_first, N, m)
t_values_full = student_test(coefficients_full, S_bi_first, N, m)

# Шаг 5: Проверка по критерию Фишера

# Для первой модели
Y_pred_first = X_first @ coefficients_first  # Предсказанные значения Y
S_ad_first = adequacy_dispersion(Y_mean, Y_pred_first, m, N, len(coefficients_first))  # Дисперсия адекватности
Fp_first = fisher_test(S_ad_first, s0_sq)  # Рассчитанный критерий Фишера

# Для второй модели
Y_pred_full = X_full @ coefficients_full  # Предсказанные значения Y
S_ad_full = adequacy_dispersion(Y_mean, Y_pred_full, m, N, len(coefficients_full))  # Дисперсия адекватности
Fp_full = fisher_test(S_ad_full, s0_sq)  # Рассчитанный критерий Фишера

# Рассчёт критического значения Фишера Ft (одно и то же для обеих моделей)
f1 = N - len(coefficients_first)  # Степени свободы числителя (N - k)
f2 = N * (m - 1)  # Степени свободы знаменателя
Ft = stats.f.ppf(1 - 0.05, f1, f2)  # Критическое значение Фишера на уровне значимости 0.05

# Вывод таблиц
print_experiment_table(X_full, Y, Y_mean, variances)
print_extended_experiment_table(X_full, Y_mean)
# Вывод всех результатов
print("\n--- Шаг 1: Построчные дисперсии и общая дисперсия ---")
print(f"Построчные дисперсии si^2: {variances}")
print(f"Общая дисперсия s0^2: {s0_sq:.4f}")

print("\n--- Шаг 2: Проверка однородности дисперсий по критерию Кохрена ---")
print(f"Рассчитанный критерий Кохрена: Gp = {Gp:.4f}")
print(f"Табличное значение критерия Кохрена: Gt = {Gt:.4f}")
if Gp < Gt:
    print("Результат: Дисперсии однородны.")
else:
    print("Результат: Дисперсии неоднородны.")

# Вывод результатов для первой модели (без взаимодействий)
format_regression_results(coefficients_first, t_values_first, "Первая модель (без взаимодействий)")
format_fisher_test_results(Fp_first, Ft, S_ad_first, "Первая модель (без взаимодействий)")

# Вывод результатов для второй модели (с взаимодействиями)
format_regression_results(coefficients_full, t_values_full, "Вторая модель (с взаимодействиями)")
format_fisher_test_results(Fp_full, Ft, S_ad_full, "Вторая модель (с взаимодействиями)")