from tabulate import tabulate
from scipy import stats

def sign(x):
    if x > 0:
        return "+"
    elif x < 0:
        return "-"
    else:
        return "0"

def print_experiment_table(X, Y, Y_mean, variances):
    """
    Выводит таблицу экспериментов с факторами, результатами экспериментов и дисперсиями.

    Parameters:
    X (list of lists): Матрица факторов (независимых переменных).
    Y (list of lists): Матрица результатов экспериментов (зависимые переменные).
    Y_mean (list): Вектор средних значений откликов.
    variances (list): Вектор дисперсий для каждого опыта.

    Returns:
    None
    """
    table_data = []
    for i in range(len(X)):
        row = [
            i + 1,            # Номер опыта
            sign(X[i][1]),    # x1
            sign(X[i][2]),    # x2
            sign(X[i][3]),    # x3
            Y[i][0],          # y(1)
            Y[i][1],          # y(2)
            Y_mean[i],        # Среднее Y
            variances[i]      # Дисперсия
        ]
        table_data.append(row)
    
    headers = ["№ опыта", "x1", "x2", "x3", "y(1)", "y(2)", "ӯ (среднее Y)", "si^2 (дисперсия)"]
    
    print("\n--- Таблица экспериментов ---")
    print(tabulate(table_data, headers, tablefmt="fancy_grid", floatfmt=".4f"))

def print_extended_experiment_table(X, Y_mean):
    """
    Выводит расширенную матрицу планирования с взаимодействиями факторов.

    Parameters:
    X (list of lists): Матрица факторов (независимых переменных).
    Y_mean (list): Вектор средних значений откликов.

    Returns:
    None
    """
    table_data = []
    for i in range(len(X)):
        row = [
            i + 1,                         # Номер опыта
            sign(1),                       # x0 (константный столбец, обычно равен 1)
            sign(X[i][1]),                 # x1
            sign(X[i][2]),                 # x2
            sign(X[i][3]),                 # x3
            sign(X[i][1] * X[i][2]),       # x1 * x2
            sign(X[i][1] * X[i][3]),       # x1 * x3
            sign(X[i][2] * X[i][3]),       # x2 * x3
            sign(X[i][1] * X[i][2] * X[i][3]),  # x1 * x2 * x3
            Y_mean[i]                      # Среднее Y
        ]
        table_data.append(row)
    
    headers = ["№", "х0", "х1", "х2", "х3", "х1х2", "х1х3", "х2х3", "х1х2х3", "y"]
    
    print("\n--- Расширенная матрица планирования ---")
    print(tabulate(table_data, headers, tablefmt="fancy_grid", floatfmt=".4f"))

def format_regression_results(coefficients, t_values, model_name):
    """
    Форматирует результаты регрессии с коэффициентами и t-значениями.

    Parameters:
    coefficients (list): Список коэффициентов регрессии.
    t_values (list): Список t-значений.
    model_name (str): Название модели.

    Returns:
    None
    """
    headers = ["Коэффициент", "Значение", "t-значение", "Значимость"]
    rows = []
    for i, (b, t) in enumerate(zip(coefficients, t_values)):
        significance = "Значим" if abs(t) > stats.t.ppf(1 - 0.025, len(coefficients) - 1) else "Не значим"
        rows.append([f"b{i}", f"{b:.4f}", f"{t:.4f}", significance])
    
    print(f"\n--- Коэффициенты регрессии для модели: {model_name} ---")
    print(tabulate(rows, headers, tablefmt="fancy_grid", floatfmt=".4f"))

def format_fisher_test_results(Fp, Ft, S_ad, model_name):
    """
    Форматирует результаты проверки по критерию Фишера.

    Parameters:
    Fp (float): Значение F-критерия.
    Ft (float): Критическое значение F-критерия.
    S_ad (float): Дисперсия адекватности.
    model_name (str): Название модели.

    Returns:
    None
    """
    headers = ["Показатель", "Значение"]
    rows = [
        ["Дисперсия адекватности", f"{S_ad:.4f}"],
        ["Критерий Фишера (Fp)", f"{Fp:.4f}"],
        ["Критическое значение Фишера (Ft)", f"{Ft:.4f}"],
        ["Адекватность модели", "Адекватна" if Fp < Ft else "Неадекватна"]
    ]
    
    print(f"\n--- Результаты проверки по критерию Фишера для модели: {model_name} ---")
    print(tabulate(rows, headers, tablefmt="fancy_grid", floatfmt=".4f"))
