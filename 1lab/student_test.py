import numpy as np

def student_test(coefficients, s_bi, N, m):
    """
    Проверяет значимость коэффициентов регрессии с помощью критерия Стьюдента.
    
    Parameters:
    coefficients (numpy array): Вектор коэффициентов регрессии.
    s_bi (float): Среднеквадратичное отклонение коэффициентов.
    N (int): Количество экспериментов.
    m (int): Количество повторных измерений для каждого эксперимента.
    
    Returns:
    t_values (numpy array): Вектор t-значений для каждого коэффициента.
    """
    # Рассчитываем стандартную ошибку для коэффициентов
    S_b = s_bi / np.sqrt(N * m)
    
    # Рассчитываем t-значения для каждого коэффициента
    t_values = np.abs(coefficients) / S_b
    
    return t_values
