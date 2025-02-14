import numpy as np
from scipy import stats

def cochran_test(variances, Y, alpha=0.05):
    """
    Проверяет однородность дисперсий по критерию Кохрена.

    Parameters:
    variances (numpy array): Массив дисперсий для каждого опыта.
    Y (numpy array): Массив с повторными измерениями для каждого опыта.
    alpha (float): Уровень значимости, по умолчанию 0.05.

    Returns:
    Gp (float): Рассчитанное значение критерия Кохрена.
    Gt (float): Табличное значение критерия Кохрена на уровне значимости alpha.
    """
    # Рассчитываем значение критерия Кохрена (Gp)
    Gp = max(variances) / np.sum(variances)
    
    # Число степеней свободы для каждого опыта (количество измерений - 1)
    f1 = len(Y[0]) - 1  # Степени свободы для каждого опыта
    
    # Количество опытов
    f2 = len(variances)  # Степени свободы для суммы
    
    # Табличное значение критерия Кохрена (Gt) через распределение Фишера-Снедекора
    Gt = stats.f.ppf(1 - alpha, f1, f1 * f2) / (stats.f.ppf(1 - alpha, f1, f1 * f2) + f2 - 1)
    
    return Gp, Gt
