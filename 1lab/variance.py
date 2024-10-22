import numpy as np

def calculate_variances(Y):
    """
    Вычисляет дисперсии для каждого опыта и среднюю дисперсию.
    
    Parameters:
    Y (numpy array): Массив с повторными измерениями для каждого опыта.
    
    Returns:
    variances (numpy array): Массив дисперсий для каждого опыта.
    s0_sq (float): Средняя дисперсия.
    """
    # Рассчитываем дисперсию по каждому опыту (по строкам)
    variances = np.var(Y, axis=1, ddof=1)
    
    # Средняя дисперсия (по всем опытам)
    s0_sq = np.mean(variances)
    
    return variances, s0_sq
