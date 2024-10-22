import numpy as np
from numpy.linalg import inv, LinAlgError

def calculate_coefficients(X, Y_mean):
    """
    Вычисляет коэффициенты линейной регрессии методом наименьших квадратов.
    
    Parameters:
    X (numpy array): Матрица с факторами (независимыми переменными).
    Y_mean (numpy array): Вектор средних значений отклика (зависимой переменной) для каждого эксперимента.
    
    Returns:
    coefficients (numpy array): Вектор коэффициентов регрессии.
    
    Raises:
    LinAlgError: Если матрица (X^T X) является вырожденной (необратимой).
    """
    try:
        # Транспонирование матрицы X
        XT = X.T
        
        # Вычисление коэффициентов регрессии по формуле (XT * X)^(-1) * XT * Y_mean
        coefficients = inv(XT @ X) @ (XT @ Y_mean)
        
        return coefficients
    
    except LinAlgError:
        raise LinAlgError("Матрица X^T X вырождена и не может быть инвертирована.")
