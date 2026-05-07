import numpy as np

def f1(X: list) -> float:
    # f1(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2 
    x, y = X[0], X[1]
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2 

def f2(X: list) -> float:
    # f2(x,y) con la simplificación (|y|) 
    x, y = X[0], X[1]
    tr1 = -(y + 47) * np.sin(np.sqrt(abs(y)))
    tr2 = -x * np.sin(np.sqrt(abs(y)))
    return tr1 + tr2

def gradf(X: list, f: callable, h: float = 0.001) -> list:
    """
    Calcula el gradiente aproximado. 
    Nota: Se usa diferencia centrada para obtener un error de orden O(h^2),
    siendo más preciso que la diferencia hacia adelante.
    """
    grad = []
    for i in range(len(X)):
        p_mas = list(X); p_mas[i] += h
        p_menos = list(X); p_menos[i] -= h
        # Formula: (f(x+h) - f(x-h)) / 2h 
        df = (f(p_mas) - f(p_menos)) / (2 * h)
        grad.append(df)
    return grad