import numpy as np

# Definimos las funciones
def f1(X: list) -> float:
    x, y = X[0], X[1]
    # f1(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2 

def f2(X: list) -> float:
    """
    Recibe una lista X = [x, y] y regresa un flotante.
    """
    x = X[0]
    y = X[1]
    
    # Primer término
    tr1 = -(y + 47) * np.sin(np.sqrt(abs(y)))
    # Segundo término: -x * sin(sqrt(|y|)
    tr2 = -x * np.sin(np.sqrt(abs(y)))
    
    return tr1 + tr2

def gradf_robusto(X: list, f: callable, limits: list, h: float = 0.000001) -> list:
    # Si X llega como [[x, y]], extraemos [x, y] para evitar el TypeError 
    if isinstance(X[0], list):
        X = X[0]
        
    grad = []
    for i in range(len(X)):
        # Uso dinámico de limits para f1 [-5, 5] y f2 [-512, 512] [cite: 8, 11]
        if X[i] + h > limits[1]:
            punto_atras = list(X); punto_atras[i] -= h
            df = (f(X) - f(punto_atras)) / h
        elif X[i] - h < limits[0]:
            punto_adelante = list(X); punto_adelante[i] += h
            df = (f(punto_adelante) - f(X)) / h
        else:
            p_mas = list(X); p_mas[i] += h
            p_menos = list(X); p_menos[i] -= h
            df = (f(p_mas) - f(p_menos)) / (2 * h) # Diferencia centrada [cite: 41]
        grad.append(df)
    return grad