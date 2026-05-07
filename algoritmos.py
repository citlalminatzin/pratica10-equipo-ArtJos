import random
import numpy as np
from funciones import gradf

def metodo_gradiente(n: int, x0: list, a: float, f: callable, limits: list) -> list:
    """
    Implementa el descenso de gradiente con proyección 
     x_actual = x_actual - a * g
     Aplicamos la relación de recurrencia: el signo negativo 
     asegura que nos movemos en dirección opuesta al descenso
    """
    x_actual = np.array(x0, dtype=float)
    historial = [x_actual.tolist()]
    
    for _ in range(n):
        # Obtenemos el gradiente numérico
        g = np.array(gradf(x_actual.tolist(), f))
        # Actualización: x_n+1 = x_n - a * grad(f) 
        x_actual = x_actual - a * g
        # Proyección: Asegurar que el punto no salga del dominio 
        x_actual = np.clip(x_actual, limits[0], limits[1])
        historial.append(x_actual.tolist())
        
    return historial

def algoritmo_evolutivo(n: int, x0: list, a: float, m: float, 
                        x_limits: list, y_limits: list, f: callable) -> list:
    """
    Algoritmo evolutivo con mutación y vecindad optimizada
    Mecanismo de Mutación: Permite la exploración global 
    y ayuda a escapar de óptimos locales.
    """
    p_actual = np.array(x0, dtype=float)
    historial = [p_actual.tolist()]
    
    for _ in range(n):
        if random.random() < m:
            # Mutación: Punto aleatorio en todo el dominio 
            p_cand = np.array([
                random.uniform(x_limits[0], x_limits[1]),
                random.uniform(y_limits[0], y_limits[1])
            ])
        else:
            # Vecindad: Generamos desplazamiento y recortamos al límite 
            p_cand = p_actual + np.random.uniform(-a, a, size=2)
            p_cand[0] = np.clip(p_cand[0], x_limits[0], x_limits[1])
            p_cand[1] = np.clip(p_cand[1], y_limits[0], y_limits[1])
        
        # Selección: Si el candidato es mejor, se acepta 
        if f(p_cand.tolist()) < f(p_actual.tolist()):
            p_actual = p_cand
            
        historial.append(p_actual.tolist())
        
    return historial