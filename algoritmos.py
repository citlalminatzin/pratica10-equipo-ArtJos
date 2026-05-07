import random
from funciones import gradf_robusto 

def metodo_gradiente(n: int, x0: list, a: float, f: callable, limits: list) -> list:
    historial = [x0]
    x_actual = list(x0) # Aseguramos que sea una copia de la lista simple
    for _ in range(n):
        grad = gradf_robusto(x_actual, f, limits) 
        # Actualización según la relación de recurrencia [cite: 54]
        x_actual = [x_actual[0] - a * grad[0], x_actual[1] - a * grad[1]]
        historial.append(x_actual)
    return historial

def algoritmo_evolutivo(n: int, x0: list, a: float, m: float, 
                        x_limits: list, y_limits: list, f: callable) -> list:
    """
    n: iteraciones 
    x0: condición inicial 
    a: tamaño de vecindad 
    m: prob. de mutación 
    """
    historial = [x0]
    p_actual = x0
    
    for _ in range(n):
        # Decidir si hay mutación o búsqueda  
        if random.random() < m:
            # Mutación
            p_candidato = [
                random.uniform(x_limits[0], x_limits[1]),
                random.uniform(y_limits[0], y_limits[1])
            ]
        else:
            # Vecindad: [x-a, x+a] asegurar que el candidato esté en la frontera 
            valido = False
            while not valido:
                p_candidato = [
                    random.uniform(p_actual[0] - a, p_actual[0] + a),
                    random.uniform(p_actual[1] - a, p_actual[1] + a)
                ]
                if (x_limits[0] <= p_candidato[0] <= x_limits[1] and 
                    y_limits[0] <= p_candidato[1] <= y_limits[1]):
                    valido = True
        
        # Comparar y seleccionar el mejor 
        if f(p_candidato) < f(p_actual):
            p_actual = p_candidato
            
        historial.append(p_actual)
        
    return historial 