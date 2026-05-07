import random
from funciones import f1, f2
from algoritmos import metodo_gradiente, algoritmo_evolutivo

"""
Realizamos n intentos para mitigar la sensibilidad 
 a la condición inicial (x0), especialmente en superficies no convexas.
"""
def realizar_pruebas(nombre, func, limites, tipo="gradiente", n_intentos=10):
    print(f"\n=== Evaluando {nombre} ({n_intentos} intentos) ===")
    mejor_global_val = float('inf')
    mejor_global_punto = None

    for i in range(n_intentos):
        # Punto inicial aleatorio dentro del dominio 
        x0 = [random.uniform(limites[0], limites[1]), 
              random.uniform(limites[0], limites[1])]
        
        if tipo == "gradiente":
            # Para f1, el gradiente es muy efectivo 
            res = metodo_gradiente(200, x0, 0.01, func, limites)
        else:
            # Para f2, el evolutivo evita mínimos locales 
            res = algoritmo_evolutivo(10000, x0, 10, 0.1, limites, limites, func)
        
        ultimo_punto = res[-1]
        valor_final = func(ultimo_punto)
        
        if valor_final < mejor_global_val:
            mejor_global_val = valor_final
            mejor_global_punto = ultimo_punto
            
        print(f"  Intento {i+1}: Valor = {valor_final:.6f}")

    print(f"\n>> RESULTADO ÓPTIMO ENCONTRADO EN {nombre}:")
    print(f"   Valor: {mejor_global_val:.8f}")
    print(f"   Punto: {mejor_global_punto}")

if __name__ == "__main__":
    # Prueba para f1 
    realizar_pruebas("f1", f1, [-5, 5], tipo="gradiente")
    
    # Prueba para f2
    realizar_pruebas("f2", f2, [-512, 512], tipo="evolutivo")