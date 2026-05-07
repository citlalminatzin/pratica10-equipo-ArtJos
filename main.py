from funciones import f1, f2
from algoritmos import metodo_gradiente, algoritmo_evolutivo
import random

if __name__ == "__main__":
    # --- Pruebas f1 ---
    print("Resultados f1 :")
    lim_f1 = [-5, 5]
    # Gradiente: convergencia rápida en funciones suaves
    res_g1 = metodo_gradiente(100, [0, 0], 0.01, f1, lim_f1) 
    print(f"  Gradiente: {f1(res_g1[-1]):.6f} en {res_g1[-1]}")
    
    # --- Pruebas f2 ---
    print("\nResultados f2 :")
    lim_f2 = [-512, 512]
    x0_f2 = [random.uniform(-512, 512), random.uniform(-512, 512)]
    
    # Evolutivo: mejor para evitar mínimos locales (Paso 2, Ejercicio 3)
    res_e2 = algoritmo_evolutivo(5000, x0_f2, 15, 0.2, lim_f2, lim_f2, f2) 
    print(f"  Evolutivo: {f2(res_e2[-1]):.6f} en {res_e2[-1]}")