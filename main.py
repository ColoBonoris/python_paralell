import numpy as np

def generate_deterministic_chaotic_mde(m, n):
    """
    Genera una matriz de m puntos con n dimensiones
    Es una función determinista pero se busca que parezca aleatoria (caótica)
    
    Parámetros:
        m (int): Número de filas (puntos) a generar.
        n (int): Dimensión del espacio en el que se encuentran los puntos.
    
    Devuelve:
        np.array: Matriz de dpuntos mxn.
    """
    # Generar puntos
    X = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            # Generar comportamiento caótico usando una función trigonométrica
            X[i, j] = np.sin((i + 1) * (j + 1) * 0.5) * np.cos(i * j * 0.3)
    return X

if __name__ == '__main__':
    # Parámetros de la matriz: m (número de puntos) y n (dimensiones)
    sizes = [512, 1024, 2048, 4096, 8192]
    # Generar MDE
    for size in sizes:
        mde_matrix = generate_deterministic_chaotic_mde(size, size)
        # Guardar la matriz en un JSON
        np.save(f'./data/mde_matrix_{size}x{size}.npy', mde_matrix)