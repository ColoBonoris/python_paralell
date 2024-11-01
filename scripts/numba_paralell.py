from numba import njit, prange
import numpy as np
import time

@njit(fastmath=True)
def euclidean_distance_matrix_numba(X):
    m = X.shape[0]
    D = np.zeros((m, m))

    # Paralelizamos el bucle exterior para mejorar el rendimiento
    for i in prange(m):
        for j in range(m):
            D[i, j] = np.sqrt(np.sum((X[i] - X[j])**2))

    return D

if __name__ == '__main__':
    '''
        Lee las 4 matrices de:
            ../data/mde_matrix_512x512.npy
            ../data/mde_matrix_1024x1024.npy
            ../data/mde_matrix_2048x2048.npy
            ../data/mde_matrix_4096x4096.npy
            ../data/mde_matrix_8192x8192.npy
        guarda el resultado en:
            ../results/numba_mde_matrix_512x512.npy
            ../results/numba_mde_matrix_1024x1024.npy
            ../results/numba_mde_matrix_2048x2048.npy
            ../results/numba_mde_matrix_4096x4096.npy
            ../results/numba_mde_matrix_8192x8192.npy
        guarda el tiempo de ejecución en:
            ../data/numba.csv agregando una fila con el tiempo para cada matriz, el encabezado es (size,512,1024,2048,4096,8192)
    '''
    # Definimos los datos
    matrixes = [
        np.load('./data/mde_matrix_512x512.npy'),
        np.load('./data/mde_matrix_1024x1024.npy'),
        np.load('./data/mde_matrix_2048x2048.npy'),
        np.load('./data/mde_matrix_4096x4096.npy'),
        np.load('./data/mde_matrix_8192x8192.npy')
    ]
    times = []
    # Nos aseguramos de que antes compile la función
    euclidean_distance_matrix_numba(matrixes[0])

    # Calcular la matriz de distancia para cada matriz
    for matrix in matrixes:
        print(f"Calculando matriz de distancia para matriz de tamaño {matrix.shape}")
        start_time = time.time()
        D = euclidean_distance_matrix_numba(matrix)
        end_time = time.time()
        
        print(f"Tiempo de ejecución (numba): {end_time - start_time:.6f} segundos")
        # Guardar la matriz
        np.save(f'./results/numba_mde_matrix_{matrix.shape[0]}x{matrix.shape[0]}.npy', D)
        # Guardar el tiempo de ejecución
        times.append(end_time - start_time)
    # Guardar los tiempos de ejecución en un archivo CSV
    with open('./results/numba.csv', 'a') as f:
        f.write(','.join(map(str, [512, 1024, 2048, 4096, 8192])) + '\n')
        f.write(','.join(map(str, times)) + '\n')
