import numpy as np
import time

def euclidean_distance_matrix(X):
    m = X.shape[0]
    D = np.zeros((m, m))
    for i in range(m):
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
            ../results/mde_matrix_512x512.npy
            ../results/mde_matrix_1024x1024.npy
            ../results/mde_matrix_2048x2048.npy
            ../results/mde_matrix_4096x4096.npy
            ../results/mde_matrix_8192x8192.npy
        guarda el tiempo de ejecución en:
            ../data/secuential.csv agregando una fila con el tiempo para cada matriz, el encabezado es (size,512,1024,2048,4096,8192)
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
    # Calcular la matriz de distancia para cada matriz
    for matrix in matrixes:
        print(f"Calculando matriz de distancia para matriz de tamaño {matrix.shape}")
        start_time = time.time()
        D = euclidean_distance_matrix(matrix)
        end_time = time.time()
        
        print(f"Tiempo de ejecución (secuencial): {end_time - start_time:.6f} segundos")
        # Guardar la matriz
        np.save(f'./results/mde_matrix_{matrix.shape[0]}x{matrix.shape[0]}.npy', D)
        # Guardar el tiempo de ejecución
        times.append(end_time - start_time)
    
    # Guardar los tiempos de ejecución en un archivo CSV
    with open('./results/secuential.csv', 'a') as f:
        f.write(','.join(map(str, [512, 1024, 2048, 4096, 8192])) + '\n')
        f.write(','.join(map(str, times)) + '\n')
        