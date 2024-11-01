import numpy as np
import time
from multiprocessing import Pool, cpu_count
import sys

def euclidean_distance_chunk(chunk):
    X_chunk, X = chunk
    D_chunk = np.zeros((len(X_chunk), len(X)))
    for i in range(len(X_chunk)):
        for j in range(len(X)):
            D_chunk[i, j] = np.sqrt(np.sum((X_chunk[i] - X[j])**2))
    return D_chunk

def euclidean_distance_matrix_mp(X, workers):
    num_workers =  workers
    chunk_size = len(X) // num_workers
    
    chunks = [(X[i:i + chunk_size], X) for i in range(0, len(X), chunk_size)]

    with Pool(num_workers) as pool:
        results = pool.map(euclidean_distance_chunk, chunks)

    return np.vstack(results)

if __name__ == '__main__':
    '''
        Lee las 4 matrices de:
            ../data/mde_matrix_512x512.npy
            ../data/mde_matrix_1024x1024.npy
            ../data/mde_matrix_2048x2048.npy
            ../data/mde_matrix_4096x4096.npy
            ../data/mde_matrix_8192x8192.npy
        guarda el resultado en:
            ../results/mp_mde_matrix_512x512.npy
            ../results/mp_mde_matrix_1024x1024.npy
            ../results/mp_mde_matrix_2048x2048.npy
            ../results/mp_mde_matrix_4096x4096.npy
            ../results/mp_mde_matrix_8192x8192.npy
        guarda el tiempo de ejecución en:
            ../data/multi_processing.csv agregando una fila con el tiempo para cada matriz, el encabezado es (size,512,1024,2048,4096,8192)
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
    workers = int(sys.argv[1])

    # Calcular la matriz de distancia para cada matriz
    for matrix in matrixes:
        print(f"Calculando matriz de distancia para matriz de tamaño {matrix.shape}")

        start_time = time.time()
        D = euclidean_distance_matrix_mp(matrix, workers)
        end_time = time.time()
        
        print(f"Tiempo de ejecución (multiprocessing): {end_time - start_time:.6f} segundos")
        # Guardar la matriz
        np.save(f'./results/mp_mde_matrix_{matrix.shape[0]}x{matrix.shape[0]}.npy', D)
        # Guardar el tiempo de ejecución
        times.append(end_time - start_time)
    # Guardar los tiempos de ejecución en un archivo CSV
    with open('./results/multi_processing.csv', 'a') as f:
        f.write(','.join(map(str, ["matriz",512, 1024, 2048, 4096, 8192])) + '\n')
        f.write(f"time {workers} cores," + ','.join(map(str, times)) + '\n')
    
