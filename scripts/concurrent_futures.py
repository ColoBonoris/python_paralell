import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import time
import sys
import psutil
import os

def euclidean_distance_chunk(chunk):
    X_chunk, X = chunk
    D_chunk = np.zeros((len(X_chunk), len(X)))
    for i in range(len(X_chunk)):
        for j in range(len(X)):
            D_chunk[i, j] = np.sqrt(np.sum((X_chunk[i] - X[j]) ** 2))
    return D_chunk

def euclidean_distance_matrix_cf(X, num_workers):
    # Dividir la matriz X en chunks dinámicos para cada proceso
    chunk_size = len(X) // num_workers
    chunks = [(X[i:i + chunk_size], X) for i in range(0, len(X), chunk_size)]
    
    # Asegurarse de que el último chunk incluya todas las filas restantes
    if len(X) % num_workers != 0:
        chunks[-1] = (X[len(chunks) * chunk_size:], X)

    D_total = np.zeros((len(X), len(X)))
    
    # Iniciar el tiempo de comunicación antes de distribuir las tareas
    start_comm_time = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(euclidean_distance_chunk, chunk) for chunk in chunks]
        end_comm_time = time.time()  # Finalizar tiempo de comunicación una vez distribuidos

        # Iniciar el tiempo de cálculo después de la distribución
        start_calc_time = time.time()
        row_start = 0
        for future in as_completed(futures):
            D_chunk = future.result()
            num_rows = len(D_chunk)
            D_total[row_start:row_start + num_rows] = D_chunk
            row_start += num_rows
        end_calc_time = time.time()
    
    # Calcular tiempos
    comm_time = end_comm_time - start_comm_time
    calc_time = end_calc_time - start_calc_time

    return D_total, calc_time, comm_time

if __name__ == '__main__':
    '''
        Lee las 5 matrices de:
            ../data/mde_matrix_512x512.npy
            ../data/mde_matrix_1024x1024.npy
            ../data/mde_matrix_2048x2048.npy
            ../data/mde_matrix_4096x4096.npy
            ../data/mde_matrix_8192x8192.npy
        guarda el resultado en:
            ../results/cf_mde_matrix_512x512.npy
            ../results/cf_mde_matrix_1024x1024.npy
            ../results/cf_mde_matrix_2048x2048.npy
            ../results/cf_mde_matrix_4096x4096.npy
            ../results/cf_mde_matrix_8192x8192.npy
        guarda el tiempo de ejecución en:
            ../data/cf_times.csv, con encabezado (size,512,1024,2048,4096,8192)
            ../data/cf_comms.csv, con encabezado (size,512,1024,2048,4096,8192)
    '''
    # Cargar matrices de ejemplo
    matrixes = [
        np.load('./data/mde_matrix_512x512.npy'),
        np.load('./data/mde_matrix_1024x1024.npy'),
        np.load('./data/mde_matrix_2048x2048.npy'),
        np.load('./data/mde_matrix_4096x4096.npy'),
        np.load('./data/mde_matrix_8192x8192.npy')
    ]
    
    # Obtener cantidad de workers de argumentos
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else cpu_count()
    times = []
    comms = []
    
    # Calcular la matriz de distancia para cada tamaño
    for matrix in matrixes:
        print(f"Calculando matriz de distancia para tamaño {matrix.shape}")
        
        # Calcular distancia Euclidiana y medir tiempos
        D, calc_time, comm_time = euclidean_distance_matrix_cf(matrix, workers)
        
        print(f"Tiempo de cálculo (concurrent.features): {calc_time:.6f} segundos")
        print(f"Tiempo de comunicación (concurrent.futures): {comm_time:.6f} segundos")
        
        # Guardar matriz y tiempos
        np.save(f'./results/cf_mde_matrix_{matrix.shape[0]}x{matrix.shape[0]}.npy', D)
        times.append(calc_time)
        comms.append(comm_time)
    
    # Guardar tiempos de ejecución en archivo CSV
    with open('./results/cf_times.csv', 'a') as f, open('./results/cf_comms.csv', 'a') as f2:
        f.write(','.join(map(str, ["units",512, 1024, 2048, 4096, 8192])) + '\n')
        f.write(f"{workers}," + ','.join(map(str, times)) + '\n')
        f2.write(','.join(map(str, ["units",512, 1024, 2048, 4096, 8192])) + '\n')
        f2.write(f"{workers}," + ','.join(map(str, comms)) + '\n')
