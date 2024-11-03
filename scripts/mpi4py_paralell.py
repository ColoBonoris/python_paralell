from mpi4py import MPI
import numpy as np
import time

def euclidean_distance_matrix_mpi(X):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    m, n = X.shape

    # Calculate rows per process, and handle uneven distribution
    rows_per_process = [m // size] * size
    for i in range(m % size):
        rows_per_process[i] += 1

    # Create displacements (offsets) for Scatterv
    displacements = [sum(rows_per_process[:i]) * n for i in range(size)]
    send_counts = [r * n for r in rows_per_process]

    # Initialize local arrays
    local_rows = rows_per_process[rank]
    local_X = np.zeros((local_rows, n))

    # Scatter the matrix X to all processes
    if rank == 0:
        start_comm_time = time.time()
    comm.Scatterv([X, send_counts, displacements, MPI.DOUBLE], local_X, root=0)
    if rank == 0:
        end_comm_time = time.time()
        comm_time = end_comm_time - start_comm_time

    # Compute the local Euclidean Distance Matrix
    local_D = np.zeros((local_rows, m))
    
    if rank == 0:
        start_calc_time = time.time()
    for i in range(local_rows):
        for j in range(m):
            local_D[i, j] = np.sqrt(np.sum((local_X[i] - X[j])**2))
    if rank == 0:
        end_calc_time = time.time()
        calc_time = end_calc_time - start_calc_time

    # Gather the results back to process 0
    if rank == 0:
        D = np.zeros((m, m))
    else:
        D = None
    
    recv_counts = [r * m for r in rows_per_process]
    recv_displacements = [sum(recv_counts[:i]) for i in range(size)]

    if rank == 0:
        start_comm_time = time.time()
    comm.Gatherv(local_D, [D, recv_counts, recv_displacements, MPI.DOUBLE], root=0)
    if rank == 0:
        end_comm_time = time.time()
        comm_time += end_comm_time - start_comm_time
        return D[:m, :], calc_time, comm_time

if __name__ == '__main__':
    '''
        Lee las 4 matrices de:
            ../data/mde_matrix_512x512.npy
            ../data/mde_matrix_1024x1024.npy
            ../data/mde_matrix_2048x2048.npy
            ../data/mde_matrix_4096x4096.npy
            ../data/mde_matrix_8192x8192.npy
        guarda el resultado en:
            ../results/mpi_mde_matrix_512x512.npy
            ../results/mpi_mde_matrix_1024x1024.npy
            ../results/mpi_mde_matrix_2048x2048.npy
            ../results/mpi_mde_matrix_4096x4096.npy
            ../results/mpi_mde_matrix_8192x8192.npy
        guarda el tiempo de ejecución en:
            ../data/mpi4py.csv agregando una fila con el tiempo para cada matriz, el encabezado es (size,512,1024,2048,4096,8192)
    '''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    matrixes = [
        np.load('./data/mde_matrix_512x512.npy'),
        np.load('./data/mde_matrix_1024x1024.npy'),
        np.load('./data/mde_matrix_2048x2048.npy'),
        np.load('./data/mde_matrix_4096x4096.npy'),
        np.load('./data/mde_matrix_8192x8192.npy')
    ]
    times = []
    comms = []

    for matrix in matrixes:
        if rank == 0:
            print(f"Calculando matriz de distancia para matriz de tamaño {matrix.shape}")
        
        result = euclidean_distance_matrix_mpi(matrix)
        
        if rank == 0:
            D, calc_time, comm_time = result
            print(f"Tiempo de cálculo (MPI4Py): {calc_time:.6f} segundos")
            print(f"Tiempo de comunicación (MPI4Py): {comm_time:.6f} segundos")
            # Guardar la matriz
            np.save(f'./results/mpi_mde_matrix_{matrix.shape[0]}x{matrix.shape[0]}.npy', D)
            # Guardar el tiempo de ejecución
            times.append(calc_time)
            comms.append(comm_time)
    
    if rank == 0:
        with open('./results/mpi4py.csv', 'a') as f, open('./results/mpi4py_comms.csv', 'a') as f2:
            f.write(','.join(map(str, ["units",512, 1024, 2048, 4096, 8192])) + '\n')
            f.write(f"{rank}" + ','.join(map(str, times)) + '\n')
            f2.write(','.join(map(str, ["units",512, 1024, 2048, 4096, 8192])) + '\n')
            f2.write(f"{rank}" + ','.join(map(str, comms)) + '\n')