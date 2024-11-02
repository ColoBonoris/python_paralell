import os
import csv

def read_times_from_csv(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Saltamos el encabezado
        times = [list(map(float, row[1:])) for row in reader]
    return times

def calculate_speedup_and_efficiency(sequential_times, parallel_times, num_cores):
    speedup = [[seq / par if par != 0 else 0 for seq, par in zip(sequential_times, p_times)] for p_times in parallel_times]
    efficiency = [[s / core if core != 0 else 0 for s in s_times] for s_times, core in zip(speedup, num_cores)]
    return speedup, efficiency

def calculate_overhead(communication_times, parallel_times):
    overhead = [[comm / par if par != 0 else 0 for comm, par in zip(c_times, p_times)] for c_times, p_times in zip(communication_times, parallel_times)]
    return overhead

def write_results_to_csv(filename, headers, rows):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

# Rutas de los archivos
sequential_file = './results/sequential.csv'
parallel_files = {
    'cf': './results/concurrent_features.csv',
    'mpi': './results/mpi.csv',
    'numba': './results/numba.csv'
}

# Leer tiempos secuenciales
sequential_times = read_times_from_csv(sequential_file)[0]  # Solo una fila

# Leer tiempos de ejecución paralela y de comunicación
parallel_data = {}
num_cores = [12, 10, 8, 4, 2]

for approach, filepath in parallel_files.items():
    parallel_times = read_times_from_csv(filepath)
    if approach != 'numba':  # `numba` no tiene tiempos de comunicación
        communication_times = read_times_from_csv(filepath)[1:]
    else:
        communication_times = None
    
    # Calcular speedup y eficiencia
    speedup, efficiency = calculate_speedup_and_efficiency(sequential_times, parallel_times, num_cores)
    if communication_times:
        overhead = calculate_overhead(communication_times, parallel_times)

    # Guardar resultados en CSV
    write_results_to_csv(f'./results/{approach}_speedup.csv', ["cores"] + [str(size) for size in (512, 1024, 2048, 4096, 8192)], [[core] + s_times for core, s_times in zip(num_cores, speedup)])
    write_results_to_csv(f'./results/{approach}_efficiency.csv', ["cores"] + [str(size) for size in (512, 1024, 2048, 4096, 8192)], [[core] + e_times for core, e_times in zip(num_cores, efficiency)])
    if communication_times:
        write_results_to_csv(f'./results/{approach}_overhead.csv', ["cores"] + [str(size) for size in (512, 1024, 2048, 4096, 8192)], [[core] + o_times for core, o_times in zip(num_cores, overhead)])
