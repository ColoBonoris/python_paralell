import csv

# Define la dimensión de cada punto (cambia esto si sabes el valor exacto)
d = 3

# Función para cargar los datos de un archivo CSV
def load_csv(file_path, has_units=True):
    data = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            if has_units:
                data.append([int(row[0])] + [float(x) for x in row[1:]])
            else:
                data.append([float(x) for x in row])
    return header, data

# Función para calcular el speedup, eficiencia, overhead y FLOPs/s
def calculate_metrics(sequential_times, parallel_times, comms=None):
    results = []
    n_sizes = len(sequential_times[0])  # Cantidad de tamaños de matrices

    for row in parallel_times:
        units = row[0]
        times = row[1:]

        # Calcular speedup
        speedup = [sequential_times[0][i] / times[i] for i in range(n_sizes)]

        # Calcular eficiencia
        efficiency = [speedup[i] / units for i in range(n_sizes)]

        # Calcular overhead (incluyendo el multiplicador de 100)
        if comms:
            overhead = [(comms[row[0]][i] / times[i]) * 100 for i in range(n_sizes)]
        else:
            overhead = ["N/A"] * n_sizes  # Para numba y mp sin tiempo de comunicación

        results.append([units, times, speedup, efficiency, overhead])

    return results

# Cargar los datos de tiempos secuenciales y paralelos
header, sequential_times_data = load_csv('./results/sequential_times.csv', has_units=False)
sequential_times = sequential_times_data[0]  # Solo una fila en tiempos secuenciales

# Cargar archivos de tiempos y comunicaciones para cada tipo de enfoque
approaches = ['cf', 'mp', 'mpi4py', 'numba']
all_metrics = []

for approach in approaches:
    time_header, parallel_times = load_csv(f'./results/{approach}_times.csv')
    comms = None

    if approach not in ['mp', 'numba']:
        comms_header, comms = load_csv(f'./results/{approach}_comms.csv')
        comms = {row[0]: row[1:] for row in comms}  # Convertir en dict para acceso fácil

    # Calcular métricas y almacenar con el tipo de enfoque
    metrics = calculate_metrics([sequential_times], parallel_times, comms)
    for units, times, speedup, efficiency, overhead in metrics:
        all_metrics.append(
            [approach, units] + times + speedup + efficiency + overhead
        )

# Guardar las métricas en un archivo CSV
with open('./results/stats_metrics.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Encabezado con las métricas
    header_row = (
        ["type", "units"] +
        [f"time_{size}" for size in header] +
        [f"speedup_{size}" for size in header] +
        [f"efficiency_{size}" for size in header] +
        [f"overhead_{size}" for size in header]
    )
    writer.writerow(header_row)

    # Escribir los datos de las métricas
    for row in all_metrics:
        writer.writerow(row)
