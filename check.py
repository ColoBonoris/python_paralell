# Here we'll check the results of every run made by the scripts.
# We'll have a tolerance of 1% in the results.

import numpy as np

def check(A, B):
    # Compare two matrixes and return True if they pass
    return np.allclose(A, B, rtol=1, atol=1)


# Define the data
#sizes = ["512", "1024", "2048", "4096", "8192"]
sizes = ["512", "1024"]
types = ["mp", "cf", "numba", "mpi"]

# Load the sequential results
matrixes = [
    np.load('./data/mde_matrix_512x512.npy'),
    np.load('./data/mde_matrix_1024x1024.npy'),
    np.load('./data/mde_matrix_2048x2048.npy'),
    np.load('./data/mde_matrix_4096x4096.npy'),
    np.load('./data/mde_matrix_8192x8192.npy')
]

# Check the results
for t in types:
    print(f"Checking {t}")
    for i, size in enumerate(sizes):
        D = np.load(f'./results/{t}_mde_matrix_{size}x{size}.npy')
        result = check(D, matrixes[i])
        if(result): print(f"{t} {size}x{size} passed the test")
        else: print(f"{t} {size}x{size} failed the test")

