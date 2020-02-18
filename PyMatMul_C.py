import numpy as np
import matmul
import pyextrae.mpi as pyextrae
print('MATRIX MULTIPLICATION (NxN) V0.2')

# Constants
N = 256
TaskMaster = 0
MPIT_MATRIX_A = 2

def wrap_matmul(A, B, chunk):
    return matmul.matmul_omp(A, B, chunk)
    #return matmul.matmul(A, B, chunk)

# Main
print('Creating matrix...')
A = np.random.randint(10, size=(N, N), dtype='int32')
B = np.random.randint(10, size=(N, N), dtype='int32')

# Matrix Multiplication
print('Multiplying...')
C = wrap_matmul(A, B, A.shape[0])

# Master checks the result
R = np.matmul(A, B)
print('MASTER: Verifying...')
for i in range(N):
    for j in range(N):
        if (C[i][j] != R[i][j]):
            print('WRONG multiplication!')
            exit(-1)
print('CORRECT multiplication!')
