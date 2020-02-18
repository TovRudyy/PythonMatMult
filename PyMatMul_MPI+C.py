import numpy as np
from mpi4py import MPI
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
# MPI variables
comm = MPI.COMM_WORLD
mpiRank = comm.Get_rank()
mpiSize = comm.Get_size()

if (N % mpiSize != 0):
    print('BAD DIMENSION: N must be divisible by the amount of Tasks')
    exit(-1)

# Matmul variables
chunk = int(N / mpiSize)
A, B = None, None

if mpiRank == TaskMaster:
    print('MASTER: Number of MPI tasks is: '+str(mpiSize))
    print('Creating matrix...')
    A = np.random.randint(10, size=(N, N), dtype='int32')
    B = np.random.randint(10, size=(N, N), dtype='int32')

# Broadcasting Matrix B
B = comm.bcast(B, root=TaskMaster)

#Distribute Matrix A
if mpiRank == TaskMaster:
    print('MASTER: Distributing matrix A')
    for i in range(1, mpiSize):
        lowerBound = i * chunk
        upperBound = (i+1) * chunk
        tmp = A[lowerBound:upperBound,:]
        comm.send(tmp, dest=i, tag=MPIT_MATRIX_A)
else:
    A = comm.recv(source=TaskMaster, tag=MPIT_MATRIX_A)

# Matrix Multiplication
print('TASK '+str(mpiRank)+' | Multiplying...')
C = wrap_matmul(A, B, chunk)

# Gather of all results in MASTER
C = comm.gather(C, root=TaskMaster)
# Master checks the result
if mpiRank == TaskMaster:
    R = np.matmul(A, B)
    print('MASTER: Verifying...')
    # Join the result matrix of all Tasks
    C = np.concatenate(C)
    for i in range(N):
        for j in range(N):
            if (C[i][j] != R[i][j]):
                print('WRONG multiplication!')
                comm.Abort(-1)
    print('CORRECT multiplication!')

comm.Barrier()