from qrdecomp import simpleQR
from substitution import forward
import numpy as np

def QR_modified(A, epsilon=1e-10):
    m, n = A.shape


    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    rank = 0
    for j in range(n):
        residual = A[:, j].copy()

        for i in range(rank):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            residual -= R[i, j] * Q[:, i]

        norm_res = np.linalg.norm(residual)
        if norm_res > epsilon:
            rank=rank+1
            R[rank-1, j] = norm_res
            Q[:, rank-1:rank] = residual / norm_res
            for k in range (j+1,n):
                R[rank-1,k]=np.transpose(Q[:,rank-1:rank],A[:,k:k+1])
    Q=Q[:,0:rank]
    R=R[0:rank,0:rank]
    return Q[:, :rank], R[:rank, :], rank

def min_norm_LSQR(A, b):
    Q, R, _ = QR_modified(A)
    b = Q.T @ b
    Q1, R1, _ = QR_modified(R.T)
    z = forward(R1.T, b)
    x = Q1 @ z
    return x
