import numpy as np

def simpleQR(A):
    m = A.shape[0]
    n = A.shape[1]

    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    R[0,0] = np.linalg.norm(A[:,0:1])
    Q[:,0:1] = A[:,0:1] / R[0,0]

    for k in range(n):
        residual = A[:, k].copy()
        
        for i in range(k):
            R[i, k] = np.matmul(np.transpose(Q[:,i:i+1],A[:,k:k+1]))
            vector_proj = R[i,k]*Q[:,i:i+1]
            residual = A[:,k:k+1]-vector_proj

        R[k, k] = np.linalg.norm(residual)
        Q[:, k] = residual / R[k, k]

    return Q, R
