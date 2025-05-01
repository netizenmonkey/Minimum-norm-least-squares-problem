import numpy as np

def backward(U, z):
    n = U.shape[0]
    
    x = np.zeros((n, 1))

    for i in range(n - 2, -1, -1):
        sum = 0
        for j in range(i + 1, n):
            sum = sum + U[i, j] * x[j]
        x[i] = (z[i] - sum) / U[i, i]

    return x

def forward(L, b):
    n = L.shape[0]
    z = np.zeros((n, 1))
    z[0] = b[0] / L[0, 0]

    for i in range(1, n): #2...n
        sum = 0  
        for j in range(i):  
            sum += L[i, j] * z[j] 
        z[i] = (b[i] - sum) / L[i, i]  

    return z
