from qrdecomp import simpleQR
from substitution import backward

def LSQR(A, b):
    Q, R = simpleQR(A)
    b = Q.T @ b
    x = backward(R, b)
    return x
