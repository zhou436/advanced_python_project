import numpy as np

def solveq(K, f, bc):
    nd = K.shape(0)
    fdof = np.transpose(np.arange(1,nd,1))
    d = np.zeros((fdof).shape)
    Q = np.zeros((fdof).shape)
    
    pdof = bc[:,1]
    dp = bc[:,2]
    fdof[pdof] = []
    s = np.linalg.inv(K[fdof,fdof])*(f[fdof]-K[fdof,pdof]*dp)
    d[pdof] = dp
    d[fdof] = s
    Q=K*d-f

    return d, Q
