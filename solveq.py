import numpy as np

def solveq(K, f, bc):
    nd = K.shape[0]
    fdof = np.arange(1,nd+1,1)-1
    
    d = np.zeros((fdof).shape)
    Q = np.zeros((fdof).shape)
    
    pdof = bc[:,0]
    pdof = pdof.reshape((pdof.size,1)).astype(int)
    dp = bc[:,1]
    dp = dp.reshape((dp.size,1))
    fdof = np.delete(fdof,pdof)
    s = np.linalg.inv(K[fdof,:][:,fdof])@(f[fdof]-np.transpose(K[fdof,pdof])@dp)
    d[pdof] = dp
    fdof = np.array(fdof)
    d[fdof] = s[:,0]
    Q=K@d-f[0]

    return d, Q
