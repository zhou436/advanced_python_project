import numpy as np

def plane_iso4ke(ex, ey, t, Dep):
    ngp = 4
    ## Gauss points
    g1 = 1/np.sqrt(3); w1 = 1
    gp = np.zeros([4,2]); w = np.zeros([4,2])
    gp[:,1] = np.transpose(np.array([-g1, g1, -g1, g1]));  gp[:,2] = np.transpose(np.array([-g1, -g1, g1, g1]))
    w[:,1] = np.transpose(np.array([w1, w1, w1, w1]));   w[:,2] = np.transpose(np.array([w1, w1, w1, w1]))
    wp = w[:,1]*w[:,2]
    xsi = gp[:,1];  eta=gp[:,2];  r2=  ngp*2
    ## shape functions
    N = np.zeros([4,4])
    N[:,1] = (1-xsi)*(1-eta)/4;  N[:,2] = (1+xsi)*(1-eta)/4
    N[:,3] = (1+xsi)*(1+eta)/4;  N[:,4] = (1-xsi)*(1+eta)/4
    dNr = np.zeros([8,4])
    dNr[1:2:r2,1] = -(1-eta)/4;     dNr[1:2:r2,2] = (1-eta)/4
    dNr[1:2:r2,3] = (1+eta)/4;     dNr[1:2:r2,4] = -(1+eta)/4
    dNr[2:2:r2+1,1] = -(1-xsi)/4;   dNr[2:2:r2+1,2] = -(1+xsi)/4
    dNr[2:2:r2+1,3] = (1+xsi)/4;   dNr[2:2:r2+1,4] = (1-xsi)/4

    Ke = np.zeros([8,8])
    JT = dNr*np.array([ex,ey])

    for ii in range(ngp):
        indx=[[2*ii-1],[2*ii]]
        detJ = np.linalg.det(JT[indx,:])
        if detJ < 10*2**-52:
            print("Jacobideterminant equal or less than zero!")
        
        JTinv = np.linalg.inv(JT[indx,:])
        dNx = JTinv*dNr[indx,:]

        B = np.zeros([3,8])
        B[1,1:2:8-1] = dNx[1,:]
        B[2,2:2:8] = dNx[2,:]
        B[3,1:2:8-1] = dNx[2,:]
        B[3,2:2:8] = dNx[1,:]

        D = Dep[ii*3-2:ii*3,:]
        Ke = Ke+np.transpose(B)*D*B*detJ*wp[ii]*t

    return Ke