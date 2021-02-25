import numpy as np

def constitutive_matrix(stress, seffmax, s0, nexp, E, v):
    ## initialize tangent stiffness matrix
    D = np.zeros(12,3)
    ## compute the elastic stiffness matrix
    C = (E/((1+v)*(1-2*v)))*[[1-v, v, v, 0, 0, 0],
                                [v, 1-v, v, 0, 0, 0],
                                [v, v, 1-v, 0, 0, 0],
                                [0, 0, 0, 1/2*(1-2*v), 0, 0],
                                [0, 0, 0, 0, 1/2*(1-2*v), 0],
                                [0, 0, 0, 0, 0, 1/2*(1-2*v)]]
    ## Compute von Mises effective stress at each integration point
    seff = np.sqrt(((stress[:,1]-stress[:,2])**2 + stress[:,1]**2 + stress[:,1]**2 + 6*stress[:,3]**2)/2)
    for ii in range(4): # loop over all integration points
        if seff[ii] < seffmax[ii]:
            Dep = C # integration point is still elastic
        else:
            seffmax[ii] = seff[ii] # Update the yield strength if the integration point has yielded
            strain = s0/E*(seff[ii]/s0)**nexp
            Et = E/nexp*(E*strain/s0)^(1/nexp-1)
            h = ((9/(4*seff[ii]**2))*(1/Et-1/E))

            dev_stress = np.zeros([6,1])
            dev_stress[1] = stress[ii,1] - (stress[ii,1]+stress[ii-2])/3
            dev_stress[2] = stress[ii,2] - (stress[ii,1]+stress[ii-2])/3
            dev_stress[3] = 0 - (stress[ii,1]+stress[ii-2])/3
            dev_stress[4] = stress[ii,3]

            Cp = h*C*dev_stress*np.transpose(dev_stress)*C/(1+h*np.transpose(dev_stress)*C*dev_stress)
            Dep = C-Cp

    Dinv = np.linalg.inv(Dep)
    Dinv[[3,5,6],:] = []
    Dinv[:,[3,5,6]] = []
    Dep2D = np.linalg.inv(Dinv)
    D[ii*3-3:ii*3-1,:] = Dep2D

    return D, seffmax
