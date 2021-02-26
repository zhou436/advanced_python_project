import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
# import calfem.vis as cfv

# local
from constitutive_matrix import constitutive_matrix
from plane_iso4ke import plane_iso4ke
from solveq import solveq
from plane_iso4s import plane_iso4s

## load the gemoetry and mesh
node = np.loadtxt("nodeHole.txt", comments='#', delimiter=",", unpack=False)
elemnode = np.loadtxt("meshHole.txt", delimiter=",", unpack=False, dtype="int")
elemnode = elemnode - 1

## several parameters
L = np.max(node[:,0])               # length of the geometry
W = np.max(node[:,1])               # width of the geometry
t = 1e-3                            # thickness of the geometry
du = 0.2e-3                         # incremental displacement load
u = [0]                              # total displacement
f = [0]                              # total force

## material properties
E = 210e9
v = 0.3
s0 = 1e9
nexp = 5

Nel = elemnode.shape[0]            # number of elements
ndof = 2 * node.shape[0]

U = np.zeros([ndof,1])                # displacement vector for each DOF
Q = np.zeros([ndof,1])                # internal force vector for each DOF
stress = np.zeros([4, 3, Nel])        # stress matrix [4 integration points, 3 components (sig_xx, sig_yy, tau_xy), in all Nel elements]
strain = np.zeros([4, 3, Nel])        # strain matrix [4 integration points, 3 components (eps_xx, eps_yy, gam_xy), in all Nel elements]
seffmax = s0 * np.ones([4, 1, Nel])   # yield surface, onset at s0

## Allocate global matrix of all integration points stiffness's
Dep = np.zeros([12,3,Nel])            # 4 inegration points/element -> 4x(3x3) for each element

## Boundary nodes B1, B3 and their dofs b1x, b1y, b3x, b3y
B1 = np.where(node[:,0]==0)[0]; b1x=2*B1; b1y=2*B1+1
B3 = np.where(node[:,0]==L)[0]; b3x=2*B3; b3y=2*B3+1

step = 0
K = np.zeros([ndof,ndof])         # Assemble the global stiffness matrix K
while u[step] < L/50:                     # Mainloop. We load in xx steps
    # Boundary conditions
    BC = np.transpose(np.hstack((np.vstack((b1x, 0*b1x)), np.vstack((b1y[0], 0)), np.vstack((b3x, 0*b3x+du)))))
    for el in range(Nel):
        [Dep[:,:,el],seffmax[:,:,el]] = constitutive_matrix(stress[:,:,el], seffmax[:,:,el], s0, nexp, E, v)
        n = elemnode[el,:]; ex = np.transpose(node[n,0]); ey = np.transpose(node[n,1])
        Ke = plane_iso4ke(ex, ey, t, Dep[:,:,el])
        elemdof = np.array([n[0]*2, n[0]*2+1, n[1]*2, n[1]*2+1, n[2]*2, n[2]*2+1, n[3]*2, n[3]*2+1])
        for ii in range(Ke.shape[0]):
            for jj in range(Ke.shape[0]):
                K[elemdof[ii],elemdof[jj]] = K[elemdof[ii],elemdof[jj]] + Ke[ii,jj]

    # internal force vector
    dF = np.zeros([ndof,1])           # No internal tractions are given
    [dU,dQ] = solveq(K, dF, BC)
    # when load increment is solved: extract incremental stresses, strains, integration points
    dstress = np.zeros([4, 3, Nel])
    dstrain = np.zeros([4, 3, Nel])
    intpoints = np.zeros([4, 2, Nel])
    for el in range(Nel):
        n = elemnode[el,:]; ex = np.transpose(node[n,0]); ey = np.transpose(node[n,1])
        elemdof = np.array([n[0]*2, n[0]*2+1, n[1]*2, n[1]*2+1, n[2]*2, n[2]*2+1, n[3]*2, n[3]*2+1])
        ed = np.transpose(dU[elemdof])
        [dstress[:,:,el], dstrain[:,:,el], intpoints[:,:,el]] = plane_iso4s(ex, ey, Dep[:,:,el], ed)

    # Update stresses and strains
    stress = stress + dstress
    strain = strain + dstrain
    # Compute von Mises effective stress
    seff = np.sqrt(stress[:,0,:]**2+  stress[:,1,:]**2 - stress[:,0,:]*stress[:,1,:] + 3*stress[:,2,:]**2)
    # Update globale load and reaction forces

    u.append(u[step]+np.mean(dU[b3x]))
    f.append(np.sum(dQ[b3x]))
    U = U + np.expand_dims(dU, axis=1)
    Q = Q + np.expand_dims(dQ, axis=1)
    step += 1

# FEM curve
plt.figure()
plt.plot(u/L,f/(t*W)/s0,'bo-')
plt.xlabel('u/L')
plt.ylabel('[f/Wt]/s0')
plt.show(block=False)
plt.title('Strain-Stress Curve')
# FEM deformation

fig, (ax0, ax1) = plt.subplots(nrows=2,figsize=(8,8))
x = np.expand_dims(node[:,0], axis=1)
y = np.expand_dims(node[:,1], axis=1)
ax0.scatter(x, y)
ax0.set_title('Original Geometry')
scale_factor = 6
x_expand = np.expand_dims(node[:,0], axis=1) + U[0:-1:2]*scale_factor
y_expand = np.expand_dims(node[:,1], axis=1) + U[1::2]*scale_factor
ax1.scatter(x_expand, y_expand)
ax1.set_title('Deformed Geometry')
# plt.axis('equal')
ax0.set_xlim([-0.005,0.070])
ax0.set_ylim([-0.005,0.025])
ax1.set_xlim([-0.005,0.070])
ax1.set_ylim([-0.005,0.025])
plt.show()
