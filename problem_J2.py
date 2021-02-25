import numpy as np
import math

# local
from constitutive_matrix import constitutive_matrix
from plane_iso4ke import plane_iso4ke
from solveq import solveq
from plane_iso4s import plane_iso4s

## load the gemoetry and mesh
# [node,elemnode]=meshhole;
# node = np.zeros([100,3])
# elemnode = np.zeros([100,3])
node = np.loadtxt("nodeHole.txt", delimiter=",", unpack=False)
elemnode = np.loadtxt("meshHole.txt", delimiter=",", unpack=False)


## several parameters
L = np.max(node[:,1])               # length of the geometry
W = np.max(node[:,2])               # width of the geometry
t = 1e-3                            # thickness of the geometry
du = 0.2e-4                         # incremental displacement load
u = [0]                              # total displacement
f = [0]                              # total force

## material properties
E = 9.87e8
v = 0.3
s0 = 1e9
nexp = 5

Nel = elemnode.shape([0])             # number of elements
ndof = 2 * node.shape([0])

U = np.zeros(ndof,1)                # displacement vector for each DOF
Q = np.zeros(ndof,1)                # internal force vector for each DOF
stress = np.zeros(4, 3, Nel)        # stress matrix [4 integration points, 3 components (sig_xx, sig_yy, tau_xy), in all Nel elements]
strain = np.zeros(4, 3, Nel)        # strain matrix [4 integration points, 3 components (eps_xx, eps_yy, gam_xy), in all Nel elements]
seffmax = s0 * np.ones(4, 1, Nel)   # yield surface, onset at s0

## Allocate global matrix of all integration points stiffness's
Dep = np.zeros(12,3,Nel)            # 4 inegration points/element -> 4x(3x3) for each element

## Boundary nodes B1, B3 and their dofs b1x, b1y, b3x, b3y
B1 = np.where(node[:,1]==0); b1x=2*B1-1; b1y=2*B1
B3 = np.where(node[:,1]==L); b3x=2*B3-1; b3y=2*B3

step = 0
while u[step] < L/50:                     # Mainloop. We load in xx steps
    step += 1
    # Boundary conditions
    BC = np.array([[b1x, 0*b1x], [b1y(1), 0], [b3x, 0*b3x+du]])
    K = np.zeros(ndof,ndof)         # Assemble the global stiffness matrix K
    for el in range(Nel):
        [Dep[:,:,el],seffmax[:,:,el]] = constitutive_matrix(stress[:,:,el], seffmax[:,:,el], s0, nexp, E, v)
        n = elemnode[el,:]; ex = np.transpose(node[n,1]); ey = np.transpose(node[n,2])
        Ke = plane_iso4ke(ex, ey, t, Dep[:,:,el])
        elemdof = np.array([n(1)*2-1, n(1)*2, n(2)*2-1, n(2)*2, n(3)*2-1, n(3)*2, n(4)*2-1, n(4)*2])
        K[elemdof,elemdof] = K[elemdof,elemdof] + Ke

    # internal force vector
    dF = np.zeros(ndof,1)           # No internal tractions are given
    [dU,dQ] = solveq(K, dF, BC)
    # when load increment is solved: extract incremental stresses, strains, integration points
    dstress = np.zeros([1, 3, Nel])
    dstrain = np.zeros([1, 3, Nel])
    intpoints = np.zeros([1, 2, Nel])
    for el in range(Nel):
        n = elemnode[el,:]; ex = np.transpose(node[n,1]); ey = np.transpose(node[n,2])
        elemdof = np.array([n(1)*2-1, n(1)*2, n(2)*2-1, n(2)*2, n(3)*2-1, n(3)*2, n(4)*2-1, n(4)*2])
        ed = np.transpose(dU[elemdof])
        [dstress[:,:,el], dstrain[:,:,el], intpoints[:,:,el]] = plane_iso4s(ex, ey, Dep[:,:,el], ed)

    # Update stresses and strains
    stress = stress + dstress
    strain = strain + dstrain
    # Compute von Mises effective stress
    seff = math.sqrt(stress[:,1,:]**2+  stress[:,2,:]**2 - stress[:,1,:]*stress[:,2,:] + 3*stress[:,3,:]**2)
    # Update globale load and reaction forces
    u[step+1] = u[step] + np.mean(dU[b3x])
    f[step+1] = f[step] + np.sum(dQ[b3x])

    # plot figures
    # figure(1)
    # subplot(2,2,1), cla, plot(u/L,f/(t*W)/s0,'bo-'), grid on, xlabel('u/L','fontsize',14); ylabel('[f/Wt]/s0','fontsize',14);
    # subplot(2,2,3), cla, plot_yielded_points(node,elemnode,intpoints,seffmax,s0)
    # subplot(2,2,4), cla, plot_deformed_mesh(node,elemnode,U)
    # subplot(2,2,2), cla, plot_plastic_zone(node,elemnode,intpoints,seff,s0)

