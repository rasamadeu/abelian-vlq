# This module contains the function extract_obs that compute the physical observables
# (quark masses + 3 mixing angles of CKM matrix + CP-violating delta phase of CKM) for
# a given pair of texture zeros for the quark mass matrices (M_u, M_d). It also computes
# the Jarlskog invariant.

import numpy as np
import numpy.matlib
import numpy.linalg
import cmath
import math
import pdb
from numba import njit  # import pdb


@njit
def compute_singular_values(M, rotation_option):

    if rotation_option == "left":
        # Compute singular values of M and U_L
        D, U = np.linalg.eig(np.nan_to_num(M @ M.conj().T))

    if rotation_option == "right":
        # Compute singular values of M and U_R
        D, U = np.linalg.eig(np.nan_to_num(M.conj().T @ M))

    # Sort the eigenvalues from lowest to highest
    D = np.sqrt(D)
    index = D.argsort()
    D = D[index]
    U = U[:, index]

    return D, U


@njit
def extract_obs(M_u, M_d=[]):

    D_u, U_L_u = compute_singular_values(M_u, "left")
    D_d, U_L_d = compute_singular_values(M_d, "left")

    # Compute the matrix V = A_u_L^{dagger} A_d_L
    A_u_L = np.empty((3, np.shape(M_u)[0]), dtype=np.complex128)
    A_d_L = np.empty((3, np.shape(M_d)[0]), dtype=np.complex128)

    for i in range(3):
        A_u_L[i] = U_L_u[i]
        A_d_L[i] = U_L_d[i]

    V = A_u_L.conj().T @ A_d_L

    # Obtain the mixing angles and Dirac Phase of V_CKM
    # NOTE: Remember that the matrix V_CKM obtained from V may not
    # be in the PDG standard parameterization. Hence, the float "delta"
    # may be zero but V_CKM can still have a non-zero CP violating phase.
    # A better measure of CP violation is the Jarlskog invariant.

    delta = -cmath.phase(V[0, 2])
    Jarlskog = (V[0, 1] * V[1, 2] * np.conj(V[0, 2]) * np.conj(V[1, 1])).imag

    return D_u, D_d, delta, Jarlskog, V


def main():

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

    M_u = np.array(
        [[0, 0, 7186.3256, 1269.73113],
         [0, 83394.7485, 0, 118.671681],
         [552.774389, 0, 172539.517, 0],
         [0, 1964314, 0, 0]]

    )

    M_d = np.array(
        [[0, 0, 91.5163108],
         [0, 4.77702501, -0.0905301173-19.18138j],
         [4180, 0, 0]]
    )

#   M_u = np.array(
#       [[0, 0, 1265.57892, 7161.12073],
#        [0, 616162.666, 0, 172939.919],
#        [2.16959375, 0, 117.067567, 0],
#        [0, 9062696.07, 0, 0]]
#   )

#   M_d = np.array(
#       [[0, 0, 91.4853216],
#        [0, 4180, 0],
#        [4.78434215, 0, -0.280434011 - 19.18715j]]
#   )

    M_u = np.array(
        [[0, 0, 32721.3707, 169555.030],
         [0, 78914.1226, 0, 1505.34039],
         [9.62166554, 0, 1260.32296, 0],
         [0, 2733123.53, 0, 0]]
    )

    M_d = np.array(
        [[0, 0, 4176.31609],
         [0, 4.67, 0],
         [93.48239, 0, 160.35214-71.10366 * 1j]]
    )

    D_u, U_L_u = compute_singular_values(M_u, "left")
    D_u, U_R_u = compute_singular_values(M_u, "right")
    D_d, U_L_d = compute_singular_values(M_d, "left")
    D_d, U_R_d = compute_singular_values(M_d, "right")

    print(M_u / D_u[2] * 172.69)
    print(M_d)
    print("U_L_u:")
    print(U_L_u)
    print("U_R_u:")
    print(U_R_u)
    print("U_L_d:")
    print(U_L_d)
    print("U_R_d:")
    print(U_R_d)
    print(D_d)
    print(D_u / D_u[2] * 172.69)

    # Compute the matrix V = A_u_L^{dagger} A_d_L
    A_u_L = np.empty([3, np.shape(M_u)[0]], dtype=complex)
    A_d_L = np.empty([3, np.shape(M_d)[0]], dtype=complex)
    for i in range(3):
        A_u_L[i] = U_L_u[i]
        A_d_L[i] = U_L_d[i]

    print(A_u_L.conj().T @ A_d_L)


if __name__ == "__main__":
    main()
