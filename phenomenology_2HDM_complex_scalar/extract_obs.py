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
    print(U[3,0].conj() * U[3,1])

    return D, U


def extract_obs(M_u, M_d=[]):

    D_u, U_L_u = compute_singular_values(M_u, "left")

    if not np.any(M_d):
        return D_u
    else:

        D_d, U_L_d = compute_singular_values(M_d, "left")

        # Compute the matrix V = A_u_L^{dagger} A_d_L
        A_u_L = np.empty([3, np.shape(M_u)[0]], dtype=complex)
        A_d_L = np.empty([3, np.shape(M_d)[0]], dtype=complex)

        for i in range(3):
            A_u_L[i] = U_L_u[i]
            A_d_L[i] = U_L_d[i]

        V = np.matrix(A_u_L).conj().T @ A_d_L

        # Obtain the mixing angles and Dirac Phase of V_CKM
        # NOTE: Remember that the matrix V_CKM obtained from V may not
        # be in the PDG standard parameterization. Hence, the float "delta"
        # may be zero but V_CKM can still have a non-zero CP violating phase.
        # A better measure of CP violation is the Jarlskog invariant.

        if(V[0, 1] == 0 or V[0, 0] == 0):
            sin_theta_12 = 0
        else:
            sin_theta_12 = math.sin(math.atan(abs(V[0, 1]) / abs(V[0, 0])))

        sin_theta_13 = abs(V[0, 2])

        if(V[1, 2] == 0 or V[2, 2] == 0):
            sin_theta_23 = 0
        else:
            sin_theta_23 = math.sin(math.atan(abs(V[1, 2]) / abs(V[2, 2])))

        delta = -cmath.phase(V[0, 2])
        Jarlskog = (V[0, 1] * V[1, 2] * np.conj(V[0, 2]) * np.conj(V[1, 1])).imag

        return D_u, D_d, sin_theta_12, sin_theta_13, sin_theta_23, delta, Jarlskog, V


def extract_obs_m_decoupled(M_u, M_d=[], decoupled = 1):

    D_u, U_L_u = compute_singular_values(M_u, "left")

    if not np.any(M_d):
        return D_u
    else:

        A_u_L = np.empty([3, np.shape(M_u)[0]], dtype=complex)
        for i in range(3):
            A_u_L[i] = U_L_u[i]

        if M_d[2, 2] == 0:
            M_d_decoupled = M_d[0:2, 1:3]
            D_d, U_L_d = compute_singular_values(M_d_decoupled, "left")
            if decoupled == 1:
                D_d = np.array([M_d[2, 0], D_d[0], D_d[1]])
                A_d_L = np.array([[0, U_L_d[0, 0], U_L_d[0, 1]], [0, U_L_d[1, 0], U_L_d[1, 1]], [1, 0, 0]])
            if decoupled == 2:
                D_d = np.array([D_d[0], M_d[2, 0], D_d[1]])
                A_d_L = np.array([[U_L_d[0, 0], 0, U_L_d[0, 1]], [U_L_d[1, 0], 0, U_L_d[1, 1]], [0, 1, 0]])
            if decoupled == 3:
                D_d = np.array([D_d[0], D_d[1], M_d[2, 0]])
                A_d_L = np.array([[U_L_d[0, 0], U_L_d[0, 1], 0], [U_L_d[1, 0], U_L_d[1, 1], 0], [0, 0, 1]])
        else:
            M_d_decoupled = M_d[0:3:2, 0:3:2]
            D_d, U_L_d = compute_singular_values(M_d_decoupled, "left")
            if decoupled == 1:
                D_d = np.array([M_d[1, 1], D_d[0], D_d[1]])
                A_d_L = np.array([[0, U_L_d[0, 0], U_L_d[0, 1]], [1, 0, 0], [0, U_L_d[1, 0], U_L_d[1, 1]]])
            if decoupled == 2:
                D_d = np.array([D_d[0], M_d[1, 1], D_d[1]])
                A_d_L = np.array([[U_L_d[0, 0], 0, U_L_d[0, 1]], [0, 1, 0], [U_L_d[1, 0], 0, U_L_d[1, 1]]])
            if decoupled == 3:
                D_d = np.array([D_d[0], D_d[1], M_d[1, 1]])
                A_d_L = np.array([[U_L_d[0, 0], U_L_d[0, 1], 0], [0, 0, 1], [U_L_d[1, 0], U_L_d[1, 1], 0]])

        # print(A_d_L)
        # print(np.sqrt(A_d_L.conj().T @ M_d @ M_d.conj().T @ A_d_L))
        V = np.matrix(A_u_L).conj().T @ A_d_L

        # Obtain the mixing angles and Dirac Phase of V_CKM
        # NOTE: Remember that the matrix V_CKM obtained from V may not
        # be in the PDG standard parameterization. Hence, the float "delta"
        # may be zero but V_CKM can still have a non-zero CP violating phase.
        # A better measure of CP violation is the Jarlskog invariant.

        if(V[0, 1] == 0 or V[0, 0] == 0):
            sin_theta_12 = 0
        else:
            sin_theta_12 = math.sin(math.atan(abs(V[0, 1]) / abs(V[0, 0])))

        sin_theta_13 = abs(V[0, 2])

        if(V[1, 2] == 0 or V[2, 2] == 0):
            sin_theta_23 = 0
        else:
            sin_theta_23 = math.sin(math.atan(abs(V[1, 2]) / abs(V[2, 2])))

        delta = -cmath.phase(V[0, 2])
        Jarlskog = (V[0, 1] * V[1, 2] * np.conj(V[0, 2]) * np.conj(V[1, 1])).imag

        return D_u, D_d, sin_theta_12, sin_theta_13, sin_theta_23, delta, Jarlskog, V


def main():

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

    M_u = np.array([[ 0.00000000e+00+0.j,  0.00000000e+00+0.j,  1.68238182e-01+0.j,  2.93436509e-02+0.j],
                    [ 0.00000000e+00+0.j, -5.66302428e-02+0.j,  0.00000000e+00+0.j, -2.86543910e-04+0.j],
                    [ 9.53763318e-06+0.j,  0.00000000e+00+0.j, -7.13503621e-03+0.j,  0.00000000e+00+0.j],
                    [ 0.00000000e+00+0.j,  1.38451276e+00+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j]])
    M_u = M_u * 1010340.1602922176

    M_d = np.array([[   0.         +0.j,            0.         +0.j,         4180.44525982 +0.j        ],
                    [   0.         +0.j,           -4.6709246  +0.j,          -33.89816159-14.87595132j],
                    [  93.4        +0.j,            0.         +0.j,            0.         +0.j        ]])

    D_u, U_L_u = compute_singular_values(M_u, "left")
    D_u, U_R_u = compute_singular_values(M_u, "right")
    D_d, U_L_d = compute_singular_values(M_d, "left")
    D_d, U_R_d = compute_singular_values(M_d, "right")

    print("U_L_u:")
    print(U_L_u)
    print("U_R_u:")
    print(U_R_u)
    print("U_L_d:")
    print(U_L_d)
    print("U_R_d:")
    print(U_R_d)
    print(D_d)
    #U_L_d = U_L_d @ np.array([[-1, 0, 0],[0, 1, 0],[0, 0, 1]])
    print(U_L_d.conj().T @ M_d @ U_R_d)
    print(D_u)
    print(U_L_u.conj().T @ M_u @ U_R_u)

    # Compute the matrix V = A_u_L^{dagger} A_d_L
    A_u_L = np.empty([3, np.shape(M_u)[0]], dtype=complex)
    A_d_L = np.empty([3, np.shape(M_d)[0]], dtype=complex)
    for i in range(3):
        A_u_L[i] = U_L_u[i]
        A_d_L[i] = U_L_d[i]

    print(A_u_L.conj().T @ A_d_L)


if __name__ == "__main__":
    main()
