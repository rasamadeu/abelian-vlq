import texture_zeros_v4 as tz
import extract_obs as ext
import numpy as np
import numpy.matlib
import math
from iminuit import Minuit
import sys
import time
import pdb
# Physical parameters taken from PDG (Revision of 2022)

MeV = 1e6
GeV = 1e9
TeV = 1e12
MASS_U = 2.16 * MeV
UPPER_SIGMA_MASS_U = 0.49 * MeV
LOWER_SIGMA_MASS_U = 0.26 * MeV

MASS_D = 4.67 * MeV
UPPER_SIGMA_MASS_D = 0.48 * MeV
LOWER_SIGMA_MASS_D = 0.17 * MeV

MASS_S = 93.4 * MeV
UPPER_SIGMA_MASS_S = 8.6 * MeV
LOWER_SIGMA_MASS_S = 3.4 * MeV

MASS_C = 1.27 * GeV
SIGMA_MASS_C = 0.02 * GeV

MASS_B = 4.18 * GeV
UPPER_SIGMA_MASS_B = 0.03 * GeV
LOWER_SIGMA_MASS_B = 0.02 * GeV

MASS_T = 172.69 * GeV
UPPER_SIGMA_MASS_T = 0.3 * GeV
LOWER_SIGMA_MASS_T = 0.3 * GeV

MASS_VLQ = 1.5 * TeV
SIGMA_MASS_VLQ = 0.15 * TeV

SIN_THETA_12 = 0.22500
SIGMA_SIN_THETA_12 = 0.00067

SIN_THETA_13 = 0.00369
SIGMA_SIN_THETA_13 = 0.00011

SIN_THETA_23 = 0.04182
UPPER_SIGMA_SIN_THETA_23 = 0.00085
LOWER_SIGMA_SIN_THETA_23 = 0.00074

DIRAC_DELTA = 1.144
SIGMA_DIRAC_DELTA = 0.027

JARLSKOG = 3.08e-5
UPPER_SIGMA_JARLSKOG = 0.15e-5
LOWER_SIGMA_JARLSKOG = 0.13e-5

#v = 174.104 * GeV
#
#MASS_U = 7.4 * 1e-6 * v
#UPPER_SIGMA_MASS_U = 1.5 * 1e-6 * v
#LOWER_SIGMA_MASS_U = 3 * 1e-6 * v
#
#MASS_D = 1.58 * 1e-5 * v
#UPPER_SIGMA_MASS_D = 0.23 * 1e-5 * v
#LOWER_SIGMA_MASS_D = 0.10 * 1e-5 * v
#
#MASS_S = 3.12 * 1e-4 * v
#UPPER_SIGMA_MASS_S = 0.17 * 1e-4 * v
#LOWER_SIGMA_MASS_S = 0.16 * 1e-4 * v
#
#MASS_C = 3.6 * 1e-3 * v
#SIGMA_MASS_C = 0.11 * 1e-3 * v
#
#MASS_B = 1.639 * 1e-2 * v
#UPPER_SIGMA_MASS_B = 0.015 * 1e-2 * v
#LOWER_SIGMA_MASS_B = 0.015 * 1e-2 * v
#
#MASS_T = 9.861 * 1e-1 * v
#UPPER_SIGMA_MASS_T = 0.086 * 1e-1 * v
#LOWER_SIGMA_MASS_T = 0.087 * 1e-1 * v
#
#SIN_THETA_12 = 0.22735
#SIGMA_SIN_THETA_12 = 0.00072
#
#SIN_THETA_13 = 3.64 * 1e-3
#SIGMA_SIN_THETA_13 = 0.13 * 1e-3
#
#SIN_THETA_23 = 4.208 * 1e-2
#UPPER_SIGMA_SIN_THETA_23 = 0.064 * 1e-2
#LOWER_SIGMA_SIN_THETA_23 = 0.064 * 1e-2
#
#DIRAC_DELTA = 1.208
#SIGMA_DIRAC_DELTA = 0.05
#
#JARLSKOG = (math.cos(SIN_THETA_12) * math.sin(SIN_THETA_12)
#            * math.cos(SIN_THETA_23) * math.sin(SIN_THETA_23)
#            * math.cos(SIN_THETA_13) ** 2 * math.sin(SIN_THETA_13)
#            * math.sin(DIRAC_DELTA))
#SIGMA_JARLSKOG = 0.05 * JARLSKOG

MAX_CHI_SQUARE = 9
# 3 up quark masses + 3 down quark masses + 3 mixing angles + 1 CP-violating phase
N_OBSERVABLES_SM = 11


def compute_chi_square(D_u, D_d, sin_theta_12, sin_theta_13, sin_theta_23, delta, Jarlskog):

    if D_u[0].real - MASS_U > 0:
        chi_square_m_u = ((D_u[0].real - MASS_U) / UPPER_SIGMA_MASS_U)**2
    else:
        chi_square_m_u = ((D_u[0].real - MASS_U) / LOWER_SIGMA_MASS_U)**2

    chi_square_m_c = ((D_u[1].real - MASS_C) / SIGMA_MASS_C)**2

    if D_u[2].real - MASS_T > 0:
        chi_square_m_t = ((D_u[2].real - MASS_T) / UPPER_SIGMA_MASS_T)**2
    else:
        chi_square_m_t = ((D_u[2].real - MASS_T) / LOWER_SIGMA_MASS_T)**2

    if D_d[0].real - MASS_D > 0:
        chi_square_m_d = ((D_d[0].real - MASS_D) / UPPER_SIGMA_MASS_D)**2
    else:
        chi_square_m_d = ((D_d[0].real - MASS_D) / LOWER_SIGMA_MASS_D)**2

    if D_d[1].real - MASS_S > 0:
        chi_square_m_s = ((D_d[1].real - MASS_S) / UPPER_SIGMA_MASS_S)**2
    else:
        chi_square_m_s = ((D_d[1].real - MASS_S) / LOWER_SIGMA_MASS_S)**2

    if D_d[2].real - MASS_B > 0:
        chi_square_m_b = ((D_d[2].real - MASS_B) / UPPER_SIGMA_MASS_B)**2
    else:
        chi_square_m_b = ((D_d[2].real - MASS_B) / LOWER_SIGMA_MASS_B)**2

    chi_square_m_VLQ = ((D_u[3].real - MASS_VLQ) / SIGMA_MASS_VLQ)**2

    try:
        chi_square_sin_theta_12 = ((math.asin(sin_theta_12) - SIN_THETA_12) / SIGMA_SIN_THETA_12)**2
    except ValueError:
        print(f"ERROR SIN {sin_theta_12}")
        chi_square_sin_theta_12 = 1e9

    try:
        chi_square_sin_theta_13 = ((math.asin(sin_theta_13) - SIN_THETA_13) / SIGMA_SIN_THETA_13)**2
    except ValueError:
        print(f"ERROR SIN {sin_theta_13}")
        chi_square_sin_theta_13 = 1e9

    if math.asin(sin_theta_23) - SIN_THETA_23 > 0:
        try:
            chi_square_sin_theta_23 = (
                (math.asin(sin_theta_23) - SIN_THETA_23) / UPPER_SIGMA_SIN_THETA_23)**2
        except ValueError:
            print(f"ERROR SIN {sin_theta_23}")
            chi_square_sin_theta_23 = 1e9
    else:
        try:
            chi_square_sin_theta_23 = (
                (math.asin(sin_theta_23) - SIN_THETA_23) / LOWER_SIGMA_SIN_THETA_23)**2
        except ValueError:
            print(f"ERROR SIN {sin_theta_23}")
            chi_square_sin_theta_23 = 1e9

    chi_square_delta = ((delta - DIRAC_DELTA) / SIGMA_DIRAC_DELTA)**2

    if Jarlskog - JARLSKOG > 0:
        chi_square_Jarlskog = ((Jarlskog - JARLSKOG) / UPPER_SIGMA_JARLSKOG)**2
    else:
        chi_square_Jarlskog = ((Jarlskog - JARLSKOG) / LOWER_SIGMA_JARLSKOG)**2

    return (chi_square_m_u, chi_square_m_c, chi_square_m_t, chi_square_m_VLQ,
            chi_square_m_d, chi_square_m_s, chi_square_m_b,
            chi_square_sin_theta_12, chi_square_sin_theta_13,
            chi_square_sin_theta_23, chi_square_delta, chi_square_Jarlskog)



def least_squares(M_u_0_0_re, M_u_0_0_im, M_u_0_1_re, M_u_0_1_im, M_u_0_2_re, M_u_0_2_im, M_u_0_3_re, M_u_0_3_im,
                  M_u_1_0_re, M_u_1_0_im, M_u_1_1_re, M_u_1_1_im, M_u_1_2_re, M_u_1_2_im, M_u_1_3_re, M_u_1_3_im,
                  M_u_2_0_re, M_u_2_0_im, M_u_2_1_re, M_u_2_1_im, M_u_2_2_re, M_u_2_2_im, M_u_2_3_re, M_u_2_3_im,
                  M_u_3_0_re, M_u_3_0_im, M_u_3_1_re, M_u_3_1_im, M_u_3_2_re, M_u_3_2_im, M_u_3_3_re, M_u_3_3_im,
                  M_d_0_0_re, M_d_0_0_im, M_d_0_1_re, M_d_0_1_im, M_d_0_2_re, M_d_0_2_im,
                  M_d_1_0_re, M_d_1_0_im, M_d_1_1_re, M_d_1_1_im, M_d_1_2_re, M_d_1_2_im,
                  M_d_2_0_re, M_d_2_0_im, M_d_2_1_re, M_d_2_1_im, M_d_2_2_re, M_d_2_2_im):

    M_u = np.array([
        [M_u_0_0_re + M_u_0_0_im * 1j, M_u_0_1_re + M_u_0_1_im * 1j, M_u_0_2_re + M_u_0_2_im * 1j, M_u_0_3_re + M_u_0_3_im * 1j],
        [M_u_1_0_re + M_u_1_0_im * 1j, M_u_1_1_re + M_u_1_1_im * 1j, M_u_1_2_re + M_u_1_2_im * 1j, M_u_1_3_re + M_u_1_3_im * 1j],
        [M_u_2_0_re + M_u_2_0_im * 1j, M_u_2_1_re + M_u_2_1_im * 1j, M_u_2_2_re + M_u_2_2_im * 1j, M_u_2_3_re + M_u_2_3_im * 1j],
        [M_u_3_0_re + M_u_3_0_im * 1j, M_u_3_1_re + M_u_3_1_im * 1j, M_u_3_2_re + M_u_3_2_im * 1j, M_u_3_3_re + M_u_3_3_im * 1j]
    ])

    M_d = np.array([
        [M_d_0_0_re + M_d_0_0_im * 1j, M_d_0_1_re + M_d_0_1_im * 1j, M_d_0_2_re + M_d_0_2_im * 1j],
        [M_d_1_0_re + M_d_1_0_im * 1j, M_d_1_1_re + M_d_1_1_im * 1j, M_d_1_2_re + M_d_1_2_im * 1j],
        [M_d_2_0_re + M_d_2_0_im * 1j, M_d_2_1_re + M_d_2_1_im * 1j, M_d_2_2_re + M_d_2_2_im * 1j],
    ])

    # Compute observables from texture zeros
    D_u, D_d, sin_theta_12, sin_theta_13, sin_theta_23, delta, Jarlskog, V = ext.extract_obs(
        M_u, M_d)

    # Check if CKM has null entries
    for i in range(3):
        for j in range(3):
            if abs(V[i, j]) < 1e-10:
                # pdb.set_trace()
                return 1e9

    # Check if masses have neglegible imaginary part from rounding errors

    for mass in D_u:
        if mass.real < 1e10 * mass.imag:
            return 1e9

    for mass in D_d:
        if mass.real < 1e10 * mass.imag:
            return 1e9

    # Compute chi square for observables
    chi_square_m_u, chi_square_m_c, chi_square_m_t, chi_square_m_VLQ, chi_square_m_d, chi_square_m_s, chi_square_m_b, chi_square_sin_theta_12, chi_square_sin_theta_13, chi_square_sin_theta_23, chi_square_delta, chi_square_Jarlskog = compute_chi_square(D_u, D_d, sin_theta_12, sin_theta_13, sin_theta_23, delta, Jarlskog)

    chi_square_total = (chi_square_m_u + chi_square_m_c + chi_square_m_t + chi_square_m_VLQ
                        + chi_square_m_d + chi_square_m_s + chi_square_m_b
                        + chi_square_sin_theta_12 + chi_square_sin_theta_13
                        + chi_square_sin_theta_23 + chi_square_delta + chi_square_Jarlskog)

    return chi_square_total


def par_values_to_np_array(par_values):

    M_u = np.array([
                    [par_values[0] + par_values[1] * 1j, par_values[2] + par_values[3] * 1j, par_values[4] + par_values[5] * 1j, par_values[6] + par_values[7] * 1j],
                    [par_values[8] + par_values[9] * 1j, par_values[10] + par_values[11] * 1j, par_values[12] + par_values[13] * 1j, par_values[14] + par_values[15] * 1j],
                    [par_values[16] + par_values[17] * 1j, par_values[18] + par_values[19] * 1j, par_values[20] + par_values[21] * 1j, par_values[22] + par_values[23] * 1j],
                    [par_values[24] + par_values[25] * 1j, par_values[26] + par_values[27] * 1j, par_values[28] + par_values[29] * 1j, par_values[30] + par_values[31] * 1j]
                   ])

    M_d = np.array([
                    [par_values[32] + par_values[33] * 1j, par_values[34] + par_values[35] * 1j, par_values[36] + par_values[37] * 1j],
                    [par_values[38] + par_values[39] * 1j, par_values[40] + par_values[41] * 1j, par_values[42] + par_values[43] * 1j],
                    [par_values[44] + par_values[45] * 1j, par_values[46] + par_values[47] * 1j, par_values[48] + par_values[49] * 1j]
                   ])

    return M_u, M_d

# This function checks if each parameter is smaller than the maximum accepted chi_square value
def check_chi_squares_limits(par_values):

    if(par_values):
        M_u, M_d = par_values_to_np_array(par_values)

        # Compute observables from texture zeros
        D_u, D_d, sin_theta_12, sin_theta_13, sin_theta_23, delta, Jarlskog, V = ext.extract_obs(
            M_u, M_d)

        chi_square_m_u, chi_square_m_c, chi_square_m_t, chi_square_m_VLQ, chi_square_m_d, chi_square_m_s, chi_square_m_b, chi_square_sin_theta_12, chi_square_sin_theta_13, chi_square_sin_theta_23, chi_square_delta, chi_square_Jarlskog = compute_chi_square(D_u, D_d, sin_theta_12, sin_theta_13, sin_theta_23, delta, Jarlskog)

        # Check deviation from experimental value for each observable
        if chi_square_m_u > MAX_CHI_SQUARE:
            return False

        if chi_square_m_c > MAX_CHI_SQUARE:
            return False

        if chi_square_m_t > MAX_CHI_SQUARE:
            return False

        if chi_square_m_VLQ > MAX_CHI_SQUARE:
            return False

        if chi_square_m_d > MAX_CHI_SQUARE:
            return False

        if chi_square_m_s > MAX_CHI_SQUARE:
            return False

        if chi_square_m_b > MAX_CHI_SQUARE:
            return False

        if chi_square_sin_theta_12 > MAX_CHI_SQUARE:
            return False

        if chi_square_sin_theta_13 > MAX_CHI_SQUARE:
            return False

        if chi_square_sin_theta_23 > MAX_CHI_SQUARE:
            return False

        if chi_square_delta > MAX_CHI_SQUARE:
            return False

        if chi_square_Jarlskog > MAX_CHI_SQUARE:
            return False

        return True

    return False


def run_Minuit(pair, par_values, control):

    if par_values:

        M_u_min, M_d_min = par_values_to_np_array(par_values)
        if control == 0:
            M_u = np.matlib.rand(4, 4) * 0.5 + 0.75
            M_d = np.matlib.rand(3, 3) * 0.5 + 0.75

            M_u = np.multiply(M_u, M_u_min)
            M_d = np.multiply(M_d, M_d_min)
        elif control == 1:
            M_u = M_u_min
            M_d = M_d_min
    else:
        # Input of Minuit optimization
        rng = np.random.default_rng()
        M_u = np.empty((4, 4), dtype=complex)
        M_d = np.empty((3, 3), dtype=complex)
        for i in range(4):
            for j in range(4):
                if pair[0][i, j] == 1:
                    M_u[i, j] = 10 ** (rng.random() * 7 - 7) + 10 ** (rng.random() * 7 - 7) * 1j
                else:
                    M_u[i, j] = 0

        for i in range(3):
            for j in range(3):
                if pair[1][i, j] == 1:
                    M_d[i, j] = 10 ** (rng.random() * 6 - 6) + 10 ** (rng.random() * 6 - 6) * 1j
                else:
                    M_d[i, j] = 0

        M_u = np.matlib.rand(4, 4) + np.matlib.rand(4, 4) * 1j
        M_d = np.matlib.rand(3, 3) + np.matlib.rand(3, 3) * 1j

        # Hadamard product
        M_u = np.multiply(M_u, pair[0])
        M_d = np.multiply(M_d, pair[1])

        M_u[0, :] = M_u[0, :] * MASS_T
        M_u[1, :] = M_u[1, :] * MASS_T
        M_u[2, :] = M_u[2, :] * MASS_T
        M_u[3, :] = M_u[3, :] * MASS_VLQ
        M_d[0, :] = M_d[0, :] * MASS_B
        M_d[1, :] = M_d[1, :] * MASS_B
        M_d[2, :] = M_d[2, :] * MASS_B

    M_u_0_0_re = M_u[0, 0].real
    M_u_0_0_im = M_u[0, 0].imag
    M_u_0_1_re = M_u[0, 1].real
    M_u_0_1_im = M_u[0, 1].imag
    M_u_0_2_re = M_u[0, 2].real
    M_u_0_2_im = M_u[0, 2].imag
    M_u_0_3_re = M_u[0, 3].real
    M_u_0_3_im = M_u[0, 3].imag

    M_u_1_0_re = M_u[1, 0].real
    M_u_1_0_im = M_u[1, 0].imag
    M_u_1_1_re = M_u[1, 1].real
    M_u_1_1_im = M_u[1, 1].imag
    M_u_1_2_re = M_u[1, 2].real
    M_u_1_2_im = M_u[1, 2].imag
    M_u_1_3_re = M_u[1, 3].real
    M_u_1_3_im = M_u[1, 3].imag

    M_u_2_0_re = M_u[2, 0].real
    M_u_2_0_im = M_u[2, 0].imag
    M_u_2_1_re = M_u[2, 1].real
    M_u_2_1_im = M_u[2, 1].imag
    M_u_2_2_re = M_u[2, 2].real
    M_u_2_2_im = M_u[2, 2].imag
    M_u_2_3_re = M_u[2, 3].real
    M_u_2_3_im = M_u[2, 3].imag

    M_u_3_0_re = M_u[3, 0].real
    M_u_3_0_im = M_u[3, 0].imag
    M_u_3_1_re = M_u[3, 1].real
    M_u_3_1_im = M_u[3, 1].imag
    M_u_3_2_re = M_u[3, 2].real
    M_u_3_2_im = M_u[3, 2].imag
    M_u_3_3_re = M_u[3, 3].real
    M_u_3_3_im = M_u[3, 3].imag

    M_d_0_0_re = M_d[0, 0].real
    M_d_0_0_im = M_d[0, 0].imag
    M_d_0_1_re = M_d[0, 1].real
    M_d_0_1_im = M_d[0, 1].imag
    M_d_0_2_re = M_d[0, 2].real
    M_d_0_2_im = M_d[0, 2].imag

    M_d_1_0_re = M_d[1, 0].real
    M_d_1_0_im = M_d[1, 0].imag
    M_d_1_1_re = M_d[1, 1].real
    M_d_1_1_im = M_d[1, 1].imag
    M_d_1_2_re = M_d[1, 2].real
    M_d_1_2_im = M_d[1, 2].imag

    M_d_2_0_re = M_d[2, 0].real
    M_d_2_0_im = M_d[2, 0].imag
    M_d_2_1_re = M_d[2, 1].real
    M_d_2_1_im = M_d[2, 1].imag
    M_d_2_2_re = M_d[2, 2].real
    M_d_2_2_im = M_d[2, 2].imag

    # Initialization of Minuit
    m = Minuit(least_squares,
               M_u_0_0_re, M_u_0_0_im, M_u_0_1_re, M_u_0_1_im, M_u_0_2_re, M_u_0_2_im,   M_u_0_3_re, M_u_0_3_im,
               M_u_1_0_re, M_u_1_0_im, M_u_1_1_re, M_u_1_1_im, M_u_1_2_re, M_u_1_2_im,   M_u_1_3_re, M_u_1_3_im,
               M_u_2_0_re, M_u_2_0_im, M_u_2_1_re, M_u_2_1_im, M_u_2_2_re, M_u_2_2_im,   M_u_2_3_re, M_u_2_3_im,
               M_u_3_0_re, M_u_3_0_im, M_u_3_1_re, M_u_3_1_im, M_u_3_2_re, M_u_3_2_im, M_u_3_3_re, M_u_3_3_im,
               M_d_0_0_re, M_d_0_0_im, M_d_0_1_re, M_d_0_1_im, M_d_0_2_re, M_d_0_2_im,
               M_d_1_0_re, M_d_1_0_im, M_d_1_1_re, M_d_1_1_im, M_d_1_2_re, M_d_1_2_im,
               M_d_2_0_re, M_d_2_0_im, M_d_2_1_re, M_d_2_1_im, M_d_2_2_re, M_d_2_2_im)

    # Fixing texture zeros of M_u
    if M_u[0, 0] == 0:
        m.fixed["M_u_0_0_re"] = True
        m.fixed["M_u_0_0_im"] = True

    if M_u[0, 1] == 0:
        m.fixed["M_u_0_1_re"] = True
        m.fixed["M_u_0_1_im"] = True

    if M_u[0, 2] == 0:
        m.fixed["M_u_0_2_re"] = True
        m.fixed["M_u_0_2_im"] = True

    if M_u[0, 3] == 0:
        m.fixed["M_u_0_3_re"] = True
        m.fixed["M_u_0_3_im"] = True

    if M_u[1, 0] == 0:
        m.fixed["M_u_1_0_re"] = True
        m.fixed["M_u_1_0_im"] = True

    if M_u[1, 1] == 0:
        m.fixed["M_u_1_1_re"] = True
        m.fixed["M_u_1_1_im"] = True

    if M_u[1, 2] == 0:
        m.fixed["M_u_1_2_re"] = True
        m.fixed["M_u_1_2_im"] = True

    if M_u[1, 3] == 0:
        m.fixed["M_u_1_3_re"] = True
        m.fixed["M_u_1_3_im"] = True

    if M_u[2, 0] == 0:
        m.fixed["M_u_2_0_re"] = True
        m.fixed["M_u_2_0_im"] = True

    if M_u[2, 1] == 0:
        m.fixed["M_u_2_1_re"] = True
        m.fixed["M_u_2_1_im"] = True

    if M_u[2, 2] == 0:
        m.fixed["M_u_2_2_re"] = True
        m.fixed["M_u_2_2_im"] = True

    if M_u[2, 3] == 0:
        m.fixed["M_u_2_3_re"] = True
        m.fixed["M_u_2_3_im"] = True

    if M_u[3, 0] == 0:
        m.fixed["M_u_3_0_re"] = True
        m.fixed["M_u_3_0_im"] = True

    if M_u[3, 1] == 0:
        m.fixed["M_u_3_1_re"] = True
        m.fixed["M_u_3_1_im"] = True

    if M_u[3, 2] == 0:
        m.fixed["M_u_3_2_re"] = True
        m.fixed["M_u_3_2_im"] = True

    if M_u[3, 3] == 0:
        m.fixed["M_u_3_3_re"] = True
        m.fixed["M_u_3_3_im"] = True

    # Fixing texture zeros of M_d
    if M_d[0, 0] == 0:
        m.fixed["M_d_0_0_re"] = True
        m.fixed["M_d_0_0_im"] = True

    if M_d[0, 1] == 0:
        m.fixed["M_d_0_1_re"] = True
        m.fixed["M_d_0_1_im"] = True

    if M_d[0, 2] == 0:
        m.fixed["M_d_0_2_re"] = True
        m.fixed["M_d_0_2_im"] = True

    if M_d[1, 0] == 0:
        m.fixed["M_d_1_0_re"] = True
        m.fixed["M_d_1_0_im"] = True

    if M_d[1, 1] == 0:
        m.fixed["M_d_1_1_re"] = True
        m.fixed["M_d_1_1_im"] = True

    if M_d[1, 2] == 0:
        m.fixed["M_d_1_2_re"] = True
        m.fixed["M_d_1_2_im"] = True

    if M_d[2, 0] == 0:
        m.fixed["M_d_2_0_re"] = True
        m.fixed["M_d_2_0_im"] = True

    if M_d[2, 1] == 0:
        m.fixed["M_d_2_1_re"] = True
        m.fixed["M_d_2_1_im"] = True

    if M_d[2, 2] == 0:
        m.fixed["M_d_2_2_re"] = True
        m.fixed["M_d_2_2_im"] = True

    # Run Minuit minimization
    m.migrad()
    return m.fval, m.values

# This function writes to a string the information regarding a minimum found by iminuit
def info_minimum(par_values):

    M_u, M_d = par_values_to_np_array(par_values)

    D_u, D_d, sin_theta_12, sin_theta_13, sin_theta_23, delta, Jarlskog, V = ext.extract_obs(
        M_u, M_d)

    chi_square_m_u, chi_square_m_c, chi_square_m_t, chi_square_m_VLQ, chi_square_m_d, chi_square_m_s, chi_square_m_b, chi_square_sin_theta_12, chi_square_sin_theta_13, chi_square_sin_theta_23, chi_square_delta, chi_square_Jarlskog = compute_chi_square(D_u, D_d, sin_theta_12, sin_theta_13, sin_theta_23, delta, Jarlskog)

    string = f"m_u = {D_u[0]} / chi_square_m_u = {chi_square_m_u}\n"
    string += f"m_c = {D_u[1]} / chi_square_m_c = {chi_square_m_c}\n"
    string += f"m_t = {D_u[2]} / chi_square_m_t = {chi_square_m_t}\n"
    string += f"m_VLQ = {D_u[3]} / chi_square_m_VLQ = {chi_square_m_VLQ}\n"
    string += f"m_d = {D_d[0]} / chi_square_m_d = {chi_square_m_d}\n"
    string += f"m_s = {D_d[1]} / chi_square_m_s = {chi_square_m_s}\n"
    string += f"m_b = {D_d[2]} / chi_square_m_b = {chi_square_m_b}\n"
    string += f"sin_theta_12 = {sin_theta_12} / chi_square_sin_theta_12 = {chi_square_sin_theta_12}\n"
    string += f"sin_theta_13 = {sin_theta_13} / chi_square_sin_theta_13 = {chi_square_sin_theta_13}\n"
    string += f"sin_theta_23 = {sin_theta_23} / chi_square_sin_theta_23 = {chi_square_sin_theta_23}\n"
    string += f"delta = {delta} / chi_square_delta = {chi_square_delta}\n"
    string += f"Jarlskog = {Jarlskog} / chi_square_Jarlskog = {chi_square_Jarlskog}\n"
    string += f"V:\n{V}\n"
    return string


# This function writes to a file the maximally restrictive pairs found by Minuit
def print_restrictive_pairs_from_minuit(set_maximally_restrictive_pairs,
                                        option,
                                        filename):
    with open(filename, option) as f:
        f.write("LIST OF MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS (AFTER MINUIT):\n\n")
        num_pairs = 0
        for restrictive_pairs_n_zeros_u in set_maximally_restrictive_pairs:
            num_pairs_n_zeros_u = 0
            f.write(
                f"PAIRS WITH {restrictive_pairs_n_zeros_u[0]} TEXTURE ZEROS FOR (M_u, M_d):\n\n")
            for pair in restrictive_pairs_n_zeros_u[1]:
                f.write(f"M_u:\n{pair[2][0]}\n")
                f.write(f"M_d:\n{pair[2][1]}\n")
                f.write(f"Minimum found:\n")
                M_u, M_d = par_values_to_np_array(pair[1])
                f.write(f"M_u:\n{M_u}\n")
                f.write(f"M_d:\n{M_d}\n")
                f.write(f"chi_square: {pair[0]}\n")
                f.write(info_minimum(pair[1]))
                f.write("\n")
                num_pairs_n_zeros_u += 1
            f.write(
                f"THERE ARE {num_pairs_n_zeros_u} PAIRS WITH {restrictive_pairs_n_zeros_u[0]} TEXTURE ZEROS\n\n")
            num_pairs += num_pairs_n_zeros_u

        f.write(
            f"\nTHERE ARE IN TOTAL {num_pairs} MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS (AFTER MINUIT). \n")

    return


def main():

    args = sys.argv[1:]
    start = time.time()
    set_maximally_restrictive_pairs = tz.read_maximmaly_restrictive_pairs(args[0])

    for set_class_pairs in set_maximally_restrictive_pairs:
        indices_pairs_to_pop = []
        for index, pair in enumerate(set_class_pairs[1]):
            print(pair[0])
            print(pair[1])
            par_values = []
            i = 0
            j = 0
            chi_square = 1e10
            while i < 100 and not check_chi_squares_limits(par_values): #15000
                if j > 10:
                    par_values = []
                    chi_square = 1e10
                    print("RESET\n")
                    j = 0
                j += 1
                print(i)
                chi_square_minuit, par_values_minuit = run_Minuit(pair, par_values, 0)
                print(chi_square_minuit)
                while chi_square_minuit < chi_square and int((chi_square - chi_square_minuit)) > 0:
                    chi_square = chi_square_minuit
                    par_values = par_values_minuit
                    print(f"NOVO CHI SQUARE = {chi_square}")
                    chi_square_minuit, par_values_minuit = run_Minuit(pair, par_values, 1)
                    j = 0
                i += 1

            if chi_square > MAX_CHI_SQUARE * N_OBSERVABLES_SM:
                indices_pairs_to_pop.append(index)
            else:
                set_class_pairs[1][index] = [chi_square, par_values, set_class_pairs[1][index]]

        for index in reversed(indices_pairs_to_pop):
            set_class_pairs[1].pop(index)

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)
    filename = args[0].replace("before", "after")
    print_restrictive_pairs_from_minuit(set_maximally_restrictive_pairs, "w", "test.txt")
    end = time.time()
    print(f"TOTAL TIME = {int((float(end) - float(start)) / 60)} min ", end="")
    print(f"{(int(float(end) - float(start)) % 60)} sec \n")

    return


if __name__ == "__main__":
    main()
