from numba import njit  # import pdb
import time
import sys
from joblib import delayed, Parallel
import phenomenology as pheno
from iminuit import Minuit
import cmath
import math
import numpy.matlib
import numpy as np
import extract_obs as ext

# Experimental values taken from PDG revision of 2022:
# https://pdg.lbl.gov/2022/reviews/rpp2022-rev-ckm-matrix.pdf (January 2023)

# The main function of this scprit takes a file containing pairs of matrices as input and writes to
# a file the pairs which manage to replicate the experimental data.
# Instead of cycling through the list of pairs with a for cycle, we use the multiproccessing
# function Pool.map().
# Each pair is analyzed with the run_Minuit(pair, par_values, control) function until it
# reaches a maximum of N tries or it satisfies the stopping condition check_chi_squares_limits(par_values).
# The extract_obs.py module contains the function that computes the theoretical values from
# a given pair of matrices.
# https://iminuit.readthedocs.io/en/stable/


##################################################################################################
#
#   INPUT PARAMETERS
#
##################################################################################################

MeV = 1e-3
GeV = 1
TeV = 1e3

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

# MASS_VLQ = 1.95 * TeV
SIGMA_MASS_VLQ = 0.01 * TeV

RATIO_MASS_DB = MASS_D / MASS_B
UPPER_SIGMA_RATIO_MASS_DB = RATIO_MASS_DB * math.sqrt((UPPER_SIGMA_MASS_D / MASS_D) ** 2
                                                      + (LOWER_SIGMA_MASS_B / MASS_B) ** 2)
LOWER_SIGMA_RATIO_MASS_DB = RATIO_MASS_DB * math.sqrt((LOWER_SIGMA_MASS_D / MASS_D) ** 2
                                                      + (UPPER_SIGMA_MASS_B / MASS_B) ** 2)

RATIO_MASS_SB = MASS_S / MASS_B
UPPER_SIGMA_RATIO_MASS_SB = RATIO_MASS_SB * math.sqrt((UPPER_SIGMA_MASS_S / MASS_S) ** 2
                                                      + (LOWER_SIGMA_MASS_B / MASS_B) ** 2)
LOWER_SIGMA_RATIO_MASS_SB = RATIO_MASS_SB * math.sqrt((LOWER_SIGMA_MASS_S / MASS_S) ** 2
                                                      + (UPPER_SIGMA_MASS_B / MASS_B) ** 2)

RATIO_MASS_UT = MASS_U / MASS_T
UPPER_SIGMA_RATIO_MASS_UT = RATIO_MASS_UT * math.sqrt((UPPER_SIGMA_MASS_U / MASS_U) ** 2
                                                      + (LOWER_SIGMA_MASS_T / MASS_T) ** 2)
LOWER_SIGMA_RATIO_MASS_UT = RATIO_MASS_UT * math.sqrt((LOWER_SIGMA_MASS_U / MASS_U) ** 2
                                                      + (UPPER_SIGMA_MASS_T / MASS_T) ** 2)

RATIO_MASS_CT = MASS_C / MASS_T
UPPER_SIGMA_RATIO_MASS_CT = RATIO_MASS_CT * math.sqrt((SIGMA_MASS_C / MASS_C) ** 2
                                                      + (LOWER_SIGMA_MASS_T / MASS_T) ** 2)
LOWER_SIGMA_RATIO_MASS_CT = RATIO_MASS_CT * math.sqrt((SIGMA_MASS_C / MASS_C) ** 2
                                                      + (UPPER_SIGMA_MASS_T / MASS_T) ** 2)

V_UD = 0.97373
UPPER_SIGMA_V_UD = 0.000313
LOWER_SIGMA_V_UD = 0.000313

V_US = 0.2243
UPPER_SIGMA_V_US = 0.0008
LOWER_SIGMA_V_US = 0.0008

V_UB = 0.00382
UPPER_SIGMA_V_UB = 0.00020
LOWER_SIGMA_V_UB = 0.00020

V_CD = 0.221
UPPER_SIGMA_V_CD = 0.004
LOWER_SIGMA_V_CD = 0.004

V_CS = 0.975
UPPER_SIGMA_V_CS = 0.006
LOWER_SIGMA_V_CS = 0.006

V_CB = 0.0408
UPPER_SIGMA_V_CB = 0.0014
LOWER_SIGMA_V_CB = 0.0014

V_TD = 0.0086
UPPER_SIGMA_V_TD = 0.0002
LOWER_SIGMA_V_TD = 0.0002

V_TS = 0.0415
UPPER_SIGMA_V_TS = 0.0009
LOWER_SIGMA_V_TS = 0.0009

V_TB = 1.014
UPPER_SIGMA_V_TB = 0.029
LOWER_SIGMA_V_TB = 0.029

# QUAL A MEDIDA A USAR? (HÁ 3 NO CKM FITTER)
GAMMA = 65.9
UPPER_SIGMA_GAMMA = 3.3
LOWER_SIGMA_GAMMA = 3.5

MAX_CHI_SQUARE = 1
MAX_CHI_SQUARE_PHENO = 9
# 4 mass ratios (u/t, c/t, d/b, s/b) + 9 CKM elements + UT gamma phase
N_OBSERVABLES = 14

N_TRIES = 1000
MASS_DECOUPLED = MASS_B
MASS_P = MASS_S
MASS_M = MASS_D
FILENAME = "2HDM_minuit_with_pheno_b_decoupled"

ZERO = 1e-10

# Texture pairs
set_maximally_restrictive_pairs = [
    ["(9,5)",
     [np.array([[0, 0, 1, 1],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 0]
                ]),
      np.array([[0, 0, 1],
                [0, 1, 2],
                [1, 0, 0]
                ])
      ]],
    ["(9,5)",
     [np.array([[0, 0, 1, 1],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 0]
                ]),
      np.array([[0, 0, 1],
                [0, 1, 0],
                [1, 0, 2]
                ])
      ]]
]

##################################################################################################

# AUXILIARY FUNCTIONS


def check_decoupled_quark(pair):

    texture = pair[1]
    positions = []
    for i in range(3):
        for j in range(3):
            if texture[i, j] != 0:
                positions.append([i, j])

    ones = np.ones(3)
    rows = []
    columns = []
    for i, num in enumerate(texture @ ones):
        if num > 1:
            rows.append(i)
    for i, num in enumerate(ones.T @ texture):
        if num > 1:
            columns.append(i)

    j = len(positions) - 1
    while j > -1:
        valid = True
        for num in rows:
            if positions[j][0] == num:
                valid = False
        for num in columns:
            if positions[j][1] == num:
                valid = False
        if not valid:
            positions.pop(j)
        j -= 1

    if positions:
        return positions[0]

    return None

##################################################################################################


@njit
def compute_chi_square(D_u, D_d, V):

    # Computation of chi squared of masses
    if abs(D_u[0]) / abs(D_u[2]) - RATIO_MASS_UT > 0:
        chi_square_ratio_ut = (
            (abs(D_u[0]) / abs(D_u[2]) - RATIO_MASS_UT) / UPPER_SIGMA_RATIO_MASS_UT) ** 2
    else:
        chi_square_ratio_ut = (
            (abs(D_u[0]) / abs(D_u[2]) - RATIO_MASS_UT) / LOWER_SIGMA_RATIO_MASS_UT) ** 2

    if abs(D_u[1]) / abs(D_u[2]) - RATIO_MASS_CT > 0:
        chi_square_ratio_ct = (
            (abs(D_u[1]) / abs(D_u[2]) - RATIO_MASS_CT) / UPPER_SIGMA_RATIO_MASS_CT) ** 2
    else:
        chi_square_ratio_ct = (
            (abs(D_u[1]) / abs(D_u[2]) - RATIO_MASS_CT) / LOWER_SIGMA_RATIO_MASS_CT) ** 2

    if abs(D_d[0]) / abs(D_d[2]) - RATIO_MASS_DB > 0:
        chi_square_ratio_db = (
            (abs(D_d[0]) / abs(D_d[2]) - RATIO_MASS_DB) / UPPER_SIGMA_RATIO_MASS_DB) ** 2
    else:
        chi_square_ratio_db = (
            (abs(D_d[0]) / abs(D_d[2]) - RATIO_MASS_DB) / LOWER_SIGMA_RATIO_MASS_DB) ** 2

    if abs(D_d[1]) / abs(D_d[2]) - RATIO_MASS_SB > 0:
        chi_square_ratio_sb = (
            (abs(D_d[1]) / abs(D_d[2]) - RATIO_MASS_SB) / UPPER_SIGMA_RATIO_MASS_SB) ** 2
    else:
        chi_square_ratio_sb = (
            (abs(D_d[1]) / abs(D_d[2]) - RATIO_MASS_SB) / LOWER_SIGMA_RATIO_MASS_SB) ** 2

    chi_square_m_VLQ = 1e5
    if abs(D_u[3] / D_u[2] * MASS_T) > 1 * TeV and abs(D_u[3] / D_u[2] * MASS_T) < 10 * TeV:
        chi_square_m_VLQ = 0

    # Computation of absolute value of entries of V_CKM
    if abs(V[0, 0]) - V_UD > 0:
        chi_square_V_ud = ((abs(V[0, 0]) - V_UD) / UPPER_SIGMA_V_UD) ** 2
    else:
        chi_square_V_ud = ((abs(V[0, 0]) - V_UD) / LOWER_SIGMA_V_UD) ** 2

    if abs(V[0, 1]) - V_US > 0:
        chi_square_V_us = ((abs(V[0, 1]) - V_US) / UPPER_SIGMA_V_US) ** 2
    else:
        chi_square_V_us = ((abs(V[0, 1]) - V_US) / LOWER_SIGMA_V_US) ** 2

    if abs(V[0, 2]) - V_UB > 0:
        chi_square_V_ub = ((abs(V[0, 2]) - V_UB) / UPPER_SIGMA_V_UB) ** 2
    else:
        chi_square_V_ub = ((abs(V[0, 2]) - V_UB) / LOWER_SIGMA_V_UB) ** 2

    if abs(V[1, 0]) - V_CD > 0:
        chi_square_V_cd = ((abs(V[1, 0]) - V_CD) / UPPER_SIGMA_V_CD) ** 2
    else:
        chi_square_V_cd = ((abs(V[1, 0]) - V_CD) / LOWER_SIGMA_V_CD) ** 2

    if abs(V[1, 1]) - V_CS > 0:
        chi_square_V_cs = ((abs(V[1, 1]) - V_CS) / UPPER_SIGMA_V_CS) ** 2
    else:
        chi_square_V_cs = ((abs(V[1, 1]) - V_CS) / LOWER_SIGMA_V_CS) ** 2

    if abs(V[1, 2]) - V_CB > 0:
        chi_square_V_cb = ((abs(V[1, 2]) - V_CB) / UPPER_SIGMA_V_CB) ** 2
    else:
        chi_square_V_cb = ((abs(V[1, 2]) - V_CB) / LOWER_SIGMA_V_CB) ** 2

    if abs(V[2, 0]) - V_TD > 0:
        chi_square_V_td = ((abs(V[2, 0]) - V_TD) / UPPER_SIGMA_V_TD) ** 2
    else:
        chi_square_V_td = ((abs(V[2, 0]) - V_TD) / LOWER_SIGMA_V_TD) ** 2

    if abs(V[2, 1]) - V_TS > 0:
        chi_square_V_ts = ((abs(V[2, 1]) - V_TS) / UPPER_SIGMA_V_TS) ** 2
    else:
        chi_square_V_ts = ((abs(V[2, 1]) - V_TS) / LOWER_SIGMA_V_TS) ** 2

    if abs(V[2, 2]) - V_TB > 0:
        chi_square_V_tb = ((abs(V[2, 2]) - V_TB) / UPPER_SIGMA_V_TB) ** 2
    else:
        chi_square_V_tb = ((abs(V[2, 2]) - V_TB) / LOWER_SIGMA_V_TB) ** 2

    gamma = cmath.phase(-V[0, 0] * V[1, 2] * np.conj(V[0, 2])
                        * np.conj(V[1, 0])) / (2 * math.pi) * 360
    if gamma - GAMMA > 0:
        chi_square_gamma = ((gamma - GAMMA) / UPPER_SIGMA_GAMMA) ** 2
    else:
        chi_square_gamma = ((gamma - GAMMA) / LOWER_SIGMA_GAMMA) ** 2

    return (chi_square_ratio_db, chi_square_ratio_sb,
            chi_square_ratio_ut, chi_square_ratio_ct, chi_square_m_VLQ,
            chi_square_V_ud, chi_square_V_us, chi_square_V_ub,
            chi_square_V_cd, chi_square_V_cs, chi_square_V_cb,
            chi_square_V_td, chi_square_V_ts, chi_square_V_tb,
            chi_square_gamma)


# Function to be mininimized
@ njit()
def least_squares(M_u_0_0_re, M_u_0_0_im, M_u_0_1_re, M_u_0_1_im, M_u_0_2_re, M_u_0_2_im, M_u_0_3_re, M_u_0_3_im,
                  M_u_1_0_re, M_u_1_0_im, M_u_1_1_re, M_u_1_1_im, M_u_1_2_re, M_u_1_2_im, M_u_1_3_re, M_u_1_3_im,
                  M_u_2_0_re, M_u_2_0_im, M_u_2_1_re, M_u_2_1_im, M_u_2_2_re, M_u_2_2_im, M_u_2_3_re, M_u_2_3_im,
                  M_u_3_0_re, M_u_3_0_im, M_u_3_1_re, M_u_3_1_im, M_u_3_2_re, M_u_3_2_im, M_u_3_3_re, M_u_3_3_im,
                  M_d_0_0_re, M_d_0_0_im, M_d_0_1_re, M_d_0_1_im, M_d_0_2_re, M_d_0_2_im,
                  M_d_1_0_re, M_d_1_0_im, M_d_1_1_re, M_d_1_1_im, M_d_1_2_re, M_d_1_2_im,
                  M_d_2_0_re, M_d_2_0_im, M_d_2_1_re, M_d_2_1_im, M_d_2_2_re, M_d_2_2_im
                  ):

    M_u = np.array([
        [M_u_0_0_re + M_u_0_0_im * 1j, M_u_0_1_re + M_u_0_1_im * 1j,
            M_u_0_2_re + M_u_0_2_im * 1j, M_u_0_3_re + M_u_0_3_im * 1j],
        [M_u_1_0_re + M_u_1_0_im * 1j, M_u_1_1_re + M_u_1_1_im * 1j,
            M_u_1_2_re + M_u_1_2_im * 1j, M_u_1_3_re + M_u_1_3_im * 1j],
        [M_u_2_0_re + M_u_2_0_im * 1j, M_u_2_1_re + M_u_2_1_im * 1j,
            M_u_2_2_re + M_u_2_2_im * 1j, M_u_2_3_re + M_u_2_3_im * 1j],
        [M_u_3_0_re + M_u_3_0_im * 1j, M_u_3_1_re + M_u_3_1_im * 1j,
            M_u_3_2_re + M_u_3_2_im * 1j, M_u_3_3_re + M_u_3_3_im * 1j]
    ])

    M_d = np.array([
        [M_d_0_0_re + M_d_0_0_im * 1j, M_d_0_1_re +
            M_d_0_1_im * 1j, M_d_0_2_re + M_d_0_2_im * 1j],
        [M_d_1_0_re + M_d_1_0_im * 1j, M_d_1_1_re +
            M_d_1_1_im * 1j, M_d_1_2_re + M_d_1_2_im * 1j],
        [M_d_2_0_re + M_d_2_0_im * 1j, M_d_2_1_re +
            M_d_2_1_im * 1j, M_d_2_2_re + M_d_2_2_im * 1j],
    ])

    # Compute observables from texture zeros
    D_u, D_d, delta, Jarlskog, V = ext.extract_obs(
        M_u, M_d)

    # Check if CKM has null entries
    for i in range(3):
        for j in range(3):
            if abs(V[i, j]) < 1e-10:
                return 1e9

    chi_square_ratio_db, chi_square_ratio_sb, chi_square_ratio_ut, chi_square_ratio_ct, chi_square_m_VLQ, chi_square_V_ud, chi_square_V_us, chi_square_V_ub, chi_square_V_cd, chi_square_V_cs, chi_square_V_cb, chi_square_V_td, chi_square_V_ts, chi_square_V_tb, chi_square_gamma = compute_chi_square(
        D_u, D_d, V)

    x = np.array([np.exp(-np.angle(V[0, 0]) * 1j),
                 np.exp(-np.angle(V[0, 1]) * 1j), 1. + 0 * 1j], dtype=np.complex128)
    V = V @ np.diag(x)

    D_u = D_u * abs(MASS_T / D_u[2])
    garbage, chi_square_pheno = pheno.phenomenology_tests(
        V, V @ V.conj().T, D_u[0], D_u[1], D_u[2], D_u[3])

    chi_square_total = (chi_square_ratio_db + chi_square_ratio_sb
                        + chi_square_ratio_ut + chi_square_ratio_ct + chi_square_m_VLQ
                        + chi_square_V_ud + chi_square_V_us + chi_square_V_ub
                        + chi_square_V_cd + chi_square_V_cs + chi_square_V_cb
                        + chi_square_V_td + chi_square_V_ts + chi_square_V_tb
                        + chi_square_gamma + chi_square_pheno
                        )

    return chi_square_total


def par_values_to_np_array(par_values):

    M_u = np.array([
        [par_values[0] + par_values[1] * 1j, par_values[2] + par_values[3] * 1j,
            par_values[4] + par_values[5] * 1j, par_values[6] + par_values[7] * 1j],
        [par_values[8] + par_values[9] * 1j, par_values[10] + par_values[11] * 1j,
            par_values[12] + par_values[13] * 1j, par_values[14] + par_values[15] * 1j],
        [par_values[16] + par_values[17] * 1j, par_values[18] + par_values[19] * 1j,
            par_values[20] + par_values[21] * 1j, par_values[22] + par_values[23] * 1j],
        [par_values[24] + par_values[25] * 1j, par_values[26] + par_values[27] * 1j,
            par_values[28] + par_values[29] * 1j, par_values[30] + par_values[31] * 1j]
    ])

    M_d = np.array([
        [par_values[32] + par_values[33] * 1j, par_values[34] +
            par_values[35] * 1j, par_values[36] + par_values[37] * 1j],
        [par_values[38] + par_values[39] * 1j, par_values[40] +
            par_values[41] * 1j, par_values[42] + par_values[43] * 1j],
        [par_values[44] + par_values[45] * 1j, par_values[46] +
            par_values[47] * 1j, par_values[48] + par_values[49] * 1j]
    ])

    return M_u, M_d


def run_Minuit(pair, par_values, control, position=None):

    # Input of Minuit optimization
    rng = np.random.default_rng()
    if par_values:
        if control == 1:
            M_u_min, M_d_min = par_values_to_np_array(par_values)
            M_u = M_u_min
            M_d = M_d_min
    else:
        M_u = np.empty((4, 4), dtype=complex)
        M_d = np.empty((3, 3), dtype=complex)
        for i in range(4):
            for j in range(4):
                if pair[0][i, j] == 2:
                    M_u[i, j] = 10 ** (rng.random() * 6 - 6) + \
                        10 ** (rng.random() * 6 - 6) * 1j
                elif pair[0][i, j] == 1:
                    M_u[i, j] = 10 ** (rng.random() * 6 - 6)
                else:
                    M_u[i, j] = 0

        x2 = MASS_M + rng.random() * (MASS_P - MASS_M)
        theta = rng.random() * 2 * np.pi
        x1 = np.sqrt(MASS_P ** 2 * MASS_M ** 2 / x2 ** 2)
        rho = np.sqrt((MASS_P ** 2 - x2 ** 2) *
                      (x2 ** 2 - MASS_M ** 2) / x2 ** 2)

        if pair[1][2, 2] == 0:
            M_d = np.array([[0,               0,                          x1],
                            [0,              x2, rho * cmath.exp(theta * 1j)],
                            [MASS_DECOUPLED,   0,                           0]])
        else:
            M_d = np.array([[0,               0,                          x1],
                            [0,  MASS_DECOUPLED,                           0],
                            [x2,               0, rho * cmath.exp(theta * 1j)]])

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
               M_u_0_0_re, M_u_0_0_im, M_u_0_1_re, M_u_0_1_im, M_u_0_2_re, M_u_0_2_im, M_u_0_3_re, M_u_0_3_im,
               M_u_1_0_re, M_u_1_0_im, M_u_1_1_re, M_u_1_1_im, M_u_1_2_re, M_u_1_2_im, M_u_1_3_re, M_u_1_3_im,
               M_u_2_0_re, M_u_2_0_im, M_u_2_1_re, M_u_2_1_im, M_u_2_2_re, M_u_2_2_im, M_u_2_3_re, M_u_2_3_im,
               M_u_3_0_re, M_u_3_0_im, M_u_3_1_re, M_u_3_1_im, M_u_3_2_re, M_u_3_2_im, M_u_3_3_re, M_u_3_3_im,
               M_d_0_0_re, M_d_0_0_im, M_d_0_1_re, M_d_0_1_im, M_d_0_2_re, M_d_0_2_im,
               M_d_1_0_re, M_d_1_0_im, M_d_1_1_re, M_d_1_1_im, M_d_1_2_re, M_d_1_2_im,
               M_d_2_0_re, M_d_2_0_im, M_d_2_1_re, M_d_2_1_im, M_d_2_2_re, M_d_2_2_im,
               )

    # Fixing texture zeros of M_u
    if M_u[0, 0] < ZERO:
        m.fixed["M_u_0_0_re"] = True
        m.fixed["M_u_0_0_im"] = True

    if M_u[0, 1] < ZERO:
        m.fixed["M_u_0_1_re"] = True
        m.fixed["M_u_0_1_im"] = True

    if M_u[0, 2] < ZERO:
        m.fixed["M_u_0_2_re"] = True
        m.fixed["M_u_0_2_im"] = True

    if M_u[0, 3] < ZERO:
        m.fixed["M_u_0_3_re"] = True
        m.fixed["M_u_0_3_im"] = True

    if M_u[1, 0] < ZERO:
        m.fixed["M_u_1_0_re"] = True
        m.fixed["M_u_1_0_im"] = True

    if M_u[1, 1] < ZERO:
        m.fixed["M_u_1_1_re"] = True
        m.fixed["M_u_1_1_im"] = True

    if M_u[1, 2] < ZERO:
        m.fixed["M_u_1_2_re"] = True
        m.fixed["M_u_1_2_im"] = True

    if M_u[1, 3] < ZERO:
        m.fixed["M_u_1_3_re"] = True
        m.fixed["M_u_1_3_im"] = True

    if M_u[2, 0] < ZERO:
        m.fixed["M_u_2_0_re"] = True
        m.fixed["M_u_2_0_im"] = True

    if M_u[2, 1] < ZERO:
        m.fixed["M_u_2_1_re"] = True
        m.fixed["M_u_2_1_im"] = True

    if M_u[2, 2] < ZERO:
        m.fixed["M_u_2_2_re"] = True
        m.fixed["M_u_2_2_im"] = True

    if M_u[2, 3] < ZERO:
        m.fixed["M_u_2_3_re"] = True
        m.fixed["M_u_2_3_im"] = True

    if M_u[3, 0] < ZERO:
        m.fixed["M_u_3_0_re"] = True
        m.fixed["M_u_3_0_im"] = True

    if M_u[3, 1] < ZERO:
        m.fixed["M_u_3_1_re"] = True
        m.fixed["M_u_3_1_im"] = True

    if M_u[3, 2] < ZERO:
        m.fixed["M_u_3_2_re"] = True
        m.fixed["M_u_3_2_im"] = True

    if M_u[3, 3] < ZERO:
        m.fixed["M_u_3_3_re"] = True
        m.fixed["M_u_3_3_im"] = True

    # Fixing texture zeros of M_d
    if M_d[0, 0] < ZERO:
        m.fixed["M_d_0_0_re"] = True
        m.fixed["M_d_0_0_im"] = True

    if M_d[0, 1] < ZERO:
        m.fixed["M_d_0_1_re"] = True
        m.fixed["M_d_0_1_im"] = True

    if M_d[0, 2] < ZERO:
        m.fixed["M_d_0_2_re"] = True
        m.fixed["M_d_0_2_im"] = True

    if M_d[1, 0] < ZERO:
        m.fixed["M_d_1_0_re"] = True
        m.fixed["M_d_1_0_im"] = True

    if M_d[1, 1] < ZERO:
        m.fixed["M_d_1_1_re"] = True
        m.fixed["M_d_1_1_im"] = True

    if M_d[1, 2] < ZERO:
        m.fixed["M_d_1_2_re"] = True
        m.fixed["M_d_1_2_im"] = True

    if M_d[2, 0] < ZERO:
        m.fixed["M_d_2_0_re"] = True
        m.fixed["M_d_2_0_im"] = True

    if M_d[2, 1] < ZERO:
        m.fixed["M_d_2_1_re"] = True
        m.fixed["M_d_2_1_im"] = True

    if M_d[2, 2] < ZERO:
        m.fixed["M_d_2_2_re"] = True
        m.fixed["M_d_2_2_im"] = True

    # Setting M_u values to positive
#   m.limits["M_u_0_0_re"] = (0, None)
#   m.limits["M_u_0_0_im"] = (0, None)
#   m.limits["M_u_0_1_re"] = (0, None)
#   m.limits["M_u_0_1_im"] = (0, None)
#   m.limits["M_u_0_2_re"] = (0, None)
#   m.limits["M_u_0_2_im"] = (0, None)
#   m.limits["M_u_0_3_re"] = (0, None)
#   m.limits["M_u_0_3_im"] = (0, None)
#   m.limits["M_u_1_0_re"] = (0, None)
#   m.limits["M_u_1_0_im"] = (0, None)
#   m.limits["M_u_1_1_re"] = (0, None)
#   m.limits["M_u_1_1_im"] = (0, None)
#   m.limits["M_u_1_2_re"] = (0, None)
#   m.limits["M_u_1_2_im"] = (0, None)
#   m.limits["M_u_1_3_re"] = (0, None)
#   m.limits["M_u_1_3_im"] = (0, None)
#   m.limits["M_u_2_0_re"] = (0, None)
#   m.limits["M_u_2_0_im"] = (0, None)
#   m.limits["M_u_2_1_re"] = (0, None)
#   m.limits["M_u_2_1_im"] = (0, None)
#   m.limits["M_u_2_2_re"] = (0, None)
#   m.limits["M_u_2_2_im"] = (0, None)
#   m.limits["M_u_2_3_re"] = (0, None)
#   m.limits["M_u_2_3_im"] = (0, None)
#   m.limits["M_u_3_0_re"] = (0, None)
#   m.limits["M_u_3_0_im"] = (0, None)
#   m.limits["M_u_3_1_re"] = (0, None)
#   m.limits["M_u_3_1_im"] = (0, None)
#   m.limits["M_u_3_2_re"] = (0, None)
#   m.limits["M_u_3_2_im"] = (0, None)
#   m.limits["M_u_3_3_re"] = (0, None)
#   m.limits["M_u_3_3_im"] = (0, None)

#   m.fixed["M_d_0_0_im"] = True
#   m.fixed["M_d_0_1_im"] = True
#   m.fixed["M_d_0_2_im"] = True
#   m.fixed["M_d_1_0_im"] = True
#   m.fixed["M_d_1_1_im"] = True
#   m.fixed["M_d_2_0_im"] = True
#   m.fixed["M_d_2_1_im"] = True

    # Run Minuit minimization
    m.migrad()
    return m.fval, m.values


def info_minimum(par_values):

    M_u, M_d = par_values_to_np_array(par_values)

    D_u, D_d, delta, Jarlskog, V = ext.extract_obs(
        M_u, M_d)

    chi_square_ratio_db, chi_square_ratio_sb, chi_square_ratio_ut, chi_square_ratio_ct, chi_square_m_VLQ, chi_square_V_ud, chi_square_V_us, chi_square_V_ub, chi_square_V_cd, chi_square_V_cs, chi_square_V_cb, chi_square_V_td, chi_square_V_ts, chi_square_V_tb, chi_square_gamma = compute_chi_square(
        D_u, D_d, V)

    x = np.array([np.exp(-np.angle(V[0, 0]) * 1j),
                 np.exp(-np.angle(V[0, 1]) * 1j), 1. + 0 * 1j], dtype=np.complex128)
    V = V @ np.diag(x)
    F = V @ V.conj().T
    D_u = abs(D_u * MASS_T / D_u[2])
    results = pheno.phenomenology_tests(V, F, D_u[0], D_u[1], D_u[2], D_u[3])

    scale_up = MASS_T / D_u[2]
    scale_down = MASS_B / D_d[2]
    string = f"m_u = {D_u[0] * scale_up / MeV} MeV/ chi_square_ratio_ut = {chi_square_ratio_ut}\n"
    string += f"m_c = {D_u[1] * scale_up / GeV} GeV/ chi_square_ratio_ct = {chi_square_ratio_ct}\n"
    string += f"m_t = {D_u[2] * scale_up / GeV} GeV\n"
    string += f"m_VLQ = {D_u[3] * scale_up / TeV} TeV / chi_square_m_VLQ = {chi_square_m_VLQ}\n"
    string += f"m_d = {D_d[0] * scale_down / MeV} MeV / chi_square_ratio_db = {chi_square_ratio_db}\n"
    string += f"m_s = {D_d[1] * scale_down / MeV} MeV/ chi_square_m_ratio_sb = {chi_square_ratio_sb}\n"
    string += f"m_b = {D_d[2] * scale_down / GeV} GeV\n"
    string += f"V_ud = {abs(V[0, 0])} / chi_square_V_ud = {chi_square_V_ud}\n"
    string += f"V_us = {abs(V[0, 1])} / chi_square_V_us = {chi_square_V_us}\n"
    string += f"V_ub = {abs(V[0, 2])} / chi_square_V_ub = {chi_square_V_ub}\n"
    string += f"V_cd = {abs(V[1, 0])} / chi_square_V_cd = {chi_square_V_cd}\n"
    string += f"V_cs = {abs(V[1, 1])} / chi_square_V_cs = {chi_square_V_cs}\n"
    string += f"V_cb = {abs(V[1, 2])} / chi_square_V_cb = {chi_square_V_cb}\n"
    string += f"V_td = {abs(V[2, 0])} / chi_square_V_td = {chi_square_V_td}\n"
    string += f"V_ts = {abs(V[2, 1])} / chi_square_V_ts = {chi_square_V_ts}\n"
    string += f"V_tb = {abs(V[2, 2])} / chi_square_V_tb = {chi_square_V_tb}\n"
    a = 1 / 0.23131
    vud = abs(V[0, 0])
    vus = abs(V[0, 1])
    chi_square_first_row = (
        (((vus - a / (a ** 2 + 1) * (vud + vus / a)) ** 2
          + (vud - a ** 2 / (a ** 2 + 1) * (vud + vus / a)) ** 2) / 0.00051 ** 2)
        + ((vus - 0.22307) / 0.00055) ** 2
        + ((vud - 0.97375) / 0.00029) ** 2
        - 7.96)
    string += f"delta = {np.sqrt(1 - vud ** 2 - vus ** 2 - abs(V[0,2]) ** 2)} / chi_square_first_row_unitarity = {np.around(chi_square_first_row, 3)}\n"
    gamma = cmath.phase(-V[0, 0] * V[1, 2] * np.conj(V[0, 2])
                        * np.conj(V[1, 0])) / (2 * math.pi) * 360
    string += f"gamma = {gamma} / chi_square_gamma = {chi_square_gamma}\n"
    string += f"V:\n{V}\n"
    string += f"F:\n{V @ V.conj().T}\n"
    string += f"PHENO:\n"
    string += f"Incompatible processes: {results[0]}\n"

    return string, scale_up, scale_down


# This function writes to a file the maximally restrictive pairs found by Minuit
def print_restrictive_pairs_from_minuit(set_maximally_restrictive_pairs,
                                        option,
                                        filename):

    with open(filename, option) as f:
        f.write(
            "LIST OF MAxIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS (AFTER MINUIT):\n\n")
        for data in set_maximally_restrictive_pairs:
            f.write(f"M_u:\n{data[0][0]}\n")
            f.write(f"M_d:\n{data[0][1]}\n")
            f.write(f"\n#################################\n")
            n_points = 0
            for data_point in data[1]:
                M_u, M_d = par_values_to_np_array(data_point[1])
                string, scale_up, scale_down = info_minimum(data_point[1])
                f.write(f"M_u:\n{M_u * scale_up}\n")
                f.write(f"M_d:\n{M_d * scale_down}\n")
                f.write(f"chi_square: {data_point[0]}\n")
                f.write(string)
                f.write(f"\n#################################\n")
                f.write("\n")
                n_points += 1
            f.write(
                f"THERE ARE {n_points} DATA POINTS\n\n")

    return


def pool_function(input_pair):

    data = []

    pair = input_pair[1]
    par_values = []
    chi_square = 1e20
    chi_square_bfv = 1e20
    par_values_bfv = []
    i = 0
    position = check_decoupled_quark(pair)

    while i < N_TRIES:
        par_values = []
        chi_square = 1e20
        chi_square_minuit, par_values_minuit = run_Minuit(
            pair, par_values, 0, position)
        while (chi_square - chi_square_minuit) / chi_square > 1e-5:
            chi_square = chi_square_minuit
            par_values = par_values_minuit
            chi_square_minuit, par_values_minuit = run_Minuit(
                pair, par_values, 1, position)

        if chi_square < chi_square_bfv:
            chi_square_bfv = chi_square
            par_values_bfv = par_values
        i += 1

    data.append([chi_square_bfv, par_values_bfv])

    return [input_pair[1], data]


def main():

    args = sys.argv[1:]
    start = time.time()

    result = Parallel(n_jobs=-1)(delayed(pool_function)(data)
                                 for data in set_maximally_restrictive_pairs)

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

    if set_maximally_restrictive_pairs == []:
        print("NO PAIRS WERE FOUND!")
    else:
        print_restrictive_pairs_from_minuit(result, "w", FILENAME)

    end = time.time()
    print(f"TOTAL TIME = {int((float(end) - float(start)) / 60)} min ", end="")
    print(f"{(int(float(end) - float(start)) % 60)} sec \n")

    return


if __name__ == "__main__":
    main()
