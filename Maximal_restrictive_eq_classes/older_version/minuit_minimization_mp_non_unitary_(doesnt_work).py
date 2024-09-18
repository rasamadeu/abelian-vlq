import extract_obs as ext
import numpy as np
import numpy.matlib
import math
import cmath
from iminuit import Minuit
from multiprocessing import Pool
import sys
import time
import pdb

# Experimental values taken from:
# http://ckmfitter.in2p3.fr/www/results/plots_spring21/num/ckmEval_results_spring21.html

# The main function of this scprit takes a file containing pairs of matrices as input and writes to
# a file the pairs which manage to replicate the experimental data.
# Instead of cycling through the list of pairs with a for cycle, we use the multiproccessing
# function Pool.map().
# Each pair is analyzed with the run_Minuit(pair, par_values, control) function until it
# reaches a maximum of N tries or it satisfies the stopping condition check_chi_squares_limits(par_values).
# The extract_obs.py module contains the function that computes the theoretical values from
# a given pair of matrices.
# https://iminuit.readthedocs.io/en/stable/

# Possible solutions to reduce runtime of the program:
# - optimise Minuit minimization process -> have no idea how this could be done
# - parallel computing -> the minimization proccess is independent, so theoretically all pairs
#                         could be analyzed at the same time. Changing from a for cycle to the
#                         function Pool.map() greatly reduced the runtime, where I use my 6
#                         cores to run. Is there a better parallel computing technique better
#                         suited for this case?

MeV = 1
GeV = 1e3
TeV = 1e6

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

MASS_VLQ = 1.95 * TeV
SIGMA_MASS_VLQ = 0.15 * TeV

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

V_UD = 0.974353
UPPER_SIGMA_V_UD = 0.000049
LOWER_SIGMA_V_UD = 0.000056

V_US = 0.22500
UPPER_SIGMA_V_US = 0.00024
LOWER_SIGMA_V_US = 0.00021

V_UB = 0.003667
UPPER_SIGMA_V_UB = 0.000088
LOWER_SIGMA_V_UB = 0.000073

V_CD = 0.22487
UPPER_SIGMA_V_CD = 0.00024
LOWER_SIGMA_V_CD = 0.00021

V_CS = 0.973521
UPPER_SIGMA_V_CS = 0.000057
LOWER_SIGMA_V_CS = 0.000062

V_CB = 0.04145
UPPER_SIGMA_V_CB = 0.00035
LOWER_SIGMA_V_CB = 0.00061

V_TD = 0.008519
UPPER_SIGMA_V_TD = 0.000075
LOWER_SIGMA_V_TD = 0.000146

V_TS = 0.04065
UPPER_SIGMA_V_TS = 0.00040
LOWER_SIGMA_V_TS = 0.00055

V_TB = 0.999142
UPPER_SIGMA_V_TB = 0.000018
LOWER_SIGMA_V_TB = 0.000023

# QUAL A MEDIDA A USAR? (HÁ 3 NO CKM FITTER)
GAMMA = 72.1
UPPER_SIGMA_GAMMA = 5.4
LOWER_SIGMA_GAMMA = 5.7

MAX_CHI_SQUARE = 9
# 4 mass ratios (u/t, c/t, d/b, s/b) + 9 CKM elements + UT gamma phase
N_OBSERVABLES = 14


def compute_chi_square(D_u, D_d, V):

    # Computation of chi squared of masses
    if (D_u[0].real / D_u[2].real) - RATIO_MASS_UT > 0:
        chi_square_ratio_ut = (((D_u[0].real / D_u[2].real) - RATIO_MASS_UT) / UPPER_SIGMA_RATIO_MASS_UT) ** 2
    else:
        chi_square_ratio_ut = (((D_u[0].real / D_u[2].real) - RATIO_MASS_UT) / LOWER_SIGMA_RATIO_MASS_UT) ** 2

    if (D_u[1].real / D_u[2].real) - RATIO_MASS_CT > 0:
        chi_square_ratio_ct = (((D_u[1].real / D_u[2].real) - RATIO_MASS_CT) / UPPER_SIGMA_RATIO_MASS_CT) ** 2
    else:
        chi_square_ratio_ct = (((D_u[1].real / D_u[2].real) - RATIO_MASS_CT) / LOWER_SIGMA_RATIO_MASS_CT) ** 2

    if (D_d[0].real / D_d[2].real) - RATIO_MASS_DB > 0:
        chi_square_ratio_db = (((D_d[0].real / D_d[2].real) - RATIO_MASS_DB) / UPPER_SIGMA_RATIO_MASS_DB) ** 2
    else:
        chi_square_ratio_db = (((D_d[0].real / D_d[2].real) - RATIO_MASS_DB) / LOWER_SIGMA_RATIO_MASS_DB) ** 2

    if (D_d[1].real / D_d[2].real) - RATIO_MASS_SB > 0:
        chi_square_ratio_sb = (((D_d[1].real / D_d[2].real) - RATIO_MASS_SB) / UPPER_SIGMA_RATIO_MASS_SB) ** 2
    else:
        chi_square_ratio_sb = (((D_d[1].real / D_d[2].real) - RATIO_MASS_SB) / LOWER_SIGMA_RATIO_MASS_SB) ** 2

    m_VLQ = D_u[3].real * MASS_T / D_u[2].real
    if m_VLQ > 1.4 * TeV:
        chi_square_m_VLQ = 0
    else:
        chi_square_m_VLQ = ((m_VLQ - MASS_VLQ) / SIGMA_MASS_VLQ)**2

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

    gamma = cmath.phase(-V[0, 0] * V[1, 2] * np.conj(V[0, 2]) * np.conj(V[1, 0])) / (2 * math.pi) * 360
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
def least_squares(M_u_0_0_re, M_u_0_0_im, M_u_0_1_re, M_u_0_1_im, M_u_0_2_re, M_u_0_2_im, M_u_0_3_re, M_u_0_3_im,
                  M_u_1_0_re, M_u_1_0_im, M_u_1_1_re, M_u_1_1_im, M_u_1_2_re, M_u_1_2_im, M_u_1_3_re, M_u_1_3_im,
                  M_u_2_0_re, M_u_2_0_im, M_u_2_1_re, M_u_2_1_im, M_u_2_2_re, M_u_2_2_im, M_u_2_3_re, M_u_2_3_im,
                  M_u_3_0_re, M_u_3_0_im, M_u_3_1_re, M_u_3_1_im, M_u_3_2_re, M_u_3_2_im, M_u_3_3_re, M_u_3_3_im,
                  M_d_0_0_re, M_d_0_0_im, M_d_0_1_re, M_d_0_1_im, M_d_0_2_re, M_d_0_2_im,
                  M_d_1_0_re, M_d_1_0_im, M_d_1_1_re, M_d_1_1_im, M_d_1_2_re, M_d_1_2_im,
                  M_d_2_0_re, M_d_2_0_im, M_d_2_1_re, M_d_2_1_im, M_d_2_2_re, M_d_2_2_im):

    M_u, M_d = par_values_to_np_array([M_u_0_0_re, M_u_0_0_im, M_u_0_1_re, M_u_0_1_im, M_u_0_2_re, M_u_0_2_im, M_u_0_3_re, M_u_0_3_im,
                                       M_u_1_0_re, M_u_1_0_im, M_u_1_1_re, M_u_1_1_im, M_u_1_2_re, M_u_1_2_im, M_u_1_3_re, M_u_1_3_im,
                                       M_u_2_0_re, M_u_2_0_im, M_u_2_1_re, M_u_2_1_im, M_u_2_2_re, M_u_2_2_im, M_u_2_3_re, M_u_2_3_im,
                                       M_u_3_0_re, M_u_3_0_im, M_u_3_1_re, M_u_3_1_im, M_u_3_2_re, M_u_3_2_im, M_u_3_3_re, M_u_3_3_im,
                                       M_d_0_0_re, M_d_0_0_im, M_d_0_1_re, M_d_0_1_im, M_d_0_2_re, M_d_0_2_im,
                                       M_d_1_0_re, M_d_1_0_im, M_d_1_1_re, M_d_1_1_im, M_d_1_2_re, M_d_1_2_im,
                                       M_d_2_0_re, M_d_2_0_im, M_d_2_1_re, M_d_2_1_im, M_d_2_2_re, M_d_2_2_im])

    # Compute observables from texture zeros
    D_u, D_d, sin_theta_12, sin_theta_13, sin_theta_23, delta, Jarlskog, V = ext.extract_obs(
        M_u, M_d)

    # Check if CKM has null entries
    for i in range(3):
        for j in range(3):
            if abs(V[i, j]) < 1e-10:
                return 1e9

    # Check if masses have neglegible imaginary part from rounding errors

    for mass in D_u:
        if mass.real < 1e10 * mass.imag:
            return 1e9

    for mass in D_d:
        if mass.real < 1e10 * mass.imag:
            return 1e9

    # Compute chi square for observables
    chi_square_ratio_db, chi_square_ratio_sb, chi_square_ratio_ut, chi_square_ratio_ct, chi_square_m_VLQ, chi_square_V_ud, chi_square_V_us, chi_square_V_ub, chi_square_V_cd, chi_square_V_cs, chi_square_V_cb, chi_square_V_td, chi_square_V_ts, chi_square_V_tb, chi_square_gamma = compute_chi_square(D_u, D_d, V)

    chi_square_total = (chi_square_ratio_db + chi_square_ratio_sb
                        + chi_square_ratio_ut + chi_square_ratio_ct + chi_square_m_VLQ
                        + chi_square_V_ud + chi_square_V_us + chi_square_V_ub
                        + chi_square_V_cd + chi_square_V_cs + chi_square_V_cb
                        + chi_square_V_td + chi_square_V_ts + chi_square_V_tb
                        + chi_square_gamma)
    return chi_square_total


def par_values_to_np_array(par_values):

    M_u = np.empty([4, 4], dtype=np.complex_)
    M_d = np.empty([3, 3], dtype=np.complex_)
    for i in range(4):
        for j in range(4):
            if par_values[2 * (j + i * 4)] == 0:
                M_u[i, j] = 0
            else:
                M_u[i, j] = 10 ** par_values[2 * (j + i * 4)] + 10 ** par_values[2 * (j + i * 4) + 1] * 1j

    for i in range(3):
        for j in range(3):
            if par_values[2 * (j + i * 4)] == 0:
                M_d[i, j] = 0
            else:
                M_d[i, j] = 10 ** par_values[2 * (j + i * 4)] + 10 ** par_values[2 * (j + i * 4) + 1] * 1j

    return M_u, M_d


def check_chi_squares_limits(par_values):

    M_u, M_d = par_values_to_np_array(par_values)

    # Compute observables from texture zeros
    D_u, D_d, sin_theta_12, sin_theta_13, sin_theta_23, delta, Jarlskog, V = ext.extract_obs(
        M_u, M_d)

    chi_square_ratio_db, chi_square_ratio_sb, chi_square_ratio_ut, chi_square_ratio_ct, chi_square_m_VLQ, chi_square_V_ud, chi_square_V_us, chi_square_V_ub, chi_square_V_cd, chi_square_V_cs, chi_square_V_cb, chi_square_V_td, chi_square_V_ts, chi_square_V_tb, chi_square_gamma = compute_chi_square(D_u, D_d, V)

    # Check deviation from experimental value for each observable
    if chi_square_ratio_db > MAX_CHI_SQUARE:
        return False

    if chi_square_ratio_sb > MAX_CHI_SQUARE:
        return False

    if chi_square_ratio_ut > MAX_CHI_SQUARE:
        return False

    if chi_square_ratio_ct > MAX_CHI_SQUARE:
        return False

    if chi_square_m_VLQ > MAX_CHI_SQUARE:
        return False

    if chi_square_V_ud > MAX_CHI_SQUARE:
        return False

    if chi_square_V_us > MAX_CHI_SQUARE:
        return False

    if chi_square_V_ub > MAX_CHI_SQUARE:
        return False

    if chi_square_V_cd > MAX_CHI_SQUARE:
        return False

    if chi_square_V_cs > MAX_CHI_SQUARE:
        return False

    if chi_square_V_cb > MAX_CHI_SQUARE:
        return False

    if chi_square_V_td > MAX_CHI_SQUARE:
        return False

    if chi_square_V_ts > MAX_CHI_SQUARE:
        return False

    if chi_square_V_tb > MAX_CHI_SQUARE:
        return False

    if chi_square_gamma > MAX_CHI_SQUARE:
        return False

    return True


def run_Minuit(pair, par_values, control):

    # Input of Minuit optimization
    if control == 0:
        rng = np.random.default_rng()
        index = 0
        for i in range(4):
            for j in range(4):
                if pair[0][i, j] == 1:
                    par_values[index] = 6 - rng.random() * 6   # 7
                    par_values[index + 1] = 6 - rng.random() * 6  # 7
                else:
                    par_values[index] = 0
                    par_values[index + 1] = 0
                index += 2

        for i in range(3):
            for j in range(3):
                if pair[1][i, j] == 1:
                    par_values[index] = 3 - rng.random() * 3  # 7
                    par_values[index + 1] = 3 - rng.random() * 3  # 7
                else:
                    par_values[index] = 0
                    par_values[index + 1] = 0
                index += 2

    # Initialization of Minuit
    m = Minuit(least_squares,
               par_values[0], par_values[1], par_values[2], par_values[3], par_values[4], par_values[5], par_values[6], par_values[7],
               par_values[8], par_values[9], par_values[10], par_values[11], par_values[12], par_values[13], par_values[14], par_values[15],
               par_values[16], par_values[17], par_values[18], par_values[19], par_values[20], par_values[21], par_values[22], par_values[23],
               par_values[24], par_values[25], par_values[26], par_values[27], par_values[28], par_values[29], par_values[30], par_values[31],
               par_values[32], par_values[33], par_values[34], par_values[35], par_values[36], par_values[37],
               par_values[38], par_values[39], par_values[40], par_values[41], par_values[42], par_values[43],
               par_values[44], par_values[45], par_values[46], par_values[47], par_values[48], par_values[49])

    # Fixing texture zeros of M_u
    if par_values[0] == 0:
        m.fixed["M_u_0_0_re"] = True
        m.fixed["M_u_0_0_im"] = True

    if par_values[2] == 0:
        m.fixed["M_u_0_1_re"] = True
        m.fixed["M_u_0_1_im"] = True

    if par_values[4] == 0:
        m.fixed["M_u_0_2_re"] = True
        m.fixed["M_u_0_2_im"] = True

    if par_values[6] == 0:
        m.fixed["M_u_0_3_re"] = True
        m.fixed["M_u_0_3_im"] = True

    if par_values[8] == 0:
        m.fixed["M_u_1_0_re"] = True
        m.fixed["M_u_1_0_im"] = True

    if par_values[10] == 0:
        m.fixed["M_u_1_1_re"] = True
        m.fixed["M_u_1_1_im"] = True

    if par_values[12] == 0:
        m.fixed["M_u_1_2_re"] = True
        m.fixed["M_u_1_2_im"] = True

    if par_values[14] == 0:
        m.fixed["M_u_1_3_re"] = True
        m.fixed["M_u_1_3_im"] = True

    if par_values[16] == 0:
        m.fixed["M_u_2_0_re"] = True
        m.fixed["M_u_2_0_im"] = True

    if par_values[18] == 0:
        m.fixed["M_u_2_1_re"] = True
        m.fixed["M_u_2_1_im"] = True

    if par_values[20] == 0:
        m.fixed["M_u_2_2_re"] = True
        m.fixed["M_u_2_2_im"] = True

    if par_values[22] == 0:
        m.fixed["M_u_2_3_re"] = True
        m.fixed["M_u_2_3_im"] = True

    if par_values[24] == 0:
        m.fixed["M_u_3_0_re"] = True
        m.fixed["M_u_3_0_im"] = True

    if par_values[26] == 0:
        m.fixed["M_u_3_1_re"] = True
        m.fixed["M_u_3_1_im"] = True

    if par_values[28] == 0:
        m.fixed["M_u_3_2_re"] = True
        m.fixed["M_u_3_2_im"] = True

    if par_values[30] == 0:
        m.fixed["M_u_3_3_re"] = True
        m.fixed["M_u_3_3_im"] = True

    # Fixing texture zeros of M_d
    if par_values[32] == 0:
        m.fixed["M_d_0_0_re"] = True
        m.fixed["M_d_0_0_im"] = True

    if par_values[34] == 0:
        m.fixed["M_d_0_1_re"] = True
        m.fixed["M_d_0_1_im"] = True

    if par_values[36] == 0:
        m.fixed["M_d_0_2_re"] = True
        m.fixed["M_d_0_2_im"] = True

    if par_values[38] == 0:
        m.fixed["M_d_1_0_re"] = True
        m.fixed["M_d_1_0_im"] = True

    if par_values[40] == 0:
        m.fixed["M_d_1_1_re"] = True
        m.fixed["M_d_1_1_im"] = True

    if par_values[42] == 0:
        m.fixed["M_d_1_2_re"] = True
        m.fixed["M_d_1_2_im"] = True

    if par_values[44] == 0:
        m.fixed["M_d_2_0_re"] = True
        m.fixed["M_d_2_0_im"] = True

    if par_values[46] == 0:
        m.fixed["M_d_2_1_re"] = True
        m.fixed["M_d_2_1_im"] = True

    if par_values[48] == 0:
        m.fixed["M_d_2_2_re"] = True
        m.fixed["M_d_2_2_im"] = True

    # Run Minuit minimization
    m.migrad()
    return m.fval, m.values


def info_minimum(par_values):

    M_u, M_d = par_values_to_np_array(par_values)

    D_u, D_d, sin_theta_12, sin_theta_13, sin_theta_23, delta, Jarlskog, V = ext.extract_obs(
        M_u, M_d)

    chi_square_ratio_db, chi_square_ratio_sb, chi_square_ratio_ut, chi_square_ratio_ct, chi_square_m_VLQ, chi_square_V_ud, chi_square_V_us, chi_square_V_ub, chi_square_V_cd, chi_square_V_cs, chi_square_V_cb, chi_square_V_td, chi_square_V_ts, chi_square_V_tb, chi_square_gamma = compute_chi_square(D_u, D_d, V)

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
    gamma = cmath.phase(-V[0, 0] * V[1, 2] * np.conj(V[0, 2]) * np.conj(V[1, 0])) / (2 * math.pi) * 360
    string += f"gamma = {gamma} / chi_square_gamma = {chi_square_gamma}\n"
    string += f"V:\n{V}\n"
    return string


# This function reads the set of maximally restrictive pairs stored in a file
def read_maximmaly_restrictive_pairs(filename):

    set_maximally_restrictive_pairs = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != "":
            if line[0] == "P":
                string = line.split(" ")[2]
                while line[0] != "T":
                    line = f.readline()
                    texture_u = []
                    if line == "M_u:\n":
                        line = f.readline()
                        while line != "M_d:\n":
                            texture_u.append([int(s) for s in line if s.isdigit()])
                            line = f.readline()
                        texture_d = []
                        line = f.readline()
                        while line != "\n":
                            texture_d.append([int(s) for s in line if s.isdigit()])
                            line = f.readline()
                        set_maximally_restrictive_pairs.append([string,
                                                                [np.array(texture_u),
                                                                 np.array(texture_d)]])
            line = f.readline()

    return set_maximally_restrictive_pairs


# This function writes to a file the maximally restrictive pairs found by Minuit
def print_restrictive_pairs_from_minuit(set_maximally_restrictive_pairs,
                                        option,
                                        filename):

    set_maximally_restrictive_pairs.sort(key=lambda x: x[0][-2])

    i = 0
    length = len(set_maximally_restrictive_pairs)
    set_maximally_restrictive_pairs_print = []
    pairs = []
    string = set_maximally_restrictive_pairs[0][0]
    while(i < length):
        pairs.append([set_maximally_restrictive_pairs[i][1],
                      set_maximally_restrictive_pairs[i][2],
                      set_maximally_restrictive_pairs[i][3]])

        if i == length - 1:
            set_maximally_restrictive_pairs_print.append([string, pairs])
            break

        if set_maximally_restrictive_pairs[i][0] != set_maximally_restrictive_pairs[i + 1][0]:
            set_maximally_restrictive_pairs_print.append([string, pairs])
            pairs = []
            string = set_maximally_restrictive_pairs[i + 1][0]

        i += 1

    with open(filename, option) as f:
        f.write("LIST OF MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS (AFTER MINUIT):\n\n")
        num_pairs = 0
        for restrictive_pairs_n_zeros_u in set_maximally_restrictive_pairs_print:
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


def pool_function(input_pair):

    pair = input_pair[1]
    par_values = np.ones(50)
    chi_square = 1e10
    i = 0
    while i < 100 and not check_chi_squares_limits(par_values): #15000
        chi_square = 1e10
        chi_square_minuit, par_values_minuit = run_Minuit(pair, par_values, 0)
        while (chi_square - chi_square_minuit) / chi_square > 0.05:
            chi_square = chi_square_minuit
            par_values = par_values_minuit
            chi_square_minuit, par_values_minuit = run_Minuit(pair, par_values, 1)
        i += 1

    if chi_square_minuit < N_OBSERVABLES * MAX_CHI_SQUARE:
        print(pair[0])
        print(pair[1])
        print(f"# de iteracoes = {i}\n")
        print(info_minimum(par_values))

    return input_pair[0], chi_square, par_values, input_pair[1]


def main():

    args = sys.argv[1:]
    start = time.time()
    set_maximally_restrictive_pairs = read_maximmaly_restrictive_pairs(args[0])
    with Pool() as p:
        set_maximally_restrictive_pairs = p.map(pool_function, set_maximally_restrictive_pairs)

    set_maximally_restrictive_pairs_after_minuit = []

    for pair in set_maximally_restrictive_pairs:
        if check_chi_squares_limits(pair[2]):
            set_maximally_restrictive_pairs_after_minuit.append(pair)

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)
    filename = args[0].replace("without_found_non_unitary", "after_MINUIT_non_unitary25")

    if set_maximally_restrictive_pairs_after_minuit == []:
        print("NO PAIRS WERE FOUND!")
        end = time.time()
        print(f"TOTAL TIME = {int((float(end) - float(start)) / 60)} min ", end="")
        print(f"{(int(float(end) - float(start)) % 60)} sec \n")
    return

    print_restrictive_pairs_from_minuit(set_maximally_restrictive_pairs_after_minuit, "w", filename)
    end = time.time()
    print(f"TOTAL TIME = {int((float(end) - float(start)) / 60)} min ", end="")
    print(f"{(int(float(end) - float(start)) % 60)} sec \n")

    return


if __name__ == "__main__":
    main()
