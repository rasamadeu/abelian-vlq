from numba import njit  # import pdb
import time
import sys
from joblib import delayed, Parallel
import cmath
import math
import numpy.matlib
import numpy as np
import extract_obs as ext

# Experimental values taken from PDG revision of 2022:
# https://pdg.lbl.gov/2022/reviews/rpp2022-rev-ckm-matrix.pdf (January 2023)

##################################################################################################
#
#   INPUT PARAMETERS
#
##################################################################################################

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

MASS_DECOUPLED = MASS_D
MASS_P = MASS_B
MASS_M = MASS_S

X2 = 93.4835644
THETA = 5.865
MASS_U = 2.161
MASS_C = 1269.8
MASS_VLQ = 8.906473 * TeV
A1_BFV = 3.25736416e+04
A2_BFV = 1.50962246e+03
A3_BFV = 9.64502145
INDEX = 1
FILENAME = "2HDM_minuit_d_decoupled"

N_POINTS = 1
A1 = np.array([A1_BFV + i / (N_POINTS + 1) * (i - N_POINTS / 2)
               for i in range(N_POINTS)])

A2 = np.array([A2_BFV + i / (N_POINTS + 1) * (i - N_POINTS / 2)
               for i in range(N_POINTS)])

A3 = np.array([A3_BFV + i / (N_POINTS + 1) * (i - N_POINTS / 2)
               for i in range(N_POINTS)])

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


def c(a1, a2, a3, mu, mc, mt, mT):
    return mu * mc * mt * mT / (a1 * a2 * a3)


def g(a1, a3, b3):
    return a1 ** 2 * a3 ** 2 / (a3 ** 2 + b3 ** 2)


def d(a1, a2, a3, b3, mu, mc, mt, mT):

    A = mu ** 2 + mc ** 2 + mt ** 2 + mT ** 2
    C = c(a1, a2, a3, mu, mc, mt, mT)
    return A - (a1 ** 2 + a2 ** 2 + a3 ** 2 + b3 ** 2 + C ** 2)


def f(a1, a2, a3, b3, mu, mc, mt, mT):

    C = c(a1, a2, a3, mu, mc, mt, mT)
    C = C ** 2
    a1 = a1 ** 2
    a2 = a2 ** 2
    a3 = a3 ** 2
    b3 = b3 ** 2
    mu = mu ** 2
    mc = mc ** 2
    mt = mt ** 2
    mT = mT ** 2

    A = mu * mc * mt + mu * mc * mT + mu * mt * mT + mc * mt * mT
    return (A - C * (a1 * a3 + a2 * (a1 + a3 + b3)) - a1 * a2 * a3) / (a3 + b3)


def b2(a1, a2, a3, b3, mu, mc, mt, mT):

    C = c(a1, a2, a3, mu, mc, mt, mT)
    G = g(a1, a3, b3)
    D = d(a1, a2, a3, b3, mu, mc, mt, mT)
    F = f(a1, a2, a3, b3, mu, mc, mt, mT)
    C = C ** 2

    return math.sqrt((G + D - C + math.sqrt((C - G - D) ** 2 - 4 * (F - C * D))) / 2)


def b1(a1, a2, a3, b3, mu, mc, mt, mT):

    C = c(a1, a2, a3, mu, mc, mt, mT)
    G = g(a1, a3, b3)
    B2 = b2(a1, a2, a3, b3, mu, mc, mt, mT)
    return math.sqrt((f(a1, a2, a3, b3, mu, mc, mt, mT) - G * B2 ** 2) / (C ** 2 + B2 ** 2))


def b3(a1, a2, a3, b3, mu, mc, mt, mT):

    C = c(a1, a2, a3, mu, mc, mt, mT)
    B2 = b2(a1, a2, a3, b3, mu, mc, mt, mT)
    B1 = b1(a1, a2, a3, b3, mu, mc, mt, mT)

    C = C ** 2
    a1 = a1 ** 2
    a2 = a2 ** 2
    a3 = a3 ** 2
    B1 = 1.6958344e5 ** 2
    B2 = 3.63064647e5 ** 2
    mu = mu ** 2
    mc = mc ** 2
    mt = mt ** 2
    mT = mT ** 2

    A = mu * mc + mu * mt + mu * mT + mc * mt + mc * mT + mt * mT

    return math.sqrt((A - (a1 * (a2 + a3) + a2 * a3 + (a1 + a2 + a3) * C + (a3 + B2 + C) * B1 + (a1 + a3) * B2)) / (a2 + B1 + B2 + C))


def compute_up_parameters(a1, a2, a3, guess, mu, mc, mt, mT):

    count = 0
    i = b3(a1, a2, a3, guess, mu, mc, mt, mT)
    while True:
        B3 = b3(a1, a2, a3, i, mu, mc, mt, mT)
        if abs((B3 - i) / i) < 1e-30:
            break
        i = B3
        count += 1

    B1 = b1(a1, a2, a3, B3, mu, mc, mt, mT)
    B2 = b2(a1, a2, a3, B3, mu, mc, mt, mT)
    C = c(a1, a2, a3, mu, mc, mt, mT)
    print(count)

    return B1, B2, B3, C


def compute_delta_unitarity(V):

    a = 1 / 0.23131
    vud = abs(V[0, 0])
    vus = abs(V[0, 1])
    chi_square_first_row = (
        (((vus - a / (a ** 2 + 1) * (vud + vus / a)) ** 2
          + (vud - a ** 2 / (a ** 2 + 1) * (vud + vus / a)) ** 2) / 0.00051 ** 2)
        + ((vus - 0.22307) / 0.00055) ** 2
        + ((vud - 0.97375) / 0.00029) ** 2
        - 7.96)

    return np.sqrt(1 - vud ** 2 - vus ** 2 - abs(V[0, 2]) ** 2), chi_square_first_row


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


@ njit()
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


def compute_mass_matrices(pair, a1, a2, a3):

    x1 = np.sqrt(MASS_P ** 2 * MASS_M ** 2 / X2 ** 2)
    rho = np.sqrt((MASS_P ** 2 - X2 ** 2) * (X2 ** 2 - MASS_M ** 2) / X2 ** 2)
    B1, B2, B3, C = compute_up_parameters(
        a1, a2, a3, MASS_U, MASS_C, MASS_T, MASS_VLQ)

    M_u = np.array([[0, 0, a1, B1],
                    [0, B2, 0, a2],
                    [B3, 0, a3, 0],
                    [0, C, 0, 0]])

    if pair[1][2, 2] == 0:
        M_d = np.array([[0,               0,                          x1],
                        [0,              X2, rho * cmath.exp(THETA * 1j)],
                        [MASS_DECOUPLED,   0,                           0]])
    else:
        M_d = np.array([[0,               0,                          x1],
                        [0,  MASS_DECOUPLED,                           0],
                        [X2,               0, rho * cmath.exp(THETA * 1j)]])

    return M_u, M_d


def info_minimum(M_u, M_d):

    D_u, D_d, sin_theta_12, sin_theta_13, sin_theta_23, delta, Jarlskog, V = ext.extract_obs(
        M_u, M_d)

    chi_square_ratio_db, chi_square_ratio_sb, chi_square_ratio_ut, chi_square_ratio_ct, chi_square_m_VLQ, chi_square_V_ud, chi_square_V_us, chi_square_V_ub, chi_square_V_cd, chi_square_V_cs, chi_square_V_cb, chi_square_V_td, chi_square_V_ts, chi_square_V_tb, chi_square_gamma = compute_chi_square(
        D_u, D_d, V)

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

    delta_unitarity, chi_square_first_row = compute_delta_unitarity(V)
    string += f"delta = {delta_unitarity} / chi_square_first_row_unitarity = {np.around(chi_square_first_row, 3)}\n"
    gamma = cmath.phase(-V[0, 0] * V[1, 2] * np.conj(V[0, 2])
                        * np.conj(V[1, 0])) / (2 * math.pi) * 360
    string += f"gamma = {gamma} / chi_square_gamma = {chi_square_gamma}\n"
    string += f"V:\n{V}\n"

    return string, scale_up, scale_down


# This function writes to a file the maximally restrictive pairs found by Minuit
def print_restrictive_pairs_from_minuit(output,
                                        option,
                                        filename,
                                        ):

    with open(filename, option) as f:
        f.write(
            "LIST OF MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS (AFTER MINUIT):\n\n")
        n = 0

        for i, data in enumerate(output):
            for datapoint in data:
                string, scale_up, scale_down = info_minimum(
                    datapoint[-2, -1])
                f.write(f"M_u:\n{datapoint[-2] * scale_up}\n")
                f.write(f"M_d:\n{datapoint[-1] * scale_down}\n")
                f.write(f"chi_square: {datapoint[0]}\n")
                f.write(string)
                f.write(f"\n#################################\n")
                f.write("\n")
                n += 1
            f.write(
                f"THERE ARE {n} DATA POINTS\n\n")

            if INDEX == 0:
                with open(f"{filename}_5_3_a{i+1}_bfv.dat", "w") as f:
                    f.write(
                        f'{"a1" : >15} {"a2" : >15} {"a3" : >15} {"chi^2" : >15} {"chi^2_gamma" : >15} {"delta" : >15} {"chi^2_delta" : >15} {"m_T(TeV)" : >15} \n\n')
                    for i in range(N_POINTS ** 2):
                        f.write(
                            f"{output[i][1]: 15.7f} {output[i][2]: 15.7f} {output[i][3]: 15.6f} {output[i][0]: 15.2f} {output[i][4]: 15.2f} {output[i][5]: 15.5f} {output[i][6]: 15.2f} {output[i][7]: 15.3f}\n")

            if INDEX == 1:
                with open(f"{filename}_5_1_a{i+1}_bfv.dat", "w") as f:
                    f.write(
                        f'{"a1" : >15} {"a2" : >15} {"a3" : >15} {"chi^2" : >15} {"chi^2_gamma" : >15} {"delta" : >15} {"chi^2_delta" : >15} {"m_T(TeV)" : >15} \n\n')
                    for i in range(N_POINTS ** 2):
                        f.write(
                            f"{output[i][1]: 15.7f} {output[i][2]: 15.6f} {output[i][3]: 15.6f} {output[i][0]: 15.2f} {output[i][4]: 15.2f} {output[i][5]: 15.5f} {output[i][6]: 15.2f} {output[i][7]: 15.3f}\n")

    return


def pool_function(input):

    pair = set_maximally_restrictive_pairs[INDEX][1]
    a1 = input[0]
    a2 = input[1]
    a3 = input[2]

    M_u, M_d = compute_mass_matrices(pair, a1, a2, a3)

    D_u, D_d, sin_theta_12, sin_theta_13, sin_theta_23, delta, Jarlskog, V = ext.extract_obs(
        M_u, M_d)

    chi_square_ratio_db, chi_square_ratio_sb, chi_square_ratio_ut, chi_square_ratio_ct, chi_square_m_VLQ, chi_square_V_ud, chi_square_V_us, chi_square_V_ub, chi_square_V_cd, chi_square_V_cs, chi_square_V_cb, chi_square_V_td, chi_square_V_ts, chi_square_V_tb, chi_square_gamma = compute_chi_square(
        D_u, D_d, V)

    delta_unitarity, chi_square_first_row = compute_delta_unitarity(V)
    return np.array([chi_square_bfv, a1, a2, a3, chi_square_gamma, delta_unitarity, chi_square_first_row, D_u[3] * MASS_T / D_u[2] / TeV, M_u, M_d], dtype='object')


def main():

    args = sys.argv[1:]
    start = time.time()

    input_a3 = []
    input_a2 = []
    input_a1 = []

    for a1 in A1:
        for a2 in A2:
            input_a3.append([a1, a2, A3_BFV])

    for a1 in A1:
        for a3 in A3:
            input_a2.append([a1, A2_BFV, a3])

    for a2 in A2:
        for a3 in A3:
            input_a1.append([A1_BFV, a2, a3])

    guess = 1260

    print(c(A1[0], A2[0],
          A3[0], MASS_U, MASS_C, MASS_T, MASS_VLQ))
    print(c(A1[0], A2[0],
          A3[0], MASS_U, MASS_C, MASS_T, MASS_VLQ) / 8.89907e6 - 1)
    print(b1(A1[0], A2[0],
          A3[0], guess, MASS_U, MASS_C, MASS_T, MASS_VLQ))
    print(b1(A1[0], A2[0],
          A3[0], guess, MASS_U, MASS_C, MASS_T, MASS_VLQ) / 1.6958344e5 - 1)
    print(b2(A1[0], A2[0],
          A3[0], guess, MASS_U, MASS_C, MASS_T, MASS_VLQ))
    print(b2(A1[0], A2[0],
          A3[0], guess, MASS_U, MASS_C, MASS_T, MASS_VLQ) / 3.63064647e5 - 1)
    print(b3(A1[0], A2[0],
          A3[0], guess, MASS_U, MASS_C, MASS_T, MASS_VLQ))
    print(b3(A1[0], A2[0],
          A3[0], guess, MASS_U, MASS_C, MASS_T, MASS_VLQ) / 1.26e3**2 - 1)
    print(compute_up_parameters(A1[0], A2[0],
          A3[0], 1000, MASS_U, MASS_C, MASS_T, MASS_VLQ))
    return

    output_a3 = Parallel(n_jobs=-1)(delayed(pool_function)(data)
                                    for data in input_a3)

    output_a2 = Parallel(n_jobs=-1)(delayed(pool_function)(data)
                                    for data in input_a2)

    output_a1 = Parallel(n_jobs=-1)(delayed(pool_function)(data)
                                    for data in input_a1)

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

    if output_a1 == []:
        print("NO PAIRS WERE FOUND!")
    else:
        print_restrictive_pairs_from_minuit(
            output_a1, output_a2, output_a3, "w", FILENAME)

    end = time.time()
    print(f"TOTAL TIME = {int((float(end) - float(start)) / 60)} min ", end="")
    print(f"{(int(float(end) - float(start)) % 60)} sec \n")

    return


if __name__ == "__main__":
    main()
