from numba import njit
from math import log, pi, sqrt
from cmath import exp
import numpy as np
import matplotlib.pyplot as plt
import pdb

##########################################################################################
#
#   PAPERS
#
##########################################################################################

# 1 - "Vector-like Singlet Quarks: a Roadmap" by Branco Nishi Penedo ... 2023
# 2 - "Are the CKM anomalies induced by vector-like quarks? Limits from flavor changing
#     and Standard Model precision tests" by Belfatto and Berezhiani 2021
# 3 - "Addressing the CKM unitarity problem with a vector-like up quark" by Branco Penedo
#     Rebelo... 2021

##########################################################################################
#
#   PARAMETERS
#
##########################################################################################

MEV = 1e-3
GEV = 1
TEV = 1e3
S = 1 / 6.582 * 1e22 * MEV ** -1

MW = 80.377 * GEV
MW_ERROR = 0.012 * GEV
MT = 172.69 * GEV
MT_ERROR = 0.30 * GEV
MZ = 91.1876 * GEV
MZ_ERROR = 0.0021 * GEV
MH = 125.25 * GEV
MH_ERROR = 0.17 * GEV
MMU = 105.66 * MEV

# PDG 2020
# MW = 80.379 * GEV

ALPHA = 1 / 137.036
SW_SQUARED = 0.23121

VTB_error = 0.029

GF = 1.1663787 * 1e-5 * GEV ** -2
ZERO = 1e-15
##########################################################################################
#
#   AUXILIARY FUNCTIONS
#
##########################################################################################


# CHECKED
@ njit()
def xm(m):

    return (m / MW) ** 2


# CHECKED
@ njit()
def r(m):

    return m / MT


# CHECKED
@ njit()
def delta(i, j):

    if i == j:
        return 1

    return 0


# CHECKED
@ njit()
def top_decay_fun(a, b):

    return 1 - 2 * (a ** 2 + b ** 2) + (a ** 2 - b ** 2) ** 2


# CHECKED
@ njit()
def VV(V, u, d1, d2):

    return np.conj(V[u, d1]) * V[u, d2]


# CHECKED
@ njit()
def S0_aux(x, y):
    return log(x) / ((x - y) * (1 - x) ** 2) * (4 - 8 * x + x ** 2)


# CHECKED
@ njit()
def S0(x1, x2=0):

    x1 = xm(x1)
    x2 = xm(x2)

    if x2 < ZERO:
        return x1 / (4 * (1 - x1) ** 2) * (4 - 11 * x1 + x1 ** 2 - 6 * x1 ** 2 / (1 - x1) * log(x1))
    else:
        return x1 * x2 / 4 * (S0_aux(x1, x2) + S0_aux(x2, x1) - 3 / ((1 - x1) * (1 - x2)))


# CHECKED
@ njit()
def Y0(x):

    x = xm(x)
    return x / 8 * ((x - 4) / (x - 1) + 3 * x / (1 - x) ** 2 * log(x))


# CHECKED
@ njit()
def N0(x1, x2):

    x1 = xm(x1)
    x2 = xm(x2)

    if x1 == 0 or x2 == 0:
        return 0

    if abs(x1 - x2) < ZERO:
        return x1 / 8

    return x1 * x2 / 8 * ((log(x1) - log(x2)) / (x1 - x2))


# CHECKED
@ njit()
def X0(x):

    x = xm(x)
    return x / 8 * ((x + 2) / (x - 1) + 3 * (x - 2) / (1 - x) ** 2 * log(x))


# CHECKED
@ njit()
def Z0(x):

    x = xm(x)
    return x * (108 - 259 * x + 163 * x ** 2 - 18 * x ** 3) / (144 * (1 - x) ** 3) + (24 * x ** 4 - 6 * x ** 3 - 63 * x ** 2 + 50 * x - 8) / (72 * (1 - x) ** 4) * log(x)


# CHECKED
@ njit()
def E0(x):

    x = xm(x)
    return x * (18 - 11 * x - x ** 2) / (12 * (1 - x) ** 3) - (4 - 16 * x + 9 * x ** 2) / (6 * (1 - x) ** 4) * log(x)


def plot_x_y(x, y, xlabel, ylabel, upper_bound=0, lower_bound=0):

    plt.plot(x, y, color="r")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if upper_bound != 0:
        plt.hlines(y=upper_bound, xmin=x[0], xmax=x[-1], color="b")
    if lower_bound != 0:
        plt.hlines(y=lower_bound, xmin=x[0], xmax=x[-1], color="b")
    plt.show()

    return

##########################################################################################
#
#   MES0N MIXING
#
##########################################################################################

# K0-K0bar, B(d/s)-B(d/s)bar
# We follow [1] since it is the only one that includes QCD correction terms. Furthermore,
# the constraints on [2] seem more "qualitative" << NOTE


# CHECKED
@ njit()
def K0K0bar(V, mc, mt, mT, return_chi):

    # Parameters:
    BK = 0.717
    BK_error = 0.024
    fK = 155.7 * MEV  # fK esta na# primeira tabela da referencia
    fK_error = 0.3 * MEV
    # eta_cc = 1.38
    eta_ct = 0.47
    eta_tt = 0.57
    eta_ct_error = 0.04
    eta_tt_error = 0.01
    lambda_c_ds = VV(V, 1, 0, 1)
    lambda_t_ds = VV(V, 2, 0, 1)
    lambda_T_ds = VV(V, 3, 0, 1)

    # Valores usados pelo paper [1]
    # abs_epsK_SM = 2.16 * 1e-3
    # abs_epsK_SM_error = 0.18 * 1e-3
    # mK = 497.611 * MEV
    # DeltaMK = 3.484 * 1e-12 * MEV
    # DeltaMK_error = 0.006 * 1e-12 * MEV
    # abs_epsK_exp = 2.257 * 1e-3
    # abs_epsK_error = 0.018 * 1e-3

    abs_epsK_SM = 2.16 * 1e-3
    abs_epsK_SM_error = 0.18 * 1e-3
    # PDG 2022 - 0.5293(9) 10^10 hbar s^-1 na seccao sobre K_L
    mK = 497.611 * MEV
    DeltaMK = 3.484 * 1e-12 * MEV  # PDG 2022
    DeltaMK_error = 0.006 * 1e-12 * MEV  # PDG 2022
    abs_epsK_exp = 2.228 * 1e-3  # PDG 2022
    abs_epsK_error = 0.011 * 1e-3  # PDG 2022

    # Computation
    C = BK * GF ** 2 * fK ** 2 * MW ** 2 * mK / (12 * pi ** 2)
    C_error = C * np.sqrt((BK_error / BK) ** 2 + 2 * (fK_error / fK) ** 2)
    # NOTE: We aproximate eta_TT = eta_tt and eta_Tc = eta_tc
    NP = eta_tt * S0(mT) * lambda_T_ds ** 2 + 2 * eta_ct * S0(mc, mT) * lambda_c_ds * \
        lambda_T_ds + 2 * eta_tt * S0(mt, mT) * lambda_t_ds * lambda_T_ds

    DeltaMK_bound = DeltaMK / 2 / C
    DeltaMK_bound_error = DeltaMK_bound * \
        np.sqrt((DeltaMK_error / DeltaMK) ** 2 + (C_error / C) ** 2)
    abs_eps_K_bound = (abs_epsK_exp - abs_epsK_SM) * \
        sqrt(2) * DeltaMK / 0.92 / C
    abs_eps_K_bound_error = np.sqrt((abs_epsK_error * sqrt(2) * DeltaMK / 0.92 / C) ** 2 + (abs_eps_K_bound / DeltaMK * DeltaMK_error)
                                    ** 2 + (abs_epsK_SM_error * sqrt(2) * DeltaMK / 0.92 / C) ** 2 + (abs_eps_K_bound / DeltaMK * DeltaMK_error) ** 2)

   # print("K0K0bar:")
   # print("DeltaMK_bound = ")
   # print(DeltaMK_bound)
   # print("DeltaMK_bound_error = ")
   # print(DeltaMK_bound_error)
   # print("NP = ")
   # print(NP)
   # print("abs(NP) = ")
   # print(abs(NP))
    chi_square_Delta = (abs(NP) - DeltaMK_bound) ** 2 / \
        DeltaMK_bound_error ** 2
   # print("chi_square_Delta_MK = ")
   # print(chi_square_Delta)
   # print("abs_eps_K_bound = ")
   # print(abs_eps_K_bound)
   # print("abs_eps_K_bound_error = ")
   # print(abs_eps_K_bound_error)
   # print("abs(np.imag(NP)) = ")
   # print(abs(np.imag(NP)))
   # print()

    chi_square_eps_K = (abs_eps_K_bound - abs(np.imag(NP))
                        ) ** 2 / abs_eps_K_bound_error ** 2

    if abs(NP) > DeltaMK_bound + DeltaMK_bound_error:
       # print("\n\nINCOMPATIBLE\n\n")
        return chi_square_Delta, chi_square_eps_K

    if abs(np.imag(NP)) > abs_eps_K_bound + abs_eps_K_bound_error:
       # print("\n\nINCOMPATIBLE\n\n")
       # print("eps_K = False")
        return chi_square_Delta, chi_square_eps_K

    if return_chi:
        return chi_square_Delta, chi_square_eps_K
    else:
        return 0, 0


# CHECKED
@ njit()
def eps_ratio(V, F, mu, mc, mt, mT, return_chi):

    B6 = 1.36
    B6_error = 0.23
    B8 = 0.79
    B8_error = 0.05

    P0 = -3.167 + 12.409 * B6 + 1.262 * B8
    PX = 0.540 + 0.023 * B6
    PY = 0.387 + 0.088 * B6
    PZ = 0.474 - 0.017 * B6 - 10.186 * B8
    PE = 0.188 - 1.399 * B6 + 0.459 * B8
    P0_error = np.sqrt((12.409 * B6_error) ** 2 + (1.262 * B8_error) ** 2)
    PX_error = 0.023 * B6_error
    PY_error = 0.088 * B6_error
    PZ_error = np.sqrt((0.017 * B6_error) ** 2 + (10.186 * B8_error) ** 2)
    PE_error = np.sqrt((1.399 * B6_error) ** 2 + (0.459 * B8_error) ** 2)

    Ft = P0 + PX * X0(mt) + PY * Y0(mt) + PZ * Z0(mt) + PE * E0(mt)
    Ft_error = np.sqrt(P0_error ** 2 + (PX_error * X0(mt)) ** 2 + (PY_error * Y0(mt))
                       ** 2 + (PZ_error * Z0(mt)) ** 2 + (PE_error * E0(mt)) ** 2)
    FT = P0 + PX * X0(mT) + PY * Y0(mT) + PZ * Z0(mT) + PE * E0(mT)

    lambda_t_sd = VV(V, 2, 1, 0)
    lambda_T_sd = VV(V, 3, 1, 0)
    # PDG 2020 and 2022
    eps_ratio = 1.66 * 1e-3
    eps_ratio_error = 0.23 * 1e-3

    bound = eps_ratio
    bound_error = np.sqrt(eps_ratio_error ** 2 +
                          abs(Ft_error * np.imag(lambda_t_sd)) ** 2)

    M = np.array([mu, mc, mt, mT])
    NP = FT * np.imag(lambda_T_sd)
    for i in range(4):
        for j in range(4):
            NP = NP + (PX + PY + PZ) * np.imag(np.conj(V[i, 1]) * (
                F[i, j] - delta(i, j)) * V[j, 0]) * N0(M[i], M[j])

   # print("eps_ratio:")
   # print("Im(lambda_t_sd) = ")
   # print(np.imag(lambda_t_sd))
   # print("Im(lambda_T_sd) = ")
   # print(np.imag(lambda_T_sd))
   # print("Im(lambda_t_sd) = ")
   # print(np.imag(lambda_t_sd))
   # print("NP = ")
   # print(NP)
   # print("bound = ")
   # print(bound)
   # print("bound_error = ")
   # print(bound_error)
    chi_square = (bound - (NP + Ft * np.imag(lambda_t_sd))
                  ) ** 2 / bound_error ** 2
   # print("chi_square_eps_ratio = ")
   # print(chi_square)
   # print()

    if chi_square > 1:
       # print("\n\nINCOMPATIBLE\n\n")
        return chi_square

    if return_chi:
        return chi_square
    else:
        return 0


# CHECKED
@ njit()
def BdBdbar(V, mt, mT, return_chi):

    # DUVIDA
    # Delta_BdBd = 3.337 \pm 0.33 10^-10 MeV (PDG 2010)

    # Parameters:
    eta_tt = 0.55
    lambda_t_db = VV(V, 2, 0, 2)
    lambda_T_db = VV(V, 3, 0, 2)

    # Computation
    M12Bd_SM = eta_tt * S0(mt) * lambda_t_db ** 2
    # NOTE: We aproximate eta_TT = eta_tt and eta_Tu = eta_tu
    M12Bd_NP = eta_tt * S0(mT) * lambda_T_db ** 2 + 2 * \
        eta_tt * S0(mt, mT) * lambda_t_db * lambda_T_db

    if abs(M12Bd_SM) == 0:
        print("ERRO: BdBd")
        return 1e10, 1e10

   # print("BdBdbar:")
   # print("M12Bd_SM = ")
   # print(M12Bd_SM)
   # print("M12Bd_NP = ")
   # print(M12Bd_NP)
   # print("np.real(M12Bd_NP / M12Bd_SM) = ")
   # print(np.real(M12Bd_NP / M12Bd_SM))
   # print("np.imag(M12Bd_NP / M12Bd_SM) = ")
   # print(np.imag(M12Bd_NP / M12Bd_SM))
    chi_square_M12Bd_re = abs(
        np.real(M12Bd_NP / M12Bd_SM) - (-0.18)) ** 2 / 0.14 ** 2
    chi_square_M12Bd_im = abs(
        np.imag(M12Bd_NP / M12Bd_SM) - (-0.199)) ** 2 / 0.062 ** 2
   # print("chi_square_M12Bd_re = ")
   # print(chi_square_M12Bd_re)
   # print("chi_square_M12Bd_im = ")
   # print(chi_square_M12Bd_im)
   # print()

    # Constraints
    if np.real(M12Bd_NP / M12Bd_SM) < (-0.18 - 0.14):
       # print("\n\nINCOMPATIBLE\n\n")
       # print("M12Bd_re = False")
        return chi_square_M12Bd_re, chi_square_M12Bd_im
    if np.real(M12Bd_NP / M12Bd_SM) > (-0.18 + 0.14):
       # print("\n\nINCOMPATIBLE\n\n")
       # print("M12Bd_re = False")
        return chi_square_M12Bd_re, chi_square_M12Bd_im
    if np.imag(M12Bd_NP / M12Bd_SM) < (-0.199 - 0.062):
       # print("\n\nINCOMPATIBLE\n\n")
       # print("M12Bd_im = False")
        return chi_square_M12Bd_re, chi_square_M12Bd_im
    if np.imag(M12Bd_NP / M12Bd_SM) > (-0.199 + 0.062):
       # print("\n\nINCOMPATIBLE\n\n")
       # print("M12Bd_im = False")
        return chi_square_M12Bd_re, chi_square_M12Bd_im

    if return_chi:
        return chi_square_M12Bd_re, chi_square_M12Bd_im
    else:
        return 0, 0


# CHECKED
@ njit()
def BsBsbar(V, mt, mT, return_chi):

    # Parameters:
    # BBs = 1.232
    # fBs = 230.3 * MEV
    # mBs = 5279.66 * MEV
    eta_tt = 0.55
    lambda_t_sb = VV(V, 2, 1, 2)
    lambda_T_sb = VV(V, 3, 1, 2)
    # DeltaMBs = 1.1693 * 1e-8 * MEV
    # DeltaMBs_error = 0.0004 * 1e-8 * MEV

    # Computation
    # C = BBs * GF ** 2 * fBs ** 2 * MW ** 2 * mBs / (12 * pi ** 2)
    M12Bs_SM = eta_tt * S0(mt) * lambda_t_sb ** 2
    # NOTE: We aproximate eta_TT = eta_tt and eta_Tu = eta_tu
    M12Bs_NP = eta_tt * S0(mT) * lambda_T_sb ** 2 + 2 * \
        eta_tt * S0(mt, mT) * lambda_t_sb * lambda_T_sb

    if abs(M12Bs_SM) == 0:
        return 1e10, 1e10

   # print("BsBsbar:")
   # print("M12Bs_SM = ")
   # print(M12Bs_SM)
   # print("M12Bs_NP = ")
   # print(M12Bs_NP)
   # print("np.real(M12Bs_NP / M12Bs_SM) = ")
   # print(np.real(M12Bs_NP / M12Bs_SM))
   # print("np.imag(M12Bs_NP / M12Bs_SM) = ")
   # print(np.imag(M12Bs_NP / M12Bs_SM))
    chi_square_M12Bs_re = abs(
        np.real(M12Bs_NP / M12Bs_SM) - (-0.03)) ** 2 / 0.13 ** 2
    chi_square_M12Bs_im = abs(np.imag(M12Bs_NP / M12Bs_SM) - 0) ** 2 / 0.1 ** 2
   # print("chi_square_M12Bs_re = ")
   # print(chi_square_M12Bs_re)
   # print("chi_square_M12Bs_im = ")
   # print(chi_square_M12Bs_im)
   # print()

    # Constraints
    if not (np.real(M12Bs_NP / M12Bs_SM) > (-0.03 - 0.13)
            and np.real(M12Bs_NP / M12Bs_SM) < (-0.03 + 0.13)):
       # print("\n\nINCOMPATIBLE\n\n")
       # print("M12Bs_re = False")
        return chi_square_M12Bs_re, chi_square_M12Bs_im

    if not (np.imag(M12Bs_NP / M12Bs_SM) > (0 - 0.1)
            and np.imag(M12Bs_NP / M12Bs_SM) < (0 + 0.1)):
       # print("\n\nINCOMPATIBLE\n\n")
       # print("M12Bs_im = False")
        return chi_square_M12Bs_re, chi_square_M12Bs_im

    if return_chi:
        return chi_square_M12Bs_re, chi_square_M12Bs_im
    else:
        return 0, 0


# CHECKED
@ njit()
def D0D0bar(F, return_chi):

    # According to [2], the SM box contributions to abs_M12D are of order 1e-17/-16
    # Thus, we can approximate M12D to only NP contributions.
    # Parameters:
    BD = 1
    BD_error = 0.3
    fD = 212 * MEV

    # PDG 2020
    mD = 1864.83 * MEV
    DeltaMD = 0.95 * 1e10 * 6.582 * 10 ** -22 * MEV
    DeltaMD_error = 0.44 * 1e10 * 6.582 * 10 ** -22 * MEV

    # PDG 2022
    # mD = 1864.84 * MEV
    # DeltaMD = 0.997 * 1e10 * 6.582 * 10 ** -22 * MEV
    # DeltaMD_error = 0.116 * 1e10 * 6.582 * 10 ** -22 * MEV
    eta = 0.59

    # Computation
    bound = DeltaMD / (2 * BD * GF ** 2 * fD ** 2 *
                       MW ** 2 * mD / (12 * pi ** 2) * 4 * pi * SW_SQUARED / ALPHA * eta)
    bound_error = bound * \
        sqrt((1 / DeltaMD * DeltaMD_error) ** 2 + (1 / BD * BD_error) ** 2)

    DeltaMD_NP = F[1, 0] ** 2
   # print("D0D0bar:")
   # print("bound = ")
   # print(bound)
   # print("bound_error = ")
   # print(bound_error)
   # print("abs(DeltaMD_NP) = ")
   # print(abs(DeltaMD_NP))
    chi_square = (abs(DeltaMD_NP) - bound) ** 2 / bound_error ** 2
   # print("chi_square_DeltaMD_NP = ")
   # print(chi_square)
   # print()

    if abs(DeltaMD_NP) > bound + bound_error:
       # print("\n\nINCOMPATIBLE\n\n")
       # print("DeltaMD = False")
        return chi_square

    if return_chi:
        return chi_square
    else:
        return 0

##########################################################################################
#
#   RARE TOP DECAYS
#
##########################################################################################


# CHECKED
@ njit()
def top_decay(V, F):

    # PDG 2020 and 2022
    Br_top_Zu = 1.7 * 1e-4
    Br_top_Zc = 2.4 * 1e-4
    Br_top_hu = 1.9 * 1e-4  # 1.2 * 1e-3
    Br_top_hc = 7.3 * 1e-4  # 1.1 * 1e-3

    C = 1 - 3 * r(MW) ** 4 + 2 * r(MW) ** 6

    # No paper [1], eles utilizam abs(V_tb) = 1.013 e Br_top_Z(u,c) < 5 * 1e-4
    top_Zu_bound = sqrt(Br_top_Zu * C * 2 * abs(VV(V, 2, 2, 2)
                                                ) / (1 - 3 * r(MZ) ** 4 + 2 * r(MZ) ** 6))
    top_Zc_bound = sqrt(Br_top_Zc * C * 2 * abs(VV(V, 2, 2, 2)
                                                ) / (1 - 3 * r(MZ) ** 4 + 2 * r(MZ) ** 6))
    top_hu_bound = sqrt(Br_top_hu * C * 2 *
                        abs(VV(V, 2, 2, 2)) / (1 - r(MH) ** 2) ** 2)
    top_hc_bound = sqrt(Br_top_hc * C * 2 *
                        abs(VV(V, 2, 2, 2)) / (1 - r(MH) ** 2) ** 2)

    if sqrt(abs(VV(V, 2, 2, 2))) == 0:
        print("ERRO: top decay")
        return 1e10

    top_Zu_bound_error = top_Zu_bound / sqrt(abs(VV(V, 2, 2, 2))) * VTB_error
    top_Zc_bound_error = top_Zc_bound / sqrt(abs(VV(V, 2, 2, 2))) * VTB_error
    top_hu_bound_error = top_hu_bound / sqrt(abs(VV(V, 2, 2, 2))) * VTB_error
    top_hc_bound_error = top_hc_bound / sqrt(abs(VV(V, 2, 2, 2))) * VTB_error

   # print("Rare top decays:")
   # print("t -> Zu bound:  +/- {top_Zu_bound_error}")
   # print(top_Zu_bound)
   # print("t -> Zc bound:  +/- {top_Zc_bound_error}")
   # print(top_Zc_bound)
   # print("t -> hu bound:  +/- {top_hu_bound_error}")
   # print(top_hu_bound)
   # print("t -> hc bound:  +/- {top_hc_bound_error}")
   # print(top_hc_bound)
   # print("F_ut: ")
   # print(F[0, 2])
   # print("F_ct: ")
   # print(F[1, 2])
   # print("chi_square_t -> Zu = ")
   # print(top_Zu_bound - abs(F[0, 2]) ** 2 / top_Zu_bound_error)
   # print("chi_square_t -> Zc = ")
   # print(top_Zc_bound - abs(F[1, 2]) ** 2 / top_Zc_bound_error)
   # print("chi_square_t -> hu = ")
   # print(top_hu_bound - abs(F[0, 2]) ** 2 / top_hu_bound_error)
   # print("chi_square_t -> hc = ")
   # print(top_hc_bound - abs(F[1, 2]) ** 2 / top_hc_bound_error)
   # print()

    if abs(F[0, 2]) > top_Zu_bound + top_Zu_bound_error:
       # print("\n\nINCOMPATIBLE\n\n")
        return (top_Zu_bound - abs(F[0, 2])) ** 2 / top_Zu_bound_error

    if abs(F[1, 2]) > top_Zc_bound + top_Zc_bound_error:
       # print("\n\nINCOMPATIBLE\n\n")
        return (top_Zc_bound - abs(F[1, 2])) ** 2 / top_Zc_bound_error

    if abs(F[0, 2]) > top_hu_bound + top_hu_bound_error:
       # print("\n\nINCOMPATIBLE\n\n")
        return (top_hu_bound - abs(F[0, 2])) ** 2 / top_hu_bound_error

    if abs(F[1, 2]) > top_hc_bound + top_hc_bound_error:
       # print("\n\nINCOMPATIBLE\n\n")
        return (top_hc_bound - abs(F[1, 2])) ** 2 / top_hc_bound_error

    return 0

##########################################################################################
#
#   RARE MES0N DECAYS
#
##########################################################################################


# CHECKED
@ njit()
def Bd_mumu_decay(V, F, mu, mc, mt, mT, return_chi):

    fBd = 190.0 * MEV
    eta_Y = 1.0113

    # PDG 2020
    # tau_Bd = 1.519 * 1e-12 * S
    # mBd = 5279.65 * MEV
    # BR_Bd_mumu = 1.1 * 1e-10
    # BR_Bd_mumu_error = 1.4 * 1e-10

    # PDG 2022
    tau_Bd = 1.519 * 1e-12 * S
    mBd = 5279.66 * MEV
    BR_Bd_mumu = 7 * 1e-11
    BR_Bd_mumu_error = 13 * 1e-11

    bound = BR_Bd_mumu / (tau_Bd * GF ** 2 / (16 * pi) * (ALPHA / (pi * SW_SQUARED)) **
                          2 * fBd ** 2 * mBd * MMU ** 2 * sqrt(1 - (4 * MMU ** 2) / mBd ** 2) * eta_Y ** 2)
    bound_error = bound * BR_Bd_mumu_error / BR_Bd_mumu
    lambda_t_db = VV(V, 2, 0, 2)
    lambda_T_db = VV(V, 3, 0, 2)
    SM = lambda_t_db * Y0(mt)
    NP = lambda_T_db * Y0(mT)
    M = np.array([mu, mc, mt, mT])
    for i in range(4):
        for j in range(4):
            NP = NP + np.conj(V[i, 0]) * (F[i, j] -
                                          delta(i, j)) * V[j, 2] * N0(M[i], M[j])

   # print("Bd_mumu_decay:")
   # print("bound = ")
   # print(bound)
   # print("bound_error = ")
   # print(bound_error)
   # print("SM = ")
   # print(SM)
   # print("NP = ")
   # print(NP)
   # print("abs(SM + NP) ** 2 = ")
   # print(abs(SM + NP) ** 2)
    chi_square = (abs(SM + NP) ** 2 - bound) ** 2 / bound_error ** 2
   # print("chi_square_Bd_mumu_decay = ")
   # print(chi_square)
   # print()

    if chi_square > 1:
       # print("\n\nINCOMPATIBLE\n\n")
       # print("Bd_mumu_decay = False")
        return chi_square

    if return_chi:
        return chi_square
    else:
        return 0


# CHECKED
@ njit()
def Bs_mumu_decay(V, F, mu, mc, mt, mT, return_chi):

    fBs = 230.3 * MEV
    eta_Y = 1.0113

    # PDG 2020
    # tau_Bs = 1.515 * 1e-12 * S
    # mBs = 5366.88 * MEV
    # BR_Bs_mumu = 3.0 * 1e-9
    # BR_Bs_mumu_error = 0.4 * 1e-9

    # PDG 2022
    tau_Bs = 1.520 * 1e-12 * S
    mBs = 5366.92 * MEV
    BR_Bs_mumu = 3.01 * 1e-9
    BR_Bs_mumu_error = 0.35 * 1e-9

    bound = BR_Bs_mumu / (tau_Bs * GF ** 2 / (16 * pi) * (ALPHA / (pi * SW_SQUARED)) **
                          2 * fBs ** 2 * mBs * MMU ** 2 * sqrt(1 - (4 * MMU ** 2) / mBs ** 2) * eta_Y ** 2)

    bound_error = bound * BR_Bs_mumu_error / BR_Bs_mumu

    lambda_t_sb = VV(V, 2, 1, 2)
    lambda_T_sb = VV(V, 3, 1, 2)
    SM = lambda_t_sb * Y0(mt)
    NP = lambda_T_sb * Y0(mT)
    M = np.array([mu, mc, mt, mT])
    for i in range(4):
        for j in range(4):
            NP = NP + np.conj(V[i, 1]) * (F[i, j] -
                                          delta(i, j)) * V[j, 2] * N0(M[i], M[j])

   # print("Bs_mumu_decay:")
   # print("bound = ")
   # print(bound)
   # print("bound_error = ")
   # print(bound_error)
   # print("SM = ")
   # print(SM)
   # print("NP = ")
   # print(NP)
   # print("abs(SM + NP) ** 2 = ")
   # print(abs(SM + NP) ** 2)
    chi_square = (abs(SM + NP) ** 2 - bound) ** 2 / bound_error ** 2
   # print("chi_square_Bs_mumu_decay = ")
   # print(chi_square)
   # print()

    if chi_square > 1:
       # print("\n\nINCOMPATIBLE\n\n")
       # print("Bs_mumu_decay = False")
        return chi_square

    if return_chi:
        return chi_square
    else:
        return 0


# CHECKED
@ njit()
# DUVIDA: QCD CORRECTIONS FOR U AND C QUARKS????
def K_decay_bound_1(V, F, mu, mc, mt, mT, return_chi):

    if abs(V[0, 1]) ** 2 == 0:
        print("ERRO: K_decay_1")
        return 1e10

    # PDG 2020
    # Br_KP_PiPnunu = 1.7 * 1e-10
    # Br_KP_PiPnunu_error = 1.1 * 1e-10
    # Br_KP_Pi0pnu = 0.0507
    # Br_KP_Pi0pnu_error = 0.04 * 1e-2

    # PDG 2022
    Br_KP_PiPnunu = 1.14 * 1e-10
    Br_KP_PiPnunu_error = 0.4 * 1e-10
    Br_KP_Pi0pnu = 5.07 * 1e-2
    Br_KP_Pi0pnu_error = 0.04 * 1e-2

    rK = 0.901

    bound = Br_KP_PiPnunu / Br_KP_Pi0pnu / \
        (ALPHA ** 2 * rK) * (2 * pi ** 2 * SW_SQUARED ** 2)
    bound_error = bound * np.sqrt((Br_KP_Pi0pnu_error / Br_KP_Pi0pnu)
                                  ** 2 + (Br_KP_PiPnunu_error / Br_KP_PiPnunu) ** 2)

    lambda_c_sd = VV(V, 1, 1, 0)
    lambda_t_sd = VV(V, 2, 1, 0)
    lambda_T_sd = VV(V, 3, 1, 0)
    X_e = 10.6 * 1e-4
    X_mu = 10.6 * 1e-4
    X_tau = 7.1 * 1e-4
    eta = 0.994

    NP = eta * lambda_T_sd * X0(mT)
    M = np.array([mu, mc, mt, mT])
    for i in range(4):
        for j in range(4):
            if i == 2 or i == 3:
                NP = NP + eta * \
                    np.conj(V[i, 1]) * (F[i, j] - delta(i, j)) * \
                    V[j, 0] * N0(M[i], M[j])
            else:
                NP = NP + np.conj(V[i, 1]) * (F[i, j] -
                                              delta(i, j)) * V[j, 0] * N0(M[i], M[j])

    leptons = [X_e, X_mu, X_tau]
    decay_value = 0
    for lep in leptons:
        decay_value += abs(lambda_c_sd * lep +
                           lambda_t_sd * eta * X0(mt) + NP) ** 2 / abs(V[0, 1]) ** 2

   # print("K_decay_bound_1:")
   # print("bound = ")
   # print(bound)
   # print("bound_error = ")
   # print(bound_error)
   # print("decay_value = ")
   # print(decay_value)
   # print("NP = ")
   # print(NP)
    chi_square = (decay_value - bound) ** 2 / bound_error ** 2
   # print("chi_square_K_decay_bound_1 = ")
   # print(chi_square)
   # print()

    if chi_square > 1:
       # print("\n\nINCOMPATIBLE\n\n")
        return chi_square

    if return_chi:
        return chi_square
    else:
        return 0


# CHECKED
# DUVIDA: QCD CORRECTIONS FOR U AND C QUARKS????
@ njit()
def K_decay_bound_2(V, F, mu, mc, mt, mT, return_chi):

    if abs(V[0, 1]) ** 2 == 0:
        print("ERRO: K_decay_2")
        return 1e10

    Br_KL_mumu = 2.5 * 1e-9  # Estimation

    # PDG 2020 and 2022
    Br_KP_munu = 6.356 * 1e-1
    Br_KP_munu_error = 0.011 * 1e-1
    tau_KL = 5.116 * 1e-8 * S
    tau_KL_error = 0.021 * 1e-8 * S
    tau_KP = 1.238 * 1e-8 * S
    tau_KP_error = 0.002 * 1e-8 * S

    # No paper [1] usa-se o valor abs(V_us) = 0.2245(8) e incluem este valor no erro
    bound = Br_KL_mumu / Br_KP_munu * tau_KP / tau_KL / ALPHA ** 2 * \
        (pi ** 2 * SW_SQUARED ** 2)
    bound_error = bound * np.sqrt((Br_KP_munu_error / Br_KP_munu) ** 2 + (
        tau_KL_error / tau_KL) ** 2 + (tau_KP_error / tau_KP) ** 2)  # + (2 / abs(V[0, 1]) * 0.0008) ** 2

    lambda_c_sd = VV(V, 1, 1, 0)
    lambda_t_sd = VV(V, 2, 1, 0)
    lambda_T_sd = VV(V, 3, 1, 0)
    Y = 2.94 * 1e-4
    eta = 1.012

    SM = Y * np.real(lambda_c_sd) + eta * np.real(lambda_t_sd) * Y0(mt)
    NP = eta * np.real(lambda_T_sd) * Y0(mT)
    M = np.array([mu, mc, mt, mT])
    for i in range(4):
        for j in range(4):
            if i == 2 or i == 3:
                NP = NP + eta * \
                    np.real(np.conj(V[i, 1]) * (F[i, j] -
                            delta(i, j)) * V[j, 0]) * N0(M[i], M[j])
            else:
                NP = NP + \
                    np.real(np.conj(V[i, 1]) * (F[i, j] -
                            delta(i, j)) * V[j, 0]) * N0(M[i], M[j])

   # print("K_decay_bound_2:")
   # print("bound = ")
   # print(bound)
   # print("bound_error = ")
   # print(bound_error)
   # print("abs(SM + NP) ** 2 = ")
   # print(abs(SM + NP) ** 2)
   # print("NP = ")
   # print(NP)
    chi_square = (
        (abs(SM + NP) / abs(np.conj(V[0, 1]) * V[0, 1])) ** 2 - bound) ** 2 / bound_error ** 2
   # print("chi_square_K_decay_bound_2 = ")
   # print(chi_square)
   # print()

    if abs(SM + NP) ** 2 > bound + bound_error:
       # print("\n\nINCOMPATIBLE\n\n")
        return chi_square

    if return_chi:
        return chi_square
    else:
        return 0


# CHECKED
# DUVIDA: QCD CORRECTIONS FOR U AND C QUARKS????
@ njit()
def K_decay_bound_3(V, F, mu, mc, mt, mT, return_chi):

    # PDG 2020 and 2022
    Br_KL_Pi0nunu = 3 * 1e-9
    Br_KL_Pi0nunu_SM = 3 * 1e-11
    Br_KL_Pi0nunu_SM_error = 0.6 * 1e-11

    lambda_c_sd = VV(V, 1, 1, 0)
    lambda_t_sd = VV(V, 2, 1, 0)
    lambda_T_sd = VV(V, 3, 1, 0)

    SM = lambda_c_sd * X0(mc) + lambda_t_sd * X0(mt)
    bound = Br_KL_Pi0nunu / Br_KL_Pi0nunu_SM * abs(np.imag(SM))**2
    bound_error = bound * (1 / Br_KL_Pi0nunu_SM * Br_KL_Pi0nunu_SM_error)

    NP = lambda_T_sd * X0(mT)
    M = np.array([mu, mc, mt, mT])
    for i in range(4):
        for j in range(4):
            NP = NP + np.conj(V[i, 1]) * (F[i, j] -
                                          delta(i, j)) * V[j, 0] * N0(M[i], M[j])

   # print("K_decay_bound_3:")
   # print("bound = ")
   # print(bound)
   # print("bound_error = ")
   # print(bound_error)
   # print("abs(np.imag(SM + NP)) ** 2 = ")
   # print(abs(np.imag(SM + NP)) ** 2)
   # print("NP = ")
   # print(NP)
    if bound_error == 0:
        print("ERRO: K_decay_3")
        return 1e10

    chi_square = (abs(np.imag(SM + NP)) ** 2 -
                  bound) ** 2 / bound_error ** 2
   # print("chi_square_K_decay_bound_3 = ")
   # print(chi_square)
   # print()

    if abs(np.imag(SM + NP)) ** 2 > bound + bound_error:
       # print("\n\nINCOMPATIBLE\n\n")
        return chi_square

    if return_chi:
        return chi_square
    else:
        return 0


##########################################################################################


@ njit()
def phenomenology_tests(V, F, mu, mc, mt, mT):

    mu = abs(mu)
    mc = abs(mc)
    mt = abs(mt)
    mT = abs(mT)
    i = 0
    chi_square = 0

    if mu > 1e-10 and mc > 1e-10 and mt > 1e-10 and mT > 1e-10:
        output = K0K0bar(V, mc, mt, mT, 0)
        if output[0]:
            chi_square += output[0] + output[1]
            i += 1

        output = eps_ratio(V, F, mu, mc, mt, mT, 0)
        if output:
            chi_square += output
            i += 1

        output = D0D0bar(F, 0)
        if output:
            chi_square += output
            i += 1

        output = BdBdbar(V, mt, mT, 0)
        if output[0]:
            chi_square += output[0] + output[1]
            i += 1

        output = BsBsbar(V, mt, mT, 0)
        if output[0]:
            chi_square += output[0] + output[1]
            i += 1

        if top_decay(V, F):
            i += 1

        output = Bd_mumu_decay(V, F, mu, mc, mt, mT, 0)
        if output:
            chi_square += output
            i += 1

        output = Bs_mumu_decay(V, F, mu, mc, mt, mT, 0)
        if output:
            chi_square += output
            i += 1

        output = K_decay_bound_1(V, F, mu, mc, mt, mT, 0)
        if output:
            chi_square += output
            i += 1

        output = K_decay_bound_2(V, F, mu, mc, mt, mT, 0)
        if output:
            chi_square += output
            i += 1

        output = K_decay_bound_3(V, F, mu, mc, mt, mT, 0)
        if output:
            chi_square += output
            i += 1
        return i, chi_square
    return 0, 1e20


def read_info(filename):

    data = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != "":
            if len(line) > 1:
                if line.split()[0] == "chi_square:":
                    chi_square = float(line.split()[1])

                    while line.split()[0] != "phase":
                        line = f.readline()
                    phase = float(line.split()[2])

                    while line.split()[0] != "m_VLQ":
                        line = f.readline()
                    m_VLQ = float(line.split()[2][1:7]) * TEV

                    while line.split()[0] != "|V_ud|^2":
                        line = f.readline()
                    unitarity = float(line.split()[-1])

                    while line[0:2] != "V:":
                        line = f.readline()

                    V = []
                    for i in range(3):
                        line = f.readline()[2:-2].split()
                        V.append([complex(line[0]), complex(
                            line[1]), complex(line[2])])
                    line = f.readline()[2:-3].split()
                    V.append([complex(line[0]), complex(
                        line[1]), complex(line[2])])

                    V = np.array(V)
                    data.append([V, V @ V.conj().T, m_VLQ,
                                phase, unitarity, chi_square])

            line = f.readline()

    return data


def main():

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

    # TESTE DAS FUNCOES (ASSUMINDO BEST FIT VALUES DO SM)
    # s12 = 0.2265
    # c12 = sqrt(1 - s12 ** 2)
    # s13 = 0.00361
    # c13 = sqrt(1 - s13 ** 2)
    # s23 = 0.04053
    # c23 = sqrt(1 - s23 ** 2)
    # delta = 1.144
    # V = np.array([[c12 * c13, s12 * c13, s13 * exp(-delta * 1j)],
    #               [-s12 * c23 - c12 * s23 * s13 *
    #                   exp(delta * 1j), c12 * c23 - s12 * s23 * s13 * exp(delta * 1j), s23 * c13],
    #               [s12 * s23 - c12 * c23 * s13 *
    #                   exp(delta * 1j), -c12 * s23 - s12 * c23 * s13 * exp(delta * 1j), c23 * c13],
    #               [0, 0, 0]])

    # V MATRIX FOR:
    V = np.array(
        [[9.74319907e-01+0.00000000e+00j, 2.25047230e-01+1.47252357e-04j, 1.32712976e-05+3.82832460e-03j],
         [-2.25005478e-01+0.00000000e+00j, 9.73093820e-01 -
          9.80331809e-06j, -3.79478147e-02+1.65783280e-02j],
            [-8.55880720e-03+0.00000000e+00j, 3.69859490e-02 +
             1.70206782e-02j, 9.99134166e-01-2.35717459e-05j],
            [3.12917247e-14+0.00000000e+00j, 2.80085045e-02-2.03445371e-10j, -1.07567460e-03+4.77162250e-04j]]
    )
    F = V @ V.conj().T
   # print("V:")
   # print(V)
   # print("F:")
   # print(F)

    mu = 2.1599 * MEV  # 2.1600 * MEV
    mc = 1.27 * GEV  # 1.27 * GEV
    mt = 172.69 * GEV
    mT = 9.847 * TEV
    results = phenomenology_tests(V, F, mu, mc, mt, mT)

#    data = read_info("best_fit_value_b_decoupled_200.txt")
#    limit = 100
#    best_fit_pheno = []
#    for point in data:
#        if point[4] > 2 and point[5] < 100:
#            j = phenomenology_tests(point[0], point[1], mu, mc, mt, point[2])
#            if j == 3:
#                best_fit_pheno.append([point[2], point[3], D0D0bar(point[1])])
#
#   # print("BEST:")
#    for i in best_fit_pheno:
#       # print(i)

    return


if __name__ == "__main__":
    main()
