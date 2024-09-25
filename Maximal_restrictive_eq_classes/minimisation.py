import numpy as np
import numpy.matlib
import math
import cmath
import sys
import time
from iminuit import Minuit
from multiprocessing import Pool
from numba import njit

import extract_obs as ext
import physical_parameters as phys
import io_mrt

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
#   MINIMISATION CONDITIONS
#
##################################################################################################

# 4 mass ratios (u/t, c/t, d/b, s/b) + 9 CKM elements + UT gamma phase
N_OBSERVABLES = 14
N_TRIES = 20
MAX_CHI_SQUARE = 9

VLQ_LOWER_BOUND = 1.4 * phys.TeV
SIGMA_MASS_VLQ = 0.15 * phys.TeV
FILENAME_OUTPUT = "output/test"


##################################################################################################

@njit
def compute_chi_square(D_u, D_d, V):

    # Computation of chi squared of masses
    chi_square_mass = np.empty((4), dtype=float)
    i = 0

    for j in range(2):
        if abs(D_u[j]) / abs(D_u[2]) - phys.RATIO_UP[j] > 0:
            chi_square_mass[i] = (
                (abs(D_u[j]) / abs(D_u[2]) - phys.RATIO_UP[j]) / phys.UPPER_SIGMA_RATIO_UP[j]) ** 2
        else:
            chi_square_mass[i] = (
                (abs(D_u[j]) / abs(D_u[2]) - phys.RATIO_UP[j]) / phys.LOWER_SIGMA_RATIO_UP[j]) ** 2
        i += 1

    for j in range(2):
        if abs(D_d[j]) / abs(D_d[2]) - phys.RATIO_DOWN[j] > 0:
            chi_square_mass[i] = (
                (abs(D_d[j]) / abs(D_d[2]) - phys.RATIO_DOWN[j]) / phys.UPPER_SIGMA_RATIO_DOWN[j]) ** 2
        else:
            chi_square_mass[i] = (
                (abs(D_d[j]) / abs(D_d[2]) - phys.RATIO_DOWN[j]) / phys.LOWER_SIGMA_RATIO_DOWN[j]) ** 2
        i += 1

    m_VLQ = abs(D_u[3]) * phys.MASS_T / abs(D_u[2])
    if m_VLQ > VLQ_LOWER_BOUND:
        chi_square_m_VLQ = 0
    else:
        chi_square_m_VLQ = ((m_VLQ - VLQ_LOWER_BOUND) / SIGMA_MASS_VLQ) ** 2

    # Computation of absolute value of entries of V_CKM
    chi_square_V = np.empty((3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            chi_square_V[i, j] = (
                (abs(V[i, j]) - phys.V_CKM[i, j]) / phys.SIGMA_V_CKM[i, j]) ** 2

    gamma = cmath.phase(-V[0, 0] * V[1, 2] * np.conj(V[0, 2])
                        * np.conj(V[1, 0])) / (2 * math.pi) * 360
    if gamma - phys.GAMMA > 0:
        chi_square_gamma = ((gamma - phys.GAMMA) / phys.UPPER_SIGMA_GAMMA) ** 2
    else:
        chi_square_gamma = ((gamma - phys.GAMMA) / phys.LOWER_SIGMA_GAMMA) ** 2

    return chi_square_mass, chi_square_m_VLQ, chi_square_V, chi_square_gamma


# Function to be mininimized
def least_squares(M_u_0_0_re, M_u_0_0_im, M_u_0_1_re, M_u_0_1_im, M_u_0_2_re, M_u_0_2_im, M_u_0_3_re, M_u_0_3_im,
                  M_u_1_0_re, M_u_1_0_im, M_u_1_1_re, M_u_1_1_im, M_u_1_2_re, M_u_1_2_im, M_u_1_3_re, M_u_1_3_im,
                  M_u_2_0_re, M_u_2_0_im, M_u_2_1_re, M_u_2_1_im, M_u_2_2_re, M_u_2_2_im, M_u_2_3_re, M_u_2_3_im,
                  M_u_3_0_re, M_u_3_0_im, M_u_3_1_re, M_u_3_1_im, M_u_3_2_re, M_u_3_2_im, M_u_3_3_re, M_u_3_3_im,
                  M_d_0_0_re, M_d_0_0_im, M_d_0_1_re, M_d_0_1_im, M_d_0_2_re, M_d_0_2_im,
                  M_d_1_0_re, M_d_1_0_im, M_d_1_1_re, M_d_1_1_im, M_d_1_2_re, M_d_1_2_im,
                  M_d_2_0_re, M_d_2_0_im, M_d_2_1_re, M_d_2_1_im, M_d_2_2_re, M_d_2_2_im):

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
    D_u, D_d, delta, Jarlskog, V = ext.extract_obs(M_u, M_d)

    # Check if CKM has null entries
    for i in range(3):
        for j in range(3):
            if abs(V[i, j]) < 1e-10:
                return 1e9

    # Compute chi square for observables
    chi_square_mass, chi_square_m_VLQ, chi_square_V, chi_square_gamma = compute_chi_square(
        D_u, D_d, V)

    return np.sum(chi_square_mass) + chi_square_m_VLQ + np.sum(chi_square_V) + chi_square_gamma


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


def check_chi_squares_limits(par_values):

    M_u, M_d = par_values_to_np_array(par_values)

    # Compute observables from texture zeros
    D_u, D_d, delta, Jarlskog, V = ext.extract_obs(
        M_u, M_d)

    chi_square_mass, chi_square_m_VLQ, chi_square_V, chi_square_gamma = compute_chi_square(
        D_u, D_d, V)

    # Check deviation from experimental value for each observable
    for chi_square in chi_square_mass:
        if chi_square > MAX_CHI_SQUARE:
            return False

    for line in chi_square_V:
        for chi_square in line:
            if chi_square > MAX_CHI_SQUARE:
                return False

    if chi_square_m_VLQ > MAX_CHI_SQUARE:
        return False

    if chi_square_gamma > MAX_CHI_SQUARE:
        return False

    return True


def run_Minuit(pair, par_values, use_par_values):

    # Input of Minuit optimization
    if use_par_values:
        M_u, M_d = par_values_to_np_array(par_values)
    else:
        rng = np.random.default_rng()
        M_u = np.zeros((4, 4), dtype=np.complex128)
        M_d = np.zeros((3, 3), dtype=np.complex128)
        for i in range(4):
            for j in range(4):
                if pair[0][i, j] == 1:
                    M_u[i, j] = 10 ** (rng.random() * 6) + \
                        10 ** (rng.random() * 6) * 1j

        for i in range(3):
            for j in range(3):
                if pair[1][i, j] == 1:
                    M_d[i, j] = 10 ** (rng.random() * 3) + \
                        10 ** (rng.random() * 3) * 1j

    # Initialization of Minuit
    m = Minuit(least_squares,
               M_u[0, 0].real, M_u[0, 0].imag, M_u[0, 1].real, M_u[0, 1].imag,
               M_u[0, 2].real, M_u[0, 2].imag, M_u[0, 3].real, M_u[0, 3].imag,
               M_u[1, 0].real, M_u[1, 0].imag, M_u[1, 1].real, M_u[1, 1].imag,
               M_u[1, 2].real, M_u[1, 2].imag, M_u[1, 3].real, M_u[1, 3].imag,
               M_u[2, 0].real, M_u[2, 0].imag, M_u[2, 1].real, M_u[2, 1].imag,
               M_u[2, 2].real, M_u[2, 2].imag, M_u[2, 3].real, M_u[2, 3].imag,
               M_u[3, 0].real, M_u[3, 0].imag, M_u[3, 1].real, M_u[3, 1].imag,
               M_u[3, 2].real, M_u[3, 2].imag, M_u[3, 3].real, M_u[3, 3].imag,
               M_d[0, 0].real, M_d[0, 0].imag, M_d[0, 1].real, M_d[0, 1].imag,
               M_d[0, 2].real, M_d[0, 2].imag,
               M_d[1, 0].real, M_d[1, 0].imag, M_d[1, 1].real, M_d[1, 1].imag,
               M_d[1, 2].real, M_d[1, 2].imag,
               M_d[2, 0].real, M_d[2, 0].imag, M_d[2, 1].real, M_d[2, 1].imag,
               M_d[2, 2].real, M_d[2, 2].imag)

    # Fixing texture zeros of M_u and M_d
    for i in range(4):
        for j in range(4):
            if M_u[i, j] == 0:
                m.fixed[f"M_u_{i}_{j}_re"] = True
                m.fixed[f"M_u_{i}_{j}_im"] = True

    for i in range(3):
        for j in range(3):
            if M_d[i, j] == 0:
                m.fixed[f"M_d_{i}_{j}_re"] = True
                m.fixed[f"M_d_{i}_{j}_im"] = True

    # Run Minuit minimization
    m.migrad()
    return m.fval, m.values


def pool_function(pair):

    for i in range(N_TRIES):
        par_values = np.zeros((50), dtype=np.float64)
        chi_square = 1e20
        chi_square_minuit, par_values_minuit = run_Minuit(
            pair, par_values, False)
        while (chi_square - chi_square_minuit) / chi_square > 1e-3:
            chi_square = chi_square_minuit
            par_values = par_values_minuit
            chi_square_minuit, par_values_minuit = run_Minuit(
                pair, par_values, True)
        if check_chi_squares_limits(par_values_minuit):
            break

    return par_values_minuit


def main():

    args = sys.argv[1:]
    start = time.time()
    n_u, n_d, set_mrt_before = io_mrt.read_mrt_before_min(args[0])

    set_mrt_after = []
    for classes in set_mrt_before:
        with Pool() as p:
            textures = p.map(pool_function, classes[2])

        length = len(textures)
        for i in reversed(range(length)):
            if not check_chi_squares_limits(textures[i]):
                textures.pop(i)

        set_mrt_after.append([classes[0], classes[1], textures])

    if set_mrt_after == []:
        print("NO PAIRS WERE FOUND!")

    io_mrt.print_mrt_after_min(
        set_mrt_after, FILENAME_OUTPUT, n_u, n_d)

    end = time.time()
    print(f"TOTAL TIME = {int((float(end) - float(start)) / 60)} min ", end="")
    print(f"{(int(float(end) - float(start)) % 60)} sec \n")

    return


if __name__ == "__main__":
    main()
