# The objective of this scprit is to remove the degrees of freedom of a given texture zero pair.
# The WB transformations that leave the Lagrangian invariant are:
#
# M_u -> K_L_u_dagger @ M_u @ K_R_u     ,    M_d -> K_L_d_dagger @ M_d @ K_R_d
#
# For the SM + 1 up isosinglet VLQ, the matrices K are given by:
#
# K_L_u = [     |0]     K_L_d = A_L
#         [ A_L |0]
#         [_ _ _|0]
#         [0 0 0|X]
#
# where A_L and K_R_d are 3 x 3 unitary matrices and K_R_u is a 4 x 4 unitary matrix.
# To remove the maximum number of complex phases from the texture zero pair, we need
# to perform rephasing of its entries. Given the WB transformations, we may perform
# the following general rephasing:
#
# (M_u)_kj -> exp(i theta_k) * (M_u)_kj * exp(i alpha_j),
# (M_u)_4j -> exp(i gamma) * (M_u)_4j * exp(i alpha_j),
# (M_d)_km -> exp(i theta_k) * (M_u)_km * exp(i beta_m),
#
# where k, m go from 1 to 3 and j from 1 to 4.
# Removing the degrees of freedom is equivalent to solving the system of linear eqs.
# obtained for the phases of the WB transformations.
# We have 3 theta + 4 alpha + 3 beta + 1 gamma = 11 rephasing angles. These are defined
# up to a overall rephasing. For example:
#
# theta_k + phase((M_u)_kj) + alpha_j = 0
#
# is invariant under theta_k -> theta_k + delta , alpha_j -> alpha_j - delta;
# thus only a maximum of 10 complex phases can be removed.
# We start by attempting to remove N = 10 complex phases for each pair.
# If a solution is not found for N complex phases, we repeat the process
# for N - 1 complex phases successively, until a solution is obtained.

import numpy as np
import numpy.linalg
from itertools import combinations
import sys
import pdb
from minuit_minimization_mp_non_unitary import read_maximmaly_restrictive_pairs


# This function returns a list non_zero_entries which contains the positions
# and phases of the non zero entries of the pair of texture zeros.
# non_zero_entries[i] = [0 for M_u and 1 for N_d, line, collumn]
def get_positions_and_phases(pair):

    non_zero_entries = []

    for i in range(4):
        for j in range(4):
            if pair[0][i, j] != 0:
                non_zero_entries.append([0, i, j])

    for i in range(3):
        for j in range(3):
            if pair[1][i, j] != 0:
                non_zero_entries.append([1, i, j])

    return non_zero_entries


# This function checks if the system of equations A @ x = b, where:
# - A is a n x 11 coefficient matrix, where n is the number of non_zero_entries;
# - x = (theta_1, theta_2, theta_3, gamma, alpha_1, alpha_2, alpha_3, alpha_4, beta_1, beta_2, beta_3);
# - b contains the complex phases of the n non_zero_entries;
# is solvable
def solve_system_equations_phases(non_zero_entries):

    n = len(non_zero_entries)
    A = np.zeros((n, 11))

    for row, position in enumerate(non_zero_entries):
        A[row][position[1]] = 1
        A[row][int((position[0] * 4 + 4) + position[2])] = 1

    rng = np.random.default_rng()
    b = rng.random((1, n))

    rank_A = np.linalg.matrix_rank(A)
    rank_A_augmented = np.linalg.matrix_rank(np.concatenate((A, b.T), axis=1))

    if rank_A_augmented > rank_A:
        return False
    else:
        return True


# This function returns the positions of the non zero elements of the texture
# zero pair and the positions of its real entries after rephasing
def remove_dof_pair(pair):

    non_zero_entries = get_positions_and_phases(pair)
    n_zeros = len(non_zero_entries)

    if n_zeros > 9:
        set_real_entries = list(combinations(non_zero_entries, 10))
        N = 10
    else:
        set_real_entries = non_zero_entries
        N = n_zeros

    i = 0
    len_set = len(set_real_entries)
    while not solve_system_equations_phases(set_real_entries[i]):
        i += 1
        if i == len_set:
            N = N - 1
            set_real_entries = list(combinations(non_zero_entries, N))
            len_set = len(set_real_entries)
            i = 0

    return non_zero_entries, set_real_entries[i]


# This function returns the pair of texture zeros after the rephasing
# in 2 formats:
# - The texture zeros with non zero entries where 1 represents real entries
#   and 2 complex entries
# - 2 strings where U/D_{positions of real entries}^{position of complex entries}
#   For example:
#
#   [0 0 0 1]   [0 0 0]
#   [0 0 1 0] , [0 1 1]    =>   U_{3,6,9,12} ; D_{4,5,6,8}^{7}
#   [0 1 0 0]   [1 2 1]
#   [1 0 0 1]
def get_matrix_after_rephasing(non_zero_entries, real_entries):

    n_real = len(real_entries)
    n = len(non_zero_entries) * 2 - n_real
    M_u = np.zeros((4, 4))
    M_d = np.zeros((3, 3))
    string_up_real = ""
    string_down_real = ""
    string_up_complex = ""
    string_down_complex = ""
    i = 0
    for pos in non_zero_entries:
        if pos == real_entries[i]:
            if pos[0] == 0:
                M_u[pos[1], pos[2]] = 1
                string_up_real = string_up_real + str(int(pos[1] * 4 + pos[2])) + ","
            else:
                M_d[pos[1], pos[2]] = 1
                string_down_real = string_down_real + str(int(pos[1] * 3 + pos[2])) + ","
            if i != n_real - 1:
                i += 1
        else:
            if pos[0] == 0:
                M_u[pos[1], pos[2]] = 2
                string_up_complex = string_up_complex + str(int(pos[1] * 4 + pos[2])) + ","
            else:
                M_d[pos[1], pos[2]] = 2
                string_down_complex = string_down_complex + str(int(pos[1] * 3 + pos[2])) + ","

    string_up = "M_u_{" + string_up_real[0:-1] + "}^{" + string_up_complex[0:-1] + "}"
    string_down = "M_d_{" + string_down_real[0:-1] + "}^{" + string_down_complex[0:-1] + "}"

    return n, [M_u, M_d], string_up, string_down


# This function writes to a file the texture zeros pairs after removing
# the maximum amount of complex phases
def print_restrictive_pairs_from_minuit(set_maximally_restrictive_pairs,
                                        option,
                                        filename):

    i = 0
    length = len(set_maximally_restrictive_pairs)
    set_maximally_restrictive_pairs_print = []
    pairs = []
    string = set_maximally_restrictive_pairs[0][0]
    while(i < length):
        pairs.append(get_matrix_after_rephasing(
                      set_maximally_restrictive_pairs[i][1][0],
                      set_maximally_restrictive_pairs[i][1][1])
                      )

        if i == length - 1:
            set_maximally_restrictive_pairs_print.append([string, pairs])
            break

        if set_maximally_restrictive_pairs[i][0] != set_maximally_restrictive_pairs[i + 1][0]:
            set_maximally_restrictive_pairs_print.append([string, pairs])
            pairs = []
            string = set_maximally_restrictive_pairs[i + 1][0]

        i += 1

    with open(filename, option) as f:
        f.write("LIST OF MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS (AFTER MINUIT) WITHOUT DEGREES OF FREEDOM:\n\n")
        f.write("1 = REAL ENTRY\n")
        f.write("2 = COMPLEX ENTRY\n")
        f.write("N = NUMBER OF PHYSICAL PARAMETERS\n\n")
        num_pairs = 0
        for restrictive_pairs_n_zeros_u in set_maximally_restrictive_pairs_print:
            num_pairs_n_zeros_u = 0
            f.write(
                f"PAIRS WITH {restrictive_pairs_n_zeros_u[0]} TEXTURE ZEROS FOR (M_u, M_d):\n\n")
            for pair in restrictive_pairs_n_zeros_u[1]:
                f.write(f"M_u:\n{pair[1][0]}\n\n")
                f.write(f"M_d:\n{pair[1][1]}\n\n")
                f.write(f"N = {pair[0]}\n")
                f.write("\n")
                f.write("====================")
                f.write("\n\n")
                num_pairs_n_zeros_u += 1
            f.write(
                f"THERE ARE {num_pairs_n_zeros_u} PAIRS WITH {restrictive_pairs_n_zeros_u[0]} TEXTURE ZEROS\n\n")
            num_pairs += num_pairs_n_zeros_u

        f.write("CONDENSED VIEW OF LIST:\n")
        f.write("NOTATION: M_(u/d)_{positions of real entries}^{positions of complex entries}\n\n")

        for restrictive_pairs_n_zeros_u in set_maximally_restrictive_pairs_print:
            num_pairs_n_zeros_u = 0
            f.write(f"(M_u, M_d) = {restrictive_pairs_n_zeros_u[0]}:\n\n")
            for pair in restrictive_pairs_n_zeros_u[1]:
                num_pairs_n_zeros_u += 1
                string = f"{num_pairs_n_zeros_u}) {pair[2]} / {pair[3]}"
                while len(string) < 55:
                    string = string + " "
                string = string + f"|N = {pair[0]}\n" 
                f.write(string)
            f.write("\n")

        f.write(
            f"\nTHERE ARE IN TOTAL {num_pairs} MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS (AFTER MINUIT). \n")

    return


def main():

    args = sys.argv[1:]
    filename = args[0]
    set_maximally_restrictive_pairs = read_maximmaly_restrictive_pairs(filename)

    for i, pair in enumerate(set_maximally_restrictive_pairs):
        set_maximally_restrictive_pairs[i] = [set_maximally_restrictive_pairs[i][0],
                                              remove_dof_pair(set_maximally_restrictive_pairs[i][1])]
    filename = args[0].replace("textures", "rephasing")
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

    print_restrictive_pairs_from_minuit(set_maximally_restrictive_pairs, "w", "test_2")
    return


if __name__ == "__main__":
    main()
