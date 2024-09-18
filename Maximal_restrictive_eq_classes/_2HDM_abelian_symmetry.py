# The objective of this scprit is to determine if a 2HDM + abelian symmetry is able
# to reproduce some of the texture zero pairs identified using the scripts
# texture_zeros_v4.py and minuit_minimization_mp_non_unitary.mp

import numpy as np
import scipy
import numpy.linalg
from itertools import combinations
import sys
import pdb
from minuit_minimization_mp_non_unitary import read_maximmaly_restrictive_pairs


# This function checks if the 4th row of M_u has only one non zero entry.
# This is a constraint coming from the requirement of the invariance of
# the Lagrangian under the abelian symmetry.
def check_fourth_row_M_u(set_maximally_restrictive_pairs):

    pairs_to_eliminate = []
    for i, pair in enumerate(set_maximally_restrictive_pairs):
        M_u = pair[1][0]
        count = 0
        for j in range(4):
            if M_u[3, j] != 0:
                count += 1
            if count > 1:
                pairs_to_eliminate.append(i)
                break

    for i in reversed(pairs_to_eliminate):
        del set_maximally_restrictive_pairs[i]

    return set_maximally_restrictive_pairs


# This function returns a list non_zero_entries which contains the positions
# of the non zero entries of the pair of texture zeros.
def get_positions(pair):

    non_zero_entries_up = []
    non_zero_entries_up_4th_row = []
    non_zero_entries_down = []

    for i in range(3):
        for j in range(4):
            if pair[0][i, j] != 0:
                non_zero_entries_up.append([i, j])

    for j in range(4):
        if pair[0][3, j] != 0:
            non_zero_entries_up_4th_row.append([3, j])

    for i in range(3):
        for j in range(3):
            if pair[1][i, j] != 0:
                non_zero_entries_down.append([i, j])

    return non_zero_entries_up, non_zero_entries_up_4th_row, non_zero_entries_down


# This function returns all possible decompositions of the mass matrices
# M_u and M_d into Y_u_1, Y_u_2, Y_d_1 and Y_d_2
def get_decompositions_mass_matrices(non_zero_entries_up, non_zero_entries_down):

    n_up = int(len(non_zero_entries_up) / 2 + 1)
    n_down = int(len(non_zero_entries_down))

    Y_u = []
    Y_d = []

    for i in range(1, n_up):
        for decomposition in combinations(non_zero_entries_up, i):
            non_zeros_Y_u_2 = non_zero_entries_up.copy()
            for non_zero in decomposition:
                j = 0
                while non_zero != non_zeros_Y_u_2[j]:
                    j += 1
                del non_zeros_Y_u_2[j]
            Y_u.append([decomposition, non_zeros_Y_u_2])

    for i in range(1, n_down):
        for decomposition in combinations(non_zero_entries_down, i):
            non_zeros_Y_d_2 = non_zero_entries_down.copy()
            for non_zero in decomposition:
                j = 0
                while non_zero != non_zeros_Y_d_2[j]:
                    j += 1
                del non_zeros_Y_d_2[j]
            Y_d.append([decomposition, non_zeros_Y_d_2])

    return Y_u, Y_d


# This function checks if a given pair of texture zeros can be
# explained by an abelian symmetry.
# For each entry of the mass matrices there will be 2 conditions, except for
# the fourth one.
# Thus, we have a total of (12 + 9) * 2 + 4 = 46 equations
# We will have 13 phases:
# - 3 alpha for left quark doublets
# - 2 theta for Higgs doublets
# - 4 gamma for right up quarks (3 SM + 1 VLQ)
# - 3 beta for right down quarks
# - 1 omega for the up left VLQ
# From these phases, we can set the values for 2. We choose theta_1 = alpha_1 = 0
# x = (theta_2, omega, alpha_2, alpha_3, gamma_1, gamma_2, gamma_3, gamma_4, beta_1, beta_2, beta_3)
def check_pair_texture_zeros(pair):

    non_zero_entries_up, non_zero_entries_up_4th_row, non_zero_entries_down = get_positions(pair)
    Y_u, Y_d = get_decompositions_mass_matrices(
        non_zero_entries_up,
        non_zero_entries_down)

    # Definition of matrix A
    #
    # Only the 3 first rows of Y_u_2 and Y_d_2 couple to theta_2
    # The fourth row comes from bare mass terms mu_j^2 * U_L* u_R_j

    A = np.zeros((46, 11))
    for i in range(12, 24):
        A[i][0] = 1
    for i in range(37, 46):
        A[i][0] = -1
    line = 0

    # First 3 rows of M_u (3 * 4 * 2 = 24 equations)
    for k in range(2):
        for n in range(3):
            for m in range(4):
                if not n == 0:
                    A[line, int(n + 1)] = 1
                A[line, int(m + 4)] = -1
                line += 1

    # Last row of M_u (4 equations)
    for m in range(4):
        A[line, 1] = 1
        A[line, int(m + 4)] = -1
        line += 1

    # 3 rows of M_d (3 * 3 * 2 = 18 equations)
    for k in range(2):
        for n in range(3):
            for m in range(3):
                if not n == 0:
                    A[line, int(n + 1)] = 1
                A[line, int(m + 8)] = -1
                line += 1

    print(A)
    zeros_list = []
    print(pair)
    print("\n")
    possible_decompositions = []
    for decomposition_u in Y_u:

        constraints_non_zero_entries = np.zeros((11, 11))
        i = 0
        for non_zero_Y_u_1 in decomposition_u[0]:
            constraints_non_zero_entries[i, :] = A[int(4 * non_zero_Y_u_1[0] + non_zero_Y_u_1[1]), :]
            i += 1

        for non_zero_Y_u_2 in decomposition_u[1]:
            constraints_non_zero_entries[i, :] = A[int(4 * non_zero_Y_u_2[0] + non_zero_Y_u_2[1] + 12), :]
            i += 1

        for non_zero in non_zero_entries_up_4th_row:
            constraints_non_zero_entries[i, :] = A[int(non_zero[1] + 24), :]
            i += 1

        for decomposition_d in Y_d:

            j = i
            for non_zero_Y_d_1 in decomposition_d[0]:
                constraints_non_zero_entries[j, :] = A[int(3 * non_zero_Y_d_1[0] + non_zero_Y_d_1[1] + 28), :]
                j += 1

            for non_zero_Y_d_2 in decomposition_d[1]:
                constraints_non_zero_entries[j, :] = A[int(3 * non_zero_Y_d_2[0] + non_zero_Y_d_2[1] + 37), :]
                j += 1

            # Compute null space of constraints_non_zero_entries
            null_space = scipy.linalg.null_space(constraints_non_zero_entries)

            # Check if null space forbids remaining terms of the Lagrangian
            if np.any(null_space):
                b = np.zeros((1, 11))
                for k in range(np.shape(null_space)[1]):
                    b += null_space[:, k]

                # Set the scale bellow which a number is zero
                scale = 0
                for elem in constraints_non_zero_entries @ b.T:
                    if abs(elem) > scale:
                        scale = abs(elem)

                b = A @ b.T
                zeros = 0
                for elem in b:
                    if abs(elem) <= scale * 1e2:
                        zeros += 1
                print(zeros)
                print("\n")
                pdb.set_trace()
                if zeros == 11:
                    possible_decompositions.append([decomposition_u,
                                                    non_zero_entries_up_4th_row,
                                                    decomposition_d,
                                                    null_space
                                                    ])
                    print(decomposition_u)
                    print("\n")
                    print(non_zero_entries_up_4th_row)
                    print("\n")
                    print(decomposition_d)
                    print("\n")
                    print(constraints_non_zero_entries)
                    print("\n")
                    print(null_space)
                    print("\n")
                    print(scale)
                    print("\n")
                    print(b)
                    #pdb.set_trace()

                zeros_list.append(zeros)

    if zeros_list != []:
        zeros_list.sort()
        print(zeros_list[0])

    if possible_decompositions == []:
        return False, possible_decompositions
    else:
        return True, possible_decompositions


# This function writes creates a matrix with non zero entries for the
# positions given as input

def decomposition_matrix_form(decomposition):

    matrix_decompositions_u = []
    matrix_decompositions_d = []
    for i, decomp_u in enumerate(decomposition[0]):
        matrix_decompositions_u.append(np.zeros((3, 4)))
        for element in decomp_u:
            matrix_decompositions_u[i][element[0], element[1]] = 1

    fourth_row = np.zeros((1, 4))
    fourth_row[0, decomposition[1][0][1]] = 1

    for i, decomp_d in enumerate(decomposition[2]):
        matrix_decompositions_d.append(np.zeros((3, 3)))
        for element in decomp_d:
            matrix_decompositions_d[i][element[0], element[1]] = 1

    return matrix_decompositions_u, fourth_row, matrix_decompositions_d

# This function writes to a file the results
def print_restrictive_pairs_from_minuit(set_maximally_restrictive_pairs,
                                        option,
                                        filename):

    with open(filename, option) as f:
        f.write("LIST OF MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS REALIZABLE WITH 2HDM + ABELIAN SYMMETRY (AFTER MINUIT):\n\n")
        num_pairs = 0
        for pair in set_maximally_restrictive_pairs:
            f.write(f"M_u:\n{pair[1][0]}\n")
            f.write(f"M_d:\n{pair[1][1]}\n")
            f.write(f"\n")
            f.write(f"POSSIBLE DECOMPOSITIONS:\n\n")
            for decomposition in pair[2]:

                matrix_decomp_u, fourth_row, matrix_decomp_d = decomposition_matrix_form(decomposition)

                f.write(f"Y_u_1:\n{matrix_decomp_u[0]}\n\n")
                f.write(f"Y_u_2:\n{matrix_decomp_u[1]}\n\n")

                f.write(f"fourth_row_u:\n{fourth_row}\n\n")

                f.write(f"Y_d_1:\n{matrix_decomp_d[0]}\n\n")
                f.write(f"Y_d_2:\n{matrix_decomp_d[1]}\n\n")

                null_space = decomposition[3]
                constant = null_space[0][0]
                for i, elem in enumerate(null_space):
                    null_space[i][0] = null_space[i][0] / constant

                f.write(f"Charges:\n{null_space}\n\n")

                f.write(f"##############################################################################################################\n\n")

            num_pairs += 1
        f.write(
            f"\nTHERE ARE IN TOTAL {num_pairs} MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS REALIZABLE WITH 2HDM + ABELIAN SYMMETRY (AFTER MINUIT). \n\n")

        f.write("CONDENSED VIEW OF LIST:\n")
        f.write("NOTATION: M_(u/d)_{positions of real entries}^{positions of complex entries}\n\n")

        for pair in set_maximally_restrictive_pairs:
            for decomposition in pair[2]:
                string = ""
                for i, decomp_u in enumerate(decomposition[0]):
                    string = string + f"Y_u_{int(i + 1)}: "
                    for element in decomp_u:
                        string = string + f"{element}/"
                    string = string[:-1]
                    while len(string) < 30 * (i + 1):
                        string = string + " "

                string = string + f"4th_row = {decomposition[1]}"
                while len(string) < 110:
                    string = string + " "

                for i, decomp_d in enumerate(decomposition[2]):
                    string = string + f"Y_d_{int(i + 1)}: "
                    for element in decomp_d:
                        string = string + f"{element}/"
                    string = string[:-1]
                    while len(string) < 30 * (i + 1) + 110:
                        string = string + " "

                f.write(f"{string}\n")
            f.write(f"#####################################\n\n")

        f.write(
            f"\nTHERE ARE IN TOTAL {num_pairs} MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS REALIZABLE WITH 2HDM + ABELIAN SYMMETRY (AFTER MINUIT). \n\n")
    return


def main():

    args = sys.argv[1:]
    filename = args[0]
    set_maximally_restrictive_pairs = read_maximmaly_restrictive_pairs(filename)

    size_set = len(set_maximally_restrictive_pairs)
    i = 0
    while i < size_set:
        found_decomposition, decompositions = check_pair_texture_zeros(set_maximally_restrictive_pairs[i][1])
        if not found_decomposition:
            set_maximally_restrictive_pairs.pop(i)
            size_set -= 1
        else:
            set_maximally_restrictive_pairs[i].append(decompositions)
            i += 1

    if set_maximally_restrictive_pairs != []:
        filename = args[0].replace("textures", "2HDM")
        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(linewidth=np.inf)
        print_restrictive_pairs_from_minuit(set_maximally_restrictive_pairs, "w", filename)
    else:
        print("NO PAIRS CAN BE EXPLAINED BY 2HDM")

    return


if __name__ == "__main__":
    main()
