# The objective of this scprit is to determine if a 2HDM + abelian symmetry is able
# to reproduce some of the texture zero pairs identified using the scripts
# texture_zeros_v4.py and minuit_minimization_mp_non_unitary.py

import numpy as np
import scipy
import numpy.linalg
from itertools import combinations, permutations
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
def get_decompositions_mass_matrices(non_zero_entries_up,
                                     non_zero_entries_up_4th_row,
                                     non_zero_entries_down):

    n_up = int(len(non_zero_entries_up) / 2 + 1)
    n_fourth_row = int(len(non_zero_entries_up_4th_row))
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
            Y_u.append([list(decomposition), non_zeros_Y_u_2])

    for i in range(1, n_down):
        for decomposition in combinations(non_zero_entries_down, i):
            non_zeros_Y_d_2 = non_zero_entries_down.copy()
            for non_zero in decomposition:
                j = 0
                while non_zero != non_zeros_Y_d_2[j]:
                    j += 1
                del non_zeros_Y_d_2[j]
            Y_d.append([list(decomposition), non_zeros_Y_d_2])

    if n_fourth_row == 1:
        return Y_u, [[0]], Y_d
    if n_fourth_row == 2:
        return Y_u, [[0, 1]], Y_d
    if n_fourth_row == 3:
        return Y_u, [[0, 1, 2], [1, 0, 2], [1, 2, 0]], Y_d


# This function checks if a given pair of texture zeros can be
# explained by an abelian symmetry.
# For each entry of the mass matrices there will be 2 conditions for the 3 first rows
# For the 4th row of M_u, there are 3 conditions for each entry:
# bare mass term and couplings to S and S*
# Thus, we have a total of (12 + 9) * 2 + 4 * 3 = 54 equations
# We will have 14 phases:
# - 3 alpha for left quark doublets
# - 2 theta for Higgs doublets
# - 4 gamma for right up quarks (3 SM + 1 VLQ)
# - 3 beta for right down quarks
# - 1 omega for the up left VLQ
# - 1 sigma for the complex scalar singlet
# From these phases, we can set the values for 2. We choose theta_1 = alpha_1 = 0
# x = (theta_2, omega, sigma, alpha_2, alpha_3, gamma_1, gamma_2, gamma_3, gamma_4, beta_1, beta_2, beta_3)
def check_pair_texture_zeros(pair):

    # We can only have up to a maximum of 3 non zero entries in the 4th row of M_u
    if pair[0][3] @ np.array([1, 1, 1, 1]).T == 4:
        return False, []

    non_zero_entries_up, non_zero_entries_up_4th_row, non_zero_entries_down = get_positions(pair)
    Y_u, forth_row_couplings, Y_d = get_decompositions_mass_matrices(
        non_zero_entries_up,
        non_zero_entries_up_4th_row,
        non_zero_entries_down)

    # Definition of matrix A
    A = np.zeros((54, 12))

    # Couplings from M_u to theta_2
    for i in range(12, 24):
        A[i][0] = 1

    # 4th row bare mass terms
    for i in range(24, 28):
        A[i][1] = 1

    # 4th row couplings to S
    for i in range(28, 32):
        A[i][1] = 1
        A[i][2] = -1

    # 4th row couplings to S*
    for i in range(32, 36):
        A[i][1] = 1
        A[i][2] = 1

    # Couplings from M_d to theta_2
    for i in range(45, 54):
        A[i][0] = -1

    line = 0
    # First 3 rows of M_u (3 * 4 * 2 = 24 equations)
    for k in range(2):
        for n in range(3):
            for m in range(4):
                if not n == 0:
                    A[line, int(n + 2)] = 1
                A[line, int(m + 5)] = -1
                line += 1

    # Last row of M_u (4 * 3 = 12 equations)
    for k in range(3):
        for m in range(4):
            A[line, int(m + 5)] = -1
            line += 1

    # 3 rows of M_d (3 * 3 * 2 = 18 equations)
    for k in range(2):
        for n in range(3):
            for m in range(3):
                if not n == 0:
                    A[line, int(n + 2)] = 1
                A[line, int(m + 9)] = -1
                line += 1

    zeros_list = []
    possible_decompositions = []
    for decomposition_u in Y_u:

        constraints_non_zero_entries = np.zeros((11, 12))
        positions_non_zero_entries = np.zeros(11)
        i = 0
        for non_zero_Y_u_1 in decomposition_u[0]:
            constraints_non_zero_entries[i, :] = A[int(4 * non_zero_Y_u_1[0] + non_zero_Y_u_1[1]), :]
            positions_non_zero_entries[i] = int(4 * non_zero_Y_u_1[0] + non_zero_Y_u_1[1])
            i += 1

        for non_zero_Y_u_2 in decomposition_u[1]:
            constraints_non_zero_entries[i, :] = A[int(4 * non_zero_Y_u_2[0] + non_zero_Y_u_2[1] + 12), :]
            positions_non_zero_entries[i] = int(4 * non_zero_Y_u_2[0] + non_zero_Y_u_2[1] + 12)
            i += 1

        # Possible couplings to 4th row: 0 - bare mass term/ 1 - coupling to S/ 2 - coupling to S*
        for decomposition_4th_row in forth_row_couplings:
            n = i
            for m, non_zero in enumerate(non_zero_entries_up_4th_row):
                constraints_non_zero_entries[n, :] = A[int(decomposition_4th_row[m] * 4 + non_zero[1] + 24), :]
                positions_non_zero_entries[n] = int(decomposition_4th_row[m] * 4 + non_zero[1] + 24)
                n += 1

            for decomposition_d in Y_d:

                j = n
                for non_zero_Y_d_1 in decomposition_d[0]:
                    constraints_non_zero_entries[j, :] = A[int(3 * non_zero_Y_d_1[0] + non_zero_Y_d_1[1] + 36), :]
                    positions_non_zero_entries[j] = int(3 * non_zero_Y_d_1[0] + non_zero_Y_d_1[1] + 36)
                    j += 1

                for non_zero_Y_d_2 in decomposition_d[1]:
                    constraints_non_zero_entries[j, :] = A[int(3 * non_zero_Y_d_2[0] + non_zero_Y_d_2[1] + 45), :]
                    positions_non_zero_entries[j] = int(3 * non_zero_Y_d_2[0] + non_zero_Y_d_2[1] + 45)
                    j += 1

                # Compute null space of constraints_non_zero_entries
                null_space = scipy.linalg.null_space(constraints_non_zero_entries)

                # Check if null space forbids remaining terms of the Lagrangian
                if np.any(null_space):
                    b = np.zeros((1, 12))
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

                    if zeros == 11:

                        # Determine charges of fields normalized to theta_2
                        for j in range(np.shape(null_space)[1]):
                            constant = null_space[0][j]
                            for k, elem in enumerate(null_space[:, j]):
                                null_space[k][j] = null_space[k][j] / constant

                        null_space = np.around(null_space, decimals=3)

                        #if np.shape(null_space)[1] > 1:
                        #    if len(non_zero_entries_up_4th_row) != 1:
                        #        null_space = symmetry_implementation(null_space,
                        #                                             A,
                        #                                             positions_non_zero_entries,
                        #                                             constraints_non_zero_entries)
                        #    else:
                        #        null_space[2][1] = -100
                        possible_decompositions.append([decomposition_u,
                                                        non_zero_entries_up_4th_row,
                                                        decomposition_4th_row,
                                                        decomposition_d,
                                                        null_space
                                                        ])
                        
                        #print(decomposition_u)
                        #print("\n")
                        #print(non_zero_entries_up_4th_row)
                        #print("\n")
                        #print(decomposition_4th_row)
                        #print("\n")
                        #print(decomposition_d)
                        #print("\n")
                        #print(constraints_non_zero_entries)
                        #print("\n")
                        #print(null_space)
                        #print("\n")
                        #print(scale)
                        #print("\n")
                        #print(b)
                        #pdb.set_trace()

                zeros_list.append(zeros)

    if zeros_list != []:
        zeros_list.sort()
        #print(zeros_list[0])

    if possible_decompositions == []:
        return False, possible_decompositions
    else:
        return True, possible_decompositions


# This function returns the U(1) implementation for the cases
# where the charges of the fields are not unique with the lowest
# natural number imposed for the charge gamma_1
def symmetry_implementation(null_space,
                            A,
                            positions_non_zero_entries,
                            constraints_non_zero_entries):

    fixed = null_space[:, 0]
    b = np.zeros(11)
    indices = []
    constraints_non_zero_entries_aux = np.copy(constraints_non_zero_entries)

    # Fix well determined charges
    for i, elem in enumerate(null_space[:, 1]):
        if fixed[i] != elem:
            fixed[i] = -100

    # Define system of linear eqs. to determine dependent charges
    for i, elem in enumerate(fixed):
        if elem != -100:
            for j, entry in enumerate(constraints_non_zero_entries[:, i]):
                if entry == 1:
                    b[j] += -fixed[i]
                elif entry == -1:
                    b[j] += fixed[i]
            indices.append(i)

    constraints_non_zero_entries_aux = np.delete(constraints_non_zero_entries_aux, 5, 1)
    j = 0
    for elem in indices:
        if elem > 5:
            constraints_non_zero_entries_aux = np.delete(constraints_non_zero_entries_aux, int(elem - j - 1), 1)
        else:
            constraints_non_zero_entries_aux = np.delete(constraints_non_zero_entries_aux, int(elem - j), 1)
        j += 1

    i = 1
    found = False
    while not found:
        for j, row in enumerate(constraints_non_zero_entries):
            if row[5] == -1:
                b[j] += 1
        # Solve system of linear eqs.
        try:
            solution = np.linalg.lstsq(constraints_non_zero_entries_aux, b, rcond=None)[0]
            found_solution = True
        except np.linalg.LinAlgError:
            print("Solution not found / ")
            found_solution = False

        # Check if found solution + fixed charges determine the decompositions
        if found_solution:

            # Define vector with charges
            new_solution = np.copy(fixed)
            new_solution[5] = i
            k = 0
            m = 0
            while k < len(solution):
                if new_solution[m] == -100:
                    new_solution[m] = solution[k]
                    k += 1
                m += 1

            # Count the number of zeros
            zeros = 0
            control = 0
            for m, elem in enumerate(A @ new_solution):
                if abs(elem) < 1e-14:
                    zeros += 1
                    if control < 11:
                        if positions_non_zero_entries[control] == m:
                            control += 1
                    else:
                        break

            #for elem in (A @ new_solution).T:
            #    print(elem)
            #print("\n")
            #print(i)
            #print(fixed)
            #print(solution)
            #print(new_solution)
            #print(zeros)
            #pdb.set_trace()
            if zeros == control:
                constant = solution[0]
                for j, elem in enumerate(solution):
                    solution[j] = elem / constant
                solution = np.around(solution, decimals=3)
                found = True
        i += 1

    null_space[:, 1] = new_solution.T

    return np.around(null_space, decimals=0)

# This function removes repeated decompositions obtained under the interchange
# of the indices of the Higgs doublets (\phi_1 <-> \phi_2)
# We only keep one example of decompositions where the 4th row allows for every
# possible combination of couplings ([0 1 2] , [1 0 2] and [1 2 0]), since these
# are invariant under S <-> S*
def remove_repeated_decompositions(set_maximally_restrictive_pairs):

    for pair in set_maximally_restrictive_pairs:
        i = 0
        while i < len(pair[2]) - 1:
            j = i + 1
            while j < len(pair[2]):
                if (pair[2][i][0][0] == pair[2][j][0][1]
                    and pair[2][i][0][1] == pair[2][j][0][0]
                    and pair[2][i][3][0] == pair[2][j][3][1]
                    and pair[2][i][3][1] == pair[2][j][3][0]
                    and pair[2][i][2] == pair[2][j][2]):
                    pair[2].pop(j)
                    break
                j += 1
            i += 1

    for pair in set_maximally_restrictive_pairs:
        if len(pair[2][0][1]) == 3:
            i = 0
            while i < len(pair[2]) - 2:
                j = i + 1
                while j < len(pair[2]) - 1:
                    if (pair[2][i][0][0] == pair[2][j][0][0]
                        and pair[2][i][0][1] == pair[2][j][0][1]
                        and pair[2][i][3][0] == pair[2][j][3][0]
                        and pair[2][i][3][1] == pair[2][j][3][1]):
                        k = j + 1
                        while k < len(pair[2]):
                            if (pair[2][i][0][0] == pair[2][k][0][0]
                                and pair[2][i][0][1] == pair[2][k][0][1]
                                and pair[2][i][3][0] == pair[2][k][3][0]
                                and pair[2][i][3][1] == pair[2][k][3][1]):
                                pair[2].pop(k)
                                pair[2].pop(j)
                                pair[2][i][2] = []
                                break
                            k += 1
                    j += 1
                i += 1
    return


# This function creates a matrix with non zero entries for the
# positions given as input
def auxiliary_decomposition_matrix_form(decomposition, option):

    if option == "u":
        length = 4
    if option == "d":
        length = 3

    matrix_decompositions = []
    for i, decomp in enumerate(decomposition):
        matrix_decompositions.append(np.zeros((3, length)))
        for element in decomp:
            matrix_decompositions[i][element[0], element[1]] = 1

    return matrix_decompositions


def decomposition_matrix_form(decomposition):

    matrix_decompositions_u = auxiliary_decomposition_matrix_form(decomposition[0], "u")
    matrix_decompositions_d = auxiliary_decomposition_matrix_form(decomposition[3], "d")

    fourth_row = np.zeros((1, 4))
    if decomposition[2] == []:
        for i, entry in enumerate(decomposition[1]):
            fourth_row[0, entry[1]] = i + 1
    else:
        for i, entry in enumerate(decomposition[1]):
            fourth_row[0, entry[1]] = decomposition[2][i] + 1

    return matrix_decompositions_u, fourth_row, matrix_decompositions_d

# This function writes to a file the results
def print_restrictive_pairs_from_minuit(set_maximally_restrictive_pairs,
                                        option,
                                        filename):

    with open(filename, option) as f:
        f.write("LIST OF MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS REALIZABLE WITH 2HDM + ABELIAN SYMMETRY (AFTER MINUIT):\n\n")
        f.write("LABELS FOR THE 4TH ROW:\n\n1 -> BARE MASS TERM\n2 -> COUPLINGS TO SINGLET SCALAR S\n3 -> COUPLINGS TO CONJUNGATE SINGLET SCALAR S*\n\n")        
        num_pairs = 0
        for i, pair in enumerate(set_maximally_restrictive_pairs):
            f.write(f"CASE {i + 1}:\n")
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

                null_space = decomposition[4]
                f.write(f"Charges:\n")
                for j in range(np.shape(null_space)[1]):
                    f.write(f"{null_space[:, j]}\n\n")

                f.write(f"##############################################################################################################\n\n")

            num_pairs += 1
        f.write(
            f"\nTHERE ARE IN TOTAL {num_pairs} MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS REALIZABLE WITH 2HDM + ABELIAN SYMMETRY (AFTER MINUIT). \n\n")

        f.write("CONDENSED VIEW OF LIST:\n")
        f.write("LABELS FOR THE 4TH ROW:\n\n1 -> BARE MASS TERM\n2 -> COUPLINGS TO SINGLET SCALAR S\n3 -> COUPLINGS TO CONJUNGATE SINGLET SCALAR S*\n\n")
        f.write("IF THERE IS NO LABEL, THEN ALL POSSIBLE COMBINATIONS OF COUPLINGS ARE POSSIBLE\n\n")

        for i, pair in enumerate(set_maximally_restrictive_pairs):
            f.write(f"CASE {i + 1}:\n")
            for decomposition in pair[2]:
                string = ""
                for i, decomp_u in enumerate(decomposition[0]):
                    string = string + f"Y_u_{int(i + 1)}: "
                    for element in decomp_u:
                        string = string + f"{element}/"
                    string = string[:-1]
                while len(string) < 20:
                    string = string + " "

                forth_row_string = " "
                for i, pos in enumerate(decomposition[1]):
                    if len(decomposition[1]) == 3 and decomposition[2] != []:
                        forth_row_string = forth_row_string + f"{pos}({int(decomposition[2][i] + 1)})/"
                    else:
                        forth_row_string = forth_row_string + f"{pos}/"
                forth_row_string = forth_row_string[:-1]
                string = string + f"4th_row = {forth_row_string}"
                while len(string) < 70:
                    string = string + " "

                for i, decomp_d in enumerate(decomposition[3]):
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


# This function returns all realizable texture pairs with a 2HDM + 1 scalar singlet
# + abelian symmetries written in a file
def realizable_pairs_textures(filename):

    set_maximally_restrictive_pairs = read_maximmaly_restrictive_pairs(filename)
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

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

    # Remove repeated decompositions
    remove_repeated_decompositions(set_maximally_restrictive_pairs)

    return set_maximally_restrictive_pairs

def main():

    args = sys.argv[1:]
    filename = args[0]

    set_maximally_restrictive_pairs = realizable_pairs_textures(filename)

    if set_maximally_restrictive_pairs != []:
        filename = "2HDM_plus_1_singlet_scalar_condensed_3"
        print_restrictive_pairs_from_minuit(set_maximally_restrictive_pairs, "w", filename)
    else:
        print("NO PAIRS CAN BE EXPLAINED BY 2HDM")

    return


if __name__ == "__main__":
    main()
