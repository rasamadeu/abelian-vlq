# The objective of this scprit is to determine if a 2HDM + abelian symmetry is able
# to reproduce some of the texture zero pairs identified using the scripts
# texture_zeros_v4.py and minuit_minimization_mp_non_unitary.mp

import numpy as np
import scipy
from itertools import combinations
import sys
import pdb
import random
import io_mrt


# This function returns a list non_zero_entries which contains the positions
# of the non zero entries of the pair of texture zeros.
def get_positions(pair, n_u, n_d):

    non_zero_entries_up_higgs = []
    non_zero_entries_up_vlq = []
    non_zero_entries_down_higgs = []
    non_zero_entries_down_vlq = []

    for i in range(3):
        for j in range(3 + n_u):
            if pair[0][i, j] != 0:
                non_zero_entries_up_higgs.append([i, j])

    for i in range(3, 3 + n_u):
        for j in range(3 + n_u):
            if pair[0][i, j] != 0:
                non_zero_entries_up_vlq.append([i, j])

    for i in range(3):
        for j in range(3 + n_d):
            if pair[1][i, j] != 0:
                non_zero_entries_down_higgs.append([i, j])

    for i in range(3, 3 + n_d):
        for j in range(3 + n_d):
            if pair[1][i, j] != 0:
                non_zero_entries_down_vlq.append([i, j])

    return non_zero_entries_up_higgs, non_zero_entries_up_vlq, \
        non_zero_entries_down_higgs, non_zero_entries_down_vlq,


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
                non_zeros_Y_u_2.remove(non_zero)
            Y_u.append([decomposition, non_zeros_Y_u_2])

    for i in range(1, n_down):
        for decomposition in combinations(non_zero_entries_down, i):
            non_zeros_Y_d_2 = non_zero_entries_down.copy()
            for non_zero in decomposition:
                non_zeros_Y_d_2.remove(non_zero)
            Y_d.append([decomposition, non_zeros_Y_d_2])

    return Y_u, Y_d


# Definition of system of linear equations where each line corresponds to the
# condition of a non-zero Yukawa entry
# Only the 3 first rows of Y_u_2 and Y_d_2 couple to theta_2
# The remaining rows (>3) come from bare mass terms
# mu_ij^2 * U_L_i* u_R_j and md_ij^2 * D_L_i* d_R_j
# For each entry in the first three rows of the mass matrices there will be 2 conditions
# Thus, we have:
#           - (6 + n_u) * (3 + n_u) equations for up mass matrix
#           - (6 + n_d) * (3 + n_d) equations for down mass matrix
#           - 2 theta for Higgs doublets
#           - 3 alpha for left quark doublets
#           - 3 + n_u gamma_u for right up quarks
#           - 3 + n_d gamma_d for right down quarks
#           - n_u omega_u for the up left VLQ
#           - n_d omega_d for the down left VLQ
# From these phases, we can set the values for 2.
# We choose theta_1 = alpha_1 = 0
# x = (theta_2, alpha_2, alpha_3,
#      gamma_u_1, ... , gamma_u_(3 + n_u), gamma_d_1, ... , gamma_d_(3 + n_d),
#      omega_u_1, ... , omega_u_n_u, omega_d_1, ... , omega_d_n_d)
def define_system_eqs(n_u, n_d):

    n_eqs_u = (6 + n_u) * (3 + n_u)
    n_eqs_d = (6 + n_d) * (3 + n_d)
    n_phases = 9 + 2 * (n_u + n_d)
    n_Yukawa_entries_u = 3 * (3 + n_u)
    n_Yukawa_entries_d = 3 * (3 + n_d)

    system = np.zeros((n_eqs_u + n_eqs_d, n_phases))

    for i in range(n_Yukawa_entries_u, 2 * n_Yukawa_entries_u):
        system[i][0] = 1
    for i in range(2 * n_Yukawa_entries_u + n_Yukawa_entries_d,
                   2 * n_Yukawa_entries_u + 2 * n_Yukawa_entries_d):
        system[i][0] = -1

    line = 0
    # First 3 rows of M_u (coming from Yukawa terms)
    for k in range(2):
        for n in range(3):
            for m in range(3 + n_u):
                if not n == 0:
                    system[line, n] = 1
                system[line, int(m + 3)] = -1
                line += 1

    # First 3 rows of M_d (coming from Yukawa terms)
    for k in range(2):
        for n in range(3):
            for m in range(3 + n_d):
                if not n == 0:
                    system[line, n] = 1
                system[line, int(m + 6 + n_u)] = -1
                line += 1

    # Remaining rows of M_u (coming from bare mass terms)
    for n in range(n_u):
        pos_vlq = int(9 + n_u + n_d + n)
        for m in range(3 + n_u):
            system[line, pos_vlq] = 1
            system[line, int(m + 3)] = -1
            line += 1

    # Remaining rows of M_d (coming from bare mass terms)
    for n in range(n_d):
        pos_vlq = int(9 + 2 * n_u + n_d + n)
        for m in range(3 + n_d):
            system[line, pos_vlq] = 1
            system[line, int(m + 6 + n_u)] = -1
            line += 1

    return system


def select_conditions_from_system(system, conditions_non_zero_entries, decomposition, pos, offset, n, is_bare_mass_term):

    for non_zero in decomposition:
        line = non_zero[0]
        if is_bare_mass_term:
            line = line - 3
        conditions_non_zero_entries[pos, :] = system[int(
            (3 + n) * line + non_zero[1] + offset), :]
        pos += 1
    return pos


# This function checks if a given pair of texture zeros can be
# imposed by an abelian symmetry with a 2HDM.
def check_pair_texture_zeros(pair, system, n_u, n_d):

    positions = get_positions(pair, n_u, n_d)
    non_zero_entries_up_higgs = positions[0]
    non_zero_entries_up_vlq = positions[1]
    non_zero_entries_down_higgs = positions[2]
    non_zero_entries_down_vlq = positions[3]
    n_non_zeros = 0
    for non_zeros in positions:
        n_non_zeros += len(non_zeros)
    n_phases = 9 + 2 * (n_u + n_d)

    Y_u, Y_d = get_decompositions_mass_matrices(
        non_zero_entries_up_higgs, non_zero_entries_down_higgs)

    possible_decompositions = []
    for decomposition_u in Y_u:

        conditions_non_zero_entries = np.zeros((n_non_zeros, n_phases))
        pos_u = 0

        # Non-zeros from Y_u_1
        pos_u = select_conditions_from_system(
            system, conditions_non_zero_entries, decomposition_u[0], pos_u, 0, n_u, False)

        # Non-zeros from Y_u_2
        pos_u = select_conditions_from_system(
            system, conditions_non_zero_entries, decomposition_u[1], pos_u, 3 * (3 + n_u), n_u, False)

        # Non-zeros from up bare mass terms
        pos_u = select_conditions_from_system(
            system, conditions_non_zero_entries, non_zero_entries_up_vlq, pos_u, 6 * (6 + n_u + n_d), n_u, True)

        for decomposition_d in Y_d:

            pos_d = pos_u
            # Non-zeros from Y_u_1
            pos_d = select_conditions_from_system(
                system, conditions_non_zero_entries, decomposition_d[0], pos_d, 6 * (3 + n_u), n_d, False)

            # Non-zeros from Y_u_2
            pos_d = select_conditions_from_system(
                system, conditions_non_zero_entries, decomposition_d[1], pos_d, 6 * (3 + n_u) + 3 * (3 + n_d), n_d, False)

            # Non-zeros from up bare mass terms
            pos_d = select_conditions_from_system(
                system, conditions_non_zero_entries, non_zero_entries_down_vlq, pos_d, 6 * (6 + n_u + n_d) + n_u * (3 + n_u), n_d, True)

            # Compute null space of conditions_non_zero_entries
            null_space = scipy.linalg.null_space(conditions_non_zero_entries)

            # Check if null space forbids remaining terms of the Lagrangian
            if np.any(null_space):
                b = np.zeros((1, n_phases))
                for k in range(np.shape(null_space)[1]):
                    b += random.uniform(1, 2) * null_space[:, k]

                # Set the scale below which a number is zero
                scale = 0
                for elem in conditions_non_zero_entries @ b.T:
                    if abs(elem) > scale:
                        scale = abs(elem)

                b = system @ b.T
                non_zeros_system = 0
                for elem in b:
                    if abs(elem) <= scale * 1e2:
                        non_zeros_system += 1
                if non_zeros_system == n_non_zeros:
                    null_space /= null_space[0]
                    null_space = np.transpose(null_space)[0]
                    n = 11 + 2 * (n_u + n_d)
                    output = np.zeros(n)
                    output[1] = null_space[0]
                    for i in range(n - 3):
                        output[3 + i] = np.round(null_space[i + 1])
                    possible_decompositions.append(output)

    if possible_decompositions == []:
        return False, possible_decompositions
    else:
        return True, possible_decompositions


# Constructs mass textures corresponding to a given set of field charges
def construct_texture_from_symmetry(charges, n_u, n_d):

    m_u = np.zeros([3 + n_u, 3 + n_u])
    m_d = np.zeros([3 + n_d, 3 + n_d])

    for i in range(2):
        for j in range(3):
            for k in range(3 + n_u):
                if not charges[i] + charges[2 + j] - charges[5 + k]:
                    m_u[j, k] = i + 1

            for k in range(3 + n_d):
                if not -charges[i] + charges[2 + j] - charges[8 + n_u + k]:
                    m_d[j, k] = i + 1

    for i in range(n_u):
        for j in range(3 + n_u):
            if not -charges[5 + j] + charges[11 + n_u + n_d + i]:
                m_u[3 + i, j] = 3

    for i in range(n_d):
        for j in range(3 + n_d):
            if not -charges[5 + n_u + j] + charges[11 + 2 * n_u + n_d + i]:
                m_d[3 + i, j] = 3

    return m_u, m_d


def main():

    args = sys.argv[1:]
    filename = args[0]
    n_u, n_d, set_mrt = io_mrt.read_mrt_after_min(filename)
    system = define_system_eqs(n_u, n_d)
    print(system)

    for textures in set_mrt:
        length = len(textures[2])
        for i in reversed(range(length)):
            found_decomposition, decompositions = check_pair_texture_zeros(
                textures[2][i], system, n_u, n_d)
            if found_decomposition:
                print("DECOMPOSITION FOUND")
                textures[2][i] = decompositions
                i += 1
            else:
                textures[2].pop(i)

    filename = "output/test_symmetry"
    io_mrt.print_mrt_after_symmetry(
        set_mrt, filename, n_u, n_d)

    return


if __name__ == "__main__":
    main()
