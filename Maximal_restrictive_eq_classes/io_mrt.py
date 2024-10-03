import numpy as np
import cmath
import sys
import pdb

import physical_parameters as phys
import extract_obs as ext
import minimisation as min

#####################################################
#
#   BEFORE MINIMISATION
#
#####################################################


# This function stores the results from texture_zeros.py in a file
def print_mrt_before_min(set_mrt,
                         filename, n_u, n_d):

    with open(filename, "w") as f:
        f.write(f"{n_u} {n_d}\n")
        for mrt_n_zeros_u in set_mrt:
            n_zeros = mrt_n_zeros_u[0].split(" ")
            f.write(f"{n_zeros[0]} {n_zeros[1]}\n")
            for pair in mrt_n_zeros_u[1]:
                for i in pair[0]:
                    for j in i:
                        f.write(f"{int(j)} ")

                for i in pair[1]:
                    for j in i:
                        f.write(f"{int(j)} ")
                f.write("\n")

    return


# This function reads the results stored in a file from textures_zeros.py
def read_mrt_before_min(filename):

    set_mrt = []
    with open(filename, "r") as f:
        line = f.readline()
        n = line.split(" ")
        n_u = int(n[0]) + 3
        n_d = int(n[1]) + 3
        m_u = np.zeros((n_u, n_u), dtype='uint8')
        m_d = np.zeros((n_d, n_d), dtype='uint8')

        i = 0
        while line != "":
            line = f.readline()
            line_split = line.split(" ")

            if line == "":
                return n_u - 3, n_d - 3, set_mrt

            elif len(line_split) != 2:
                for j in range(n_u):
                    for k in range(n_u):
                        m_u[j, k] = int(line_split[i])
                        i += 1

                for j in range(n_d):
                    for k in range(n_d):
                        m_d[j, k] = int(line_split[i])
                        i += 1

                set_mrt[-1][2].append([np.copy(m_u), np.copy(m_d)])
                i = 0

            else:
                n_zeros_u = int(line_split[0])
                n_zeros_d = int(line_split[1])
                set_mrt.append([n_zeros_u, n_zeros_d, []])


# This function stores the set of maximally restrictive pairs in a file
def pretty_print_set_mrt_before(filename_input, filename_output):

    _, _, set_mrt = read_mrt_before_min(filename_input)
    with open(filename_output, "w") as f:
        f.write(
            "LIST OF MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS (BEFORE MINUIT):\n\n")
        num_pairs = 0
        for mrt_n_zeros_u in set_mrt:
            n_zeros_u = mrt_n_zeros_u[0]
            n_zeros_d = mrt_n_zeros_u[1]
            num_pairs_n_zeros_u = 0
            f.write(
                f"PAIRS WITH ({n_zeros_u},{n_zeros_d}) TEXTURE ZEROS FOR (M_u, M_d):\n\n")
            for pair in mrt_n_zeros_u[2]:
                f.write(f"M_u:\n{pair[0]}\n")
                f.write(f"M_d:\n{pair[1]}\n")
                f.write("\n")
                num_pairs_n_zeros_u += 1
            f.write(
                f"THERE ARE {num_pairs_n_zeros_u} PAIRS WITH ({n_zeros_u},{n_zeros_d}) TEXTURE ZEROS\n\n")
            num_pairs += num_pairs_n_zeros_u

        f.write(
            f"\nTHERE ARE IN TOTAL {num_pairs} MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS (BEFORE MINUIT). \n")

    return


#####################################################
#
#   AFTER MINIMISATION
#
#####################################################


# This function stores the results from minuit_minimisation.py in a file
def print_mrt_after_min(set_mrt, filename, n_u, n_d):

    with open(filename, "w") as f:
        f.write(f"{n_u} {n_d}\n")
        for mrt_n_zeros_u in set_mrt:
            f.write(f"{mrt_n_zeros_u[0]} {mrt_n_zeros_u[1]}\n")
            for parameters in mrt_n_zeros_u[2]:
                for value in parameters:
                    f.write(f"{value} ")
                f.write("\n")

    return


# This function reads the results stored in a file from minuit_minimisation.py
def read_mrt_after_min(filename):

    set_mrt = []
    with open(filename, "r") as f:
        line = f.readline()
        n = line.split(" ")
        n_u = int(n[0]) + 3
        n_d = int(n[1]) + 3
        m_u = np.zeros((n_u, n_u), dtype='complex128')
        m_d = np.zeros((n_d, n_d), dtype='complex128')

        i = 0
        while True:
            line = f.readline()
            line_split = line.split(" ")

            if line == "":
                return n_u - 3, n_d - 3, set_mrt

            elif len(line_split) != 2:
                for j in range(n_u):
                    for k in range(n_u):
                        m_u[j, k] = float(line_split[i]) + \
                            float(line_split[i+1]) * 1j
                        i += 2

                for j in range(n_d):
                    for k in range(n_d):
                        m_d[j, k] = float(line_split[i]) + \
                            float(line_split[i+1]) * 1j
                        i += 2

                set_mrt[-1][2].append([np.copy(m_u), np.copy(m_d)])
                i = 0

            else:
                n_zeros_u = int(line_split[0])
                n_zeros_d = int(line_split[1])
                set_mrt.append([n_zeros_u, n_zeros_d, []])


# These two functions are used to pretty print the results after minimisation
def info_minimum(M_u, M_d):

    D_u, D_d, delta, Jarlskog, V = ext.extract_obs(
        M_u, M_d)

    chi_square_mass, chi_square_m_VLQ, chi_square_V, chi_square_gamma = min.compute_chi_square(
        D_u, D_d, V)

    scale_up = phys.MASS_T / D_u[2]
    scale_down = phys.MASS_B / D_d[2]

    string = f"chi_square = {np.sum(chi_square_mass) + np.sum(chi_square_m_VLQ) + np.sum(chi_square_V) + chi_square_gamma}\n"
    string += f"scale_up = {scale_up}\n"
    string += f"scale_down = {scale_down}\n"
    string += f"m_u = {D_u[0] * scale_up / phys.MeV} MeV/ chi_square_ratio_ut = {chi_square_mass[0]}\n"
    string += f"m_c = {D_u[1] * scale_up / phys.GeV} GeV/ chi_square_ratio_ct = {chi_square_mass[1]}\n"
    string += f"m_t = {D_u[2] * scale_up / phys.GeV} GeV\n"
    string += f"m_VLQ = {D_u[3] * scale_up / phys.TeV} TeV / chi_square_m_VLQ = {chi_square_m_VLQ}\n"
    string += f"m_d = {D_d[0] * scale_down / phys.MeV} MeV / chi_square_ratio_db = {chi_square_mass[2]}\n"
    string += f"m_s = {D_d[1] * scale_down / phys.MeV} MeV/ chi_square_m_ratio_sb = {chi_square_mass[3]}\n"
    string += f"m_b = {D_d[2] * scale_down / phys.GeV} GeV\n"
    string += f"V_ud = {abs(V[0, 0])} / chi_square_V_ud = {chi_square_V[0, 0]}\n"
    string += f"V_us = {abs(V[0, 1])} / chi_square_V_us = {chi_square_V[0, 1]}\n"
    string += f"V_ub = {abs(V[0, 2])} / chi_square_V_ub = {chi_square_V[0, 2]}\n"
    string += f"V_cd = {abs(V[1, 0])} / chi_square_V_cd = {chi_square_V[1, 0]}\n"
    string += f"V_cs = {abs(V[1, 1])} / chi_square_V_cs = {chi_square_V[1, 1]}\n"
    string += f"V_cb = {abs(V[1, 2])} / chi_square_V_cb = {chi_square_V[1, 2]}\n"
    string += f"V_td = {abs(V[2, 0])} / chi_square_V_td = {chi_square_V[2, 0]}\n"
    string += f"V_ts = {abs(V[2, 1])} / chi_square_V_ts = {chi_square_V[2, 1]}\n"
    string += f"V_tb = {abs(V[2, 2])} / chi_square_V_tb = {chi_square_V[2, 2]}\n"

    first_row_unitarity = abs(V[0, 0])**2 + abs(V[0, 1])**2 + abs(V[0, 2])**2
    string += f"|V_ud|^2 + |V_us|^2 + |V_ub|^2 = {first_row_unitarity} / chi_square_first_row_unitarity = {((first_row_unitarity - 1) / phys.FIRST_ROW_SIGMA) ** 2}\n"
    gamma = cmath.phase(-V[0, 0] * V[1, 2] * np.conj(V[0, 2])
                        * np.conj(V[1, 0])) / (2 * cmath.pi) * 360
    string += f"gamma = {gamma} / chi_square_gamma = {chi_square_gamma}\n"
    string += f"V:\n{V}\n"
    return string


def pretty_print_set_mrt_after(filename_input, filename_output):

    _, _, set_mrt = read_mrt_after_min(filename_input)
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

    with open(filename_output, "w") as f:
        f.write(
            "LIST OF MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS (AFTER MINUIT):\n\n")
        num_pairs = 0
        for n_zeros_u in set_mrt:
            n_pairs_n_zeros_u = 0
            f.write(
                f"PAIRS WITH ({n_zeros_u[0]},{n_zeros_u[1]}) TEXTURE ZEROS FOR (M_u, M_d):\n\n")
            for pair in n_zeros_u[2]:
                M_u = pair[0]
                M_d = pair[1]
                f.write("Minimum found:\n")
                f.write(f"M_u:\n{M_u}\n")
                f.write(f"M_d:\n{M_d}\n")
                f.write(info_minimum(M_u, M_d))
                f.write("\n")
                n_pairs_n_zeros_u += 1
            f.write(
                f"THERE ARE {n_pairs_n_zeros_u} PAIRS WITH ({n_zeros_u[0]},{n_zeros_u[1]}) TEXTURE ZEROS\n\n")
            num_pairs += n_pairs_n_zeros_u

        f.write(
            f"\nTHERE ARE IN TOTAL {num_pairs} MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS (AFTER MINUIT). \n")

    return


#####################################################
#
#   AFTER SYMMETRY REALISATION
#
#####################################################


# This function stores the results from texture_zeros.py in a file
def print_mrt_after_symmetry(set_mrt,
                             filename, n_u, n_d):

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)
    with open(filename, "w") as f:
        f.write(f"{n_u} {n_d}\n")
        for mrt_n_zeros_u in set_mrt:
            f.write(f"{mrt_n_zeros_u[0]} {mrt_n_zeros_u[1]}\n")
            for decomp in mrt_n_zeros_u[2]:
                for symmetry in decomp:
                    for field_charge in symmetry:
                        f.write(f"{np.round(field_charge)} ")
                    f.write("\n")
                f.write("\n")

    return


# This function creates a matrix with non zero entries for the
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
# def print_restrictive_pairs_from_minuit(set_mrt,
#                                        option,
#                                        filename):
#
#    with open(filename, option) as f:
#        f.write("LIST OF MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS REALIZABLE WITH 2HDM + ABELIAN SYMMETRY (AFTER MINUIT):\n\n")
#        num_pairs = 0
#        for pair in set_mrt:
#            f.write(f"M_u:\n{pair[1][0]}\n")
#            f.write(f"M_d:\n{pair[1][1]}\n")
#            f.write(f"\n")
#            f.write(f"POSSIBLE DECOMPOSITIONS:\n\n")
#            for decomposition in pair[2]:
#
#                matrix_decomp_u, fourth_row, matrix_decomp_d = decomposition_matrix_form(
#                    decomposition)
#
#                f.write(f"Y_u_1:\n{matrix_decomp_u[0]}\n\n")
#                f.write(f"Y_u_2:\n{matrix_decomp_u[1]}\n\n")
#
#                f.write(f"fourth_row_u:\n{fourth_row}\n\n")
#
#                f.write(f"Y_d_1:\n{matrix_decomp_d[0]}\n\n")
#                f.write(f"Y_d_2:\n{matrix_decomp_d[1]}\n\n")
#
#                null_space = decomposition[3]
#                constant = null_space[0][0]
#                for i, elem in enumerate(null_space):
#                    null_space[i][0] = null_space[i][0] / constant
#
#                f.write(f"Charges:\n{null_space}\n\n")
#
#                f.write(
#                    f"##############################################################################################################\n\n")
#
#            num_pairs += 1
#        f.write(
#            f"\nTHERE ARE IN TOTAL {num_pairs} MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS REALIZABLE WITH 2HDM + ABELIAN SYMMETRY (AFTER MINUIT). \n\n")
#
#        f.write("CONDENSED VIEW OF LIST:\n")
#        f.write(
#            "NOTATION: M_(u/d)_{positions of real entries}^{positions of complex entries}\n\n")
#
#        for pair in set_mrt:
#            for decomposition in pair[2]:
#                string = ""
#                for i, decomp_u in enumerate(decomposition[0]):
#                    string = string + f"Y_u_{int(i + 1)}: "
#                    for element in decomp_u:
#                        string = string + f"{element}/"
#                    string = string[:-1]
#                    while len(string) < 30 * (i + 1):
#                        string = string + " "
#
#                string = string + f"4th_row = {decomposition[1]}"
#                while len(string) < 110:
#                    string = string + " "
#
#                for i, decomp_d in enumerate(decomposition[2]):
#                    string = string + f"Y_d_{int(i + 1)}: "
#                    for element in decomp_d:
#                        string = string + f"{element}/"
#                    string = string[:-1]
#                    while len(string) < 30 * (i + 1) + 110:
#                        string = string + " "
#
#                f.write(f"{string}\n")
#            f.write(f"#####################################\n\n")
#
#        f.write(
#            f"\nTHERE ARE IN TOTAL {num_pairs} MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS REALIZABLE WITH 2HDM + ABELIAN SYMMETRY (AFTER MINUIT). \n\n")
#    return


def print_error_msg():

    print("ERROR: invalid input")
    print(
        "USAGE: python3 option filename_input filename_output")
    print("option: [b]efore (minimisation) || [a]fter (minimisation)")
    return


def main():

    args = sys.argv[1:]
    args_len = len(args)

    if (args_len != 3):
        print_error_msg()
        return 1

    b = {"before", "b", "B"}
    a = {"after", "a", "A"}
    if args[0] in b:
        pretty_print_set_mrt_before(args[1], args[2])
    elif args[0] in a:
        pretty_print_set_mrt_after(args[1], args[2])
    else:
        print_error_msg()

    return


if __name__ == "__main__":
    main()
