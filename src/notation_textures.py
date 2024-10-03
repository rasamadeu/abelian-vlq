from minuit_minimization_mp_non_unitary import read_maximmaly_restrictive_pairs
import _2HDM_1_scalar_singlet_abelian_symmetry as _2HDM
import numpy as np
import sys
import string
import pdb


def auxiliary_notation(set_pairs, option):

    if option == "u":
        index = 0
    if option == "d":
        index = 1

    textures = []
    for pair in set_pairs:
        num_zeros = pair[0].split(",")
        if option == "u":
            num_zeros = num_zeros[0][1::]
        if option == "d":
            num_zeros = num_zeros[1][:-1:]

        textures.append([num_zeros, pair[1][index]])

    i = 0
    k = 1
    while i < len(textures):

        # Eliminate repeated textures
        j = i + 1
        while j < len(textures):
            if np.array_equal(textures[i][1], textures[j][1]):
                textures.pop(j)
            else:
                j += 1

        num = textures[i][0]
        textures[i][0] = textures[i][0] + f"^{option}_{{{k}}}"
        # Add k to notation
        if i < len(textures) - 1:
            if num != textures[int(i + 1)][0]:
                k = 1
            else:
                k += 1
        i += 1

    textures = sorted(textures, key=lambda x: int(x[0].split("^")[0]))

    return textures


def define_notation_list(set_pairs):

    return [auxiliary_notation(set_pairs, "u"), auxiliary_notation(set_pairs, "d")]


def search_texture_notation(texture, notation_list):

    for elem in notation_list:
        if np.array_equal(elem[1], texture):
            return elem[0]


def get_pair_notation(pair, notation_list):

    return [search_texture_notation(pair[0], notation_list[0]),
            search_texture_notation(pair[1], notation_list[1])]


def search_decomp_notation(decomposition, notation_list, option):

    if option == "u":
        Y_u_1, Y_u_2 = _2HDM.auxiliary_decomposition_matrix_form(decomposition[0], "u")
        _, fourth_row = fourth_row_notation(decomposition[1],
                                            decomposition[2])
    if option == "d":
        Y_d_1, Y_d_2 = _2HDM.auxiliary_decomposition_matrix_form(decomposition[3], "d")

    for elem in notation_list:
        if option == "u":
            if (np.array_equal(Y_u_1, elem[1])
                and np.array_equal(Y_u_2, elem[2])
                and np.array_equal(fourth_row, elem[3])):
                return elem[0]
        if option == "d":
            if (np.array_equal(Y_d_1, elem[1])
                and np.array_equal(Y_d_2, elem[2])):
                return elem[0]


def get_decomposition_notation(decomposition, notation_decomp):

    return [search_decomp_notation(decomposition, notation_decomp[0], "u"),
            search_decomp_notation(decomposition, notation_decomp[1], "d")]


# This function returns a list with all decompositions compatible with 2HDM + 1 scalar singlet.
# The aim of this function is to simplify the analysis of the results
def auxiliary_decomposition(set_maximally_restrictive_pairs, notation_list, option):

    set_decompositions = []
    indices = list(range(len(set_maximally_restrictive_pairs)))
    if option == "u":
        index = 0
        index_decom = 0
    if option == "d":
        index = 1
        index_decom = 3

    i = 0
    while indices != []:

        # Construct list with all decompositions of M
        decompositions = []
        for elem in set_maximally_restrictive_pairs[indices[i]][2]:
            decompositions.append([elem[index_decom], elem[1], elem[2]])
        j = i + 1
        while j < len(indices):
            if np.array_equal(set_maximally_restrictive_pairs[indices[i]][1][index],
                              set_maximally_restrictive_pairs[indices[j]][1][index]):
                for elem in set_maximally_restrictive_pairs[indices[j]][2]:
                    if option == "u":
                        decompositions.append([elem[index_decom], elem[1], elem[2]])
                    if option == "d":
                        decompositions.append([elem[index_decom]])
                indices.pop(j)
            else:
                j += 1

        # Remove repeated decompositions
        j = 0
        while j < len(decompositions) - 1:
            k = j + 1
            while k < len(decompositions):
                if (np.array_equal(decompositions[j][0][0],
                                   decompositions[k][0][0])
                    and np.array_equal(decompositions[j][0][1],
                                       decompositions[k][0][1])):
                    if option == "u":
                        if (decompositions[j][1] == decompositions[k][1]
                            and decompositions[j][2] == decompositions[k][2]):
                            decompositions.pop(k)
                        else:
                            k += 1
                    else:
                        decompositions.pop(k)
                else:
                    k += 1
            j += 1

        notation_decomp = search_texture_notation(set_maximally_restrictive_pairs[indices[i]][1][index], notation_list)
        notation_decomp = notation_decomp.split("_")

        for order, elem in enumerate(decompositions):
            elem_matrix_form_1, elem_matrix_form_2 = _2HDM.auxiliary_decomposition_matrix_form(elem[0], option)
            elem_latex_form_1 = texture_latex_format(elem_matrix_form_1)
            elem_latex_form_2 = texture_latex_format(elem_matrix_form_2)
            if option == "u":
                fourth_row_latex_format, fourth_row = fourth_row_notation(elem[1],
                                                                          elem[2])
                set_decompositions.append([notation_decomp[0] + "_{" + notation_decomp[1][1:-1] + f",{order + 1}" + "}",
                                           elem_matrix_form_1,
                                           elem_matrix_form_2,
                                           fourth_row,
                                           elem_latex_form_1,
                                           elem_latex_form_2,
                                           fourth_row_latex_format])
            if option == "d":
                set_decompositions.append([notation_decomp[0] + "_{" + notation_decomp[1][1:-1] + f",{order + 1}" + "}",
                                           elem_matrix_form_1,
                                           elem_matrix_form_2,
                                           elem_latex_form_1,
                                           elem_latex_form_2])
        indices.pop(i)

    set_decompositions = sorted(set_decompositions, key=lambda x: int(x[0].split("^")[0]))
    return set_decompositions


def decomposition_notation(set_maximally_restrictive_pairs, notation_list):

    return [auxiliary_decomposition(set_maximally_restrictive_pairs, notation_list[0], "u"),
            auxiliary_decomposition(set_maximally_restrictive_pairs, notation_list[1], "d")]


def fourth_row_notation(positions, decomposition):

    fourth_row = np.zeros((1, 4))
    if decomposition == []:
        for i, entry in enumerate(positions):
            fourth_row[0, entry[1]] = i + 1
    else:
        for i, entry in enumerate(positions):
            fourth_row[0, entry[1]] = decomposition[i] + 1

    latex_format = "\\begin{pmatrix} "
    for j in range(4):
        if len(positions) == 3:
            if fourth_row[0, j] == 3:
                latex_format = latex_format + "S^{\\ast} & "
            if fourth_row[0, j] == 2:
                latex_format = latex_format + "S & "
            if fourth_row[0, j] == 1:
                latex_format = latex_format + "\\mathrm{b.m.} & "
        else:
            if fourth_row[0, j] != 0:
                latex_format = latex_format + "\\times & "
        if fourth_row[0, j] == 0:
            latex_format = latex_format + "0 & "

    latex_format = latex_format[0:-3]
    latex_format = latex_format + "\\\\"
    latex_format = latex_format + "\\end{pmatrix}"

    return latex_format, fourth_row


def texture_latex_format(texture):

    latex_format = "\\begin{pmatrix} "
    size = np.shape(texture)

    for i in range(size[0]):
        for j in range(size[1]):
            if texture[i, j] == 1:
                latex_format = latex_format + "\\times & "
            if texture[i, j] == 0:
                latex_format = latex_format + " 0 & "

        latex_format = latex_format[0:-3]
        if i != size[0] - 1:
            latex_format = latex_format + "\\\\   "

    latex_format = latex_format + "\\end{pmatrix}"

    return latex_format


# This function writes to a file the maximally restrictive pairs found by Minuit
def print_notation_and_latex(set_pairs_3_sigma,
                             set_pairs_1_sigma,
                             notation_list,
                             set_decompositions_3_sigma,
                             set_decompositions_1_sigma,
                             notation_decomp,
                             option,
                             filename):

    with open(filename, option) as f:
        f.write("NOTATION FOR M_U TEXTURE ZEROS:\n\n")
        for elem in notation_list[0]:
            f.write(f"{elem[0]}\n")
            f.write(f"{elem[1]}\n")
            f.write("\n")

        f.write(f"#####################################\n\n")

        f.write("NOTATION FOR M_D TEXTURE ZEROS:\n\n")
        for elem in notation_list[1]:
            f.write(f"{elem[0]}\n")
            f.write(f"{elem[1]}\n")
            f.write("\n")

        f.write(f"#####################################\n\n")

        f.write("NOTATION FOR DECOMPOSITIONS FOUND FOR M_U TEXTURE ZEROS:\n\n")
        for elem in notation_decomp[0]:
            f.write(f"{elem[0]}\n")
            f.write(f"{elem[1]}\n")
            f.write(f"{elem[2]}\n")
            f.write(f"{elem[3]}\n")
            f.write("\n")

        f.write(f"#####################################\n\n")

        f.write("NOTATION FOR DECOMPOSITIONS FOUND FOR M_D TEXTURE ZEROS:\n\n")
        for elem in notation_decomp[1]:
            f.write(f"{elem[0]}\n")
            f.write(f"{elem[1]}\n")
            f.write(f"{elem[2]}\n")
            f.write("\n")

        f.write(f"#####################################\n\n")

        f.write("LATEX TABLES:\n\n")

        f.write("TABLE WITH TEXTURE FOR M^U:\n\n")
        i = 0
        j = 0
        table = ""
        n_zeros = "8"
        while i < len(notation_list[0]):
            while (j < 4
                   and i + j < len(notation_list[0])
                   and n_zeros == notation_list[0][i + j][0].split("^")[0]):
                table = table + f"        {string.punctuation[3]}{notation_list[0][i + j][0]} \\sim {texture_latex_format(notation_list[0][i + j][1])}{string.punctuation[3]} &\n"
                j += 1

            n_zeros = notation_list[0][i][0].split("^")[0]
            table = table[0:-3] + f"\\\\\n\n"
            i += j
            j = 0
        f.write(table)

        f.write("\n\nTABLE WITH TEXTURE FOR M^D:\n\n")
        i = 0
        j = 0
        table = ""
        n_zeros = "3"
        while i < len(notation_list[1]):
            while (j < 4
                   and i + j < len(notation_list[1])
                   and n_zeros == notation_list[1][i + j][0].split("^")[0]):
                table = table + f"        {string.punctuation[3]}{notation_list[1][i + j][0]} \\sim {texture_latex_format(notation_list[1][i + j][1])}{string.punctuation[3]} &\n"
                j += 1

            n_zeros = notation_list[1][i][0].split("^")[0]
            table = table[0:-3] + f"\\\\\n\n"
            i += j
            j = 0
        f.write(table)

        # Remove from set_pairs_3_sigma the pairs of set_pairs_1_sigma
        for pair in set_pairs_1_sigma:
            i = 0
            while i < len(set_pairs_3_sigma):
                if (np.array_equal(pair[1][0], set_pairs_3_sigma[i][1][0])
                and np.array_equal(pair[1][1], set_pairs_3_sigma[i][1][1])):
                    set_pairs_3_sigma.pop(i)
                    break
                i += 1

        # Construct list to write the table in latex format
        list_table_1 = []
        for i in range(50, 60):
            down_1 = []
            for pair_1 in set_pairs_1_sigma:
                if np.array_equal(notation_list[0][i][1], pair_1[1][0]):
                    down_1.append(search_texture_notation(pair_1[1][1], notation_list[1]))

            if down_1 != []:
                down_1 = sorted(down_1, key=lambda x: int(x.split("^")[0]) + int(x.split("^")[1][3:-1]))
                list_table_1.append([notation_list[0][i][0], down_1])

        for i in range(15, 22):

            up_1 = []
            for pair_1 in set_pairs_1_sigma:
                if np.array_equal(notation_list[1][i][1], pair_1[1][1]):
                    up_1.append(search_texture_notation(pair_1[1][0], notation_list[0]))

            if up_1 != []:
                up_1 = sorted(up_1, key=lambda x: int(x.split("^")[0]) + int(x.split("^")[1][3:-1]))
                list_table_1.append([up_1, notation_list[1][i][0]])


        list_table_3 = []

        for i in range(50, 60):
            down_3 = []
            for pair_3 in set_pairs_3_sigma:
                if np.array_equal(notation_list[0][i][1], pair_3[1][0]):
                    down_3.append(search_texture_notation(pair_3[1][1], notation_list[1]))

            if down_3 != []:
                down_3 = sorted(down_3, key=lambda x: int(x.split("^")[0]) + int(x.split("^")[1][3:-1]))
                list_table_3.append([notation_list[0][i][0], down_3])

        for i in range(15, 22):

            up_3 = []
            for pair_3 in set_pairs_3_sigma:
                if np.array_equal(notation_list[1][i][1], pair_3[1][1]):
                    up_3.append(search_texture_notation(pair_3[1][0], notation_list[0]))

            if up_3 != []:
                up_3 = sorted(up_3, key=lambda x: int(x.split("^")[0]) + int(x.split("^")[1][3:-1]))
                list_table_3.append([up_3, notation_list[1][i][0]])

        # Table with pairs of textures at 3 and 1 sigma
        f.write("\n\n TABLE WITH PAIRS OF TEXTURES COMPATIBLE AT 1 SIGMA\n")
        f.write("M_U | M_D \n\n")
        for i, info in enumerate(list_table_1):
            line = f"        "
            if not isinstance(info[0], list):
                line = line + f"{string.punctuation[3]}{info[0]}{string.punctuation[3]} "
            else:
                for notation_up in info[0]:
                    line = line + f"{string.punctuation[3]}{notation_up}{string.punctuation[3]} "
            line = line + " & "
            if not isinstance(info[1], list):
                line = line + f"{string.punctuation[3]}{info[1]}{string.punctuation[3]} "
            else:
                for notation_down in info[1]:
                    line = line + f"{string.punctuation[3]}{notation_down}{string.punctuation[3]} "
            line = line + "\\\\\n"
            f.write(line)

        f.write("\n\n TABLE WITH PAIRS OF TEXTURES COMPATIBLE AT 3 SIGMA\n")
        f.write("M_U | M_D \n\n")
        for i, info in enumerate(list_table_3):
            line = f"        "
            if not isinstance(info[0], list):
                line = line + f"{string.punctuation[3]}{info[0]}{string.punctuation[3]} "
            else:
                for notation_up in info[0]:
                    line = line + f"{string.punctuation[3]}{notation_up}{string.punctuation[3]} "
            line = line + " & "
            if not isinstance(info[1], list):
                line = line + f"{string.punctuation[3]}{info[1]}{string.punctuation[3]} "
            else:
                for notation_down in info[1]:
                    line = line + f"{string.punctuation[3]}{notation_down}{string.punctuation[3]} "
            line = line + "\\\\\n"
            f.write(line)

        # Remove from set_decompositions_3_sigma the pairs of set_decompositions_1_sigma
        for pair in set_decompositions_1_sigma:
            i = 0
            while i < len(set_decompositions_3_sigma):
                if (np.array_equal(pair[1][0], set_decompositions_3_sigma[i][1][0])
                and np.array_equal(pair[1][1], set_decompositions_3_sigma[i][1][1])):
                    set_decompositions_3_sigma.pop(i)
                    break
                i += 1

        # Write table of decompositions found at 3 sigma in latex format
        f.write("\n\n TABLE WITH DECOMPOSITIONS OF TEXTURES FOR M_U:\n\n")
        i = 0
        for decomposition in notation_decomp[0]:
            f.write(f"        {string.punctuation[3]}{decomposition[0]}{string.punctuation[3]} &\n")
            f.write(f"        {string.punctuation[3]}{decomposition[4]}{string.punctuation[3]} &\n")
            f.write(f"        {string.punctuation[3]}{decomposition[5]}{string.punctuation[3]} &\n")
            f.write(f"        {string.punctuation[3]}{decomposition[6]}{string.punctuation[3]} ")
            if i < 1:
                f.write(f"&\n\n")
                i += 1
            else:
                f.write(f"\\\\\n\n")
                i = 0

        # Write table of decompositions found at 3 sigma in latex format
        f.write("\n\n TABLE WITH DECOMPOSITIONS OF TEXTURES FOR M_D:\n\n")
        i = 0
        for decomposition in notation_decomp[1]:
            f.write(f"        {string.punctuation[3]}{decomposition[0]}{string.punctuation[3]} &\n")
            f.write(f"        {string.punctuation[3]}{decomposition[3]}{string.punctuation[3]} &\n")
            f.write(f"        {string.punctuation[3]}{decomposition[4]}{string.punctuation[3]} ")
            if i < 2:
                f.write(f"&\n\n")
                i += 1
            else:
                f.write(f"\\\\\n\n")
                i = 0

        # Write table with decompositions found at 3 sigma:
        f.write("\n\n TABLE WITH DECOMPOSITIONS FOUND AT 3 SIGMA:\n\n")
        set_viable_decomp = []
        for pair in set_decompositions_3_sigma:
            for decomposition in pair[2]:
                set_viable_decomp.append(get_decomposition_notation(decomposition, notation_decomp))

        i = 0
        while i < len(set_viable_decomp):
            j = i + 1
            set_viable_decomp[i][0] = f"{string.punctuation[3]}" + set_viable_decomp[i][0] + f"{string.punctuation[3]}" 
            while j < len(set_viable_decomp):
                if set_viable_decomp[i][1] == set_viable_decomp[j][1]:
                    set_viable_decomp[i][0] = set_viable_decomp[i][0] + f" {string.punctuation[3]}{set_viable_decomp[j][0]}{string.punctuation[3]}"
                    set_viable_decomp.pop(j)
                else:
                    j += 1

            f.write(f"        {set_viable_decomp[i][0]} & {string.punctuation[3]}{set_viable_decomp[i][1]}{string.punctuation[3]}\\\\\n\n")
            i += 1

        # Write table with decompositions found at 1 sigma:
        f.write("\n\n TABLE WITH DECOMPOSITIONS FOUND AT 1 SIGMA:\n\n")
        set_viable_decomp = []
        for pair in set_decompositions_1_sigma:
            for decomposition in pair[2]:
                set_viable_decomp.append(get_decomposition_notation(decomposition, notation_decomp))

        i = 0
        while i < len(set_viable_decomp):
            j = i + 1
            set_viable_decomp[i][0] = f"{string.punctuation[3]}" + set_viable_decomp[i][0] + f"{string.punctuation[3]}" 
            while j < len(set_viable_decomp):
                if set_viable_decomp[i][1] == set_viable_decomp[j][1]:
                    set_viable_decomp[i][0] = set_viable_decomp[i][0] + f" {string.punctuation[3]}{set_viable_decomp[j][0]}{string.punctuation[3]}"
                    set_viable_decomp.pop(j)
                else:
                    j += 1

            f.write(f"        {set_viable_decomp[i][0]} & {string.punctuation[3]}{set_viable_decomp[i][1]}{string.punctuation[3]}\\\\\n\n")
            i += 1
    return


def main():

    args = sys.argv[1:]

    set_pairs_3_sigma = read_maximmaly_restrictive_pairs(args[0])
    set_pairs_1_sigma = read_maximmaly_restrictive_pairs(args[1])
    set_decompositions_3_sigma = _2HDM.realizable_pairs_textures(args[0])
    set_decompositions_1_sigma = _2HDM.realizable_pairs_textures(args[1])

    # Define notation list
    notation_list = define_notation_list(set_pairs_3_sigma)
    notation_decomp = decomposition_notation(set_decompositions_3_sigma, notation_list)

    print_notation_and_latex(set_pairs_3_sigma,
                             set_pairs_1_sigma,
                             notation_list,
                             set_decompositions_3_sigma,
                             set_decompositions_1_sigma,
                             notation_decomp,
                             "w",
                             "output/Sem_verificar_massas_2_vez/Sort massas + 20000reps/Values_taken_from_PDG_2022/non_unitary_results/notation.txt")

    return


if __name__ == "__main__":
    main()
