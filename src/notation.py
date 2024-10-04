import numpy as np
import sys
import pdb

import io_mrt
import texture_zeros as text
import abelian_symmetry_2HDM as abelian

# Output files with data ready for Latex
TABLE_NOTATION = "output/notation.dat"
TABLE_PAIRS = "output/table_pairs.dat"
TABLE_CHARGES = "output/table_charges.dat"
TABLE_DECOMP = "output/table_decomp.dat"

# Number of columns for table with textures notation
N_COLS_TABLE_NOTATION = 4

# Character to represent non-zero entries
NON_ZEROS = "\\times"


def auxiliary_notation(set_mrt, dim, option):

    i = 0
    if option == 'd':
        i = 1

    notation = []
    for n_zeros_u in set_mrt:
        list_pairs = n_zeros_u[2]
        textures = np.zeros(len(list_pairs))
        for j, pair in enumerate(list_pairs):
            textures[j] = text.get_index_of_texture(pair[i], dim, n_zeros_u[i])
        textures_set = set(textures)
        notation.append([n_zeros_u[i], sorted(textures_set)])

    return notation


def define_notation_list(set_mrt, dim_u, dim_d):

    return auxiliary_notation(set_mrt, dim_u, "u"), auxiliary_notation(set_mrt, dim_d, "d")


def texture_latex_format(texture, non_zero=''):

    latex_format = "\\begin{pmatrix} "
    size = np.shape(texture)

    for i in range(size[0]):
        for j in range(size[1]):
            if texture[i, j] != 0:
                if non_zero != '':
                    latex_format = latex_format + f"{non_zero} & "
                else:
                    latex_format = latex_format + f"{texture[i, j]} & "
            else:
                latex_format = latex_format + "0 & "

        latex_format = latex_format[0:-3]
        if i != size[0] - 1:
            latex_format = latex_format + "\\\\   "

    latex_format = latex_format + " \\end{pmatrix} "

    return latex_format


def write_table_notation(notation, dim, superscript, n_cols, filename, option):

    with open(filename, option) as f:
        header = "\\begin{tabular}{| "
        for i in range(n_cols):
            header += "c "
        header += "|}\n\\hline\n"
        f.write(header)
        for n_zeros in notation:
            for i, texture in enumerate(n_zeros[1]):
                string = f"${n_zeros[0]}^{superscript}_{{{i + 1}}} \\sim "
                string += texture_latex_format(
                    text.get_texture_from_index(texture, dim, n_zeros[0]), NON_ZEROS)
                f.write(string + '$ ')
                if not (i + 1) % n_cols:
                    f.write(" \\\\ \n")
                else:
                    f.write(" & \n")
            if (i + 1) % n_cols:
                i += 1
                while (i + 1) % n_cols:
                    f.write(" & \n")
                    i += 1
                f.write(" \\\\ \n")
            f.write("\\hline\n")
        f.write("\\end{tabular} \n\n")

    return


def write_notation_pair(list_pairs, dim_u, dim_d, main_notation, other_notation, main_text):

    if main_text == "u":
        main_dim = dim_u
        main_index = 0
        other_dim = dim_d
        other_index = 1
        other_text = "d"
    else:
        main_dim = dim_d
        main_index = 1
        other_dim = dim_u
        other_index = 0
        other_text = "u"

    lines = []
    main_notation[1] = list(main_notation[1])
    other_notation[1] = list(other_notation[1])
    for i in range(len(main_notation[1])):
        lines.append(f"${main_notation[0]}^{main_text}_{{{i + 1}}}$ &")

    for pair in list_pairs[2]:
        i = main_notation[1].index(text.get_index_of_texture(
            pair[main_index], main_dim, main_notation[0]))
        j = other_notation[1].index(text.get_index_of_texture(
            pair[other_index], other_dim, other_notation[0]))
        lines[i] += f" ${other_notation[0]}^{other_text}_{{{j + 1}}}$"

    for i in range(len(main_notation[1])):
        lines[i] += "\\\\ \n"

    return lines


def write_table_pairs(set_mrt_min, dim_u, dim_d, notation_u, notation_d, filename):

    with open(filename, 'w') as f:
        f.write("\\begin{tabular}{|c|c|}\n")
        f.write("\\hline\n")
        for i, n_zeros_u in enumerate(set_mrt_min):
            if len(notation_u[i][1]) < len(notation_d[i][1]):
                lines = write_notation_pair(
                    n_zeros_u, dim_u, dim_d, notation_u[i], notation_d[i], "u")
            else:
                lines = write_notation_pair(
                    n_zeros_u, dim_u, dim_d, notation_d[i], notation_u[i], "d")
            for line in lines:
                f.write(line)

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
    return


def write_table_charges(set_mrt_sym, notation_u, notation_d, n_u, n_d, filename):

    with open(filename, 'w') as f:
        header = "\\begin{tabular} {|c|cc|ccc|"
        charges = "Pair & $\Phi_1$ & $\Phi_2$ & ${q_L}_1$ & ${q_L}_2$ & ${q_L}_3$"
        for i in range(3 + n_u):
            header += "c"
            charges += f"& ${{u_R}}_{{{i + 1}}}$ "
        header += "|"
        for i in range(3 + n_d):
            header += "c"
            charges += f"& ${{d_R}}_{{{i + 1}}}$ "
        header += "|"
        if n_u:
            for i in range(n_u):
                header += "c"
                charges += f"& ${{T_L}}_{{{i + 1}}}$ "
            header += "|"
        if n_d:
            for i in range(n_d):
                header += "c"
                charges += f"& ${{B_L}}_{{{i + 1}}}$ "
            header += "|"
        header += "}\n"
        charges += "\\\\ \n"
        f.write(header)
        f.write("\\hline\n")
        f.write(charges)
        f.write("\\hline\n")
        for i, n_zeros_u in enumerate(set_mrt_sym):
            for list_charges in n_zeros_u[2]:
                m_u, m_d = abelian.construct_texture_from_symmetry(
                    list_charges[0], n_u, n_d)
                j = notation_u[i][1].index(text.get_index_of_texture(
                    m_u, n_u + 3, notation_u[i][0]))
                k = notation_d[i][1].index(text.get_index_of_texture(
                    m_d, n_d + 3, notation_d[i][0]))
                for n, charges in enumerate(list_charges):
                    f.write(
                        f" $\\left({notation_u[i][0]}^u_{{{j + 1}}}, {notation_d[i][0]}^d_{{{k + 1}}}\\right)_{{{n + 1}}}$")
                    for charge in charges:
                        f.write(f" & {charge}")
                    f.write(" \\\\ \n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
    return


def write_table_decomp(set_mrt_sym, notation_u, notation_d, n_u, n_d, filename):

    with open(filename, 'w') as f:
        f.write("\\begin{tabular} {|c|cc|}\n")
        f.write("\\hline\n")
        f.write("Pair & $M_u$ & $M_d$ \\\\\n")
        f.write("\\hline\n")
        for i, n_zeros_u in enumerate(set_mrt_sym):
            for list_charges in n_zeros_u[2]:
                for n, charges in enumerate(list_charges):
                    m_u, m_d = abelian.construct_texture_from_symmetry(
                        charges, n_u, n_d)
                    j = notation_u[i][1].index(text.get_index_of_texture(
                        m_u, n_u + 3, notation_u[i][0]))
                    k = notation_d[i][1].index(text.get_index_of_texture(
                        m_d, n_d + 3, notation_d[i][0]))
                    f.write(
                        f" $\\left({notation_u[i][0]}^u_{{{j + 1}}}, {notation_d[i][0]}^d_{{{k + 1}}}\\right)_{{{n + 1}}}$ & ")
                    f.write("$" + texture_latex_format(
                        m_u) + "$ & $" + texture_latex_format(m_d) + "$ \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
    return


def main():

    args = sys.argv[1:]

    n_u, n_d, set_mrt_min = io_mrt.read_mrt_after_min(args[0])
    _, _, set_mrt_sym = io_mrt.read_mrt_after_symmetry(args[1])

    # Define notation list
    dim_u = n_u + 3
    dim_d = n_d + 3
    notation_u, notation_d = define_notation_list(set_mrt_min, dim_u, dim_d)

    write_table_notation(notation_u, dim_u, "u",
                         N_COLS_TABLE_NOTATION, TABLE_NOTATION, "w")
    write_table_notation(notation_d, dim_d, "d",
                         N_COLS_TABLE_NOTATION, TABLE_NOTATION, "a")
    write_table_pairs(set_mrt_min, dim_u, dim_d,
                      notation_u, notation_d, TABLE_PAIRS)
    write_table_charges(set_mrt_sym, notation_u,
                        notation_d, n_u, n_d, TABLE_CHARGES)
    write_table_decomp(set_mrt_sym, notation_u,
                       notation_d, n_u, n_d, TABLE_DECOMP)

    return


if __name__ == "__main__":
    main()
