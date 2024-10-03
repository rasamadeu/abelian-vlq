import numpy as np
import sys

import io_mrt
import texture_zeros as text

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


def texture_latex_format(texture, non_zero):

    latex_format = "\\begin{pmatrix} "
    size = np.shape(texture)

    for i in range(size[0]):
        for j in range(size[1]):
            if texture[i, j] != 0:
                latex_format = latex_format + f"{non_zero} & "
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


def main():

    args = sys.argv[1:]

    n_u, n_d, set_mrt_min = io_mrt.read_mrt_after_min(args[0])
    # _, _, set_mrt_sym = io_mrt.read_mrt_after_sym(args[0])

    # Define notation list
    dim_u = n_u + 3
    dim_d = n_d + 3
    notation_u, notation_d = define_notation_list(set_mrt_min, dim_u, dim_d)

    write_table_notation(notation_u, dim_u, "u",
                         N_COLS_TABLE_NOTATION, TABLE_NOTATION, "w")
    write_table_notation(notation_d, dim_d, "d",
                         N_COLS_TABLE_NOTATION, TABLE_NOTATION, "a")
    # write_table_pairs(set_mrt_min, notation_u, notation_d, TABLE_PAIRS)
    # write_table_charges(set_mrt_sym, notation_list, TABLE_CHARGES)
    # write_table_decomp(set_mrt_sym, notation_list, TABLE_DECOMP)

    return


if __name__ == "__main__":
    main()
