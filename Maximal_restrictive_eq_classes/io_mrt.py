import numpy as np
import sys
import pdb

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
            n_zeros = mrt_n_zeros_u[0].split("_")
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
                return set_mrt

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

    set_mrt = read_mrt_before_min(filename_input)
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
def print_mrt_after_min():
    return


# This function reads the results stored in a file from minuit_minimisation.py
def read_mrt_after_min():
    return


# This function reads the results stored in a file from minuit_minimisation.py
def pretty_print_set_mrt_after(filename_input, filename_output):
    return


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
