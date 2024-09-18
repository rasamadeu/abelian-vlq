import numpy as np
import sys
import pdb
import texture_zeros_v4
import minuit_minimization_mp_non_unitary


# This function reads the set of maximally restrictive pairs stored in a file
def read_maximmaly_restrictive_pairs(filename, option):

    set_maximally_restrictive_pairs = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != "":
            if line[0] == "P":
                string = line.split(" ")[2]
                list_pairs = []
                while line.split(" ")[0] != "THERE":
                    line = f.readline()
                    info_pairs = "M_u:\n"
                    texture_u = []
                    if line == "M_u:\n":
                        line = f.readline()
                        info_pairs += line
                        while line != "M_d:\n":
                            texture_u.append([int(s) for s in line if s.isdigit()])
                            line = f.readline()
                            info_pairs += line
                        texture_d = []
                        line = f.readline()
                        info_pairs += line
                        while line != "Minimum found:\n":
                            texture_d.append([int(s) for s in line if s.isdigit()])
                            line = f.readline()
                            info_pairs += line
                        while line != "\n":
                            line = f.readline()
                            info_pairs += line
                            if line[0] == "c":
                                chi_square = float(line.split(" ")[1])
                        list_pairs.append([[np.array(texture_u), np.array(texture_d)],
                                           chi_square,
                                           info_pairs])

                set_maximally_restrictive_pairs.append([string, list_pairs])
            line = f.readline()

    return set_maximally_restrictive_pairs


# This function writes to a file the maximally restrictive pairs found by Minuit
def print_restrictive_pairs_from_minuit(set_maximally_restrictive_pairs,
                                        option,
                                        filename):

    with open(filename, option) as f:
        f.write("LIST OF MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS (AFTER MINUIT):\n\n")
        num_pairs = 0
        for restrictive_pairs_n_zeros_u in set_maximally_restrictive_pairs:
            num_pairs_n_zeros_u = 0
            f.write(
                f"PAIRS WITH {restrictive_pairs_n_zeros_u[0]} TEXTURE ZEROS FOR (M_u, M_d):\n\n")
            for i in range(1, len(restrictive_pairs_n_zeros_u)):
                f.write(f"M_u:\n")
                f.write(f"{restrictive_pairs_n_zeros_u[i][0][0][0]}")
                f.write(f"\nM_d:\n")
                f.write(f"{restrictive_pairs_n_zeros_u[i][0][0][1]}")
                #f.write(f"{restrictive_pairs_n_zeros_u[i][0][2]}")
                #f.write(f"This pair was found in the following files:\n")
                #for j in range(1, len(restrictive_pairs_n_zeros_u[i])):
                #    f.write(f"{restrictive_pairs_n_zeros_u[i][j]}\n")
                f.write("\n\n")
                num_pairs_n_zeros_u += 1
            f.write(
                f"THERE ARE {num_pairs_n_zeros_u} PAIRS WITH {restrictive_pairs_n_zeros_u[0]} TEXTURE ZEROS\n\n")
            num_pairs += num_pairs_n_zeros_u
        f.write(
            f"\nTHERE ARE IN TOTAL {num_pairs} MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS (AFTER MINUIT). \n")

    return


def compile_info(info_files):

    length = len(info_files)
    num_classes = len(info_files[0][1])
    processed_info = []

    for k in range(num_classes):
        class_of_pairs = [info_files[0][1][k][0]]
        for i in range(length):
            filename = info_files[i][0]
            for pair in info_files[i][1][k][1]:
                min_pair = [pair.copy(), filename]
                for j in range(i + 1, length):
                    for m, pair_compare in enumerate(info_files[j][1][k][1]):
                        if (np.array_equal(pair[0][0], pair_compare[0][0])
                                and np.array_equal(pair[0][1], pair_compare[0][1])):
                            min_pair.append(info_files[j][0])
                            if(pair_compare[1] < pair[1]):
                                min_pair[0] = pair_compare.copy()
                            info_files[j][1][k][1].pop(m)
                            break
                class_of_pairs.append(min_pair)
        class_of_pairs[1:len(class_of_pairs)] = (
            sorted(class_of_pairs[1:len(class_of_pairs)], key=lambda x: x[0][1]))
        processed_info.append(class_of_pairs)

    filename = info_files[0][0].split(".")
    filename_str = filename[0][0:-1] + "compiled_info." + filename[1]
    print_restrictive_pairs_from_minuit(processed_info,
                                        "w",
                                        filename_str)


def eliminate_minimized_pairs(filename, info_files):

    for i, classes in enumerate(info_files[1]):
        for pair in classes[1]:
            k = 0
            while not (np.array_equal(pair[0][0], info_files[0][i][1][k][0])
                and np.array_equal(pair[0][1], info_files[0][i][1][k][1])):
                k += 1
            info_files[0][i][1].pop(k)

    for pair in info_files[0]:
        pair[0] = pair[0].replace(",", "_")
        pair[0] = pair[0][1:len(pair[0]) - 1]

    filename = filename.replace("after_MINUIT", "without_found")
    texture_zeros_v4.print_maximmaly_restrictive_pairs(info_files[0], "w", filename)
    return


def main():

    args = sys.argv[2:]
    info_files = []

    if sys.argv[1] == "c":
        for file in args:
            info_files.append([file, read_maximmaly_restrictive_pairs(file, sys.argv[0])])
            compile_info(info_files)
    elif sys.argv[1] == "d":
        info_files.append(texture_zeros_v4.read_maximmaly_restrictive_pairs(args[0]))
        info_files.append(read_maximmaly_restrictive_pairs(args[1], sys.argv[0]))
        eliminate_minimized_pairs(args[0], info_files)

    return


if __name__ == "__main__":
    main()
