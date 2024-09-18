# This module contains functions needed to compute the non-equivalent
# classes of texture zeros pairs of (M_up, M_down), where M_up and M_down
# correspond to the mass matrices of the up and down type quarks,
# respectively, for a SM extension with n_up and n_down isosinglets VLQ
# of type up and down.

import numpy as np
from itertools import combinations, permutations
from math import comb
import copy
import pdb
import time

# This function returns the texture zeros of square matrices of
# dimension dim with n_zeros. We write the positions of elements
# of a square matrix of dim n as:
#
# [0  1   2  ...  n-1]
# [n n+1 n+2 ... 2n-1]
# [.      .        . ]
# [.          .    . ]
# [.              .. ]
# [n(n-1) ..... n^2-1]


def get_texture_zeros(dim, n_zeros):

    zeros_positions = list(combinations(range(dim**2), n_zeros))
    size = len(zeros_positions)
    texture = np.ones((size, dim, dim), dtype="uint8")

    for i in range(size):
        for pos in zeros_positions[i]:
            texture[i, int(pos / dim), int(pos % dim)] = 0

    return list(texture)


# This function returns the permutation matrices of dimension dim
def get_permutation(dim):

    ones_positions = list(permutations(range(dim), dim))
    size = len(ones_positions)
    permutation = np.zeros((size, dim, dim), dtype="uint8")

    for i in range(size):
        for j in range(dim):
            permutation[i, j, ones_positions[i][j]] = 1

    return permutation


# The function get_weak_basis_permutations returns the weak basis
# permutations for a SM extension with n_up and n_down VLQ isosinglets
# of type up and down, respectively. If given a 3x3 permutation
# matrix P, the function get_left_weak_basis_permutations returns all
# permutation matrices of dim n+3 where the first 3x3 block is P.

def get_block_permutation(left_weak_basis_permutation):

    permutation = np.zeros((3, 3), dtype="uint8")
    for i in range(3):
        for j in range(3):
            permutation[i][j] = left_weak_basis_permutation[i][j]

    return permutation


def get_left_weak_basis_permutations(n, permutation_3_3=None):

    permutation_L = get_permutation(n)
    n_perm = np.shape(permutation_L)[0]

    if permutation_3_3 is None:
        permutation_3_3 = get_permutation(3)
        length = np.shape(permutation_3_3)[0]
        K_L = np.zeros((length * n_perm, 3 + n, 3 + n))
        for i in range(length):
            for j in range(n_perm):
                K_L[i * n_perm + j] = np.block([
                    [permutation_3_3[i], np.zeros((3, n), dtype="uint8")],
                    [np.zeros((n, 3), dtype="uint8"), permutation_L[j]]
                ])

    else:
        K_L = np.zeros((n_perm, 3 + n, 3 + n))
        for j in range(n_perm):
            K_L[j] = np.block([
                [permutation_3_3, np.zeros((3, n), dtype="uint8")],
                [np.zeros((n, 3), dtype="uint8"), permutation_L[j]]
            ])

    return K_L


def get_weak_basis_permutations(n, permutation_3_3=None):

    if permutation_3_3 is None:
        K_L = get_left_weak_basis_permutations(n)
    else:
        K_L = get_left_weak_basis_permutations(n, permutation_3_3)

    return K_L, get_permutation(3 + n)


# This function returns all the texture zeros which are equivalent to the
# case of no zeros imposed in a mass matrix of size dim
def get_non_restrictive_texture_zeros(dim):

    n_total_zeros = int((dim**2 - dim) / 2)
    n_textures = 0
    for i in range(1, n_total_zeros + 1):
        n_textures = n_textures + comb(n_total_zeros, i)

    zeros_positions = np.empty((n_total_zeros))
    k = 0
    for i in range(1, dim):
        for j in range(i):
            zeros_positions[k] = int(i * dim + j)
            k = k + 1

    texture = np.ones((2 * n_textures, dim, dim), dtype="uint8")
    k = 0

    for n_zeros in range(1, n_total_zeros + 1):

        combinations_n_zeros = list(combinations(zeros_positions, n_zeros))
        length_combinations_n_zeros = len(combinations_n_zeros)

        for j in range(length_combinations_n_zeros):
            for pos in combinations_n_zeros[j]:
                texture[k, int(pos / dim), int(pos % dim)] = 0

            texture[k + 1] = texture[k].T
            k = k + 2

    return texture


# This function returns the equivalence classes of the texture zeros for
# n VLQ isosinglets in the up sector or down sector. The return variable
# "set_equivalence_classes" is a list organized in the following way:
# set_equivalence_classes = [[1_zero], ... , [n_zeros], ... , [n_total_zeros]]
# where [n_zeros] = [[n_zeros_1], [n_zeros_2], ... , [n_zeros_N]] is a list
# containing all equivalence classes of n_zeros texture zeros.
# Each equivalence class is a list [label, texture], where label is a string
# given by "{n_zeros_texture_order}" that labels the equivalence class and
# texture is a numpy array containing a single texture of the equivalence
# class.

def get_equivalence_classes_texture_zeros(n, string, massless_tree_level=0):

    dim = 3 + n
    rank = dim - massless_tree_level
    n_total_zeros = dim ** 2 - rank

    if string == "up":
        set_K_L, set_K_R = get_weak_basis_permutations(n)
    elif string == "down":
        set_K_L, set_K_R = get_weak_basis_permutations(n, np.identity(3))

    set_non_restrictive_texture = get_non_restrictive_texture_zeros(
        dim)

    set_equivalence_classes = []
    test = 0

    # The matrix rank_check will be used to check rank of texture zeros
    rng = np.random.default_rng()
    rank_check = rng.random((dim, dim))

    # We calculate the equivalence classes for a given number of zeros
    # (n_zeros = 1, 2, ... , n_total_zeros) in each loop
    for n_zeros in range(1, n_total_zeros + 1):

        textures_n_zeros = get_texture_zeros(dim, n_zeros)
        equivalence_class_n_zeros = []
        n_zeros_order = 0

        # We compute all equivalence classes for the set of texture zeros with
        # n_zeros
        while(textures_n_zeros):

            equivalence_class = []
            # Construct the class of equivalent texture zeros.
            for K_R in set_K_R:
                for K_L in set_K_L:
                    equivalent_texture = K_L @ textures_n_zeros[0] @ K_R
                    equivalence_class.append(equivalent_texture)

            # Remove repeated textures from equivalence class
            equivalence_class = np.unique(equivalence_class, axis=0)

            valid_equivalence_class = True
            for equivalent_texture in equivalence_class:

                if valid_equivalence_class:

                    # Check if equivalence class contains texture zeros
                    # equivalent to no zeros imposed
                    if n_zeros <= n_total_zeros / 2:
                        for non_restrictive_texture in set_non_restrictive_texture:
                            if np.array_equal(equivalent_texture, non_restrictive_texture):
                                valid_equivalence_class = False
                                break
                    # Check rank of equivalent_texture
                    if (np.linalg.matrix_rank(
                            np.multiply(equivalent_texture, rank_check)) < rank):
                        valid_equivalence_class = False

                # Eliminate from the array "texture_n_zeros"
                # the textures in the equivalence class
                k = 0
                while not np.array_equal(equivalent_texture,
                                         textures_n_zeros[k]):
                    test += 1
                    k += 1
                textures_n_zeros.pop(k)

            if valid_equivalence_class:
                n_zeros_order += 1
                equivalence_class_n_zeros.append([f"{n_zeros}_{n_zeros_order}",
                                                  equivalence_class[0]])
                print(f"{n_zeros}_{n_zeros_order}")

        if equivalence_class_n_zeros:
            set_equivalence_classes.append(equivalence_class_n_zeros)

    print(set_equivalence_classes)
    return set_equivalence_classes


def print_equivalence_classes(equivalence_class, string, filename, option):

    with open(filename, option) as f:
        f.write(f"EQUIVALENCE CLASSES FOR M_{string}:\n\n")
        length = 0
        for equivalence_class_n_zeros in equivalence_class:
            length += len(equivalence_class_n_zeros)
            num_zeros = equivalence_class_n_zeros[0][0].split("_")[0]
            f.write(f"{len(equivalence_class_n_zeros)} EQUIVALENCE CLASSES WITH ")
            f.write(f"{num_zeros} TEXTURE ZEROS:\n\n")
            for text in equivalence_class_n_zeros:
                f.write(f"{text[0]}:\n")
                f.write(f"{text[1]}\n\n")

        f.write(f"THERE ARE IN TOTAL {length} EQUIVALENCE CLASSES for M_{string}.\n")
        f.write("##################################################################\n\n")

    return


# The function "get_pairs_equivalence_classes" returns the list "pairs_equivalence_classes"
# which stores the pairs of textures zeros (M_up, M_down). A pair of textures consists of
# a list containing a

def print_pairs_equivalence_classes(pairs, redundant_pairs, option="", filename=""):

    with open(filename, option) as f:
        f.write("LIST OF NON-EQUIVALENT PAIRS OF TEXTURE ZEROS:\n")
        num_pairs = 0
        for equivalence_class_n_zeros_up in pairs:
            for equivalence_class_n_zeros_order_up in equivalence_class_n_zeros_up:
                f.write(f"\n\n{equivalence_class_n_zeros_order_up[0][0]}:\n")
                for n_zeros_down in equivalence_class_n_zeros_order_up[1]:
                    for n_zeros_down_order in n_zeros_down:
                        f.write(f"{n_zeros_down_order[0]} / ")
                        num_pairs += 1
                    f.write("\n")

        num_redundant_pairs = 0
        f.write("\n\nLIST OF REDUNDANT PAIRS OF TEXTURE ZEROS:\n")
        if redundant_pairs is not None:
            for pairs in redundant_pairs:
                f.write(f"{pairs[0]}:\n")
                for i in range(1, len(pairs)):
                    f.write(f"{pairs[i]} / ")
                    num_redundant_pairs += 1
                f.write("\n\n")

        f.write(
            f"\nTHERE ARE IN TOTAL {num_pairs} NON-EQUIVALENT CLASSES OF TEXTURE ZEROS PAIRS (M_u, M_d). \n")

        f.write(
            f"THERE ARE IN TOTAL {num_redundant_pairs} REDUNDANT EQUIVALENCE CLASSES OF TEXTURE ZEROS PAIRS (M_u, M_d).\n")

    return


def get_pairs_equivalence_classes(n_up, n_down, option="", massless_up=0, massless_down=0,
                                  filename=""):

    set_equivalence_classes_up = get_equivalence_classes_texture_zeros(
        n_up, "up", massless_up)
    set_equivalence_classes_down = get_equivalence_classes_texture_zeros(
        n_down, "down", massless_down)

    # Eliminate redundant pairs of texture zeros
    set_K_L_up, set_K_R_up = get_weak_basis_permutations(n_up)
    set_K_R_down = get_permutation(3 + n_down)
    redundant_classes = []
    pairs_equivalence_classes = []

    # Cycle through all equivalence classes_up found for M_up
    for equivalence_class_n_zeros_up in set_equivalence_classes_up:
        pair_n_zeros_up = []
        for i in range(len(equivalence_class_n_zeros_up)):

            texture_up = equivalence_class_n_zeros_up[i][1]
            redundant_class_n_zeros_up = [equivalence_class_n_zeros_up[i][0]]
            pair_n_zeros_i_up = [equivalence_class_n_zeros_up[i],
                                 copy.deepcopy(set_equivalence_classes_down)]

            for K_R_up in set_K_R_up:
                for K_L_up in set_K_L_up:
                    equivalent_texture_up = K_L_up @ texture_up @ K_R_up
                    if np.array_equal(equivalent_texture_up, texture_up):
                        A_L = get_block_permutation(K_L_up)
                        set_K_L_down = get_left_weak_basis_permutations(n_down,
                                                                        A_L)
                        # Cycle through all equivalence classes found for M_down
                        for equivalence_class_n_zeros_down in pair_n_zeros_i_up[1]:
                            index = 0
                            while index < len(equivalence_class_n_zeros_down):
                                # Compute all textures in a given equivalence_class_down
                                # with n_zeros
                                texture_down = (
                                    equivalence_class_n_zeros_down[index][1])
                                for K_L_down in set_K_L_down:
                                    for K_R_down in set_K_R_down:
                                        equivalent_texture_down = (
                                            K_L_down @ texture_down @ K_R_down)  # AVISO:
                                        for n in range(index + 1,
                                                       len(equivalence_class_n_zeros_down)):
                                            if (np.array_equal(equivalent_texture_down,
                                                               equivalence_class_n_zeros_down[n][1])):
                                                redundant_class_n_zeros_up.append(
                                                    equivalence_class_n_zeros_down.pop(n)[0])
                                                break

                                index += 1

            size = len(redundant_class_n_zeros_up)
            redundant_class_n_zeros_up[1:size] = sorted(redundant_class_n_zeros_up[1:size])
            redundant_classes.append(redundant_class_n_zeros_up)
            pair_n_zeros_up.append(pair_n_zeros_i_up)

        if pair_n_zeros_up:
            pairs_equivalence_classes.append(pair_n_zeros_up)

    if option == "w":
        if filename == "":
            filename = f"output/{n_up}_up_{n_down}_down_isosinglets_VLQ.txt"
            if(massless_up != 0):
                filename = filename[:-4] + f"_{massless_up}_massless_up" + filename[-4:]
            if(massless_down != 0):
                filename = filename[:-4] + f"_{massless_down}_massless_down" + filename[-4:]
        print_equivalence_classes(set_equivalence_classes_up, "up", filename, "w")
        print_equivalence_classes(set_equivalence_classes_down, "down", filename, "a")
        print_pairs_equivalence_classes(
            pairs_equivalence_classes, redundant_classes, "a", filename)

    return pairs_equivalence_classes, redundant_classes


def main():

    start = time.time()
    get_pairs_equivalence_classes(0, 0, "w", 1, 0)
    end = time.time()

    print(f"TOTAL TIME = {int((float(end) - float(start)) / 60)} min ", end="")
    print(f"{(int(float(end) - float(start)) % 60)} sec \n")

    # start = time.time()
    # get_pairs_equivalence_classes(0, 0, "w")
    # end = time.time()
    # print(f"TOTAL TIME = {int((float(end) - float(start)) / 60)} min ", end="")
    # print(f"{(int(float(end) - float(start)) % 60)} sec \n")


if __name__ == "__main__":
    main()
