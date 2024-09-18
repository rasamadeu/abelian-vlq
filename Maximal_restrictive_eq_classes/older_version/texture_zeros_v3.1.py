# This module contains functions needed to compute the non-equivalent
# classes of maximally restrictive texture zeros pairs of (M_u, M_d),
# where M_u and M_d correspond to the mass matrices of the up and down
# type quarks, respectively, for a SM extension with n_u and n_d
# isosinglets VLQ of type up and down.

# This module is a faster implementation of texture_zeros_v3.py

import numpy as np
from itertools import combinations, permutations
from scipy.special import comb
import extract_obs as ext
import time
import sys
import os
import pdb

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
# permutations for a SM extension with n_u and n_d VLQ isosinglets
# of type u and d, respectively. If given a 3x3 permutation
# matrix P, the function get_left_weak_basis_permutations returns all
# permutation matrices of dim n+3 where the first 3x3 block is P.
def get_block_matrix(matrix):

    block_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            block_matrix[i][j] = matrix[i][j]

    return block_matrix


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
# case of no zeros imposed in a mass matrix of size dim (Theorem 1 in 2014
# Ludl and Grimus paper "A complete survey of texture zeros in the lepton
# mass matrices")
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

    texture = np.ones((2 * int(n_textures), dim, dim), dtype="uint8")
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


# This function returns a boolean value and can take either a single texture zero
# or a pair of texture zeros as arguments. If the argument fulfills the requirements
# for compatibility with data, the function returns True.
def compatible_with_data(texture_u, massless_u, texture_d=[], massless_d=0):

    dim_u = np.shape(texture_u)[0]
    rank_u = dim_u - massless_u
    if np.any(texture_d):
        dim_d = np.shape(texture_d)[0]

    compatible = False
    zero = 1e-10  # We consider that any value <1e-10  is zero

    if np.any(texture_d):
        for i in range(50):
            M_u = np.matlib.rand(dim_u, dim_u) + np.matlib.rand(dim_u, dim_u) * 1j
            M_d = np.matlib.rand(dim_d, dim_d) + np.matlib.rand(dim_d, dim_d) * 1j

            # Hadamard product
            M_u = np.multiply(M_u, texture_u)
            M_d = np.multiply(M_d, texture_d)

            D_u, D_d, sin_theta_12, sin_theta_13, sin_theta_23, delta, Jarlskog, V = ext.extract_obs(
                M_u, M_d)

            # Check the mixing angles of V_CKM
            if sin_theta_12 < zero or sin_theta_12 >= 1:
                continue
            if sin_theta_13 < zero or sin_theta_13 >= 1:
                continue
            if sin_theta_23 < zero or sin_theta_23 >= 1:
                continue

            # Check the CP-violating phase delta via the Jarlskog invariant
            if Jarlskog < zero:
                continue

            # Check if matrix V has at least one non-zero mixing between VLQ and SM quarks.
            # Thus the matrix V must have at least
            # 9 (V_CKM) + 1 (VLQ <-> SM quarks mixing) = 10 non-zero elements

            valid_CKM = True
            for i in range(3):
                for j in range(3):
                    if abs(V[i, j]) < zero:
                        valid_CKM = False

            if not valid_CKM:
                continue

            if dim_u * dim_d != 9:
                size = np.shape(V)
                mixing = 0
                for i in range(size[0]):
                    for j in range(size[1]):
                        if abs(V[i, j]) != 0:
                            mixing += 1

                if mixing != dim_u * dim_d:
                    continue

            compatible = True
            break
    else:
        for i in range(10):
            M_u = np.matlib.rand(dim_u, dim_u) + np.matlib.rand(dim_u, dim_u) * 1j

            # Hadamard product
            M_u = np.multiply(M_u, texture_u)

            # The number of non-zero singular values of a matrix A is equal to its rank.
            # Check the number of massive quarks
            if np.linalg.matrix_rank(M_u) != rank_u:
                return False

            # D_u = ext.extract_obs(M_u)
            # Check if VLQ are massive
            # massive = True
            # for i in range(3, dim_u):
            #    if(D_u[i] < zero):
            #        massive = False
            #        break
            # if massive:
            compatible = True
            break

    return compatible


# This function returns the index of a texture in the list
# obtained from the function get_texture_zeros
def get_index_of_texture(texture, dim, n_zeros):

    texture = np.reshape(texture, dim**2)
    index = 0
    i = 0
    while n_zeros > 0:
        if texture[i] == 0:
            n_zeros -= 1
        else:
            index += comb(dim**2 - (i + 1), n_zeros - 1)
        i += 1

    return index


# This function returns the index of a pair
def get_index_of_pair(pair, dim_u, n_zeros_u, dim_d, n_zeros_d):

    index_u = get_index_of_texture(pair[0], dim_u, n_zeros_u)
    index_d = get_index_of_texture(pair[1], dim_d, n_zeros_d)

    return index_u * comb(dim_d**2, n_zeros_d) + index_d


# This function returns a list with the indices of the pairs inside
# the list set_pairs
def get_list_indices(set_pairs, dim_u, n_zeros_u, dim_d, n_zeros_d):

    indices = []
    for pair in set_pairs:
        indices.append(get_index_of_pair(pair, dim_u, n_zeros_u, dim_d, n_zeros_d))

    return indices


def binary_search(index, list_indices):

    maximum = len(list_indices) - 1
    minimum = 0

    while minimum <= maximum:

        pos = (maximum + minimum) // 2

        if index > list_indices[pos]:
            minimum = pos + 1
        elif index < list_indices[pos]:
            maximum = pos - 1
        else:
            return pos

    return -1


# This function returns the texture that corresponds to a index
# from the list obtained from the function get_texture_zeros
def get_texture_from_index(index, dim, n_zeros):

    texture = np.ones(dim**2, dtype="uint8")
    i = 0
    while n_zeros > 0:
        position = comb(dim**2 - (i + 1), n_zeros - 1)
        if index < position:
            texture[i] = 0
            n_zeros -= 1
        else:
            index -= position
        i += 1

    return np.reshape(texture, (dim, dim))


# This function returns the texture that corresponds to a index
# from the list obtained from the function get_texture_zeros
def get_pair_from_index(index, dim_u, n_zeros_u, dim_d, n_zeros_d):

    texture_u = get_texture_from_index(index // comb(dim_d**2, n_zeros_d), dim_u, n_zeros_u)
    texture_d = get_texture_from_index(index % comb(dim_d**2, n_zeros_d), dim_d, n_zeros_d)

    return [texture_u, texture_d]


# This function computes the equivalence classes of pairs with
# n_u texture zeros for M_u and n_d texture zeros for M_d
def get_set_non_equivalent_pairs(pairs, n_u, n_d,
                                 n_zeros_u, n_zeros_d):

    n_total_zeros_u = ((n_u + 3)**2 - (n_u + 3)) / 2
    n_total_zeros_d = ((n_d + 3)**2 - (n_d + 3)) / 2

    set_K_L_u, set_K_R_u = get_weak_basis_permutations(n_u)
    set_K_R_d = get_permutation(n_d + 3)

    set_non_restrictive_texture_u = get_non_restrictive_texture_zeros(n_u + 3)
    set_non_restrictive_texture_d = get_non_restrictive_texture_zeros(n_d + 3)

    set_non_equivalent_pairs = []
    indices_pairs = get_list_indices(pairs, n_u + 3, n_zeros_u, n_d + 3, n_zeros_d)
    if os.path.exists("log"):
        os.remove("log")
    while(indices_pairs):
        print(f"Number of pairs: {len(indices_pairs)}")
        set_equivalent_pairs = []
        pair = get_pair_from_index(indices_pairs[0], n_u + 3, n_zeros_u, n_d + 3, n_zeros_d)
        # Construct the class of equivalent pairs of texture zeros.
        for K_R_u in set_K_R_u:  # ATENCAO: AQUI AS ROTACOES DE DIREITA SAO DO MESMO TIPO QUE AS DE ESQUERDA
            for K_R_d in set_K_R_d:
                for K_L_u in set_K_L_u:
                    A_L = get_block_matrix(K_L_u)
                    set_K_L_d = get_left_weak_basis_permutations(n_d, A_L)
                    for K_L_d in set_K_L_d:
                        # print(K_R_u)
                        # print(K_R_d)
                        # print(K_L_u)
                        # print(K_L_d)
                        set_equivalent_pairs.append([K_L_u @ pair[0] @ K_R_u,
                                                     K_L_d @ pair[1] @ K_R_d])

        # Remove repeated pairs from set_equivalent_pairs
        length = len(set_equivalent_pairs)
        with open("log", "a") as f:
            f.write(f"BEFORE: {length}\n")
            for pair in set_equivalent_pairs:
                f.write(f"{pair[0]}\n")
                f.write(f"{pair[1]}\n")
                f.write("\n")

        list_indices_set_equivalent_pairs = get_list_indices(set_equivalent_pairs,
                                                             n_u + 3, n_zeros_u,
                                                             n_d + 3, n_zeros_d)
        list_indices_set_equivalent_pairs = sorted(list_indices_set_equivalent_pairs)

        i = 0
        while i < length - 1:
            while (i < length - 1
                   and list_indices_set_equivalent_pairs[i]
                   == list_indices_set_equivalent_pairs[i + 1]):
                list_indices_set_equivalent_pairs.pop(i + 1)
                length -= 1
            i += 1

        with open("log", "a") as f:
            f.write(f"AFTER: {length}\n")
            for pair in set_equivalent_pairs:
                f.write(f"{pair[0]}\n")
                f.write(f"{pair[1]}\n")
                f.write("\n")

        # Check if set_equivalent_pairs contains texture zeros equivalent to
        # no zeros imposed
        valid_pair = True
        for index in list_indices_set_equivalent_pairs:
            pair = get_pair_from_index(index, n_u + 3, n_zeros_u, n_d + 3, n_zeros_d)
            if valid_pair:
                if n_zeros_u <= n_total_zeros_u / 2:
                    for non_restrictive_texture in set_non_restrictive_texture_u:
                        if np.array_equal(pair[0], non_restrictive_texture):
                            valid_pair = False
                            break

                if n_zeros_d <= n_total_zeros_d / 2:
                    for non_restrictive_texture in set_non_restrictive_texture_d:
                        if np.array_equal(pair[1], non_restrictive_texture):
                            valid_pair = False
                            break

            # Eliminate from the list indices_pairs the index
            # in list_indices_set_equivalent_pairs
            with open("log", "a") as f:
                f.write(f"ELIMINATED PAIR\n")
                f.write(f"{pair[0]}\n")
                f.write(f"{pair[1]}\n")
                f.write("\n")
            indices_pairs.pop(binary_search(index, indices_pairs))

        if(valid_pair):
            set_non_equivalent_pairs.append(set_equivalent_pairs[0])

    return set_non_equivalent_pairs


# This function stores the set of maximally restrictive pairs in a file
def print_maximmaly_restrictive_pairs(set_maximally_restrictive_pairs,
                                      option,
                                      filename):
    with open(filename, option) as f:
        f.write("LIST OF MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS (BEFORE MINUIT):\n\n")
        num_pairs = 0
        for restrictive_pairs_n_zeros_u in set_maximally_restrictive_pairs:
            n_zeros = restrictive_pairs_n_zeros_u[0].split("_")
            num_pairs_n_zeros_u = 0
            f.write(f"PAIRS WITH ({n_zeros[0]},{n_zeros[1]}) TEXTURE ZEROS FOR (M_u, M_d):\n\n")
            for pair in restrictive_pairs_n_zeros_u[1]:
                f.write(f"M_u:\n{pair[0]}\n")
                f.write(f"M_d:\n{pair[1]}\n")
                f.write("\n")
                num_pairs_n_zeros_u += 1
            f.write(
                f"THERE ARE {num_pairs_n_zeros_u} PAIRS WITH ({n_zeros[0]},{n_zeros[1]}) TEXTURE ZEROS\n\n")
            num_pairs += num_pairs_n_zeros_u

        f.write(
            f"\nTHERE ARE IN TOTAL {num_pairs} MAXIMALLY RESTRICTIVE PAIRS OF TEXTURE ZEROS (BEFORE MINUIT). \n")

    return


# This function reads the set of maximally restrictive pairs stored in a file
def read_maximmaly_restrictive_pairs(filename):

    set_maximally_restrictive_pairs = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != "":
            if line[0] == "P":
                string = line.split(" ")[2]
                maximally_restrictive_pairs = []
                while line[0] != "T":
                    line = f.readline()
                    texture_u = []
                    if line == "M_u:\n":
                        line = f.readline()
                        while line != "M_d:\n":
                            texture_u.append([int(s) for s in line if s.isdigit()])
                            line = f.readline()
                        texture_d = []
                        line = f.readline()
                        while line != "\n":
                            texture_d.append([int(s) for s in line if s.isdigit()])
                            line = f.readline()
                        maximally_restrictive_pairs.append([np.array(texture_u),
                                                            np.array(texture_d)])
                set_maximally_restrictive_pairs.append([string,
                                                        maximally_restrictive_pairs])
            line = f.readline()

    return set_maximally_restrictive_pairs


# This function computes the set of non-equivalent classes of maximally
# restrictive pairs of texture zeros.
def get_maximmaly_restrictive_pairs(n_u, n_d,
                                    massless_u, massless_d,
                                    option="", filename=""):

    dim_u = n_u + 3
    dim_d = n_d + 3
    n_zeros_u_max = dim_u**2 - (dim_u - massless_u)
    n_zeros_d_max = dim_d**2 - (dim_d - massless_d)
    n_zeros_u = n_zeros_u_max
    n_zeros_d = n_zeros_d_max

    set_maximally_restrictive_pairs = []

    while n_zeros_d <= n_zeros_d_max:
        maximally_restrictive_pair = False
        texture_n_zeros_u = get_texture_zeros(dim_u, n_zeros_u)

        # Remove textures of M_u incompatible with data
        i = 0
        length = len(texture_n_zeros_u)
        while i < length:
            if(not compatible_with_data(texture_n_zeros_u[i], massless_u)):
                texture_n_zeros_u.pop(i)
                i -= 1
                length -= 1
            i += 1
        pairs_n_zeros_u = []
        print(f"n_zeros_u = {n_zeros_u}")

        while not maximally_restrictive_pair and n_zeros_d > 0:
            texture_n_zeros_d = get_texture_zeros(dim_d, n_zeros_d)
            print(f"n_zeros_d = {n_zeros_d}")

            # Remove textures of M_d incompatible with data
            i = 0
            length = len(texture_n_zeros_d)
            while i < length:
                if(not compatible_with_data(texture_n_zeros_d[i], massless_d)):
                    texture_n_zeros_d.pop(i)
                    i -= 1
                    length -= 1
                i += 1

            # Store compatible pairs of textures
            for texture_u in texture_n_zeros_u:
                for texture_d in texture_n_zeros_d:
                    if compatible_with_data(texture_u, massless_u,
                                            texture_d, massless_d):
                        maximally_restrictive_pair = True
                        pairs_n_zeros_u.append([texture_u, texture_d])

            n_zeros_d -= 1

        if n_zeros_d > 0:
            n_zeros_d += 1
            print(f"{n_zeros_u}_{n_zeros_d}\n")

            # Compute and store non-equivalent classes of texture zeros
            # pairs with (n_zeros_u, n_zeros_d)
            set_maximally_restrictive_pairs.append([
                f"{n_zeros_u}_{n_zeros_d}",
                get_set_non_equivalent_pairs(pairs_n_zeros_u,
                                             n_u, n_d,
                                             n_zeros_u, n_zeros_d)
            ])
            # n_zeros_d += 1
            # If (n_zeros_u, n_zeros_d) contains maximally_restrictive_pairs, the next
            # set of maximally_restrictive_pairs is given by (n_zeros_u - 1, n_zeros_d + 1)

        if n_zeros_d == n_zeros_d_max:
            break

        n_zeros_d = n_zeros_d_max
        n_zeros_u -= 1

    if option == "w":
        if filename == "":
            filename = f"output/{n_u}_up_{n_d}_down_MRT_before_MINUIT.txt"
            if(massless_u != 0):
                filename = filename[:-4] + f"_{massless_u}_massless_up" + filename[-4:]
            if(massless_d != 0):
                filename = filename[:-4] + f"_{massless_d}_massless_down" + filename[-4:]
        print_maximmaly_restrictive_pairs(set_maximally_restrictive_pairs,
                                          option,
                                          filename)

    return set_maximally_restrictive_pairs


def main():

    args = sys.argv[1:]
    start = time.time()
    if args[0].isnumeric():
        get_maximmaly_restrictive_pairs(int(args[0]), int(args[1]),
                                        int(args[2]), int(args[3]),
                                        "w")
    else:
        get_maximmaly_restrictive_pairs(int(args[1]), int(args[2]),
                                        int(args[3]), int(args[4]),
                                        "w", args[0])
    end = time.time()
    print(f"TOTAL TIME = {int((float(end) - float(start)) / 60)} min ", end="")
    print(f"{(int(float(end) - float(start)) % 60)} sec \n")


if __name__ == "__main__":
    main()
