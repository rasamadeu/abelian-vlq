import numpy as np
import sys
import time
import os
import pdb
import extract_obs as ext
from math import sqrt

GeV = 1
VEV = 246.22 * GeV
MASS_T = 172.69 * GeV
MASS_B = 4.18 * GeV


# This function computes the Higgs FCNC matrices N_u and N_d
def compute_Higgs_FCNC(M_u, M_d, decomposition_u, decomposition_d, tanbeta, sign):

    v1 = VEV / sqrt(1 + tanbeta ** 2)
    v2 = VEV / sqrt(1 + 1 / tanbeta ** 2)

    D_u, U_L_u = ext.compute_singular_values(M_u, "left")
    D_u, U_R_u = ext.compute_singular_values(M_u, "right")
    D_d, U_L_d = ext.compute_singular_values(M_d, "left")
    D_d, U_R_d = ext.compute_singular_values(M_d, "right")

    # Compute the matrix V = A_u_L^{dagger} A_d_L
    A_u_L = np.empty([3, np.shape(M_u)[0]], dtype=complex)
    A_d_L = np.empty([3, np.shape(M_d)[0]], dtype=complex)
    for i in range(3):
        A_u_L[i] = U_L_u[i]
        A_d_L[i] = U_L_d[i]

    scale_up = MASS_T / D_u[2]
    scale_down = MASS_B / D_d[2]
    M_u = scale_up * M_u
    M_d = scale_down * M_d

    Y_u_1 = np.multiply(M_u[:3, :], decomposition_u[0]) / v1
    Y_u_2 = np.multiply(M_u[:3, :], decomposition_u[1]) / v2
    Y_d_1 = np.multiply(M_d[:3, :], decomposition_d[0]) / v1
    Y_d_2 = np.multiply(M_d[:3, :], decomposition_d[1]) / v2

    # sign corresponds to +(-) for theta = 0(pi)
    N_u = 1 / sqrt(2) * A_u_L.conj().T @ (v2 * Y_u_1 - sign * v1 * Y_u_2) @ U_R_u
    N_d = 1 / sqrt(2) * A_d_L.conj().T @ (v2 * Y_d_1 + sign * v1 * Y_d_2) @ U_R_d

    return N_u, N_d


# This function converts a list of strings to a texture zero
def convert_text_to_texture(list_strings):

    texture = []
    for line in list_strings:
        texture.append([int(s) for s in line if s.isdigit()])

    return np.array(texture)


# This function converts a list of strings to a texture zero
def convert_text_to_mass_matrix(list_strings):

    print(list_strings)
    pdb.set_trace()
    texture = []
    for k, line in enumerate(list_strings):

        line = line.split(" ")
        num_string = []
        for elem in line:
            elem = elem.replace("\n", "")
            elem = elem.replace("[", "")
            elem = elem.replace("]", "")
            if elem:
                num_string.append(elem)

        length = len(num_string)
        num = []
        i = 0
        while i < length:
            if num_string[i][-1] != "j":
                num.append(complex(num_string[i] + num_string[i + 1]))
                i += 2
            else:
                num.append(complex(num_string[i]))
                i += 1
        texture.append(num)

    return np.array(texture)


# This function reads the set of maximally restrictive pairs stored in a file
def read_set_pair_textures(filename, n_textures):

    set_pair_textures = []
    with open(filename, "r") as f:
        file_data = f.readlines()
        for i in range(n_textures):
            k = i * 3550
            texture_pair = [convert_text_to_texture(file_data[int(k + 4):int(k + 8)]),
                            convert_text_to_texture(file_data[int(k + 9):int(k + 12)])]
            data_points = []
            for j in range(85): # 101 / 28
                M_u = convert_text_to_mass_matrix(file_data[int(k + j * 35 + 16):int(k + j * 35 + 20)])
                M_d = convert_text_to_mass_matrix(file_data[int(k + j * 35 + 21):int(k + j * 35 + 24)])
                m_VLQ = file_data[int(k + j * 35 + 30)]
                m_VLQ = m_VLQ[:int(m_VLQ.find("/") - 1)]
                m_VLQ = m_VLQ.replace(" ", "_")
                data_points.append([m_VLQ, M_u, M_d])
            set_pair_textures.append([texture_pair, data_points])

    pdb.set_trace()
    return set_pair_textures


def print_texture_pair(filename, texture_pair):

    with open(filename, "w") as f:
        f.write("TEXTURE PAIR:\n\n")
        f.write(f"M_u:\n{texture_pair[0]}\n")
        f.write(f"M_d:\n{texture_pair[1]}\n")

    return


def print_decomposition(decomposition_u, decomposition_d, texture, filename):

    for point in texture:
        with open(filename + f"/theta_0/{point[0]}.txt", "w") as f:
            for tanbeta in range(1, 101):
                N_u, N_d = compute_Higgs_FCNC(point[1], point[2], decomposition_u, decomposition_d, tanbeta, 1)
                f.write(f"tanbeta = {tanbeta}\n")
                f.write(f"N_u:\n{N_u}\n")
                f.write(f"N_d:\n{N_d}\n\n")
        print("theta_0")
        print(filename + f"/{point[0]}.txt")

    for point in texture:
        with open(filename + f"/theta_pi/{point[0]}.txt", "w") as f:
            for tanbeta in range(1, 101):
                N_u, N_d = compute_Higgs_FCNC(point[1], point[2], decomposition_u, decomposition_d, tanbeta, -1)
                f.write(f"tanbeta = {tanbeta}\n")
                f.write(f"N_u:\n{N_u}\n")
                f.write(f"N_d:\n{N_d}\n\n")
        print("theta_pi")
        print(filename + f"/{point[0]}.txt")

    return


def main():

    args = sys.argv[1:]
    start = time.time()
    set_pair_textures = read_set_pair_textures(args[0], int(args[1]))
    decomposition_u = [
                        np.array([[0, 0, 1, 0],
                                  [0, 0, 0, 1],
                                  [1, 0, 0, 0]]),
                        np.array([[0, 0, 0, 1],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0]])
                        ]
    decomposition_d = [[
                        np.array([[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]]),
                        np.array([[0, 0, 1],
                                  [0, 0, 0],
                                  [1, 0, 0]])
                        ],
                        [
                        np.array([[0, 0, 1],
                                  [0, 1, 0],
                                  [0, 0, 0]]),
                        np.array([[0, 0, 0],
                                  [0, 0, 1],
                                  [1, 0, 0]])
                        ]]

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

    if set_pair_textures == []:
        print("NO PAIRS WERE FOUND!")
    else:
        for i, texture in enumerate(set_pair_textures):
            path = args[0][:args[0].rindex("/")] + f"/case_{i}"
            print_texture_pair(path + f"/pair_textures.txt", texture[0])
            print_decomposition(decomposition_u, [decomposition_d[i][0], decomposition_d[i][1]], texture[1], path + f"/normal_decomp")
            print_decomposition(decomposition_u, [decomposition_d[i][1], decomposition_d[i][0]], texture[1], path + f"/inverse_decomp")

    end = time.time()
    print(f"TOTAL TIME = {int((float(end) - float(start)) / 60)} min ", end="")
    print(f"{(int(float(end) - float(start)) % 60)} sec \n")

    return


if __name__ == "__main__":
    main()
