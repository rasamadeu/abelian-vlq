from minuit_minimization_mp_non_unitary import read_maximmaly_restrictive_pairs
import numpy as np
import numpy.matlib
import sys
import pdb


def main():

    args = sys.argv[1:]
    for i, num in enumerate(args):
        args[i] = float(num)
    # program input = charges of (phi2, VLQ_L, Singlet scalar, q_L_2, q_L_3, u_R_1, u_R_2, u_R_3, VLQ_R, d_R_1, d_R_2, d_R_3) 
    scalars = [0, args[0], args[2]]
    left_VLQ = args[1]
    left_SM = [0, args[3], args[4]]
    right_u = [args[5], args[6], args[7], args[8]]
    right_d = [args[9], args[10], args[11]]

    up = np.zeros((4,4))
    down = np.zeros((3,3))
    for i in range(3):
        for j in range(4):
            up[i, j] = -left_SM[i] + right_u[j]

    for i in range(3):
        for j in range(3):
            down[i, j] = -left_SM[i] + right_d[j]

    for j in range(4):
        up[3, j] = -left_VLQ + right_u[j]

    print(f"up = {scalars[0]}/{scalars[1]}\n")
    print(f"down = {- scalars[0]}/{- scalars[1]}\n")
    print(f"sigma = {- scalars[2]}\n")
    print(up)
    print("\n")
    print(down)
    return

if __name__ == "__main__":
    main()
