# The aim of this module is to write a .txt file containing all parameters needed to
# compute the Wilson coefficients for 2HDM + 1 up isosinglet VLQ + abelian symmetries
# model using SPheno.
# For a given set of:
# - mass matrices (M_u, M_d),
# - Yukawa texture zeros (Y_u_1_tex, Y_u_2_tex, Y_d_1_tex, Y_d_2_tex),
# we compute N_POINTS data points compatible with physical constraints (BFB and perturbativity),
# each one corresponding to:
# - a set of scalar masses (m_R, m_I, m_h, m_Hcharged)
# - a tan(beta) value.
# We assume the alignment limit where beta = alpha + pi/2, where beta is the angle
# of the rotation to the Higgs basis and alpha the rotation angle of the CP-even mass
# matrix.

import numpy as np
import math
import pdb
import time

##################################################################################################
#
#   INPUT PARAMETERS
#
##################################################################################################

# Physical quantities are written in units of GeV
VEV = 246.22  # Vacuum expectation value
MH = 125.09   # SM higgs mass

# Information about the set of mass matrices and Yukawa decompositions
# m_VLQ = 1.4 TeV
# Decoupled down type quark = down

# Mass matrices

M_U = np.array([[ 0.00000000e+00+0.j,  0.00000000e+00+0.j,  7.48169502e-03+0.j,  3.89889413e-02+0.j],
                [ 0.00000000e+00+0.j, -1.31416994e-02+0.j,  0.00000000e+00+0.j,  3.47422838e-04+0.j],
                [ 2.21639761e-06+0.j,  0.00000000e+00+0.j,  2.89724026e-04+0.j,  0.00000000e+00+0.j],
                [ 0.00000000e+00+0.j,  3.21595130e-01+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j]])

M_D = np.array([[   0.         +0.j,            0.         +0.j,         4175.7358899  +0.j        ],
                [   0.         +0.j,            4.67       +0.j,            0.         +0.j        ],
                [  93.47294976 +0.j,            0.         +0.j,          162.13587918-71.24972603j]])

# NOTE: The minuit code conducts the minimization with respect to the quark masses ratios.
# In order to compute the quark masses correctly, we need to rescale the mass matrices by a
# constant k. TENHO DE MUDAR ISTO NO FICHEIRO DE MINUIT
M_U = M_U * 4349677.423418475 * 1e-3
M_D = M_D * 1.0001216287000756 * 1e-3

# Yukawa texture zeros
Y_U_1_TEX = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [1, 0, 0, 0]])

Y_U_2_TEX = np.array([[0, 0, 0, 1],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0]])

Y_D_1_TEX = np.array([[0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])

Y_D_2_TEX = np.array([[0, 0, 1],
                      [0, 0, 0],
                      [1, 0, 0]])

# Scalar masses ranges
LOWER_BOUND_M_R = 1e2
UPPER_BOUND_M_R = 1e4

LOWER_BOUND_M_I = 1e2
UPPER_BOUND_M_I = 1e4

LOWER_BOUND_M_HCHARGED = 80
UPPER_BOUND_M_HCHARGED = 1e3

LOWER_BOUND_TANBETA = 1e-2
UPPER_BOUND_TANBETA = 1e2

# Perturbativity limit condition
PERTUBATIVITY_LIMIT_YUKAWA = math.sqrt(4 * math.pi)
PERTUBATIVITY_LIMIT_PARAMETERS = math.sqrt(4 * math.pi)

# PM_THETA = (1/-1) corresponds to a phase difference between v1 and v2 of (0/pi)
PM_THETA = 1

# There are 2 possible solutions for L1, L2 and L3.
# The variable PM_SOLUTIONS = (1/-1) controls which solution we are considering.
PM_SOLUTIONS = 1

# Number of points
N_POINTS = 3

# Filename with program output
FILENAME_SPHENO_INPUT = "SPheno_Input_test.txt"
# Filename with scalar and VLQ masses program output
FILENAME_SPHENO_MASSES = "SPheno_Input_masses_test.txt"

##################################################################################################
#
#   AUXILIARY FUNCTIONS
#
##################################################################################################


# Returns random value in interval [lower_bound, upper_bound]
def generate_random_in(lower_bound, upper_bound, rng_generator):

    return lower_bound + rng_generator.random() * (upper_bound - lower_bound)


# Returns the singular values and left or right rotation of SVD
def compute_singular_values(M, rotation_option):

    if rotation_option == "left":
        # Compute singular values of M and U_L
        D, U = np.linalg.eig(np.nan_to_num(M @ M.conj().T))

    if rotation_option == "right":
        # Compute singular values of M and U_R
        D, U = np.linalg.eig(np.nan_to_num(M.conj().T @ M))

    # Sort the eigenvalues from lowest to highest
    D = np.sqrt(D)
    index = D.argsort()
    D = D[index]
    U = U[:, index]

    return D, U

##################################################################################################
#
#   RECONSTRUCTION FUNCTIONS
#
##################################################################################################


# This function reconstructs the scalar potential parameters for a given set of
# squared scalar masses and beta, defined as tan(beta)= v2/v1.
# The input variable pm_theta is = 1/-1 for theta = 0/Pi, where theta is the
# difference between the complex phases of the VEVS of the Higgs doublets.
# The input variable pm_solutions is = 1/-1 to choose between the 2 solutions for L1,
# L2 and L3 that we obtain from a quadratic equation.
def reconstruct_paramaters_from_masses(mh, mR, mHcharged, mI, v1, v2, pm_theta, pm_solutions):

    # Square the masses
    mh = mh ** 2
    mR = mR ** 2
    mHcharged = mHcharged ** 2
    mI = mI ** 2

    # Definition of trigonometric functions of beta
    cb = v1 / VEV
    sb = v2 / VEV
    cbs = cb ** 2
    sbs = sb ** 2
    c2b = 2 * cbs - 1
    s2b = 2 * cb * sb
    tb = v2 / v1
    ctb = v1 / v2

    # NOTE: The variables mh, mR, mHcharged and mI are squared
    L1 = 1 / (2 * VEV ** 2 * cbs) * ((c2b - 1) * mI + pm_theta * pm_solutions * c2b * (mR - mh) + mR + mh)
    L2 = 1 / (2 * VEV ** 2 * sbs) * (-(c2b + 1) * mI - pm_theta * pm_solutions * c2b * (mR - mh) + mR + mh)
    L3 = 1 / VEV ** 2 * (2 * mHcharged - mI + pm_solutions * (mR - mh))
    L4 = 2 / VEV ** 2 * (mI - mHcharged)
    mu12s = -pm_theta * s2b * mI / 2
    mu11s = -1 / 2 * VEV ** 2 * (L1 * cbs + (L3 + L4) * sbs) - pm_theta * mu12s * tb
    mu22s = -1 / 2 * VEV ** 2 * (L2 * sbs + (L3 + L4) * cbs) - pm_theta * mu12s * ctb

    return mu11s, mu22s, L1, L2, L3, L4, mu12s


# This function reconstructs the squared scalar masses for a given set of
# scalar potential parameters and beta, defined as tan(beta)= v2/v1.
# The input variable pm_theta is = 1/-1 for theta = 0/Pi, where theta is the
# difference between the complex phases of the VEVS of the Higgs doublets.
def reconstruct_masses_from_paramaters(L1, L2, L3, L4, mu12s, v1, v2, pm_theta):

    cb = v1 / VEV
    sb = v2 / VEV
    cbs = cb ** 2
    sbs = sb ** 2
    s2b = 2 * cb * sb
    tb = v2 / v1
    ctb = v1 / v2

    # NOTE: The variables mh, mR, mHcharged and mI are squared
    mI = -pm_theta * 2 * mu12s / s2b
    mHcharged = - VEV ** 2 * L4 / 2 - pm_theta * 2 * mu12s / s2b
    M11 = VEV ** 2 * cbs * L1 - pm_theta * tb * mu12s
    M22 = VEV ** 2 * sbs * L2 - pm_theta * ctb * mu12s
    M12 = pm_theta * VEV ** 2 * cb * sb * (L3 + L4) + mu12s
    mR = 1 / 2 * (M11 + M22 + math.sqrt((M11 - M22) ** 2 + 4 * M12 ** 2))
    mh = 1 / 2 * (M11 + M22 - math.sqrt((M11 - M22) ** 2 + 4 * M12 ** 2))

    return math.sqrt(mh), math.sqrt(mR), math.sqrt(mHcharged), math.sqrt(mI)


# This function computes the Yukawa matrices for a given set of mass matrices,
# Yukawa texture zeros and VEVS v1 and v2. Note that the 4th row of M_u does not originate
# from a Yukawa coupling term.
def reconstruct_Yukawa_matrices(M_u, M_d, Y_u_1_tex, Y_u_2_tex, Y_d_1_tex, Y_d_2_tex, v1, v2):

    Y_u_1 = np.multiply(M_u[:3, :], Y_u_1_tex) / v1 * math.sqrt(2)
    Y_u_2 = np.multiply(M_u[:3, :], Y_u_2_tex) / v2 * math.sqrt(2)
    Y_d_1 = np.multiply(M_d[:3, :], Y_d_1_tex) / v1 * math.sqrt(2)
    Y_d_2 = np.multiply(M_d[:3, :], Y_d_2_tex) / v2 * math.sqrt(2)

    return Y_u_1, Y_u_2, Y_d_1, Y_d_2


##################################################################################################
#
#   PHYSICAL CONSTRAINTS CHECKS (BFB AND PERTURBATIVITY)
#
##################################################################################################


# This function checks if a given set of scalar potential parameters satisfy the
# boundedness from below (BFB) requirement for the scalar potential
def check_BFB_potential(L1, L2, L3, L4):

    if not L1 > 0:
        return False

    if not L2 > 0:
        return False

    if not L3 + math.sqrt(L1 * L2) > 0:
        return False

    if not L3 + L4 + math.sqrt(L1 * L2) > 0:
        return False

    return True


# This function checks if a given set of scalar potential parameters satisfy the
# perturbative unitarity requirement for scalar-scalar scattering
def check_perturbative_unitarity(L1, L2, L3, L4):

    a_plus = 3 / 2 * (L1 + L2) + math.sqrt(9 / 4 * (L1 - L2) ** 2 + (2 * L3 - L4) ** 2)
    a_minus = 3 / 2 * (L1 + L2) - math.sqrt(9 / 4 * (L1 - L2) ** 2 + (2 * L3 - L4) ** 2)
    b_plus = 1 / 2 * (L1 + L2) + 1 / 2 * math.sqrt((L1 - L2) ** 2 + 4 * L4 ** 2)
    b_minus = 1 / 2 * (L1 + L2) - 1 / 2 * math.sqrt((L1 - L2) ** 2 + 4 * L4 ** 2)
    c_plus = L1
    c_minus = L2
    e1 = L3 + 2 * L4
    e2 = L3
    f1 = L3 + L4
    p1 = L3 - L4

    if not abs(a_plus) < 8 * math.pi:
        return False

    if not abs(a_minus) < 8 * math.pi:
        return False

    if not abs(b_plus) < 8 * math.pi:
        return False

    if not abs(b_minus) < 8 * math.pi:
        return False

    if not abs(c_plus) < 8 * math.pi:
        return False

    if not abs(c_minus) < 8 * math.pi:
        return False

    if not abs(e1) < 8 * math.pi:
        return False

    if not abs(e2) < 8 * math.pi:
        return False

    if not abs(f1) < 8 * math.pi:
        return False

    if not abs(p1) < 8 * math.pi:
        return False

    return True


# This function checks if a given set of scalar potential parameters satisfy the
# perturbativity requirement
def check_perturbativity_parameters(L1, L2, L3, L4, perturbativity_limit):

    if not abs(L1) < perturbativity_limit:
        return False

    if not abs(L2) < perturbativity_limit:
        return False

    if not abs(L3) < perturbativity_limit:
        return False

    if not abs(L4) < perturbativity_limit:
        return False

    return True


# This function checks if a given set of scalar potential parameters satisfy the
# perturbativity requirement
def check_perturbativity_Yukawa(Y_u_1, Y_u_2, Y_d_1, Y_d_2, perturbativity_limit):

    for row in Y_u_1:
        for elem in row:
            if not abs(elem) < perturbativity_limit:
                return False

    for row in Y_u_2:
        for elem in row:
            if not abs(elem) < perturbativity_limit:
                return False

    for row in Y_d_1:
        for elem in row:
            if not abs(elem) < perturbativity_limit:
                return False

    for row in Y_d_2:
        for elem in row:
            if not abs(elem) < perturbativity_limit:
                return False

    return True


##################################################################################################
#
#   I/O FUNCTIONS
#
##################################################################################################

def print_SPheno_input(datapoints, filename):

    with open(filename, "w") as f:

        # File header
        f.write(f"Lambda1 Lambda2 Lambda3 Lambda4 MU12 TanBeta ")
        for k in range(1, 3):
            for i in range(1, 5):
                for j in range(1, 5):
                    if j < 4:
                        f.write(f"REYu{k}{i}{j} ")
                        f.write(f"IMYu{k}{i}{j} ")
                    else:
                        f.write(f"REYuT{k}{i} ")
                        f.write(f"IMYuT{k}{i} ")
        for k in range(1, 3):
            for i in range(1, 4):
                for j in range(1, 4):
                    f.write(f"REYd{k}{i}{j} ")
                    f.write(f"REYd{k}{i}{j} ")

        for i in range(1, 4):
            f.write(f"REMTu{i} ")
            f.write(f"IMMTu{i} ")
        f.write(f"REMT0 ")
        f.write(f"IMMT0 ")
        f.write("\n")

        f.write("####################################################################################\n")
        for point in datapoints:
            for i in range(6):
                f.write(f"{point[i]} ")
            for i in range(6, 10):
                for row in point[i]:
                    for elem in row:
                        f.write(f"{elem.real} ")
                        f.write(f"{elem.imag} ")
            for elem in point[10]:
                f.write(f"{elem.real} ")
                f.write(f"{elem.imag} ")
            f.write("\n")

    return


def print_SPheno_input_masses(datapoints, filename):

    D_u, U = compute_singular_values(M_U, "left")
    with open(filename, "w") as f:

        # File header
        f.write(f"mh mR mI mHcharged mVLQ \n")

        f.write("####################################################################################\n")

        for point in datapoints:
            f.write(f"{MH} ")
            for i in range(11, 14):
                f.write(f"{point[i]} ")
            f.write(f"{abs(D_u[3])}")
            f.write("\n")

    return


##################################################################################################


def main():

    n = 0
    rng = np.random.default_rng()
    datapoints = []
    start = time.time()

    while n < N_POINTS:

        # Generate random values for scalar masses
        mR = generate_random_in(LOWER_BOUND_M_R, UPPER_BOUND_M_R, rng)
        mI = generate_random_in(LOWER_BOUND_M_I, UPPER_BOUND_M_I, rng)
        mHcharged = generate_random_in(LOWER_BOUND_M_HCHARGED, UPPER_BOUND_M_HCHARGED, rng)
        tanbeta = generate_random_in(LOWER_BOUND_TANBETA, UPPER_BOUND_TANBETA, rng)

        v1 = VEV / math.sqrt(1 + tanbeta ** 2)
        v2 = VEV / math.sqrt(1 + 1 / tanbeta ** 2)

        # Reconstruction of Higgs potential
        mu11s, mu22s, L1, L2, L3, L4, mu12s = reconstruct_paramaters_from_masses(MH, mR, mHcharged, mI, v1, v2, PM_THETA, PM_SOLUTIONS)
        # Reconstruction of Yukawa matrices
        Y_u_1, Y_u_2, Y_d_1, Y_d_2 = reconstruct_Yukawa_matrices(M_U, M_D, Y_U_1_TEX, Y_U_2_TEX, Y_D_1_TEX, Y_D_2_TEX, v1, v2)

        # Check BFB and perturbativity limits
        if not check_BFB_potential(L1, L2, L3, L4):
            continue
        if not check_perturbative_unitarity(L1, L2, L3, L4):
            continue
        if not check_perturbativity_parameters(L1, L2, L3, L4, PERTUBATIVITY_LIMIT_PARAMETERS):
            continue
        if not check_perturbativity_Yukawa(Y_u_1, Y_u_2, Y_d_1, Y_d_2, PERTUBATIVITY_LIMIT_YUKAWA):
            continue

        datapoints.append([L1, L2, L3, L4, mu12s, tanbeta, Y_u_1, Y_u_2, Y_d_1, Y_d_2, M_U[3, :], mR, mI, mHcharged])
        n += 1
        print(f"N_points = {n}")

    print_SPheno_input(datapoints, FILENAME_SPHENO_INPUT)
    print_SPheno_input_masses(datapoints, FILENAME_SPHENO_MASSES)

    end = time.time()
    print(f"TOTAL TIME = {int((float(end) - float(start)) / 60)} min ", end="")
    print(f"{(int(float(end) - float(start)) % 60)} sec \n")

    return


if __name__ == "__main__":
    main()
