import numpy as np
import math

##################################################################################################
#
#   PHYSICAL PARAMETERS
#
##################################################################################################

# Experimental values taken from PDG revision of 2022:
# https://pdg.lbl.gov/2022/reviews/rpp2022-rev-ckm-matrix.pdf (January 2023)

MeV = 1
GeV = 1e3
TeV = 1e6

# QUARK MASSES
MASS_U = 2.16 * MeV
UPPER_SIGMA_MASS_U = 0.49 * MeV
LOWER_SIGMA_MASS_U = 0.26 * MeV

MASS_D = 4.67 * MeV
UPPER_SIGMA_MASS_D = 0.48 * MeV
LOWER_SIGMA_MASS_D = 0.17 * MeV

MASS_S = 93.4 * MeV
UPPER_SIGMA_MASS_S = 8.6 * MeV
LOWER_SIGMA_MASS_S = 3.4 * MeV

MASS_C = 1.27 * GeV
UPPER_SIGMA_MASS_C = 0.02 * GeV
LOWER_SIGMA_MASS_C = 0.02 * GeV

MASS_B = 4.18 * GeV
UPPER_SIGMA_MASS_B = 0.03 * GeV
LOWER_SIGMA_MASS_B = 0.02 * GeV

MASS_T = 172.69 * GeV
UPPER_SIGMA_MASS_T = 0.30 * GeV
LOWER_SIGMA_MASS_T = 0.30 * GeV

# Absolute values of CKM matrix
V_CKM = np.array([
    [0.97373, 0.2243, 0.00382],
    [0.22100, 0.9750, 0.04080],
    [0.00860, 0.0415, 1.01400]
])

SIGMA_V_CKM = np.array([
    [0.00031, 0.0008, 0.00020],
    [0.00400, 0.0060, 0.00140],
    [0.00020, 0.0009, 0.02900]
])

FIRST_ROW_SIGMA = 0.0007
GAMMA = 65.9
UPPER_SIGMA_GAMMA = 3.3
LOWER_SIGMA_GAMMA = 3.5

# DEFINE ARRAYS USED FOR COMPUTATIONS
MASS_UP = np.array([MASS_U, MASS_C, MASS_T])
UPPER_SIGMA_UP = np.array(
    [UPPER_SIGMA_MASS_U, UPPER_SIGMA_MASS_C, UPPER_SIGMA_MASS_T])
LOWER_SIGMA_UP = np.array(
    [LOWER_SIGMA_MASS_U, LOWER_SIGMA_MASS_C, LOWER_SIGMA_MASS_T])

MASS_DOWN = np.array([MASS_D, MASS_S, MASS_B])
UPPER_SIGMA_DOWN = np.array(
    [UPPER_SIGMA_MASS_D, UPPER_SIGMA_MASS_S, UPPER_SIGMA_MASS_B])
LOWER_SIGMA_DOWN = np.array(
    [LOWER_SIGMA_MASS_D, LOWER_SIGMA_MASS_S, LOWER_SIGMA_MASS_B])

RATIO_UP = np.empty([2])
UPPER_SIGMA_RATIO_UP = np.empty([2])
LOWER_SIGMA_RATIO_UP = np.empty([2])
for i in range(2):
    RATIO_UP[i] = MASS_UP[i] / MASS_T
    UPPER_SIGMA_RATIO_UP[i] = RATIO_UP[i] * math.sqrt(
        (UPPER_SIGMA_UP[i] / MASS_UP[i]) ** 2 + (LOWER_SIGMA_MASS_T / MASS_T) ** 2)
    LOWER_SIGMA_RATIO_UP[i] = RATIO_UP[i] * math.sqrt(
        (LOWER_SIGMA_UP[i] / MASS_UP[i]) ** 2 + (UPPER_SIGMA_MASS_T / MASS_T) ** 2)


RATIO_DOWN = np.empty([2])
UPPER_SIGMA_RATIO_DOWN = np.empty([2])
LOWER_SIGMA_RATIO_DOWN = np.empty([2])
for i in range(2):
    RATIO_DOWN[i] = MASS_DOWN[i] / MASS_B
    UPPER_SIGMA_RATIO_DOWN[i] = RATIO_DOWN[i] * math.sqrt(
        (UPPER_SIGMA_DOWN[i] / MASS_DOWN[i]) ** 2 + (LOWER_SIGMA_MASS_B / MASS_B) ** 2)
    LOWER_SIGMA_RATIO_DOWN[i] = RATIO_DOWN[i] * math.sqrt(
        (LOWER_SIGMA_DOWN[i] / MASS_DOWN[i]) ** 2 + (UPPER_SIGMA_MASS_B / MASS_B) ** 2)
