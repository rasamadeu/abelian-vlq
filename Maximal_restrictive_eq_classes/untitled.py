import numpy as np
import numpy.matlib
import math
from cmath import exp

DIRAC_DELTA = 1.144
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

V = np.array([[ 9.73726043e-01+0.00000000e+00j,  1.83821567e-01-1.28331086e-01j, -2.63436916e-03-2.76835342e-03j],
              [-1.83475530e-01-1.28269733e-01j,  9.73677742e-01-9.95147257e-06j,  3.53241128e-02-2.24444828e-02j],
              [ 6.17285512e-03+5.95847678e-03j, -3.42963703e-02-2.27195637e-02j,  9.99116534e-01-2.34589496e-05j],
              [-3.32763826e-02+2.36701139e-02j,  5.40157979e-08-5.72324775e-08j, -3.40813102e-07+1.82333044e-06j]])
s12 = math.sin(math.atan(abs(V[0, 1]) / abs(V[0, 0])))
s13 = abs(V[0, 2])
s23 = math.sin(math.atan(abs(V[1, 2]) / abs(V[2, 2])))
with open("observables_non_unitarity", "w") as f:
    f.write(f"s12 = {s12}\n")
    f.write(f"s13 = {s13}\n")
    f.write(f"s23 = {s23}\n")
    f.write("\n")

    c12 = math.sqrt(1 - s12 * s12)
    c13 = math.sqrt(1 - s13 * s13)
    c23 = math.sqrt(1 - s23 * s23)
    
    f.write("abs(V):\n")
    f.write(f"{abs(V)}\n")
    f.write("\n")
    phases = np.angle(V)
    f.write("phases(V):\n")
    f.write(f"{phases}\n")
    CKM = np.empty([3, 3], dtype=complex)
    for i in range(3):
        CKM[i] = V[i]
    f.write("\n")
    f.write("F_u = V @ V_dag:\n")
    f.write(f"{abs(V @ np.matrix(V).conj().T)}\n")
    f.write("F_u = V @ V_dag:\n")
    f.write(f"{(V @ np.matrix(V).conj().T)}\n")
    f.write(f"{(V @ np.matrix(V).conj().T) - (V @ np.matrix(V).conj().T) @ (V @ np.matrix(V).conj().T) @ (V @ np.matrix(V).conj().T)}\n")
    f.write("\n")
    f.write("F_d = V_dag @ V:\n")
    f.write(f"{abs(np.matrix(V).conj().T @ V)}\n")
    f.write("\n")
    f.write("CKM:\n")
    f.write(f"{CKM}\n")
    f.write("\n")
    f.write("CKM @ CKM_dag:\n")
    f.write(f"{abs(CKM @ np.matrix(CKM).conj().T)}\n")
    f.write("\n")
    f.write("CKM_dag @ CKM:\n")
    f.write(f"{abs(np.matrix(CKM).conj().T @ CKM)}\n")
    f.write("\n")
    R = np.array([[exp(-(phases[0, 0] * 1j)), 0, 0],
                  [0, exp(-(phases[0, 1] * 1j)), 0],
                  [0, 0, exp((-phases[0, 2] - DIRAC_DELTA) * 1j)]])
    phases = np.angle(V @ R)
    L = np.array([[1, 0, 0, 0],
                  [0, exp(-(phases[1, 2] * 1j)), 0, 0],
                  [0, 0, exp((-phases[2, 2] * 1j)), 0],
                  [0, 0, 0, 1]])
    phases = np.angle(L @ V @ R)
    V = L @ V @ R
    f.write("V after rephasing:\n")
    f.write(f"{phases}\n")
    f.write("\n")
    f.write("phase(V[1,0]):\n")
    f.write(f"{np.angle(-s12 * c23 - c12 * s23 * s13 * exp(DIRAC_DELTA * 1j))}\n")
    f.write("\n")
    f.write("phase(V[1,1]):\n")
    f.write(f"{np.angle(c12 * c23 - s12 * s23 * s13 * exp(DIRAC_DELTA * 1j))}\n")
    f.write("\n")
    f.write("phase(V[2,0]):\n")
    f.write(f"{np.angle(s12 * s23 - c12 * c23 * s13 * exp(DIRAC_DELTA * 1j))}\n")
    f.write("\n")
    f.write("phase(V[2,1]):\n")
    f.write(f"{np.angle(-c12 * s23 - s12 * c23 * s13 * exp(DIRAC_DELTA * 1j))}\n")
    f.write("\n")
    f.write("gamma:\n")
    f.write(f"{np.angle(-V[0, 0] * V[1, 2] * np.conj(V[0, 2]) * np.conj(V[1, 0]))/(2*math.pi)*360}\n")
    f.write("\n")
    f.write("alpha:\n")
    f.write(f"{np.angle(-V[2, 0] * V[0, 2] * np.conj(V[2, 2]) * np.conj(V[0, 0]))/(2*math.pi)*360}\n")
    f.write("\n")
    f.write("beta:\n")
    f.write(f"{np.angle(-V[1, 0] * V[2, 2] * np.conj(V[1, 2]) * np.conj(V[2, 0]))/(2*math.pi)*360}\n")
    f.write("\n")
    f.write("Jarlskog:\n")
    f.write(f"{(V[0, 1] * V[1, 2] * np.conj(V[0, 2]) * np.conj(V[1, 1])).imag}\n")
    f.write("\n")
