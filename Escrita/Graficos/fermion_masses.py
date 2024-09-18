import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# Set latex writing
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 100

MeV = 1
GeV = 1e3
TeV = 1e6

MASS_E = 0.511 * MeV
MASS_MU = 105.7 * MeV
MASS_TAU = 1777 * MeV
MASS_U = 2.16 * MeV
MASS_D = 4.67 * MeV
MASS_S = 93.4 * MeV
MASS_C = 1.27 * GeV
MASS_B = 4.18 * GeV
MASS_T = 172.69 * GeV


def main():

    fig, ax = plt.subplots()
    ax.annotate(r'$u$', (1, MASS_U), (1.05, 0.9 * MASS_U), color="red")
    ax.annotate(r'$c$', (1, MASS_C), (1.05, 0.9 * MASS_C), color="red")
    ax.annotate(r'$t$', (1, MASS_T), (1.05, 0.9 * MASS_T), color="red")
    ax.annotate(r'$d$', (1.5, MASS_D), (1.55, 0.9 * MASS_D), color="blue")
    ax.annotate(r'$s$', (1.5, MASS_S), (1.55, 0.9 * MASS_S), color="blue")
    ax.annotate(r'$b$', (1.5, MASS_B), (1.55, 0.9 * MASS_B), color="blue")
    ax.annotate(r'$e$', (2, MASS_E), (2.05, 0.9 * MASS_E), color="green")
    ax.annotate(r'$\mu$', (2, MASS_MU), (2.05, 0.9 * MASS_MU), color="green")
    ax.annotate(r'$\tau$', (2, MASS_TAU),
                (2.05, 0.9 * MASS_TAU), color="green")
    ax.set_yscale('log')
    ax.set_xlim([0.5, 2.5])
    ax.set_title('MeV', loc='left')
    ax.grid(True, axis='y', which='major', linestyle='dotted')
    ax.tick_params(axis='x', which="both", bottom=False, labelbottom=False)
    ax = plt.scatter([1, 1, 1, 1.5, 1.5, 1.5, 2, 2, 2], [
                     MASS_U, MASS_C, MASS_T, MASS_D, MASS_S, MASS_B, MASS_E, MASS_MU, MASS_TAU],
                     color=["red", "red", "red", "blue", "blue", "blue", "green", "green", "green"])
    plt.show()
    return 0


if __name__ == "__main__":
    main()
