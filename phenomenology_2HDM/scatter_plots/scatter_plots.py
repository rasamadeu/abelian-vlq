import matplotlib.pyplot as plt
import sys
import numpy as np

# Set latex writing
plt.rcParams['text.usetex'] = True

TEV = 1e3

##########################################################################################
#
#   I/O FUNCTIONS
#
##########################################################################################


def read_info(filename):

    data = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != "":
            if len(line) > 1:
                if line.split()[0] == "chi_square:":
                    chi_square = np.round(float(line.split()[1]), 2)
                if line.split()[0] == "phase":
                    phase = np.round(float(line.split()[2]), 2)
                if line.split()[0] == "m_VLQ":
                    m_VLQ = np.round(float(line.split()[2][1:7]) * TEV)
                    data.append([m_VLQ, phase, chi_square])

            line = f.readline()

    return data


def write_info(filename, data):

    with open(filename, "w") as f:
        f.write("m_VLQ phase chi_square\n\n")
        for point in data:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

    return

##########################################################################################


def scatter_plot(data):

    x = []
    y = []
    z = []

    for i, point in enumerate(data):
        if point[0] < 3 * TEV:
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])

    # Plot
    fig, ax = plt.subplots() #subplot_kw={"projection": "3d"}v
    pc = ax.scatter(x, y, c=z, cmap='RdBu_r')
    ax.set_title("Quark b decoupled")
    ax.set_xlabel("$m_T$ (TeV)")
    ax.set_ylabel(r'$\alpha$')
    fig.colorbar(pc, ax=ax, extend='both', label=r'$\chi^2$')
    #ax.set_zlabel("chi_squared")
    plt.show()
    return


def main():

    args = sys.argv[1:]

    output_file = args[0][0:-4] + "_simple.txt"
    data = read_info(args[0])
    write_info(output_file, data)
    scatter_plot(data)

    return


if __name__ == "__main__":
    main()
