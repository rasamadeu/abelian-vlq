import sys
import numpy as np
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from math import isnan

# Set latex writing
plt.rcParams['text.usetex'] = True

TEV = 1e3
N_POINTS = 100

##########################################################################################
#
#   I/O FUNCTIONS
#
##########################################################################################


def read_info(filename):

    x = []
    theta = []
    chi_square = []
    chi_square_gamma = []
    delta = []
    chi_square_delta = []
    m_T = []

    i = 0
    j = 0
    with open(filename, "r") as f:
        line = f.readline()  # ignore file header
        line = f.readline()
        line = f.readline()
        while line != "":
            info = line.split()
            x.append(float(info[0]))
            theta.append(float(info[1]))
            chi_square.append(float(info[2]))
            chi_square_gamma.append(float(info[3]))
            delta.append(float(info[4]))
            chi_square_delta.append(float(info[5]))
            m_T.append(float(info[6][0:5]))
            line = f.readline()
            if chi_square[-1] < chi_square[j]:
                a = x[-1]
                b = theta[-1]
                c = chi_square[-1]
                j = i
            i += 1

    print(a, b, c)
    x = np.array(x, dtype=float)
    theta = np.array(theta, dtype=float)
    chi_square = np.array(chi_square, dtype=float)
    chi_square_gamma = np.array(chi_square_gamma, dtype=float)
    delta = np.array(delta, dtype=float)
    chi_square_delta = np.array(chi_square_delta, dtype=float)
    m_T = np.array(m_T, dtype=float)
    return x, theta, chi_square, chi_square_gamma, delta, chi_square_delta, m_T


##########################################################################################


def scatter_plot(filename, ax):

    x2, theta, chi_square, chi_square_gamma, delta, chi_square_delta, m_T = read_info(
        filename)

    # plot chi_square heatmap with theta vs x2
# u   ax.scatter(x2, theta, c=chi_square, marker='.', norm=colors.LogNorm(
# u       vmin=1, vmax=1000), cmap='RdBu_r')  # vmin=0, vmax=0.05
# u   ax.set_xlabel("$x_2$ (MeV)")
# u   ax.set_ylabel(r'$\theta$')
# u   ax.xaxis.get_major_formatter().set_useOffset(False)
# u   ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
# u   ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
# u
# u   levels = np.array([2.3, 6.18, 11.83])
# u   X = np.reshape(x2, (100, 100))
# u   Y = np.reshape(theta, (100, 100))
# u   Z = np.reshape(chi_square, (100, 100))
# u   CS = ax.contour(X, Y, Z, levels=levels, colors='k')
# u   fmt = {}
# u   fmt[levels[0]] = r'$1\sigma$'
# u   fmt[levels[1]] = r'$2\sigma$'
# u   fmt[levels[2]] = r'$3\sigma$'
# u   ax.clabel(CS, inline=True, fontsize=10, fmt=fmt, manual=True)
# u
# u   Z = np.reshape(chi_square_delta, (100, 100))
# u   CS = ax.contour(X, Y, Z, levels=levels, colors='lawngreen')
# u   fmt = {}
# u   fmt[levels[0]] = r'$1\sigma$'
# u   fmt[levels[1]] = r'$2\sigma$'
# u   fmt[levels[2]] = r'$3\sigma$'
# u   ax.clabel(CS, inline=True, fontsize=10, fmt=fmt, manual=True, colors='k')

    # plot delta heatmap with theta vs x2
#   fig, ax = plt.subplots()
#   pc = ax.scatter(x2, theta, c=delta, marker='.', cmap='RdBu_r')
#   ax.set_xlabel("$x_2$ (MeV)")
#   ax.set_ylabel(r'$\theta$')
#   fig.colorbar(pc, ax=ax, extend='both', label=r'$\delta$')
#   plt.show()
#
#   # plot chi_square_gamma heatmap with theta vs x2
#   fig, ax = plt.subplots()
#   print(chi_square_gamma.min())
#   print(chi_square_gamma.max())
#   pc = ax.scatter(x2, theta, c=chi_square_gamma, marker='.', cmap='RdBu_r')
#   ax.set_xlabel("$x_2$ (MeV)")
#   ax.set_ylabel(r'$\theta$')
#   fig.colorbar(pc, ax=ax, extend='both', label=r'$\chi^2_\gamma$')
#   plt.show()

    # plot m_T heatmap with theta vs x2
    x = []
    y = []
    z = []
    for i, elem in enumerate(chi_square):
        if elem < 11.83:
            x.append(m_T[i])
            y.append(delta[i])
            z.append(elem)
    ax.scatter(x, y, c=z, marker='.', norm=colors.Normalize(
        vmin=1, vmax=11.83), cmap='RdBu_r')
    ax.set_xlabel("$m_T$ (TeV)")
    ax.set_ylabel(r'$\delta$')
#
#   # plot m_T heatmap with theta vs x2
#   fig, ax = plt.subplots()
#   pc = ax.scatter(m_T, x2, c=delta, marker='.', cmap='RdBu_r')
#   ax.set_xlabel("$m_T$ (TeV)")
#   ax.set_ylabel(r'$x_2$ (MeV)')
#   fig.colorbar(pc, ax=ax, extend='both', label=r'$\delta$')
#   plt.show()

#   x = []
#   y = []
#   z = []
#   for i, elem in enumerate(delta):
#       if chi_square[i] < 100:
#           x.append(m_T[i])
#           y.append(delta[i])
#           z.append(theta[i])
#
#   x = np.array(x, dtype=float)
#   y = np.array(y, dtype=float)
#   z = np.array(z, dtype=float)
#   fig, ax = plt.subplots()
#   pc = ax.scatter(x, y, c=z, marker='.',  # norm=colors.LogNorm(
#                   # vmin=z.min(), vmax=z.max()),
#                   cmap='RdBu_r')
#   ax.set_xlabel(r'$m_T$''(TeV)')
#   ax.set_ylabel(r'$\delta_\mathcal{uni}$')
#   fig.colorbar(pc, ax=ax, extend='both', label=r'$\theta$')
#   plt.show()
    return


def main():

    # args = sys.argv[1:]
    fig, ax = plt.subplots(nrows=3, ncols=2)
    ax = ax.flatten()
    scatter_plot(
        "./d_decoupled_zoom/2HDM_minuit_d_decoupled_zoom_5_1.dat", ax[0])
    scatter_plot(
        "./d_decoupled_zoom/2HDM_minuit_d_decoupled_zoom_5_3.dat", ax[1])
    scatter_plot(
        "./s_decoupled_zoom/2HDM_minuit_s_decoupled_zoom_5_1.dat", ax[2])
    scatter_plot(
        "./s_decoupled_zoom/2HDM_minuit_s_decoupled_zoom_5_3.dat", ax[3])
    scatter_plot(
        "./b_decoupled_zoom/2HDM_minuit_b_decoupled_zoom_5_1.dat", ax[4])
    scatter_plot(
        "./b_decoupled_zoom/2HDM_minuit_b_decoupled_zoom_5_3.dat", ax[5])

    plt.subplots_adjust(left=0.05, bottom=0.1, top=0.98, right=0.98)
    cax = plt.axes((0.3, 0.04, 0.4, 0.02))
    fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(
        vmin=1, vmax=11.83), cmap='RdBu_r'), orientation='horizontal', cax=cax)  # , label=r'$\chi^2$')
    plt.show()
    return


if __name__ == "__main__":
    main()
