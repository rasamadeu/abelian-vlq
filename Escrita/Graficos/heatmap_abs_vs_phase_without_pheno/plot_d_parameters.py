import sys
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

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
            theta.append(float(info[1]) / (2 * np.pi) * 360)
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
    return x, theta, chi_square, chi_square_gamma, delta, chi_square_delta, m_T, a, b, c


##########################################################################################


def scatter_plot(filename, ax, label, vaxis=True):

    x2, theta, chi_square, chi_square_gamma, delta, chi_square_delta, m_T, x2_bfv, theta_bfv, chi_square_bfv = read_info(
        filename)

    # plot chi_square heatmap with theta vs x2
    # ax.scatter(x2, theta, c=chi_square, marker='.', norm=colors.LogNorm(
    #    vmin=1, vmax=1000), cmap='RdBu_r')  # vmin=0, vmax=0.05
    # ax.set_xlabel("$x_2$ (MeV)")
    # ax.set_ylabel(r'$\theta$')
    # ax.xaxis.get_major_formatter().set_useOffset(False)
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(6))

    levels = np.array([0, 2.3, 6.18, 11.83])
    X = np.reshape(x2, (100, 100))
    Y = np.reshape(theta, (100, 100))
    Z = np.reshape(chi_square, (100, 100)) - chi_square_bfv
    ax.contourf(X, Y, Z, levels=levels, colors=[
                'lightgrey', 'blue', 'magenta'])

    CS = ax.contour(X, Y, Z, levels=levels, colors='k', linewidths=1)
    fmt = {}
    fmt[levels[0]] = r'$1\sigma$'
    fmt[levels[1]] = r'$2\sigma$'
    fmt[levels[2]] = r'$3\sigma$'
    ax.clabel(CS, inline=True, fontsize=10, fmt=fmt, manual=True)

    # bfv point
    print(x2_bfv, theta_bfv)
    ax.scatter(x2_bfv, theta_bfv, marker='.', c='k')
    ax.set_xlabel("$x_2$ (MeV)")
    ax.xaxis.get_major_formatter().set_useOffset(False)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.set_ylabel(r'$\theta (^{\circ})$')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.minorticks_on()
    if not vaxis:
        ax.set_yticklabels([])
        ax.set_ylabel('')

    levels = np.array([2.3, 6.18, 11.83])
    Z = np.reshape(chi_square_delta, (100, 100))
    CS = ax.contour(X, Y, Z, levels=levels, colors='lawngreen')
    fmt = {}
    fmt[levels[0]] = r'$1\sigma$'
    fmt[levels[1]] = r'$2\sigma$'
    fmt[levels[2]] = r'$3\sigma$'
    ax.clabel(CS, inline=True, fontsize=10, fmt=fmt, manual=True, colors='k')

    ax.text(0.02, 0.9, label, transform=ax.transAxes, fontsize=20)
    ax.grid(True, which='major', linestyle='dotted')
    ax.grid(True, which='minor', linestyle='dotted', alpha=0.5)
    ax.tick_params(which='both', direction='in')
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
#   x = []
#   y = []
#   z = []
#   for i, elem in enumerate(chi_square):
#       if elem < 11.83:
#           x.append(m_T[i])
#           y.append(delta[i])
#           z.append(elem)
#   ax.scatter(x, y, c=z, marker='.', norm=colors.Normalize(
#       vmin=1, vmax=11.83), cmap='RdBu_r')
#   ax.set_xlabel("$m_T$ (TeV)")
#   ax.set_ylabel(r'$\delta$')
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
        "./d_decoupled_zoom/2HDM_minuit_d_decoupled_zoom_5_1.dat", ax[0], r'$5_1^d$')
    scatter_plot(
        "./d_decoupled_zoom/2HDM_minuit_d_decoupled_zoom_5_3.dat", ax[1], r'$5_3^d$', False)
    scatter_plot(
        "./s_decoupled_zoom/2HDM_minuit_s_decoupled_zoom_5_1.dat", ax[2], r'$5_1^s$')
    scatter_plot(
        "./s_decoupled_zoom/2HDM_minuit_s_decoupled_zoom_5_3.dat", ax[3], r'$5_3^s$', False)
    scatter_plot(
        "./b_decoupled_zoom/2HDM_minuit_b_decoupled_zoom_5_1.dat", ax[4], r'$5_1^b$')
    scatter_plot(
        "./b_decoupled_zoom/2HDM_minuit_b_decoupled_zoom_5_3.dat", ax[5], r'$5_3^b$', False)

    plt.subplots_adjust(left=0.05, bottom=0.05, top=0.98,
                        right=0.98, wspace=0.05)
   # cax = plt.axes((0.3, 0.04, 0.4, 0.02))
   # fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(
   #    vmin=1, vmax=11.83), cmap='RdBu_r'), orientation='horizontal', cax=cax)  # , label=r'$\chi^2$')
    plt.show()
    return


if __name__ == "__main__":
    main()
