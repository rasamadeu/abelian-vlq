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
    chi_square_K1 = []
    chi_square_K2 = []
    chi_square_eps = []
    chi_square_D = []
    chi_square_Bd_re = []
    chi_square_Bd_im = []
    chi_square_Bs_re = []
    chi_square_Bs_im = []
    chi_square_Bd_decay = []
    chi_square_Bs_decay = []
    chi_square_K1_decay = []
    chi_square_K2_decay = []
    chi_square_K3_decay = []
    chi_square_Vub = []

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
            chi_square_K1.append(float(info[7]))
            chi_square_K2.append(float(info[8]))
            chi_square_eps.append(float(info[9]))
            chi_square_D.append(float(info[10]))
            chi_square_Bd_re.append(float(info[11]))
            chi_square_Bd_im.append(float(info[12]))
            chi_square_Bs_re.append(float(info[13]))
            chi_square_Bs_im.append(float(info[14]))
            chi_square_Bd_decay.append(float(info[15]))
            chi_square_Bs_decay.append(float(info[16]))
            chi_square_K1_decay.append(float(info[17]))
            chi_square_K2_decay.append(float(info[18]))
            chi_square_K3_decay.append(float(info[19]))
            chi_square_Vub.append(float(info[-1]))
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
    chi_square_K1 = np.array(chi_square_K1, dtype=float)
    chi_square_K2 = np.array(chi_square_K2, dtype=float)
    chi_square_eps = np.array(chi_square_eps, dtype=float)
    chi_square_D = np.array(chi_square_D, dtype=float)
    chi_square_Bd_re = np.array(chi_square_Bd_re, dtype=float)
    chi_square_Bd_im = np.array(chi_square_Bd_im, dtype=float)
    chi_square_Bs_re = np.array(chi_square_Bs_re, dtype=float)
    chi_square_Bs_im = np.array(chi_square_Bs_im, dtype=float)
    chi_square_Bd_decay = np.array(chi_square_Bd_decay, dtype=float)
    chi_square_Bs_decay = np.array(chi_square_Bs_decay, dtype=float)
    chi_square_K1_decay = np.array(chi_square_K1_decay, dtype=float)
    chi_square_K2_decay = np.array(chi_square_K2_decay, dtype=float)
    chi_square_K3_decay = np.array(chi_square_K3_decay, dtype=float)
    chi_square_Vub = np.array(chi_square_Vub, dtype=float)

    m_T = np.array(m_T, dtype=float)
    return x, theta, chi_square, chi_square_gamma, delta, chi_square_delta, m_T, chi_square_delta, chi_square_K1, chi_square_K2, chi_square_eps, chi_square_D, chi_square_Bd_re, chi_square_Bd_im, chi_square_Bs_re, chi_square_Bs_im, chi_square_Bd_decay, chi_square_Bs_decay, chi_square_K1_decay, chi_square_K2_decay, chi_square_K3_decay, chi_square_Vub, a, b, c


##########################################################################################


def scatter_plot(filename, ax, label, vaxis=True):

    x2, theta, chi_square, chi_square_gamma, delta, chi_square_delta, m_T, chi_square_delta, chi_square_K1, chi_square_K2, chi_square_eps, chi_square_D, chi_square_Bd_re, chi_square_Bd_im, chi_square_Bs_re, chi_square_Bs_im, chi_square_Bd_decay, chi_square_Bs_decay, chi_square_K1_decay, chi_square_K2_decay, chi_square_K3_decay, chi_square_Vub, x2_bfv, theta_bfv, chi_square_bfv = read_info(
        filename)
#   Z = np.reshape(chi_square_Vub, (100, 100))
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
    Z = np.reshape(chi_square, (100, 100)) - 18.12
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

#   Z = np.reshape(chi_square_Vub, (100, 100))
#   CS = ax.contour(X, Y, Z, levels=levels, colors='m')
#   fmt = {}
#   fmt[levels[0]] = r'$1\sigma$'
#   fmt[levels[1]] = r'$2\sigma$'
#   fmt[levels[2]] = r'$3\sigma$'
#   ax.clabel(CS, inline=True, fontsize=10, fmt=fmt, manual=True, colors='k')

#   Z = np.reshape(chi_square_Bd_re, (100, 100))
#   CS = ax.contour(X, Y, Z, levels=levels, colors='m')
#   fmt = {}
#   fmt[levels[0]] = r'$1\sigma$'
#   fmt[levels[1]] = r'$2\sigma$'
#   fmt[levels[2]] = r'$3\sigma$'
#   ax.clabel(CS, inline=True, fontsize=10, fmt=fmt, manual=True, colors='k')

#   Z = np.reshape(chi_square_Bd_im, (100, 100))
#   CS = ax.contour(X, Y, Z, levels=levels, colors='g')
#   fmt = {}
#   fmt[levels[0]] = r'$1\sigma$'
#   fmt[levels[1]] = r'$2\sigma$'
#   fmt[levels[2]] = r'$3\sigma$'
#   ax.clabel(CS, inline=True, fontsize=10, fmt=fmt, manual=True, colors='k')

#   Z = np.reshape(chi_square_Bs_decay, (100, 100))
#   CS = ax.contour(X, Y, Z, levels=levels, colors='tab:brown')
#   fmt = {}
#   fmt[levels[0]] = r'$1\sigma$'
#   fmt[levels[1]] = r'$2\sigma$'
#   fmt[levels[2]] = r'$3\sigma$'
#   ax.clabel(CS, inline=True, fontsize=10, fmt=fmt, manual=True, colors='k')

#   Z = np.reshape(chi_square_K1_decay, (100, 100))
#   CS = ax.contour(X, Y, Z, levels=levels, colors='dodgerblue')
#   fmt = {}
#   fmt[levels[0]] = r'$1\sigma$'
#   fmt[levels[1]] = r'$2\sigma$'
#   fmt[levels[2]] = r'$3\sigma$'
#   ax.clabel(CS, inline=True, fontsize=10, fmt=fmt, manual=True, colors='k')

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
        "./d_decoupled/2HDM_with_pheno_d_decoupled_5_1_final.dat", ax[0], r'$5_1^d$')
    scatter_plot(
        "./d_decoupled/2HDM_with_pheno_d_decoupled_5_3_final.dat", ax[1], r'$5_3^d$', False)
    scatter_plot(
        "./s_decoupled/2HDM_with_pheno_s_decoupled_5_1_final.dat", ax[2], r'$5_1^s$')
    scatter_plot(
        "./s_decoupled/2HDM_with_pheno_s_decoupled_5_3_final.dat", ax[3], r'$5_3^s$', False)
    scatter_plot(
        "./b_decoupled/2HDM_with_pheno_b_decoupled_5_1_final.dat", ax[4], r'$5_1^b$')
    scatter_plot(
        "./2500_tries/b_decoupled_5_3_final", ax[5], r'$5_3^b$', False)

    plt.subplots_adjust(left=0.05, bottom=0.05, top=0.98,
                        right=0.98, wspace=0.05)
    plt.show()
    return


if __name__ == "__main__":
    main()
