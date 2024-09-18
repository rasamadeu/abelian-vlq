import phenomenology as pheno
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from math import log

# Set latex writing
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 100


def main():

    fig, ax = plt.subplots()
    xmin = 0.05e3
    xmax = 10e3
    delta = 5000

    x = np.linspace(xmin, xmax, delta)
    f_linear = np.zeros(delta)
    S0_points = np.zeros(delta)
    S0_xc_points = np.zeros(delta)
    S0_xt_points = np.zeros(delta)
    Y0_points = np.zeros(delta)
    X0_points = np.zeros(delta)
    E0_points = np.zeros(delta)
    Z0_points = np.zeros(delta)

    for i in range(delta):
        S0_points[i] = pheno.S0(xmin + i * (xmax - xmin)/delta)
        S0_xc_points[i] = pheno.S0(1.27, xmin + i * (xmax - xmin)/delta)
        S0_xt_points[i] = pheno.S0(172.69, xmin + i * (xmax - xmin)/delta)
        Y0_points[i] = pheno.Y0(xmin + i * (xmax - xmin)/delta)
        X0_points[i] = pheno.X0(xmin + i * (xmax - xmin)/delta)
        E0_points[i] = pheno.E0(xmin + i * (xmax - xmin)/delta)
        Z0_points[i] = pheno.Z0(xmin + i * (xmax - xmin)/delta)
        val = (xmin + i * (xmax - xmin)/delta)
        f_linear[i] = (val / pheno.MW) ** 2 / 8

    ax = plt.subplot2grid((1, 2), (0, 1))
    plt.yscale("log")
    plt.axhline(y=pheno.S0(1.27), linestyle='dotted')
    plt.axhline(y=pheno.S0(172.69), linestyle='dotted')
    plt.text(8, pheno.S0(1.27)+0.0001, r'$S_0(x_c)$', fontsize=12)
    plt.text(8, pheno.S0(172.69)+1, r'$S_0(x_t)$', fontsize=12)
    ax.plot(x/1e3, S0_points, color='red', label=r'$S_0(x_T)$')
    ax.plot(x/1e3, S0_xc_points, color='blueviolet', label=r'$S_0(x_c, x_T)$')
    ax.plot(x/1e3, S0_xt_points, color='lawngreen', label=r'$S_0(x_t, x_T)$')
    ax.plot(x/1e3, Y0_points, color='blue', label=r'$Y_0(x_T)$')
    ax.plot(x/1e3, X0_points, color='k', label=r'$X_0(x_T)$')
    ax.plot(x/1e3, E0_points, color='magenta', label=r'$E_0(x_T)$')
    ax.plot(x/1e3, Z0_points, color='brown', label=r'$Z_0(x_T)$')
    ax.plot(x/1e3, f_linear, color='k', linestyle='dashed', label=r'$x_T/8$')
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(11)))
    ax.set_xlabel(r'$m_T$(TeV)', fontsize=10)
    ax.grid(True, which='major', linestyle='dotted')
    ax.grid(True, which='minor', linestyle='dotted', alpha=0.5)
    ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=False,
                   labelright=True, bottom=True, top=True, left=True, right=True,
                   direction='in')

    i_min = 0
    i_max = 250
    ax_zoom = plt.subplot2grid((1, 2), (0, 0))
    plt.yscale("log")
    plt.axhline(y=pheno.S0(1.27), linestyle='dotted')
    plt.axhline(y=pheno.S0(172.69), linestyle='dotted')
    plt.text(0.4, pheno.S0(1.27)+0.0001/2, r'$S_0(x_c)$', fontsize=12)
    plt.text(0.4, pheno.S0(172.69)-0.7, r'$S_0(x_t)$', fontsize=12)
    ax_zoom.plot(x[i_min:i_max]/1e3, S0_points[i_min:i_max],
                 color='red', label=r'$S_0(x_T)$')
    ax_zoom.plot(x[i_min:i_max]/1e3, S0_xc_points[i_min:i_max],
                 color='blueviolet', label=r'$S_0(x_c, x_T)$')
    ax_zoom.plot(x[i_min:i_max]/1e3, S0_xt_points[i_min:i_max],
                 color='lawngreen', label=r'$S_0(x_t, x_T)$')
    ax_zoom.plot(x[i_min:i_max]/1e3, Y0_points[i_min:i_max],
                 color='blue', label=r'$Y_0(x_T)$')
    ax_zoom.plot(x[i_min:i_max]/1e3, X0_points[i_min:i_max],
                 color='k', label=r'$X_0(x_T)$')
    ax_zoom.plot(x[i_min:i_max]/1e3, E0_points[i_min:i_max],
                 color='magenta', label=r'$E_0(x_T)$')
    ax_zoom.plot(x[i_min:i_max]/1e3, Z0_points[i_min:i_max],
                 color='brown', label=r'$Z_0(x_T)$')
    ax_zoom.plot(x[i_min:i_max]/1e3, f_linear[i_min:i_max],
                 color='k', linestyle='dashed', label=r'$x_T/8$')
    ax_zoom.set_xlabel(r'$m_T$(TeV)', fontsize=10)
    ax_zoom.grid(True, which='major', linestyle='dotted')
    ax_zoom.grid(True, which='minor', linestyle='dotted', alpha=0.5)
    ax_zoom.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True,
                        labelright=False, bottom=True, top=True, left=True, right=True,
                        direction='in')
    ax.indicate_inset_zoom(ax_zoom, edgecolor="black")
    plt.subplots_adjust(left=0.05, bottom=0.05, top=0.99,
                        right=0.95, wspace=0.03)
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    main()
