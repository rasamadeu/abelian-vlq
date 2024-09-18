import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

# Set latex writing
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 100

# Belfatto
VUD = 0.97372
VUD_error = 0.00026
VUS_VUD = 0.23131
VUS_VUD_error = 0.00051
VUS = 0.22308
VUS_error = 0.00055


def compute_V_elements():

    K = 8120.2762
    K_error = 0.0003
    G_F = 1.1663788
    G_F_error = 0.0000006
    F_t = 3072.24
    F_t_error = 1.85
    Delta_R = 0.02467
    Delta_R_error = 0.00022
    tau_n = 878.4
    tau_n_error = 0.5
    gA = 1.2754  # 1.27624 # 1.2754
    gA_error = 0.0013  # 0.0005 # 0.0013
    fp = 0.9698
    fp_error = 0.0017
    fK = 1.1932
    fK_error = 0.0021
    fn = 1.6887
    fn_error = 0.0002
    delta_R = 0.014902
    delta_R_error = 0.000002

    Kud = K / (2 * G_F ** 2 * F_t)
    Kud_error = np.sqrt((Kud / K * K_error) ** 2
                        + 2 * (Kud / G_F * G_F_error) ** 2
                        + (Kud / F_t * F_t_error) ** 2)
    Vudb = np.sqrt(Kud / (1 + Delta_R))
    Vudb_error = 1 / 2 * np.sqrt((Vudb / Kud * Kud_error) ** 2
                                 + (Vudb / (1 + Delta_R) * Delta_R_error) ** 2)

    print(Kud)
    print(Kud_error)
    print(Vudb)
    print(Vudb_error)

    Fn = fn * (1 + delta_R)
    Fn_error = (1 + delta_R) * fn_error + fn * delta_R_error
    Kud = K / (np.log(2) * G_F ** 2 * Fn)
    Kud_error = np.sqrt((Kud / K * K_error) ** 2
                        + 2 * (Kud / G_F * G_F_error) ** 2
                        + (Kud / Fn * Fn_error) ** 2
                        )
    Vudn = np.sqrt(Kud / (tau_n * (1 + 3 * gA ** 2) * (1 + Delta_R)))
    Vudn_error = 1 / 2 * np.sqrt((Vudn / Kud * Kud_error) ** 2
                                 + (Vudn / tau_n * tau_n_error) ** 2
                                 + (Vudn * 6 * gA / (1 + 3 * gA ** 2)
                                    * gA_error) ** 2
                                 + (Vudn / (1 + Delta_R) * Delta_R_error) ** 2)

    print(Kud)
    print(Kud_error)
    print(Vudn)
    print(Vudn_error)

    # WE FOLLOW THE PROCEDURE FROM THE PDG
    Vud = (1 / Vudb_error ** 2 * Vudb + 1 / Vudn_error ** 2 * Vudn) / \
        (1 / Vudn_error ** 2 + 1 / Vudb_error ** 2)
    Vud_error = (1 / Vudn_error ** 2 + 1 / Vudb_error ** 2) ** -0.5
    chi_square = ((Vud - Vudb) / Vudb_error) ** 2 + \
        ((Vud - Vudn) / Vudn_error) ** 2
    print(f"chi_square/(N-1) = {chi_square / (2 - 1)}")
    # IN OUR CASE, WE GET CHI_SQUARE / N - 1 < 1 SO WE ACCEPT THE RESULTS
    print(Vud)
    print(Vud_error)

    Vus_num = 0.21634
    Vus_num_error = 0.00038
    Vus = Vus_num / fp
    Vus_error = np.sqrt((Vus / Vus_num * Vus_num_error) ** 2
                        + (Vus / fp * fp_error) ** 2)

    Vus_Vud_num = 0.27600
    Vus_Vud_num_error = 0.00037
    Vus_Vud = Vus_Vud_num / fK
    Vus_Vud_error = np.sqrt((Vus_Vud / Vus_Vud_num * Vus_Vud_num_error) ** 2
                            + (Vus_Vud / fK * fK_error) ** 2)

    print(Vus)
    print(Vus_error)
    print(Vus_Vud)
    print(Vus_Vud_error)

    return Vud, Vud_error, Vus, Vus_error, Vus_Vud, Vus_Vud_error


def main():

    VUD, VUD_error, VUS, VUS_error, VUS_VUD, VUS_VUD_error = compute_V_elements()

    fig = plt.figure()
    # ax = plt.subplot2grid((11, 5), (3, 0), colspan=5, rowspan=5)
    ax = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    xmin = 0.224
    xmax = 0.2254
    ymin = 0.9736
    ymax = 0.9745
    delta = 1000

    x = np.linspace(xmin, xmax, delta)
    y = np.linspace(ymin, ymax, delta)

    # Vud
    Vud = np.ones(delta) * VUD
    Vud_upper = VUD + VUD_error
    Vud_lower = VUD - VUD_error
    # ax.plot(x, Vud, color='red')
    # ax.fill_between(x, Vud_upper, Vud_lower, color='red', alpha=0.5)

    # Vus
    # plt.axvline(x=VUS, color='purple')
    # plt.axvspan((VUS - VUS_error), (VUS + VUS_error), color='purple', alpha=0.5)

    # Vus/Vud
    Vus_Vud = x / VUS_VUD
    Vus_Vud_lower = x / (VUS_VUD + VUS_VUD_error)
    Vus_Vud_upper = x / (VUS_VUD - VUS_VUD_error)
    # ax.plot(x, Vus_Vud, color='blue')
    # ax.fill_between(x, Vus_Vud_upper, Vus_Vud_lower, color='blue', alpha=0.5)

    # Unitarity condition
    Unitarity = np.sqrt(1 - x ** 2)
    ax.plot(x, Unitarity, color='black')

    # Sigma countour plots
    X, Y = np.meshgrid(x, y)
    a = 1 / VUS_VUD
    # Best fit value
    Z = (
        (((X - a / (a ** 2 + 1) * (Y + X / a)) ** 2
          + (Y - a ** 2 / (a ** 2 + 1) * (Y + X / a)) ** 2) / VUS_VUD_error ** 2)
        + ((X - VUS) / VUS_error) ** 2
        + ((Y - VUD) / VUD_error) ** 2
    )

    bfv_chi = Z[0, 0]
    bfv_point = [0, 0]
    for i, line in enumerate(Z):
        for j, el in enumerate(line):
            if Z[i, j] < bfv_chi:
                bfv_chi = Z[i, j]
                bfv_point = [X[i, j], Y[i, j]]

    print(bfv_chi)
    print(bfv_point)
    ax.plot(bfv_point[0], bfv_point[1], color='lawngreen', marker='x')

    # Unitarity deviation
    print(1 - bfv_point[0] ** 2 - bfv_point[1] ** 2)
    Unitarity_deviation = np.sqrt(
        bfv_point[0] ** 2 + bfv_point[1] ** 2 - x ** 2)
    ax.plot(x, Unitarity_deviation, linestyle='dashed',
            color='black', alpha=0.5)

    levels = np.array([2.3, 6.18, 11.83])
    Z = Z - bfv_chi
    ax.contour(X, Y, Z, levels=levels, colors=[
               'lawngreen', 'lawngreen', 'lawngreen'])
    plt.arrow(0.22622, 0.97402, -0.000135, -0.00065,
              head_width=0.00005, width=0.00001, color='black')
    # ax.text(0.22625, 0.9736, r'$\delta_\mathrm{CKM}$', fontsize=14)
    # ax.text(0.2278, 0.9738, r'$\beta$-decays', color='red', fontsize=14)
    # ax.text(0.2256, 0.9748, r'$\frac{K_{\mu\nu}}{\pi_{\mu\nu}}$', color='blue', fontsize=18)
    # ax.text(0.2226, 0.97520, r'$Kl_3$', color='purple', fontsize=14)

    # Plot settings
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r'$|(\mathbf{V}_\mathrm{CKM})_{us}|$', fontsize=24)
    ax.set_ylabel(r'$|(\mathbf{V}_\mathrm{CKM})_{ud}|$', fontsize=24)
    ax.grid(True, which='major', linestyle='dotted')
    ax.grid(True, which='minor', linestyle='dotted', alpha=0.5)
    ax.tick_params(which='both', direction='in', bottom=True,
                   top=True, left=True, right=True, labelbottom=False, labeltop=True, labelleft=True,
                   labelright=False, labelsize=18)
    ax.minorticks_on()
#    plt.show()

    # Values of Vus assuming unitarity
    print("Vud:")
    C = np.sqrt(1 - VUD ** 2)
    C_error = VUD * VUD_error / np.sqrt(1 - VUD ** 2)
    print(C)
    print(C_error)
    print("Vus/Vud:")
    w = VUS_VUD
    B = np.sqrt(1 / (1 + 1 / w ** 2))
    B_error = 1 / (1 + 1 / w ** 2) ** (3 / 2) * w ** (-3) * VUS_VUD_error
    print(B)
    print(B_error)

    # plt.axvline(x=B, color='blue', linestyle = '--', ymax=(np.sqrt(1 - B ** 2) - ymin) / (ymax - ymin))
    # plt.axvline(x=C, color='red', linestyle = '--', ymax=(VUD - ymin) / (ymax - ymin))
    # plt.fill_between(x, Unitarity, where = (x > B - B_error) & (x <= B + B_error), color='blue', alpha=0.5)
    # plt.fill_between(x, Unitarity, where = (x > C - C_error) & (x <= C + C_error), color='red', alpha=0.5)

    # Minuit points
    x = [0.224738, 0.224185, 0.224180, 0.224793, 0.224183, 0.225054,
         0.225045, 0.224636, 0.224633, 0.225028, 0.225058, 0.225073]
    y = [0.974159, 0.973726, 0.973725, 0.974196, 0.973733, 0.974339,
         0.974319, 0.974029, 0.974031, 0.974323, 0.974329, 0.974334]
    # texture 5_3 / no pheno / d decoupled
    ax.plot(x[0], y[0], marker='.', color="red")
    # texture 5_1 / no pheno / d decoupled
    ax.plot(x[1], y[1], marker='.', color="blue")
    # texture 5_3 / no pheno / s decoupled
    ax.plot(x[2], y[2], marker='.', color="red")
    # texture 5_1 / no pheno / s decoupled
    ax.plot(x[3], y[3], marker='.', color="blue")
    # texture 5_3 / no pheno / b decoupled
    ax.plot(x[4], y[4], marker='.', color="red")
    # texture 5_1 / no pheno / b decoupled
    ax.plot(x[5], y[5], marker='.', color="blue")
    # texture 5_3 / with pheno / d decoupled
    ax.plot(x[6], y[6], marker='^', markersize=3, color="red")
    # texture 5_1 / with pheno / d decoupled
    ax.plot(x[7], y[7], marker='^', markersize=3, color="blue")
    # texture 5_3 / with pheno / s decoupled
    ax.plot(x[8], y[8], marker='^', markersize=3, color="red")
    # texture 5_1 / with pheno / s decoupled
    ax.plot(x[9], y[9], marker='^', markersize=3, color="blue")
    # texture 5_3 / with pheno / b decoupled
    ax.plot(x[10], y[10], marker='^', markersize=3, color="red")
    # texture 5_1 / with pheno / b decoupled
    ax.plot(x[11], y[11], marker='^', markersize=3, color="blue")
    ax.errorbar(0.22500, 0.97435, 0.00016, 0.00067,
                marker='.', color="darkviolet", capsize=5)
    # texture 5_3 / no pheno / d decoupled
    ax.annotate(r'$5_3^d$', (x[0], y[0]),
                (x[0] + 2e-5, y[0] - 3e-5), color="red", fontsize=24)
    # texture 5_1 / no pheno / d decoupled
    ax.annotate(r'$5_1^d$', (x[1], y[1]),
                (x[1] + 2e-5, y[1] - 3e-5), color="blue", fontsize=24)
    # texture 5_3 / no pheno / s decoupled
    ax.annotate(r'$5_3^s$', (x[2], y[2]),
                (x[2] - 5e-5, y[2] - 3e-5), color="red", fontsize=24)
    # texture 5_1 / no pheno / s decoupled
    ax.annotate(r'$5_1^s$', (x[3], y[3]),
                (x[3] + 2e-5, y[3] - 3e-5), color="blue", fontsize=24)
    # texture 5_3 / no pheno / b decoupled
    ax.annotate(r'$5_3^b$', (x[4], y[4]),
                (x[4] + 2e-5, y[4] + 2e-5), color="red", fontsize=24)
    # texture 5_1 / no pheno / b decoupled
    ax.annotate(r'$5_1^b$', (x[5], y[5]),
                (x[5] - 2e-5, y[5] + 3e-5), color="blue", fontsize=24)
    # texture 5_3 / with pheno / d decoupled
    ax.annotate(r'$5_3^d$', (x[6], y[6]),
                (x[6] + 0e-5, y[6] - 5e-5), color="red", fontsize=24)
    # texture 5_1 / with pheno / d decoupled
    ax.annotate(r'$5_1^d$', (x[7], y[7]),
                (x[7] + 1e-5, y[7] - 5e-5), color="blue", fontsize=24)
    # texture 5_3 / with pheno / s decoupled
    ax.annotate(r'$5_3^s$', (x[8], y[8]),
                (x[8] + 1e-5, y[8] + 3e-5), color="red", fontsize=24)
    # texture 5_1 / with pheno / s decoupled
    ax.annotate(r'$5_1^s$', (x[9], y[9]),
                (x[9] - 2e-5, y[9] - 5e-5), color="blue", fontsize=24)
    # texture 5_3 / with pheno / b decoupled
    ax.annotate(r'$5_3^b$', (x[10], y[10]),
                (x[10] + 2e-5, y[10] - 4e-5), color="red", fontsize=24)
    # texture 5_1 / with pheno / b decoupled
    ax.annotate(r'$5_1^b$', (x[11], y[11]),
                (x[11] + 3e-5, y[11] + 3e-5), color="blue", fontsize=24)
    ax.annotate("SM", (0.2250, 0.97435), (0.2250 - 7e-5, 0.97435 - 5e-5),
                color="darkviolet", fontsize=24)  # texture 5_1 / with pheno / b decoupled

    # Points with modified eta
    # texture 5_1 / with pheno / d decoupled / eta = 0.25
    ax.plot(0.224418, 0.973869, marker='d',
            color="magenta", markersize=3)
    # texture 5_1 / with pheno / d decoupled / eta = 1
    ax.plot(0.224666, 0.974164, marker='d',
            color="brown", markersize=3)
    # texture 5_3 / with pheno / s decoupled / eta = 0.25
    ax.plot(0.224403, 0.973866, marker='d',
            color="magenta", markersize=3)
    # texture 5_3 / with pheno / s decoupled / eta = 1
    ax.plot(0.224736, 0.974101, marker='d',
            color="brown", markersize=3)
    # texture 5_1 / with pheno / d decoupled / eta = 0.25
    ax.annotate(r'$5_1^d$', (0.224418, 0.973869),
                (0.224418 + 1e-5, 0.973869 + 1e-5), color="magenta", fontsize=24)
    # texture 5_1 / with pheno / d decoupled / eta = 1
    ax.annotate(r'$5_1^d$', (0.224666, 0.974164),
                (0.224666 + 1e-5, 0.974164 + 1e-5), color="brown", fontsize=24)
    # texture 5_3 / with pheno / s decoupled / eta = 0.25
    ax.annotate(r'$5_3^s$', (0.224403, 0.973866),
                (0.224403 - 5e-5, 0.973866 - 3e-5), color="magenta", fontsize=24)
    # texture 5_3 / with pheno / s decoupled / eta = 1
    ax.annotate(r'$5_3^s$', (0.224736, 0.974101),
                (0.224736 + 1e-5, 0.974101 - 3e-5), color="brown", fontsize=24)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.minorticks_on()
    legend_elements = [Line2D([0], [0], marker='.', color='w', label=r'$5_3^q$ w/o pheno', markerfacecolor='r', markersize=15),
                       Line2D([0], [0], marker='.', color='w',
                              label=r'$5_1^q$ w/o pheno', markerfacecolor='b', markersize=15),
                       Line2D([0], [0], marker='^', color='w',
                              label=r'$5_3^q$ w pheno', markerfacecolor='r', markersize=15),
                       Line2D([0], [0], marker='^', color='w', label=r'$5_1^q$ w pheno', markerfacecolor='b', markersize=15)]
    plt.legend(handles=legend_elements, fontsize=18)

    # Subplot 1
    w = (0.97445-0.9736)
    h = (0.97445-0.9736)
    # subaxis_a = plt.subplot2grid((11, 5), (8, 1), colspan=3, rowspan=3)
    subaxis_a = plt.subplot2grid((3, 3), (2, 0))
    subaxis_a.tick_params(axis='both')

    # texture 5_1 / no pheno / d decoupled
    subaxis_a.plot(x[1], y[1], marker='.', color="blue")
    # texture 5_3 / no pheno / s decoupled
    subaxis_a.plot(x[2], y[2], marker='.', color="red")
    # texture 5_3 / no pheno / b decoupled
    subaxis_a.plot(x[4], y[4], marker='.', color="red")
    # texture 5_1 / no pheno / d decoupled
    subaxis_a.annotate(r'$5_1^d$', (x[1], y[1]),
                       (x[1] + 0.5e-5, y[1] - 1e-5), color="blue", fontsize=24)
    # texture 5_1 / no pheno / d decoupled
    subaxis_a.annotate(r'$5_3^s$', (x[2], y[2]),
                       (x[2] - 0.75e-5, y[2] - 1e-5), color="red", fontsize=24)
    # texture 5_1 / no pheno / d decoupled
    subaxis_a.annotate(r'$5_3^b$', (x[4], y[4]),
                       (x[4] + 0.5e-5, y[4] + 0.5e-5), color="red", fontsize=24)

    subaxis_a.set_xlim(0.22416, 0.22422)
    subaxis_a.set_ylim(0.97366, 0.97376)
    subaxis_a.grid(True, which='major', linestyle='dotted')
    subaxis_a.grid(True, which='minor', linestyle='dotted', alpha=0.5)
    x_unitarity = np.linspace(xmin, xmax, delta)
    subaxis_a.plot(x_unitarity, Unitarity_deviation,
                   linestyle='dashed', color='black', alpha=0.5)
    subaxis_a.plot(bfv_point[0], bfv_point[1], color='lawngreen', marker='x')
    subaxis_a.xaxis.get_major_formatter().set_useOffset(False)
    subaxis_a.xaxis.set_major_locator(ticker.MaxNLocator(4))
    subaxis_a.yaxis.set_major_locator(ticker.MaxNLocator(4))
    subaxis_a.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    subaxis_a.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    subaxis_a.minorticks_on()
    subaxis_a.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True,
                          labelright=False, bottom=True, top=True, left=True, right=True, direction='in', labelsize=18)

    # Subplot 2
    w = (0.2254-0.224)
    h = (0.97445-0.9736)
    # subaxis_b = plt.subplot2grid((11, 5), (0, 1), colspan=3, rowspan=3)
    subaxis_b = plt.subplot2grid((3, 3), (0, 2), rowspan=2)

    # texture 5_1 / no pheno / b decoupled
    subaxis_b.plot(x[5], y[5], marker='.', color="blue")
    # texture 5_3 / with pheno / d decoupled
    subaxis_b.plot(x[6], y[6], marker='^', markersize=3, color="red")
    # texture 5_1 / with pheno / s decoupled
    subaxis_b.plot(x[9], y[9], marker='^', markersize=3, color="blue")
    # texture 5_3 / with pheno / b decoupled
    subaxis_b.plot(x[10], y[10], marker='^', markersize=3, color="red")
    # texture 5_1 / with pheno / b decoupled
    subaxis_b.plot(x[11], y[11], marker='^', markersize=3, color="blue")
    subaxis_b.plot(0.22500, 0.97435, marker='+', color="darkviolet")

    # texture 5_1 / no pheno / b decoupled
    subaxis_b.annotate(
        r'$5_1^b$', (x[5], y[5]), (x[5] - 1.2e-5, y[5] - 0.5e-5), color="blue", fontsize=24)
    # texture 5_3 / with pheno / d decoupled
    subaxis_b.annotate(r'$5_3^d$', (x[6], y[6]),
                       (x[6] - 1.2e-5, y[6] - 0.5e-5), color="red", fontsize=24)
    # texture 5_1 / with pheno / s decoupled
    subaxis_b.annotate(
        r'$5_1^s$', (x[9], y[9]), (x[9] - 1e-5, y[9] - 0.5e-5), color="blue", fontsize=24)
    # texture 5_3 / with pheno / b decoupled
    subaxis_b.annotate(
        r'$5_3^b$', (x[10], y[10]), (x[10] - 1.2e-5, y[10] - 0.5e-5), color="red", fontsize=24)
    # texture 5_1 / with pheno / b decoupled
    subaxis_b.annotate(
        r'$5_1^b$', (x[11], y[11]), (x[11] + 0.35e-5, y[11] - 0.5e-5), color="blue", fontsize=24)
    subaxis_b.annotate("SM", (0.2250, 0.97435), (0.2250 -
                       0.5e-5, 0.97435 - 1.25e-5), color="darkviolet", fontsize=24)

    subaxis_b.set_xlim(0.22495, 0.22510)
    subaxis_b.set_ylim(0.9743, 0.9744)
    subaxis_b.grid(True, which='major', linestyle='dotted')
    subaxis_b.grid(True, which='minor', linestyle='dotted', alpha=0.5)
    x_unitarity = np.linspace(xmin, xmax, delta)
    subaxis_b.plot(x_unitarity, Unitarity, color='black')
    subaxis_b.tick_params(axis='x')  # labelsize=8)
    subaxis_b.tick_params(axis='y')  # labelsize=8)
    subaxis_b.xaxis.get_major_formatter().set_useOffset(False)
    subaxis_b.yaxis.get_major_formatter().set_useOffset(False)
    subaxis_b.xaxis.set_major_locator(ticker.MaxNLocator(4))
    subaxis_b.yaxis.set_major_locator(ticker.MaxNLocator(4))
    subaxis_b.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    subaxis_b.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    subaxis_b.minorticks_on()
    subaxis_b.tick_params(which='both', labelbottom=False, labeltop=True, labelleft=False,
                          labelright=True, bottom=True, top=True, left=True, right=True, direction='in', labelsize=18)

    ax.indicate_inset_zoom(subaxis_a, edgecolor="black")
    ax.indicate_inset_zoom(subaxis_b, edgecolor="black")
    plt.subplots_adjust(left=0.07, bottom=0.05, top=0.95,
                        right=0.95, hspace=0.2)
    plt.show()
    return


if __name__ == "__main__":
    main()
