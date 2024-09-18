import matplotlib.pyplot as plt
import numpy as np

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
    gA = 1.2754 # 1.27624 # 1.2754
    gA_error = 0.0013 # 0.0005 # 0.0013
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
                                 + (Vudn * 6 * gA / (1 + 3 * gA ** 2) * gA_error) ** 2
                                 + (Vudn / (1 + Delta_R) * Delta_R_error) ** 2)

    print(Kud)
    print(Kud_error)
    print(Vudn)
    print(Vudn_error)

    # WE FOLLOW THE PROCEDURE FROM THE PDG
    Vud = (1 / Vudb_error ** 2 * Vudb + 1 / Vudn_error ** 2 * Vudn) / (1 / Vudn_error ** 2 + 1 / Vudb_error ** 2)
    Vud_error = (1 / Vudn_error ** 2 + 1 / Vudb_error ** 2) ** -0.5
    chi_square = ((Vud - Vudb) / Vudb_error) ** 2 + ((Vud - Vudn) / Vudn_error) ** 2
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


VUD, VUD_error, VUS, VUS_error, VUS_VUD, VUS_VUD_error = compute_V_elements()


def main():

    fig, ax = plt.subplots()
    xmin = 0.222
    xmax = 0.229
    ymin = 0.972
    ymax = 0.9756
    delta = 1000

    x = np.linspace(xmin, xmax, delta)
    y = np.linspace(ymin, ymax, delta)

    # Vud
    Vud = np.ones(delta) * VUD
    Vud_upper = VUD + VUD_error
    Vud_lower = VUD - VUD_error
    ax.plot(x, Vud, color='red')
    ax.fill_between(x, Vud_upper, Vud_lower, color='red', alpha=0.5)

    # Vus
    plt.axvline(x=VUS, color='purple')
    plt.axvspan((VUS - VUS_error), (VUS + VUS_error), color='purple', alpha=0.5)

    # Vus/Vud
    Vus_Vud = x / VUS_VUD
    Vus_Vud_lower = x / (VUS_VUD + VUS_VUD_error)
    Vus_Vud_upper = x / (VUS_VUD - VUS_VUD_error)
    ax.plot(x, Vus_Vud, color='blue')
    ax.fill_between(x, Vus_Vud_upper, Vus_Vud_lower, color='blue', alpha=0.5)

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
    Unitarity_deviation = np.sqrt(bfv_point[0] ** 2 + bfv_point[1] ** 2 - x ** 2)
    ax.plot(x, Unitarity_deviation, linestyle='dashed', color='black', alpha=0.5)

    levels = np.array([2.3, 6.18, 11.83])
    Z = Z - bfv_chi
    ax.contour(X, Y, Z, levels=levels, colors=['lawngreen', 'lawngreen', 'lawngreen'])
    plt.arrow(0.22622, 0.97402, -0.000135, -0.00065,
              head_width=0.00005, width=0.00001, color='black')
    ax.text(0.22625, 0.9736, r'$\delta_\mathrm{CKM}$', fontsize=14)
    ax.text(0.2278, 0.9738, r'$\beta$-decays', color='red', fontsize=14)
    ax.text(0.2256, 0.9748, r'$\frac{K_{\mu\nu}}{\pi_{\mu\nu}}$', color='blue', fontsize=18)
    ax.text(0.2226, 0.97520, r'$Kl_3$', color='purple', fontsize=14)

    # Plot settings
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r'$|(\mathbf{V}_\mathrm{CKM})_{us}|$', fontsize=14)
    ax.set_ylabel(r'$|(\mathbf{V}_\mathrm{CKM})_{ud}|$', fontsize=14)
    ax.grid(True, which='major', linestyle='dotted')
    ax.grid(True, which='minor', linestyle='dotted', alpha=0.5)
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

    plt.axvline(x=B, color='blue', linestyle = '--', ymax=(np.sqrt(1 - B ** 2) - ymin) / (ymax - ymin))
    plt.axvline(x=C, color='red', linestyle = '--', ymax=(VUD - ymin) / (ymax - ymin))
    plt.fill_between(x, Unitarity, where = (x > B - B_error) & (x <= B + B_error), color='blue', alpha=0.5)
    plt.fill_between(x, Unitarity, where = (x > C - C_error) & (x <= C + C_error), color='red', alpha=0.5)
    plt.show()
    return


if __name__ == "__main__":
    main()
