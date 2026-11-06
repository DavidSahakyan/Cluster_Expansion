import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *

file_to_read = "/home/david/Desktop/project/generated_data_files/SGC_data.txt"

# Ts = range(150, 500, 25)
Ts = [325]
# critical_T = 350


def phase_diagram_SGC(file_to_read, Ts, critical_T):
    T_GGIs = {}
    phase_diagram_T = []
    phase_diagram_GGI_right = []
    phase_diagram_GGI_left = []
    for T in Ts:

        GGI = []
        dmu = []
        E = []

        with open(file_to_read) as f:
            for line in f:
                data = line.split()

                if float(data[0]) == T:
                    GGI.append(float(data[1]))
                    E.append(float(data[2]))
                    dmu.append(float(data[3]))

        if not len(GGI):
            print(f"WARNING! TARGET TEMPERATURE {T} IS NOT FOUND")
            continue

        GGI = np.array(GGI)
        dmu = np.array(dmu)
        E = np.array(E)

        # Sort by chemical potential
        indices = np.argsort(dmu)

        dmu = dmu[indices]
        GGI = GGI[indices]
        E = E[indices]

        edges_mask = []

        if T <= 250:
            for i in range(len(GGI)):
                if GGI[i] < 0.05 or GGI[i] > 0.95:
                    edges_mask.append(i)

            dmu = dmu[edges_mask]
            GGI = GGI[edges_mask]
            E = E[edges_mask]
        elif T < critical_T:
            for i in range(len(GGI)):
                if GGI[i] < 0.3 or GGI[i] > 0.7:
                    edges_mask.append(i)

            dmu = dmu[edges_mask]
            GGI = GGI[edges_mask]
            E = E[edges_mask]



        # Find jumps in composition
        delta_GGI = np.abs(np.diff(GGI))

        # Index of largest jump
        jump_index = np.argmax(delta_GGI)

        # Compositions before and after the jump
        GGI_left = min(GGI[jump_index], GGI[jump_index + 1])
        GGI_right = max(GGI[jump_index], GGI[jump_index + 1])

        # Store phase boundaries
        phase_diagram_T.append(T)
        phase_diagram_GGI_left.append(GGI_left)
        phase_diagram_GGI_right.append(GGI_right)

        print(
            f"T = {T} K: "
            f"GGI_left = {GGI_left:.4f}, "
            f"GGI_right = {GGI_right:.4f}, "
            f"Jump = {delta_GGI[jump_index]:.4f}"
        )

        plt.scatter(GGI, dmu, label=f"T = {T} K")


    plt.ylabel(r"$\Delta\mu$ (eV)")
    plt.xlabel("GGI")
    plt.title("SGC composition vs chemical potential")
    plt.legend()
    plt.show()
    
    phase_diagram_T         = np.array(phase_diagram_T)
    phase_diagram_GGI_left  = np.array(phase_diagram_GGI_left)
    phase_diagram_GGI_right = np.array(phase_diagram_GGI_right)

    indices = np.argsort(phase_diagram_T)


    phase_diagram_T = phase_diagram_T[indices]
    phase_diagram_GGI_left  = phase_diagram_GGI_left[indices]
    phase_diagram_GGI_right = phase_diagram_GGI_right[indices]

    for i in range(len(phase_diagram_T)):
        T_GGIs[phase_diagram_T[i]] = [phase_diagram_GGI_left[i], phase_diagram_GGI_right[i]] 

    return T_GGIs
    # plt.plot(
    #     phase_diagram_GGI_left,
    #     phase_diagram_T,
    #     'o-',
    #     label="Left boundary"
    # )

    # plt.plot(
    #     phase_diagram_GGI_right,
    #     phase_diagram_T,
    #     'o-',
    #     label="Right boundary"
    # )

    # plt.xlabel("GGI")
    # plt.ylabel("Temperature (K)")
    # plt.title("Phase diagram from SGC ensemble")
    # plt.legend()
    # plt.grid()
    # plt.show()

phase_diagram_SGC(file_to_read, Ts, 325)