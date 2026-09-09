import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *
from icet import ClusterExpansion
from scipy.interpolate import griddata

ce = ClusterExpansion.read("cluster_expansion.ce")

GGI_directories = [
                    "0.00000000000",
                    "0.11111111110",
                    "0.22222222220",
                    "0.33333333330",
                    "0.44444444440",
                    "0.55555555550",
                    "0.66666666660",
                    "0.77777777770",
                    "0.88888888880",
                    "1.00000000000",
                  ]

AAC_directories = [
                    "0.00000000000",
                    "0.11111111110",
                    "0.22222222220",
                    "0.33333333330",
                    "0.44444444440",
                    "0.55555555550",
                    "0.66666666660",
                    "0.77777777770",
                    "0.88888888880",
                    "1.00000000000",
                  ]

GGI_list = []
AAC_list = []
energy_delta = []
energy = []
prev_index = 0

for i in AAC_directories:
    for j in GGI_directories:
        AAC_list.append(float(i))
        GGI_list.append(float(j))
        energy.append(ce.predict(parse_ATAT_strout(j + "/" + i + "/str.out")))

    k, b = fit_linear(GGI_list[prev_index],  energy[prev_index], 
                      GGI_list[-1],          energy[-1])

    predicted_line = [k * i + b for i in GGI_list]

    for i in range(prev_index, len(energy)):
        energy_delta.append((energy[i] - predicted_line[i]) * 4000 / 216)

    prev_index = len(energy)

X = np.array(GGI_list).reshape(10, 10)
Y = np.array(AAC_list).reshape(10, 10)
Z = np.array(energy_delta).reshape(10, 10)

plt.contourf(X, Y, Z, levels = 50, cmap = "RdBu")

cbar = plt.colorbar()
cbar.set_label("Energy delta")

plt.xlabel("GGI")
plt.ylabel("AAC")

plt.show()