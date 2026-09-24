import matplotlib.pyplot as plt
from helper_functions import *
import re


#First is the temperature, second is the Boltzmann constant
target_values_list = [
                        [323.15, 8.617333262e-05], 
                        [289.15, 8.617333262e-05], 
                        [273.15, 8.617333262e-05], 
                        [248.15, 8.617333262e-05]
                     ]

data_file = "generated_data_files/E_over_GGI.txt"

for target_t, target_k_B in target_values_list:
    GGI_list = []
    free_energies = []
    fitted_values = []
    delta_E = []

    data = read_target_data(data_file, target_t, target_k_B)
    GGI_list = data[0]
    free_energies = data[1]

    k, b = fit_linear(GGI_list[0], free_energies[0], GGI_list[-1], free_energies[-1])
    for i in GGI_list:
        fitted_values.append(k * i + b)

    for i in range(len(GGI_list)):
        delta_E.append((free_energies[i] - fitted_values[i]) * 4000 / 216)

    leg = "T: " + str(target_t) + "K"
    plt.scatter(GGI_list, delta_E, label = leg)

plt.legend()
plt.show()