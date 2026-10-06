import matplotlib.pyplot as plt
from helper_functions import *
import re


#First is the target temperature, second is the Boltzmann constant, third is the integration temperature 
target_values_list = [
                       248.15
                      ,273.15
                      ,289.15
                      ,323.15 
                    ]

thermo_data_file = "generated_data_files/thermo_E_over_GGI.txt"
temperature_data_file = "generated_data_files/temperature_E_over_GGI.txt"

for i in range(len(target_values_list)):
    GGI_list = []
    free_energies = []
    fitted_values = []
    delta_E = []

    data = read_data(
                        thermo_data_file, 
                        target_values_list[i]
                    )
    
    GGI_list = data[0]
    free_energies = data[1]

    k, b = fit_linear(GGI_list[0], free_energies[0], GGI_list[-1], free_energies[-1])
    for j in GGI_list:
        fitted_values.append(k * j + b)

    for j in range(len(GGI_list)):
        delta_E.append((free_energies[j] - fitted_values[j]) * 4000 / 216)
    
    leg = "T: " + str(target_values_list[i]) + "K, T0: " + str(target_values_list[i]) + "K."
    plt.scatter(GGI_list, delta_E, label = leg)

plt.legend()
plt.title("Thermodynamic integration")
plt.show()

for i in range(len(target_values_list)):
    GGI_list = []
    free_energies = []
    fitted_values = []
    delta_E = []

    data = read_data(
                        temperature_data_file, 
                        target_values_list[i]
                    )
    GGI_list = data[0]
    free_energies = data[1]

    k, b = fit_linear(GGI_list[0], free_energies[0], GGI_list[-1], free_energies[-1])
    for j in GGI_list:
        fitted_values.append(k * j + b)

    for j in range(len(GGI_list)):
        delta_E.append((free_energies[j] - fitted_values[j]) * 4000 / 216)
    
    leg = "T: " + str(target_values_list[i]) + "K, T0: " + str(target_values_list[i]) + "K."
    
    plt.scatter(GGI_list, delta_E, label = leg, s = i + 1 * 50)
    plt.legend(loc = "best")

plt.title("Temperature integration")
plt.show()
