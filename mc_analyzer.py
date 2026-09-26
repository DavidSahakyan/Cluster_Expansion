import matplotlib.pyplot as plt
from helper_functions import *
import re


#First is the target temperature, second is the Boltzmann constant, third is the integration temperature 
target_values_list = [
                        [323.15, 8.617333262e-05, 100.15], 
                        [289.15, 8.617333262e-05, 100.15], 
                        [273.15, 8.617333262e-05, 100.15], 
                        [248.15, 8.617333262e-05, 100.15],
                        [323.15, 8.617333262e-05, 200.15], 
                        [289.15, 8.617333262e-05, 200.15], 
                        [273.15, 8.617333262e-05, 200.15], 
                        [248.15, 8.617333262e-05, 200.15],
                        [323.15, 8.617333262e-05, 300.15], 
                        [289.15, 8.617333262e-05, 300.15], 
                        [273.15, 8.617333262e-05, 300.15], 
                        [248.15, 8.617333262e-05, 300.15],
                        [323.15, 8.617333262e-05, 500.15], 
                        [289.15, 8.617333262e-05, 500.15], 
                        [273.15, 8.617333262e-05, 500.15], 
                        [248.15, 8.617333262e-05, 500.15],
                        [323.15, 8.617333262e-05, 600.15], 
                        [289.15, 8.617333262e-05, 600.15], 
                        [273.15, 8.617333262e-05, 600.15], 
                        [248.15, 8.617333262e-05, 600.15],
                        [323.15, 8.617333262e-05, 700.15], 
                        [289.15, 8.617333262e-05, 700.15], 
                        [273.15, 8.617333262e-05, 700.15], 
                        [248.15, 8.617333262e-05, 700.15],
                     ]

data_file = "generated_data_files/E_over_GGI.txt"

fig = plt.figure()
gs = fig.add_gridspec(3, 2)
axs = gs.subplots(sharex=True)

plot_numbers = [
                [0, 0], 
                [1, 0], 
                [2, 0], 
                [0, 1],
                [1, 1], 
                [2, 1]
                ]

for i in range(int(len(target_values_list))):
    GGI_list = []
    free_energies = []
    fitted_values = []
    delta_E = []

    data = read_target_data(
                            data_file, 
                            target_values_list[i][0], 
                            target_values_list[i][1], 
                            target_values_list[i][2]
                            )
    GGI_list = data[0]
    free_energies = data[1]

    k, b = fit_linear(GGI_list[0], free_energies[0], GGI_list[-1], free_energies[-1])
    for j in GGI_list:
        fitted_values.append(k * j + b)

    for j in range(len(GGI_list)):
        delta_E.append((free_energies[j] - fitted_values[j]) * 4000 / 216)
    
    leg = "T: " + str(target_values_list[i][0]) + "K, T0: " + str(target_values_list[i][2]) + "K."
    plot_num = plot_numbers[int(i / 4)]
    
    axs[plot_num[0], plot_num[1]].scatter(GGI_list, delta_E, label = leg)
    axs[plot_num[0], plot_num[1]].legend(loc = "best")

plt.show()