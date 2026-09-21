import matplotlib.pyplot as plt
from helper_functions import *
import re

GGI_list = []
free_energies = []
fitted_values = []
delta_E = []

target_t = 323.15
target_k_B = 8.617333262e-05
data_file = "generated_data_files/E_over_GGI.txt"

data = read_target_data(data_file, target_t, target_k_B)
GGI_list = data[0]
free_energies = data[1]

print(len(free_energies))

k, b = fit_linear(GGI_list[0], free_energies[0], GGI_list[-1], free_energies[-1])
for i in GGI_list:
    fitted_values.append(k * i + b)

plt.plot(GGI_list, fitted_values, color = "red", linestyle = "--", label = "fitted")
plt.scatter(GGI_list, free_energies, color = "blue", label = "actual data")
plt.legend()
plt.show()

for i in range(len(GGI_list)):
    delta_E.append((free_energies[i] - fitted_values[i]) * 4000 / 216)

plt.plot(GGI_list, delta_E)
plt.show()