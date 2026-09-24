from icet import ClusterExpansion
from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

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
                    # "0.11111111110",
                    # "0.22222222220",
                    # "0.33333333330",
                    # "0.44444444440",
                    # "0.55555555550",
                    # "0.66666666660",
                    # "0.77777777770",
                    # "0.88888888880",
                    # "1.00000000000",
                  ]

clusted_expansion_file_path = "generated_data_files/cluster_expansion.ce"

final_directories         = ["initial_data_files/" + i + "/" + j for i in GGI_directories for j in AAC_directories]
files_for_prediction      = [i + "/str.out" for i in final_directories]
files_for_GGI_CALCULATION = [i + "/CONTCAR" for i in final_directories]

ce = ClusterExpansion.read(clusted_expansion_file_path)

actual_energy_list = []
number_of_atoms    = 0

for i in final_directories:  
  number_of_atoms = get_struct_and_energy(i)[2]
  actual_energy_list.append(get_struct_and_energy(i)[1])

predicted_energy_list = []

for str_file in files_for_prediction:
    atoms, number_of_atoms = parse_ATAT_strout(str_file)
    predicted_energy_list.append(ce.predict(atoms) * number_of_atoms) # THIS IS DONE TO BE CONSISTENT WITH THE DATA IN THE PAPER, 
                                                                      # THAT IS E/Formula

GGI_list = [GGI_AAC(contcar_file)[0] for contcar_file in files_for_GGI_CALCULATION]

plt.xlabel("GGI")
plt.ylabel("E")
plt.scatter(GGI_list, predicted_energy_list, color = "red",  label = "predicted")
plt.scatter(GGI_list, actual_energy_list,    color = "blue", label = "actual", s = 50)

k, b = fit_linear(GGI_list[0],  predicted_energy_list[0], 
                  GGI_list[-1], predicted_energy_list[-1])
predicted_line = [k * i + b for i in GGI_list]
plt.plot(GGI_list, predicted_line, color = "red")

k, b = fit_linear(GGI_list[0],  actual_energy_list[0], 
                  GGI_list[-1], actual_energy_list[-1])
actual_line = [k * i + b for i in GGI_list]

print(k)
print(b)

plt.plot(GGI_list, actual_line, color = "blue")
plt.legend()
plt.show()

predicted_delta_E = []
actual_delta_E    = []
for i in range(len(GGI_list)):
    predicted_delta_E.append(predicted_energy_list[i] - predicted_line[i])
    actual_delta_E   .append(   actual_energy_list[i] -    actual_line[i])

actual_delta_E    = [4000 * i / number_of_atoms for i in actual_delta_E]
predicted_delta_E = [4000 * i / number_of_atoms for i in predicted_delta_E]

plt.scatter(GGI_list, predicted_delta_E, color = "red", s = 50, label = "predicted")
plt.scatter(GGI_list, actual_delta_E,    color = "blue", label = "actual")

omega = curve_fit(linear_fit_function, GGI_list, predicted_delta_E)[0]
x = np.linspace(0, 1, 20)
fit = [omega * i * (1 - i) for i in x]
plt.plot(x, fit, label = f"predicted data fit. Ω = {omega}", color = "red")
print(f"predicted omega: {omega[0]}")

omega = curve_fit(linear_fit_function, GGI_list, actual_delta_E)[0]
fit = [omega * i * (1 - i) for i in x]
x = np.linspace(0, 1, 20)
plt.plot(x, fit, label = f"actual data fit. Ω = {omega}", color = "blue")
print(f"actual omega: {omega[0]}")

plt.legend()
plt.show()
