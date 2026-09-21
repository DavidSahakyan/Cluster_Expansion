import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *
from icet import ClusterExpansion
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

ce = ClusterExpansion.read("generated_data_files/cluster_expansion.ce")

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

GGI_list_for_fit = [] 
AAC_list_for_fit = [] 

predicted_energy_for_fit = []
predicted_energy = []
predicted_energy_delta = []

for i in AAC_directories:
    for j in GGI_directories:

        AAC_list.append(float(i))
        GGI_list.append(float(j))
        predicted_energy.append(ce.predict(parse_ATAT_strout("initial_data_files/" + j + "/" + i + "/str.out")) * 4000 / 216)

        if i == "0.00000000000" and j == "0.00000000000":
            AAC_list_for_fit.append(float(i))
            GGI_list_for_fit.append(float(j))
            predicted_energy_for_fit.append(ce.predict(parse_ATAT_strout("initial_data_files/" + j + "/" + i + "/str.out")) * 4000 / 216)                                    
        elif i == "1.00000000000" and j == "0.00000000000":
            AAC_list_for_fit.append(float(i))
            GGI_list_for_fit.append(float(j))
            predicted_energy_for_fit.append(ce.predict(parse_ATAT_strout("initial_data_files/" + j + "/" + i + "/str.out")) * 4000 / 216)                                    
        elif i == "0.00000000000" and j == "1.00000000000":
            AAC_list_for_fit.append(float(i))
            GGI_list_for_fit.append(float(j))
            predicted_energy_for_fit.append(ce.predict(parse_ATAT_strout("initial_data_files/" + j + "/" + i + "/str.out")) * 4000 / 216)                                    
        elif i == "1.00000000000" and j == "1.00000000000":
            AAC_list_for_fit.append(float(i))
            GGI_list_for_fit.append(float(j))
            predicted_energy_for_fit.append(ce.predict(parse_ATAT_strout("initial_data_files/" + j + "/" + i + "/str.out")) * 4000 / 216)                                    
                
prediction = [43.8, -146.8, 60, -1781.9] 
A, B, C, D = curve_fit(surface_fit_function, (GGI_list_for_fit, AAC_list_for_fit), predicted_energy_for_fit, prediction)[0]
for i in range(len(GGI_list)):
    predicted_energy_delta.append(predicted_energy[i] - 
                           surface_fit_function((GGI_list[i], AAC_list[i]), A, B, C, D)) 
print(f"Predicted coefficients: A: {A}, B: {B}, C: {C}, D: {D}")

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")

ax.set_xlabel("GGI")
ax.set_ylabel("AAC")
ax.set_zlabel("Energy")

X = np.array(GGI_list).reshape(10, 10)
Y = np.array(AAC_list).reshape(10, 10)
Z = np.array(predicted_energy_delta).reshape(10, 10)
ax.plot_surface(X, Y, Z)
plt.show()

plt.contourf(X, Y, Z, levels = 50, cmap = "RdBu")

cbar = plt.colorbar()
cbar.set_label("Energy delta")

plt.xlabel("GGI")
plt.ylabel("AAC")

plt.show()