from helper_functions import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

temps_GGIs = {}
target_ts = range(175, 600, 25)
file_to_read = "/home/david/Desktop/project/generated_data_files/VCSGC_data.txt"

for target_t in target_ts:
    GGI = []
    dmu = []
    energy = []
    with open(file_to_read) as f:
        line = f.readline()

        while line:
            
            values = line.split(" ")
            if float(values[0]) != target_t: 
                line = f.readline()
                continue
            
            if (float(values[1]) != 0 and  
                float(values[1]) != 1):
                
                GGI.   append(float(values[1]))
                energy.append(float(values[2]))
                dmu.   append(float(values[3]))

            line = f.readline()

    coeffs = np.polyfit(GGI, dmu, 3)[::-1]
    
    dmu_fit_polinom = np.polynomial.Polynomial(coeffs)

    ggi_fit = np.linspace(0, 1, 50)
    dmu_fit = [dmu_fit_polinom(i) for i in ggi_fit]

    lab = "T = " + str(target_t) + " K"
    plt.scatter(GGI, dmu, label = lab)
    err = [abs(i) * 0.05 for i in dmu]
    plt.errorbar(GGI, dmu, yerr = err, linestyle = ' ')
    plt.plot(ggi_fit, dmu_fit, linestyle = "--", color = "red")
    plt.legend()
    plt.show()

    temps_GGIs[target_t] = find_maxwell_points(dmu_fit_polinom)     
    print(temps_GGIs)

ts = []
GGIs = []
for temperature in temps_GGIs.keys():

    for i in range(len(temps_GGIs[temperature])):
        ts.append(temperature)

    for GGI in temps_GGIs[temperature]:
        GGIs.append(GGI)

plt.ylabel("T")
plt.xlabel("GGI")
plt.scatter(GGIs, ts)
plt.show()
