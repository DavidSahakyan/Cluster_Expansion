import matplotlib.pyplot as plt
from helper_functions import *

file_to_read = "/home/david/Desktop/project/generated_data_files/SGC_data.txt"

temperatures_to_read = [300, 350, 400, 100, 150, 200]

for target_temperature in temperatures_to_read:
    
    GGI = []
    dmu = []
    E   = []
    
    with open(file_to_read) as f:
        line = f.readline()

        while line:
            words = line.split(" ")

            if float(words[0]) == target_temperature:   
                GGI.append(float(words[1]))
                E  .append(float(words[2]))
                dmu.append(float(words[3]))

            line = f.readline()

    if not len(GGI):
        print(f"WARNING! TRAGET TEMPERATURE {target_temperature} IS NOT FOUND\n")
        continue

    k, b = fit_linear(0, max(E), 1, min(E))
    lab = "T = " + str(target_temperature)

    plt.title("E")
    plt.scatter(GGI, E, label = lab)
    plt.legend()
    plt.show()
    
    plt.title("dmu")
    plt.scatter(GGI, dmu, label = lab)
    plt.legend()
    plt.show()