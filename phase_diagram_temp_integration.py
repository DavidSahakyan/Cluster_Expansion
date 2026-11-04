from helper_functions import *
import numpy as np

Ts = range(150, 500, 25)
# Ts = [150]

temperature_data_file = "generated_data_files/temperature_E_over_GGI.txt"
thermo_data_file = "generated_data_files/thermo_E_over_GGI.txt"
VCSGC_data_file = "/home/david/Desktop/project/generated_data_files/VCSGC_data.txt"

T_free_energy = get_canonical_free_energy_fits_over_T(temperature_data_file, Ts, False)
T_GGIs_temp   = analyze_free_energy_fits_over_T(T_free_energy, False)

T_free_energy = get_canonical_free_energy_fits_over_T(thermo_data_file, Ts, False)
T_GGIs_thermo = analyze_free_energy_fits_over_T(T_free_energy, False)

T_free_energy = get_VCSGC_free_energy_fits_over_T(VCSGC_data_file, Ts, False, 175)
T_GGIs_VCSGC  = analyze_free_energy_fits_over_T(T_free_energy, False, 175)

plot_phase_diagrams([
                        T_GGIs_thermo, 
                        T_GGIs_temp, 
                        T_GGIs_VCSGC
                    ], 
                    [
                        "Thermo integration", 
                        "Temp integration", 
                        "VCSGC"
                    ])

