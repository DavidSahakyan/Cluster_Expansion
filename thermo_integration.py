from helper_functions import *
from numpy import mean
import matplotlib.pyplot as plt

from ase import Atoms
from icet import ClusterSpace, StructureContainer
from trainstation import Optimizer
from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import CanonicalEnsemble
from mchammer.free_energy_tools import get_free_energy_thermodynamic_integration
from mchammer.ensembles import ThermodynamicIntegrationEnsemble

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

cutoffs = [6, 4.5]
target_t_list = [
                    # 323.15, 
                    # 289.15, 
                    # 273.15, 
                    248.15
                ]

data_from_thermal = []
data_from_thermal_per_atom = []

GGI = []

ATAT_format_lattice_path = "initial_data_files/lat.in"

cell, positions, chemical_symbols = parse_ATAT_lat(ATAT_format_lattice_path)

initial_symbols = [i[0] for i in chemical_symbols]

primitive = Atoms(
                   symbols = initial_symbols,
                   scaled_positions = positions,
                   cell = cell,
                   pbc = True
                 )

cluster_space = ClusterSpace(
                              primitive,
                              cutoffs = cutoffs,
                              chemical_symbols = chemical_symbols
                            )

cluster_expansion = ClusterExpansion.read("generated_data_files/cluster_expansion.ce")

supercell = primitive.repeat((3, 3, 3))
calc = ClusterExpansionCalculator(supercell, cluster_expansion)

mc_n_integration_steps = 800000
n_equilibration_steps = 20000
thermo_n_integration_steps = 80000

temperature_max = 373.15 #100 C     
temperature_min_list = [
                        # 100.15,
                        # 200.15,
                        # 300.15,
                        # 500.15,
                        # 600.15,
                        700.15,
                       ] 
temperature_max_plot_limit = 800

k_B = 8.617333262e-5

start_configuration = supercell.copy()
sublattices = cluster_space.get_sublattices(start_configuration)
Ga_condition = False
Cu_condition = False 
Se_condition = False

for sublattice in sublattices:
    if 'Ga' in sublattice.chemical_symbols:
        ga_sites = sublattice.indices
        if Cu_condition and Se_condition:
            break
        Ga_condition = True     

    elif 'Cu' in sublattice.chemical_symbols:
        cu_sites = sublattice.indices
        if Ga_condition and Se_condition:
             break
        Cu_condition = True

    elif 'Se' in sublattice.chemical_symbols:
        se_sites = sublattice.indices
        if Ga_condition and Cu_condition:
            break
        Se_condition = True


for i in ga_sites:
    start_configuration[i].symbol = "In"

for i in cu_sites:
    start_configuration[i].symbol = "Cu"

for i in se_sites:
    start_configuration[i].symbol = "Se"

max_Ga_In_number = int(len(supercell) / 4)

for temperature_min in temperature_min_list:
    for target_t in target_t_list: 
        potential_from_mc = []
        GGI = []

        start_text = "START OF THE DATA SET, T = " + str(target_t) + " K, k_B = " + str(k_B) + " eV, T0 = " + str(temperature_min) + " K.\n"
        with open("generated_data_files/E_over_GGI.txt", "a") as f:
            f.write("==============================================\n")
            f.write(start_text)
            f.write("==============================================\n")

        for Ga_number in range(0, max_Ga_In_number + 1):
            for i in ga_sites[0 : Ga_number]:
                start_configuration[i].symbol = 'Ga'

            for i in ga_sites[Ga_number : len(ga_sites)]:
                start_configuration[i].symbol = 'In'

            if Ga_number == 0:
                with open("generated_data_files/E_over_GGI.txt", "a") as f:
                    str_to_write = "0.0000000000000000 " + str(cluster_expansion.predict(start_configuration)) + '\n'
                    f.write(str_to_write)
                    GGI.append(0)
                    potential_from_mc.append(cluster_expansion.predict(start_configuration))
                    continue

            if Ga_number == max_Ga_In_number:
                with open("generated_data_files/E_over_GGI.txt", "a") as f:
                    str_to_write = "1.0000000000000000 " + str(cluster_expansion.predict(start_configuration)) + '\n'
                    f.write(str_to_write)
                    GGI.append(1)
                    potential_from_mc.append(cluster_expansion.predict(start_configuration))
                    continue
                
            mc = CanonicalEnsemble(
                    structure = start_configuration,
                    calculator = calc,
                    temperature = temperature_max,
                    boltzmann_constant = k_B,
                    trajectory_write_interval = None,
                    ensemble_data_write_interval = 200
                    )

            mc.run(n_equilibration_steps)

            n_before = len(mc.data_container.get('potential'))

            mc.run(mc_n_integration_steps)

            potential_from_mc.append(mean(mc.data_container.get('potential')[n_before:]) / len(supercell))
            GGI.append(Ga_number / max_Ga_In_number)

            mc = CanonicalEnsemble(
                    structure = start_configuration,
                    calculator = calc,
                    temperature = temperature_max,
                    boltzmann_constant = k_B,
                    trajectory_write_interval = None,
                    ensemble_data_write_interval = 200
                    )

            mc.run(n_equilibration_steps)

            mc = ThermodynamicIntegrationEnsemble(
                structure = mc.structure, calculator=calc,
                temperature = temperature_min,
                forward = True,
                ensemble_data_write_interval = 1,
                boltzmann_constant = k_B ,
                n_steps = thermo_n_integration_steps)
            mc.run()
            data_container = mc.data_container

            (_, free_energy_integration_forward) = \
                    get_free_energy_thermodynamic_integration(data_container, cluster_space,
                                                            forward = True,
                                                            max_temperature = temperature_max_plot_limit,
                                                            boltzmann_constant = k_B )

            mc = CanonicalEnsemble(
                    structure = mc.structure,
                    calculator = calc,
                    temperature = temperature_min,
                    boltzmann_constant = k_B ,
                    trajectory_write_interval = None,
                    ensemble_data_write_interval = 200)
            mc.run(n_equilibration_steps)

            mc = ThermodynamicIntegrationEnsemble(
                structure = mc.structure,
                calculator = calc,
                temperature = temperature_min,
                forward = False,
                ensemble_data_write_interval = 1,
                boltzmann_constant = k_B,
                n_steps = thermo_n_integration_steps)
            
            mc.run()
            data_container = mc.data_container

            (temperatures_integration, free_energy_integration_backward) = \
                    get_free_energy_thermodynamic_integration(data_container, cluster_space,
                                                            forward = False,
                                                            max_temperature = temperature_max_plot_limit,
                                                            boltzmann_constant = k_B )

            free_energy_integration_average = 0.5 * (free_energy_integration_forward +
                                                    free_energy_integration_backward)


            i = np.argmin(
                np.abs(temperatures_integration - target_t)
            )

            data_from_thermal_per_atom.append(free_energy_integration_average[i] / len(supercell))

            string_to_write = str(Ga_number / max_Ga_In_number)  + " " + str(data_from_thermal_per_atom[-1]) + '\n'

            with open("generated_data_files/E_over_GGI.txt", "a") as f:
                f.write(string_to_write)

            print(f"GGI: {GGI[-1]}, delta_e: {data_from_thermal_per_atom[-1]}\n")

        # delta_e_fitted = []
        # fitted_values = []

        # k, b = fit_linear(GGI[0], potential_from_mc[0], GGI[-1], potential_from_mc[-1])
        # for i in GGI:
        #     fitted_values.append(k * i + b)

        # for i in range(len(GGI)):
        #     delta_e_fitted.append(potential_from_mc[i] - fitted_values[i])

        # lab = "Fitted Values, T:" + str(target_t)