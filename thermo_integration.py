from helper_functions import *

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
target_t = 323.15 
free_energies = []
free_energies_per_atom = []
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

n_integration_steps = 400000
n_equilibration_steps = 1000
temperature_max = 324.3 
temperature_min = 322 
temperature_max_plot_limit = 323.15 
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


start_text = "START OF THE DATA SET, T = " + str(target_t) + " K, k_B = " + str(k_B) + " eV\n"
with open("generated_data_files/E_over_GGI.txt", "a") as f:
        f.write("==============================================\n")
        f.write(start_text)
        f.write("==============================================\n")

for i in cu_sites:
    start_configuration[i].symbol = "Cu"

for i in se_sites:
    start_configuration[i].symbol = "Se"

max_Ga_In_number = 54

for Ga_number in range(1, max_Ga_In_number):
    for i in ga_sites[0 : Ga_number]:
        start_configuration[i].symbol = 'Ga'

    for i in ga_sites[Ga_number : len(ga_sites)]:
        start_configuration[i].symbol = 'In'

    mc = CanonicalEnsemble(
            structure=start_configuration,
            calculator=calc,
            temperature=temperature_max,
            boltzmann_constant=k_B ,
            trajectory_write_interval=None,
            ensemble_data_write_interval=200)

    mc.run(n_equilibration_steps)


    mc = ThermodynamicIntegrationEnsemble(
        structure=mc.structure, calculator=calc,
        temperature=temperature_min,
        forward=True,
        ensemble_data_write_interval=1,
        boltzmann_constant=k_B ,
        n_steps=n_integration_steps)
    mc.run()
    data_container = mc.data_container

    (forward_temps, free_energy_integration_forward) = \
            get_free_energy_thermodynamic_integration(data_container, cluster_space,
                                                    forward=True,
                                                    max_temperature=temperature_max_plot_limit,
                                                    boltzmann_constant=k_B )

    mc = CanonicalEnsemble(
            structure=mc.structure,
            calculator=calc,
            temperature=temperature_min,
            boltzmann_constant=k_B ,
            trajectory_write_interval=None,
            ensemble_data_write_interval=200)
    mc.run(n_equilibration_steps)

    mc = ThermodynamicIntegrationEnsemble(
        structure=mc.structure, calculator=calc,
        temperature=temperature_min,
        forward=False,
        ensemble_data_write_interval=1,
        boltzmann_constant=k_B ,
        n_steps=n_integration_steps)
    mc.run()
    data_container = mc.data_container

    (backward_temp, free_energy_integration_backward) = \
            get_free_energy_thermodynamic_integration(data_container, cluster_space,
                                                    forward=False,
                                                    max_temperature=temperature_max_plot_limit,
                                                    boltzmann_constant=k_B )

    free_energy_integration_average = 0.5 * (free_energy_integration_forward +
                                            free_energy_integration_backward)

    i = np.argmin(
        np.abs(forward_temps - target_t)
    )

    free_energies.append(free_energy_integration_average[i])
    free_energies_per_atom.append(free_energy_integration_average[i] / len(supercell))
    GGI.append(Ga_number / 54)

    string_to_write = str(Ga_number / 54)  + " " + str(free_energies_per_atom[-1]) + '\n'

    with open("generated_data_files/E_over_GGI.txt", "a") as f:
        f.write(string_to_write)

    print(f"GGI: {GGI[-1]}, delta_e: {free_energies_per_atom[-1]}\n")