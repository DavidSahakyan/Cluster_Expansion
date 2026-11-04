from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import VCSGCEnsemble
import numpy as np
from icet import ClusterSpace
from helper_functions import *

ATAT_format_lattice_path = "initial_data_files/lat.in"
cell, positions, chemical_symbols = parse_ATAT_lat(ATAT_format_lattice_path)

initial_symbols = [i[0] for i in chemical_symbols]

primitive = Atoms(
                   symbols = initial_symbols,
                   scaled_positions = positions,
                   cell = cell,
                   pbc = True
                 )

cutoffs = [6, 4.5]

cluster_space = ClusterSpace(
                              primitive,
                              cutoffs = cutoffs,
                              chemical_symbols = chemical_symbols
                            )

cluster_expansion = ClusterExpansion.read("generated_data_files/cluster_expansion.ce")

supercell = primitive.repeat((5, 5, 5))
calc = ClusterExpansionCalculator(supercell, cluster_expansion)

k_B = 8.617333262e-5
kappa = 200

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


file_to_write   = "/home/david/Desktop/project/generated_data_files/VCSGC_data.txt"
figure_save_dir = "/home/david/Desktop/project/generated_data_files/VCSGC_figs/"
iter = 0

temperatures = [325]

for temperature in temperatures:
    phis = np.linspace(-2.5, 0.5, 300)

    structure_fwd = supercell.copy()
    structure_bck = supercell.copy()

    ggi1_obtained_fwd = False
    ggi0_obtained_fwd = False

    ggi1_obtained_bck = False
    ggi0_obtained_bck = False
    
    GGI_fwd = []
    E_fwd   = []
    E_deriv_fwd = []

    GGI_bck = []
    E_bck   = []
    E_deriv_bck = []

    for phi in phis:

        mc = VCSGCEnsemble(
            structure   = structure_fwd,
            calculator  = calc,
            temperature = temperature,
            ensemble_data_write_interval = 10,
            phis={'Ga': phi, 'Ag': 0},
            kappa = kappa,
            sublattice_probabilities = [0, 1.0, 0.0]
            )

        mc.run(150000)
        structure_fwd = mc.structure.copy()

        data_cutoff = int(0.7 * len(mc.data_container.get("Ga_count")))
        ga_number =   np.mean(mc.data_container.get("Ga_count") [data_cutoff:])
        in_number =   np.mean(mc.data_container.get("In_count") [data_cutoff:])

        ggi = ga_number / (ga_number + in_number)
        print(f"iter, ggi: {iter, ggi}")
        iter += 1
        
        GGI_fwd.    append(ggi)
        E_fwd  .    append(np.mean(mc.data_container.get("potential")[data_cutoff:]) / len(supercell))
        E_deriv_fwd.append(np.mean(mc.data_container.get("free_energy_derivative_Ga")[data_cutoff:])  * len(ga_sites) / len(supercell))

    for phi in phis[::-1]:

        mc = VCSGCEnsemble(
            structure   = structure_bck,
            calculator  = calc,
            temperature = temperature,
            ensemble_data_write_interval = 10,
            phis={'Ga': phi, 'Ag': 0},
            kappa = kappa,
            sublattice_probabilities = [0, 1.0, 0.0]
            )

        mc.run(150000)
        structure_bck = mc.structure.copy()

        data_cutoff = int(0.7 * len(mc.data_container.get("Ga_count")))
        ga_number =   np.mean(mc.data_container.get("Ga_count") [data_cutoff:])
        in_number =   np.mean(mc.data_container.get("In_count") [data_cutoff:])

        ggi = ga_number / (ga_number + in_number)
        print(f"iter, ggi: {iter} {ggi}")
        iter += 1
        
        GGI_bck.    append(ggi)
        E_bck  .    append(np.mean(mc.data_container.get("potential")[data_cutoff:]) / len(supercell))
        E_deriv_bck.append(np.mean(mc.data_container.get("free_energy_derivative_Ga")[data_cutoff:])  * len(ga_sites) / len(supercell))

    GGI = []
    E = []
    E_deriv = []

    for i in range(len(GGI_fwd)):
        GGI    .append((GGI_fwd[i] + GGI_bck[- i - 1]) / 2)
        E      .append((E_fwd[i] + E_bck[- i - 1]) / 2)
        E_deriv.append((E_deriv_fwd[i] + E_deriv_bck[- i - 1]) / 2)

    # plt.ylabel("E_derivative")
    # plt.xlabel("GGI")
    # plt.scatter(GGI_fwd, E_deriv_fwd, label = f"FWD, T = {temperature}")
    # plt.scatter(GGI_bck, E_deriv_bck, label = f"BCK, T = {temperature}")
    # plt.scatter(GGI_fwd, E_deriv,     label = f"AVG, T = {temperature}")
    # plt.legend()
    # plt.show()

    # plt.title("E")
    # plt.xlabel("GGI")
    # plt.scatter(GGI_fwd, E_fwd, label = f"FWD, T = {temperature}")
    # plt.scatter(GGI_bck, E_bck, label = f"BCK, T = {temperature}")
    # plt.scatter(GGI_fwd, E,     label = f"AVG, T = {temperature}")
    # plt.legend()
    # plt.show()
    
    with open(file_to_write, "a") as f:

        for i in range(len(GGI)):
            str_to_write = str(GGI[i]) + " " + str(temperature) + " " + \
                           str(E[i])   + " " + str(E_deriv[i]) + " \n"
        
            f.write(str_to_write)