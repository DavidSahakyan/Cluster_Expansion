from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import VCSGCEnsemble
from mchammer.ensembles import CanonicalEnsemble
import numpy as np
from icet import ClusterSpace
from helper_functions import *
import matplotlib.pyplot as plt

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
phis = np.linspace(-2, 0.5, 40)

start_configuration = supercell.copy()
sublattices = cluster_space.get_sublattices(start_configuration)
file_to_write = "/home/david/Desktop/project/generated_data_files/VCSGC_data.txt"
figure_save_dir = "/home/david/Desktop/project/generated_data_files/VCSGC_figs/"

temperatures = range(225, 1500, 50)

for temperature in temperatures:
    GGI_1_obtained = False
    GGI_0_obtained = False
        
    GGI = []
    E = []
    psi = []
    should_be_delta_myu = []

    for phi in phis:

        mc = VCSGCEnsemble(
            structure=start_configuration,
            calculator=calc,
            temperature=temperature,
            ensemble_data_write_interval = 10,
            phis={'Ga': phi, 'Ag': 0},
            kappa=kappa,
            sublattice_probabilities=[0.0, 1.0, 0.0]
            )

        mc.run(1000)
        start_configuration = mc.structure.copy()
        print(mc.data_container)

        data_cutoff = int(0.2 * len(mc.data_container.get("Ga_count")))
        ga_number =   np.mean(mc.data_container.get("Ga_count") [data_cutoff:])
        in_number =   np.mean(mc.data_container.get("In_count") [data_cutoff:])

        print(np.mean(mc.data_container.get("Ag_count") [data_cutoff:]))
        if (ga_number / (ga_number + in_number)) == 1 and GGI_1_obtained:
            continue

        elif (ga_number / (ga_number + in_number)) == 0 and GGI_0_obtained:
            continue

        elif (ga_number / (ga_number + in_number)) == 1 and not GGI_1_obtained:
            GGI_1_obtained = True

        elif (ga_number / (ga_number + in_number)) == 0 and not GGI_0_obtained:
            GGI_0_obtained = True

        GGI.   append(ga_number / (ga_number + in_number))
        psi   .append(np.mean(mc.data_container.get("potential")[data_cutoff:]))
        should_be_delta_myu.append(np.mean(mc.data_container.get("free_energy_derivative_Ga")[data_cutoff:]))

        E.append(psi[-1] - temperature * k_B * kappa * len(supercell) * (GGI[-1] + phi / 2) ** 2)
    
    k, b = fit_linear(0, max(psi), 1, min(psi))
    for i in range(len(GGI)):
        psi[i] -= (k * GGI[i] + b)

    k, b = fit_linear(0, max(E), 1, min(E))
    for i in range(len(GGI)):
        E[i] -= (k * GGI[i] + b)

    # with open(file_to_write, "a") as f:

    #     # for i in range(len(GGI)):
    #     #     str_to_write = str(temperature) + " " + str(GGI[i]) + " " + str(psi[i]) + " " + \
    #     #                    str(should_be_delta_myu[i]) + " " + str(E[i])   +  " \n"
    #     #     f.write(str_to_write)

    # fig_name = figure_save_dir + "psi" + str(temperature)
    # lab = "T = " + str(temperature)
    # plt.title("PSI") 
    # plt.scatter(GGI, psi, label = lab)
    # plt.legend()
    # plt.savefig(fig_name)
    # plt.clf()

    # fig_name = figure_save_dir + "dmu" + str(temperature)
    # lab = "T = " + str(temperature)
    # plt.title("dmu")
    # plt.scatter(GGI, should_be_delta_myu, label = lab)
    # plt.legend()
    # plt.savefig(fig_name)
    # plt.clf()

    # fig_name = figure_save_dir + "E" + str(temperature)
    # lab = "T = " + str(temperature)
    # plt.title("E")
    # plt.scatter(GGI, E, label = lab)
    # plt.legend()
    # plt.savefig(fig_name)
    # plt.clf()
