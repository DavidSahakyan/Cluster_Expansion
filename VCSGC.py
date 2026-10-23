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

supercell = primitive.repeat((10, 10, 10))
calc = ClusterExpansionCalculator(supercell, cluster_expansion)

k_B = 8.617333262e-5
kappa = 200
phis = np.linspace(-1.8, 0, 50)

start_configuration = supercell.copy()
sublattices = cluster_space.get_sublattices(start_configuration)
file_to_write = "/home/david/Desktop/project/generated_data_files/VCSGC_data.txt"
figure_save_dir = "/home/david/Desktop/project/generated_data_files/VCSGC_figs/"

temperatures = [200]

for temperature in temperatures:
    GGI_1_obtained = False
    GGI_0_obtained = False
    structure = supercell.copy()

    std = []
    GGI = []
    E = []
    E_deriv = []

    for phi in phis:

        mc = VCSGCEnsemble(
            structure=structure,
            calculator=calc,
            temperature=temperature,
            ensemble_data_write_interval = 10,
            phis={'Ga': phi, 'Ag': 0},
            kappa=kappa,
            sublattice_probabilities=[0, 1.0, 0.0]
            )

        mc.run(10000)
        structure = mc.structure.copy()

        data_cutoff = int(0.7 * len(mc.data_container.get("Ga_count")))
        ga_number =   np.mean(mc.data_container.get("Ga_count") [data_cutoff:])
        in_number =   np.mean(mc.data_container.get("In_count") [data_cutoff:])

        ggi = ga_number / (ga_number + in_number)
        print(ggi)
        print(np.std(mc.data_container.get("free_energy_derivative_Ga") [data_cutoff:]))
        print(np.std(mc.data_container.get("Ga_count") [data_cutoff:]))

        if (ggi) <= 0.2: #Change to 0 if no data cutoff needed
            continue
        
        elif (ggi) >= 0.95: #Change to 1 if no data cutoff needed
            continue
        
        elif (ggi) == 1 and GGI_1_obtained:
            continue

        elif (ggi) == 0 and GGI_0_obtained:
            continue

        elif (ggi) == 1 and not GGI_1_obtained:
            GGI_1_obtained = True

        elif (ggi) == 0 and not GGI_0_obtained:
            GGI_0_obtained = True
        
        GGI.    append(ggi)
        E  .    append(np.mean(mc.data_container.get("potential")[data_cutoff:]))
        E_deriv.append(np.mean(mc.data_container.get("free_energy_derivative_Ga")[data_cutoff:]))

    k, b = fit_linear(0, max(E), 1, min(E))
    for i in range(len(GGI)):
        E[i] -= (k * GGI[i] + b)

    with open(file_to_write, "a") as f:

        for i in range(len(GGI)):
            str_to_write = str(temperature) + " " + str(GGI[i]) + " " + \
                            str(E_deriv[i]) + " " + str(E[i]) +  " \n"
            f.write(str_to_write)

    fig_name = figure_save_dir + "Kappa" + str(kappa)
    lab = "Kappa = " + str(kappa)
    plt.title("dmu")
    plt.scatter(GGI, E_deriv, label = lab)
    plt.legend()
    plt.savefig(fig_name)
    plt.clf()