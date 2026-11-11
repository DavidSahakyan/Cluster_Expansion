from helper_functions import *
from numpy import mean

from ase import Atoms
from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import CanonicalEnsemble
from mchammer.free_energy_tools import get_free_energy_temperature_integration
from icet import ClusterSpace
from ase.visualize import view

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
Ts = range(50, 800, 25)

temperature_max_plot_limit = 800

k_B = 8.617333262e-5

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

supercell = primitive.repeat((5, 5, 5))
calc = ClusterExpansionCalculator(supercell, cluster_expansion)

equilibration_steps = 50000

T_max = 1500
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

Ga_In_number =  int(max_Ga_In_number / 2)

for i in ga_sites[0 : Ga_In_number]:
    start_configuration[i].symbol = 'Ga'

for i in ga_sites[Ga_In_number : len(ga_sites)]:
    start_configuration[i].symbol = 'In'


mc = CanonicalEnsemble(
        structure = start_configuration,
        calculator = calc,
        temperature = T_max,
        boltzmann_constant = k_B,
        trajectory_write_interval = None,
        ensemble_data_write_interval = 10)

mc.run(equilibration_steps)
energies1 = mc.data_container.get("potential")

mc = CanonicalEnsemble(
        structure = start_configuration,
        calculator = calc,
        temperature = T_max,
        boltzmann_constant = k_B,
        trajectory_write_interval = None,
        ensemble_data_write_interval = 10)

mc.run(equilibration_steps)
energies2 = mc.data_container.get("potential")

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm

# energies1 = np.array(energies1)
# energies1 = energies1 / len(supercell)

# energies2 = np.array(energies2)
# energies2 = energies2 / len(supercell)

# # Remove equilibration data if needed
# # energies1 = energies1[1000:]
# # energies2 = energies2[1000:]

# # Fit Gaussian distributions
# mu1, sigma1 = norm.fit(energies1)
# mu2, sigma2 = norm.fit(energies2)

# # Common histogram bins
# bins = np.linspace(
#     min(energies1.min(), energies2.min()),
#     max(energies1.max(), energies2.max()),
#     60
# )

# x = np.linspace(bins[0], bins[-1], 1000)

# plt.figure(figsize=(10, 6))

# # Histograms
# plt.hist(
#     energies1,
#     bins=bins,
#     density=True,
#     alpha=0.4,
#     color="blue",
#     label="Simulation 1"
# )

# plt.hist(
#     energies2,
#     bins=bins,
#     density=True,
#     alpha=0.4,
#     color="orange",
#     label="Simulation 2"
# )

# # Gaussian fits
# plt.plot(
#     x,
#     norm.pdf(x, mu1, sigma1),
#     color="blue",
#     linewidth=2,
#     label=f"Fit 1: μ={mu1:.4f}, σ={sigma1:.4f}"
# )

# plt.plot(
#     x,
#     norm.pdf(x, mu2, sigma2),
#     color="red",
#     linewidth=2,
#     label=f"Fit 2: μ={mu2:.4f}, σ={sigma2:.4f}"
# )

# plt.xlabel("Gibbs energy (meV)", fontsize = 25)
# plt.ylabel("Probability density",    fontsize = 25)
# plt.title(f"Canonical Monte Carlo Energy Distribution at T = {T_max} K", fontsize = 25)

# plt.tick_params(axis='both', which='major', labelsize=20)
# plt.legend(fontsize = 20)
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.xlim(-3.830, -3.8282)
# plt.show()
