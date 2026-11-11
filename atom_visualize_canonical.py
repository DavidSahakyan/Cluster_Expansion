from helper_functions import *
from numpy import mean

from ase import Atoms
from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import CanonicalEnsemble
from mchammer.free_energy_tools import get_free_energy_temperature_integration
from icet import ClusterSpace
from ase.visualize import view
from matplotlib.patches import Patch

import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

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

supercell = primitive.repeat((3, 3, 3))
calc = ClusterExpansionCalculator(supercell, cluster_expansion)

equilibration_steps = 100000

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

atoms = start_configuration[ga_sites]

colors = {'Ga': 'red', 'In': 'blue'}

mc = CanonicalEnsemble(
        structure = start_configuration,
        calculator = calc,
        temperature = T_max,
        boltzmann_constant = k_B,
        trajectory_write_interval = None,
        ensemble_data_write_interval = 10)

fig, ax = plt.subplots()

atoms = start_configuration[ga_sites].copy()
plot_atoms(
    atoms,
    ax,
    radii=[0.3 if atom.symbol == 'Ga' else 0.5 for atom in atoms],
    colors=[colors[atom.symbol] for atom in atoms],
    rotation='0x,45y,0z'
)

colors = {'Ga': 'red', 'In': 'blue'}

legend_elements = [
    Patch(facecolor=color, label=element)
    for element, color in colors.items()
]

ax.legend(handles=legend_elements, loc='lower right')
fig.savefig("generated_data_files/lattices_canonical/before_mc.png")

mc.run(equilibration_steps)

ax.clear()
atoms = mc.structure[ga_sites].copy()
plot_atoms(
    atoms,
    ax,
    radii=[0.3 if atom.symbol == 'Ga' else 0.5 for atom in atoms],
    colors=[colors[atom.symbol] for atom in atoms],
    rotation='0x,45y,0z'
)
from matplotlib.patches import Patch

colors = {'Ga': 'red', 'In': 'blue'}

legend_elements = [
    Patch(facecolor=color, label=element)
    for element, color in colors.items()
]

ax.legend(handles=legend_elements, loc='lower right')
fig.savefig("generated_data_files/lattices_canonical/after_mc.png")
