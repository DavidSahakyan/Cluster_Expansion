from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import SemiGrandCanonicalEnsemble
from icet import ClusterSpace
from helper_functions import *
import matplotlib.pyplot as plt

from ase import Atoms
from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from icet import ClusterSpace
from ase.visualize import view

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms


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
filename = "/home/david/Desktop/project/generated_data_files/SGC_data.txt"


start_configuration = supercell.copy()
sublattices = cluster_space.get_sublattices(start_configuration)
Ga_condition   = False
Cu_condition   = False 
Se_condition   = False

colors = {'Ga': 'red', 'In': 'blue'}

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

structure = start_configuration.copy()

mc = SemiGrandCanonicalEnsemble(
    structure=structure,
    calculator=ClusterExpansionCalculator(
        structure, cluster_expansion
    ),
    temperature = 475,
    ensemble_data_write_interval=10,
    chemical_potentials={
        "In": 0.0,
        "Ga": float(-0.55),
    },
    sublattice_probabilities=[0.0, 1.0, 0.0]
)

mc.run(25000)
structure = mc.structure.copy()

fig, ax = plt.subplots()

atoms = structure[ga_sites].copy()
ax.text(x = 0, y = -1, s = r"$\Delta \mu = -0.55, T = 475K$")
plot_atoms(
    atoms,
    ax,
    radii=[0.3 if atom.symbol == 'Ga' else 0.5 for atom in atoms],
    colors=[colors[atom.symbol] for atom in atoms],
    rotation='0x,45y,0z'
)

legend_elements = [
    Patch(facecolor=color, label=element)
    for element, color in colors.items()
]

ax.legend(handles=legend_elements, loc='lower right')
fig.savefig("generated_data_files/lattices_SGC/before_mc.png")

ax.clear()

mc = SemiGrandCanonicalEnsemble(
    structure=start_configuration,
    calculator=ClusterExpansionCalculator(
        structure, cluster_expansion
    ),
    temperature = 475,
    ensemble_data_write_interval=10,
    chemical_potentials={
        "In": 0.0,
        "Ga": float(-0.52),
    },
    sublattice_probabilities=[0.0, 1.0, 0.0]
)

mc.run(25000)
structure = mc.structure.copy()
atoms = structure[ga_sites]
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
ax.text(x = 0, y = -1, s = r"$\Delta \mu = -0.52, T = 475K$")

ax.legend(handles=legend_elements, loc='lower right')
fig.savefig("generated_data_files/lattices_SGC/after_mc.png")
