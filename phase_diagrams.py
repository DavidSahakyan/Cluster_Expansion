from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import SemiGrandCanonicalEnsemble
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

start_configuration = supercell.copy()
sublattices = cluster_space.get_sublattices(start_configuration)
Ga_condition   = False
Cu_condition   = False 
Se_condition   = False
GGI_1_obtained = False

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

temperature_data = {}
for temperature in [700]:
    # Evolve configuration through the entire composition range
    GGI = []
    energy = []
    chem_pot = []
    for dmu in np.arange(-1.04, 1.04, 0.002):

        mc = SemiGrandCanonicalEnsemble(
            structure=start_configuration,
            calculator=calc,
            temperature=temperature,
            ensemble_data_write_interval = 10,
            chemical_potentials = {
                    "In": 0.0,
                    "Ga": dmu, 
                },
                sublattice_probabilities=[0.0, 1.0, 0.0]
            )

        mc.run(10000)
        structure = mc.structure


        data_cutoff = int(0.2 * len(mc.data_container.get("Ga_count")))
        ga_number =   np.mean(mc.data_container.get("Ga_count") [data_cutoff:])
        in_number =   np.mean(mc.data_container.get("In_count") [data_cutoff:])

        if(ga_number / (ga_number + in_number) >= 1 and \
           GGI_1_obtained == False):
           GGI_1_obtained = True
        elif(ga_number / (ga_number + in_number) >= 1 and \
             GGI_1_obtained == True):
            continue

        GGI.   append(ga_number / (ga_number + in_number))
        energy.append(np.mean(mc.data_container.get("potential")[data_cutoff:]))
        chem_pot.append(dmu)
        
        energy[-1] = energy[-1] + chem_pot[-1] * GGI[-1]

    k, b = fit_linear(0, max(energy), 1, min(energy))
    for i in range(len(GGI)):
        energy[i] -= (k * GGI[i] + b)

    print(min(energy))
    temperature_data[temperature] = [GGI, energy, chem_pot]

for i in temperature_data.keys():
    lab = "T = " + str(i)
    plt.scatter(temperature_data.get(i)[0], 
                temperature_data.get(i)[1], 
                label = lab)

plt.legend()
plt.show()

for i in temperature_data.keys():
    lab = "T = " + str(i)
    plt.scatter(temperature_data.get(i)[0], 
                temperature_data.get(i)[2], 
                label = lab)


plt.legend()
plt.show()
