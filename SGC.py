from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import SemiGrandCanonicalEnsemble
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
filename = "/home/david/Desktop/project/generated_data_files/SGC_data.txt"


start_configuration = supercell.copy()
sublattices = cluster_space.get_sublattices(start_configuration)
Ga_condition   = False
Cu_condition   = False 
Se_condition   = False

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

# for temperature in range(150, 500, 25):
for temperature in range(300, 400, 25):
    results = []

    # dmus = np.linspace(-1.04, 1.04, 1041)
    dmus = np.linspace(-0.6, -0.35, 200)

    for direction in [1, -1]:

        GGI = []
        energy = []
        chem_pot = []

        structure = start_configuration.copy()

        if direction == 1:
            for i in ga_sites:
                structure[i].symbol = "In"
            dmu_values = dmus

        else:
            for i in ga_sites:
                structure[i].symbol = "Ga"
            dmu_values = dmus[::-1]

        for dmu in dmu_values:

            mc = SemiGrandCanonicalEnsemble(
                structure=structure,
                calculator=ClusterExpansionCalculator(
                    structure, cluster_expansion
                ),
                temperature=temperature,
                ensemble_data_write_interval=10,
                chemical_potentials={
                    "In": 0.0,
                    "Ga": float(dmu),
                },
                sublattice_probabilities=[0.0, 1.0, 0.0]
            )

            mc.run(25000)
            structure = mc.structure.copy()

            data_cutoff = int(
                0.7 * len(mc.data_container.get("Ga_count"))
            )

            ga_number = np.mean(
                mc.data_container.get("Ga_count")[data_cutoff:]
            )

            in_number = np.mean(
                mc.data_container.get("In_count")[data_cutoff:]
            )
            print("IN process")
            GGI.append(ga_number / (ga_number + in_number))

            energy.append(
                np.mean(
                    mc.data_container.get("potential")[data_cutoff:]
                )
            )

            chem_pot.append(dmu)

        # Reverse backward results to match forward chemical potentials
        if direction == -1:
            GGI.reverse()
            energy.reverse()
            chem_pot.reverse()

        results.append((np.array(GGI), np.array(energy)))

    # Average forward and backward simulations
    GGI = (results[0][0] + results[1][0]) / 2
    energy = (results[0][1] + results[1][1]) / 2
    chem_pot = dmus

    # Same energy subtraction as your original code
    k, b = fit_linear(0, max(energy), 1, min(energy))

    for i in range(len(GGI)):
        energy[i] -= (k * GGI[i] + b)

    # Exactly the same output format:
    # Temperature GGI Energy Chemical_potential
    with open(filename, "a") as f:
        for i in range(len(GGI)):
            string_to_write = (
                str(temperature) + " " +
                str(GGI[i]) + " " +
                str(energy[i]) + " " +
                str(chem_pot[i]) + " \n"
            )
            f.write(string_to_write)
    print(f"T = {temperature} K: data saved successfully")