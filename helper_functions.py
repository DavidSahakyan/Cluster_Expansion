import numpy as np
from ase import Atoms
import re

def parse_ATAT_strout(filename):
    """
    Reads an ATAT str.out file and converts it into an ASE Atoms object.

    ATAT format:
        first 3 lines  = lattice vectors
        next 3 lines   = supercell transformation
        remaining lines = atomic coordinates + symbol
    """

    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # First 3 rows: parent lattice vectors
    lattice = np.array([
        [float(x) for x in lines[i].split()[:3]]
        for i in range(3)
    ])

    # Next 3 rows: supercell matrix
    supercell_matrix = np.array([
        [float(x) for x in lines[i].split()[:3]]
        for i in range(3, 6)
    ])

    # Build actual supercell
    supercell = supercell_matrix @ lattice
    positions = []
    symbols = []

    number_of_atoms = 0

    for line in lines[6:]:
        parts = line.split()
        
        x, y, z = map(float, parts[:3])
        symbol = parts[3]

        positions.append([x, y, z])
        symbols.append(symbol)
        number_of_atoms += 1

    # Convert ATAT coordinates to Cartesian coordinates
    positions = np.array(positions) @ lattice

    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=supercell,
        pbc=True
    )

    return atoms, number_of_atoms

def parse_ATAT_lat(filename):

    cell     = []
    coords   = []
    elements = []
    coordinates_number = 0
    
    with open(filename, "r") as f:

        for i in range(3):
            line = f.readline().strip()

            try:
                unit_vector_coords = [float(i) for i in line.split(" ")]
                cell.append(unit_vector_coords)
            except:
                print("SOME ERROR IN PARSING UNIT VECTOR COORDINATES INTO NUMBERS. CHECK initial_data_files/lat.in FILE\n")

            if not line:
                print("SMTH IS WRONG WITH LATTICE FILE OR PARSING FUNCTION\n")

        for i in range(3):
            line = f.readline()

            if not line:
                print("SMTH IS WRONG WITH LATTICE FILE OR PARSING FUNCTION")

        while line:
            line = f.readline().strip()

            if not line:
                print(f"{coordinates_number} coordinates read")
                break

            try:
                atoms_coords = [float(i) for i in line.split(" ")[0:3]]
                coords.append(atoms_coords)
            except:
                print("SOME ERROR IN PARSING ATOM COORDINATES INTO NUMBERS. CHECK initial_data_files/lat.in FILE\n")

            coordinates_number += 1
            elements.append(line.split(" ")[3].split(","))

    return cell, coords, elements

def get_struct_and_energy(directory_name):

    try:
        with open(directory_name + "/energy") as f:
            energy = float(f.read().strip())
    except Exception as e:
        print(e)
        print(f"SMTH WRONG WITH energy FILE IN {directory_name}\nNOTE: YOUR ENERGY FILE'S NAME SHOULD BE energy\n")

    try:
        struct, number_of_atoms = parse_ATAT_strout(directory_name + "/str.out")
    except:
        print(f"SMTH WRONG WITH ATAT'S STRUCTURE FILE IN {directory_name}\nNOTE: YOUR ATAT's STRUCTURE FILE'S NAME SHOULD BE str.out")

    return struct, energy, number_of_atoms

def GGI_AAC(file_name):
    elements = []
    number_of_elements = []
    In_number = 0
    Ga_number = 0
    Ag_number = 0
    Cu_number = 0
    

    with open(file_name, "r") as f:
        for i in range(5):
            f.readline()

        line = f.readline().strip().split()
        elements = line

        line = f.readline().strip().split()        
        number_of_elements = [float(i.strip()) for i in line]

    for index in range(len(elements)):
        if elements[index] == "In":
            In_number = number_of_elements[index]
        elif elements[index] == "Ga":
            Ga_number = number_of_elements[index]
        elif elements[index] == "Ag":
            Ag_number = number_of_elements[index]
        elif elements[index] == "Cu":
            Cu_number = number_of_elements[index]

    return [(Ga_number/(Ga_number + In_number)), (Ag_number/(Ag_number + Cu_number))]

def fit_linear(x1, y1, x2, y2):
    k = (y2 - y1) / (x2 - x1)

    b = (y1 * x2 - y2 * x1) / (x2 - x1)

    return k,b

def linear_fit_function(x, omega):
    return omega * x * (1 - x)

def surface_fit_function(GGI_AAC_data, A, B, C, D):
    x = GGI_AAC_data[0]
    y = GGI_AAC_data[1]
    return (A * x * y) + (B * x) + (C * y) + D

def read_target_data(filename, target_t, target_k_B):
    with open(filename, "r") as f:

        line = f.readline()
        in_desired_range = False
        GGI_list    = []
        energy_list = []
        
        while line:
            if re.search(r"START OF THE DATA SET.*", line) and \
                           (not in_desired_range):

                start_symbol = re.search(r'T = ', line).end()
                end_symbol   = re.search(r' K', line).start()

                detected_temperature = line[start_symbol:end_symbol].strip()

                start_symbol = re.search(r'k_B = ', line).end()
                end_symbol   = re.search(r' eV',    line).start()

                detected_k_B = line[start_symbol:end_symbol].strip()

                if (float(detected_k_B)         == target_k_B) and \
                   (float(detected_temperature) == target_t):

                    in_desired_range = True
                    f.readline()

            elif in_desired_range and \
                 not re.search(r".*=.*", line):

                GGI_list   .append(float(line.split()[0]))
                energy_list.append(float(line.split()[1]))

            elif in_desired_range and \
                 re.search(r".*=.*", line):

                return [GGI_list, energy_list]

            line = f.readline()

    if in_desired_range:
        return [GGI_list, energy_list]

    raise LookupError("TARGET TEMPERATURE OR BOLTZMANN CONSTANT ARE NOT FOUND. YOU SHOULD RUN MC SIMULATION FOR THAT VALUES FIRST\n")