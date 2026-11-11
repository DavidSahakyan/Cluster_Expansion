import numpy as np
from ase import Atoms
from mchammer.free_energy_tools import _ideal_mixing_entropy, _get_atoms_on_sublattice
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

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

def get_ref_energy(mc, cluster_space, k_B, T):
    Enthalpy_of_random = np.average(mc.data_container.get("potential")[1000:])
    sublattices = cluster_space.get_sublattices(mc.structure)
    sublattice_probabilities = [True] * len(sublattices)
    atoms_on_sublattices = _get_atoms_on_sublattice(mc.structure,
                                                    sublattices)
    ideal_mixing_entropy = _ideal_mixing_entropy(sublattice_probabilities,
                                                atoms_on_sublattices,
                                                k_B)
    free_energy_reference = Enthalpy_of_random - ideal_mixing_entropy * T

    return free_energy_reference

def read_data(filename, target_T):
    GGIs = []
    Es   = []
    E_derivatives = []
    with open(filename) as f:
        line = f.readline()

        while line:
            data = line.split()
            if float(data[1]) == target_T:
                GGIs.append(float(data[0]))
                Es.append(float(data[2]))
                if len(data) == 4:
                    E_derivatives.append(float(data[3]))

            line = f.readline()

    if not len(GGIs):
        err_string = "No data for specified T. T = " + str(target_T) + "\n"
        KeyError(err_string)
    
    return GGIs, Es, E_derivatives

def find_maxwell_points(free_energy_polynom, tol=1e-6):

    first_derivative = free_energy_polynom.deriv()
    second_derivative = free_energy_polynom.deriv(2)

    # Find extrema of the first derivative
    extrema = second_derivative.roots()
    extrema = sorted(
        r.real for r in extrema
        if abs(r.imag) < 1e-10 and 0 < r.real < 1
    )

    if len(extrema) < 2:
        return []

    # Candidate chemical potential range
    values = [first_derivative(x) for x in extrema]

    mu_min = min(values)
    mu_max = max(values)

    def maxwell_condition(mu):

        roots = (first_derivative - mu).roots()

        roots = sorted(
            r.real for r in roots
            if abs(r.imag) < 1e-10 and 0 < r.real < 1
        )

        if len(roots) < 3:
            return np.nan

        x1, x2 = roots[0], roots[-1]

        return (
            free_energy_polynom(x2)
            - free_energy_polynom(x1)
            - mu * (x2 - x1)
        )

    mu_values = np.linspace(mu_min, mu_max, 1000)

    for mu1, mu2 in zip(mu_values[:-1], mu_values[1:]):

        f1 = maxwell_condition(mu1)
        f2 = maxwell_condition(mu2)

        if not (np.isfinite(f1) and np.isfinite(f2)):
            continue

        if f1 * f2 <= 0:

            mu = brentq(maxwell_condition, mu1, mu2, xtol=tol)

            roots = (first_derivative - mu).roots()

            roots = sorted(
                r.real for r in roots
                if abs(r.imag) < 1e-10 and 0 < r.real < 1
            )

            if len(roots) >= 3:
                return [roots[0], roots[-1]]

    return []

def find_maxwell_points_2_polynoms(free_energy_left, free_energy_right, left_region=(0, 0.1), right_region=(0.9, 1), tol=1e-6):

    dF_left = free_energy_left.deriv()
    dF_right = free_energy_right.deriv()

    # Find extrema of free energy in the corresponding regions
    def get_roots(polynom, region):

        roots = polynom.roots()

        return sorted(
            r.real for r in roots
            if abs(r.imag) < 1e-10
            and region[0] <= r.real <= region[1]
        )

    # Common chemical potential range
    x_left = np.linspace(*left_region, 1000)
    x_right = np.linspace(*right_region, 1000)

    mu_left = dF_left(x_left)
    mu_right = dF_right(x_right)

    mu_min = max(np.min(mu_left), np.min(mu_right))
    mu_max = min(np.max(mu_left), np.max(mu_right))

    if mu_min >= mu_max:
        return []

    def maxwell_condition(mu):

        roots_left = get_roots(dF_left - mu, left_region)
        roots_right = get_roots(dF_right - mu, right_region)

        if not roots_left or not roots_right:
            return np.nan

        x1 = roots_left[0]
        x2 = roots_right[-1]

        # Common tangent condition
        return (
            free_energy_right(x2)
            - free_energy_left(x1)
            - mu * (x2 - x1)
        )

    # Search for sign changes
    mu_values = np.linspace(mu_min, mu_max, 1000)

    for mu1, mu2 in zip(mu_values[:-1], mu_values[1:]):

        f1 = maxwell_condition(mu1)
        f2 = maxwell_condition(mu2)

        if not (np.isfinite(f1) and np.isfinite(f2)):
            continue

        if f1 * f2 <= 0:

            mu = brentq(
                maxwell_condition,
                mu1,
                mu2,
                xtol=tol
            )

            roots_left = get_roots(dF_left - mu, left_region)
            roots_right = get_roots(dF_right - mu, right_region)

            if roots_left and roots_right:
                return [roots_left[0], roots_right[-1]]

    return []

def deriv(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)

# Function below reads the datafile from canonical simulations and analyzes it. 
# Argument "Ts" is the temperatures list for which the analysis should be done, note that the temperatures should be in datafile.
# Critical temperature is the temperature below and equal which the free energy data fit will be done separately for region around 1 and 0.
# "range_around_0", "range_around_1" variables define the range for which the separate fits are done.
def get_canonical_free_energy_fits_over_T(datafile, Ts, plot_or_no = True, critical_temperature = 150, range_around_0 = 0.1, range_around_1 = 0.9):
    
    T_free_energies = {}
    
    for i in range(len(Ts)):
        fitted_values = []
        free_energies = []

        free_energies_around_0 = []
        free_energies_around_1 = []
        
        data = read_data(
                            datafile, 
                            Ts[i]
                        )

        GGIs = data[0]
        Es   = data[1]

        GGIs = np.asarray(data[0], dtype = float)
        Es   = np.asarray(data[1], dtype = float)

        indices = np.argsort(GGIs)

        GGIs = GGIs[indices]
        Es = Es[indices]

        GGIs_around_0 = []
        GGIs_around_1 = []

        Es_around_0 = []
        Es_around_1 = []

        k, b = fit_linear(GGIs[0], Es[0], GGIs[-1], Es[-1])
        for j in GGIs:
            fitted_values.append(k * j + b)

        for j in range(len(GGIs)):
            free_energies.append((Es[j] - fitted_values[j]) * 4000 / 216)

            if GGIs[j] < range_around_0:
                free_energies_around_0.append((Es[j] - fitted_values[j]) * 4000 / 216)
            elif GGIs[j] > range_around_1:
                free_energies_around_1.append((Es[j] - fitted_values[j]) * 4000 / 216)

        leg = "T: "     + str(Ts[i]) + " K"

        if Ts[i] > critical_temperature:
            coeffs = np.polyfit(GGIs, free_energies, 12)[::-1]
            free_energy_fit = np.polynomial.Polynomial(coeffs)
            T_free_energies[Ts[i]] = free_energy_fit
        
        else:
            for ggi_index in range(len(GGIs)):
                if GGIs[ggi_index] < range_around_0:
                    GGIs_around_0.append(GGIs[ggi_index])
                    Es_around_0.append(Es[ggi_index])
                elif GGIs[ggi_index] > range_around_1:
                    GGIs_around_1.append(GGIs[ggi_index])
                    Es_around_1.append(Es[ggi_index])

            coeffs_around_0 = np.polyfit(GGIs_around_0, free_energies_around_0, 4)[::-1]
            coeffs_around_1 = np.polyfit(GGIs_around_1, free_energies_around_1, 4)[::-1]
            free_energy_fit_around_0 = np.polynomial.Polynomial(coeffs_around_0)
            free_energy_fit_around_1 = np.polynomial.Polynomial(coeffs_around_1)
            T_free_energies[Ts[i]] = [free_energy_fit_around_0, 
                                      free_energy_fit_around_1]

        if plot_or_no:
            if Ts[i] > critical_temperature:
                x = np.linspace(0, 1, 50)
                y = [free_energy_fit(i) for i in x]
                plt.plot(x, y, linestyle = "--")
                plt.scatter(GGIs, free_energies, label = f"T = {Ts[i]} K")
            
            else:
                lab = "T = " + str(Ts[i]) + " K"
                plt.scatter(GGIs, free_energies, label = lab)
                x0 = np.linspace(0, range_around_0, 100)
                x1 = np.linspace(range_around_1, 1, 100)
                plt.plot(x0, free_energy_fit_around_0(x0))
                plt.plot(x1, free_energy_fit_around_1(x1))

    if plot_or_no:
        plt.grid()
        plt.xlabel("GGI", fontsize = 25)
        plt.ylabel(r"$G_{mix}(meV/atom)$", fontsize = 25)
        plt.title("TEI", fontsize = 25)
        plt.legend(fontsize = 20)
        plt.tick_params(axis = "both", which = "major", labelsize = 20)
        # plt.show()

    return T_free_energies

def get_VCSGC_free_energy_fits_over_T(datafile, Ts, plot_or_no = True, critical_temperature = 150, range_around_0 = 0.1, range_around_1 = 0.9):
    
    T_free_energies = {}
    
    for i in range(len(Ts)):
        fitted_values = []
        free_energies = []

        free_energies_around_0 = []
        free_energies_around_1 = []
        
        data = read_data(
                            datafile, 
                            Ts[i]
                        )

        GGIs = data[0]
        E_deriv = data[2] 

        GGIs_around_0 = []
        GGIs_around_1 = []

        Es_around_0 = []
        Es_around_1 = []

        GGIs = np.array(GGIs)
        E_deriv = np.array(E_deriv)
        
        indices = np.argsort(GGIs)       
        x = GGIs[indices]
        dEdx = E_deriv[indices]
    
        # Remove repeated GGI values by averaging derivatives
        x_unique, inverse = np.unique(x, return_inverse=True)
    
        dEdx_unique = (
            np.bincount(inverse, weights=dEdx)
            / np.bincount(inverse)
        )
    
        # Integrate
        Es = cumulative_trapezoid(
            dEdx_unique,
            x_unique,
            initial=0
        )

        # if plot_or_no:
        #     plt.figure(figsize=(10, 6))

        #     plt.scatter(x_unique, Es, s=15, label = f"T = {Ts[i]} K")
        #     plt.xlabel("GGI")
        #     plt.ylabel("Free energy (eV/atom)")
        #     plt.grid()
        #     plt.legend()
        #     plt.show()
    
        ind = np.argsort(x_unique)
        x_unique = x_unique[ind]
        Es = Es[ind]

        k, b = fit_linear(x_unique[0], Es[0], x_unique[-1], Es[-1])
        for j in x_unique:
            fitted_values.append(k * j + b)

        for j in range(len(x_unique)):
            free_energies.append((Es[j] - fitted_values[j]) * 4000 / 216)

            if x_unique[j] < range_around_0:
                free_energies_around_0.append((Es[j] - fitted_values[j]) * 4000 / 216)
            elif x_unique[j] > range_around_1:
                free_energies_around_1.append((Es[j] - fitted_values[j]) * 4000 / 216)

        if Ts[i] > critical_temperature:
            coeffs = np.polyfit(x_unique, free_energies, 12)[::-1]
            free_energy_fit = np.polynomial.Polynomial(coeffs)
            T_free_energies[Ts[i]] = free_energy_fit
        
        else:
            for ggi_index in range(len(x_unique)):
                if x_unique[ggi_index] < range_around_0:
                    GGIs_around_0.append(x_unique[ggi_index])
                    Es_around_0.append(Es[ggi_index])
                elif x_unique[ggi_index] > range_around_1:
                    GGIs_around_1.append(x_unique[ggi_index])
                    Es_around_1.append(Es[ggi_index])

            coeffs_around_0 = np.polyfit(GGIs_around_0, free_energies_around_0, 4)[::-1]
            coeffs_around_1 = np.polyfit(GGIs_around_1, free_energies_around_1, 4)[::-1]
            free_energy_fit_around_0 = np.polynomial.Polynomial(coeffs_around_0)
            free_energy_fit_around_1 = np.polynomial.Polynomial(coeffs_around_1)
            T_free_energies[Ts[i]] = [free_energy_fit_around_0, 
                                      free_energy_fit_around_1]

        if plot_or_no:
            if Ts[i] > critical_temperature:
                x = np.linspace(0, 1, 50)
                y = [free_energy_fit(i) for i in x]
                plt.plot(x, y, linestyle = "--")
                plt.scatter(x_unique, free_energies, label = f"T = {Ts[i]} K")
                
            else:
                lab = "T = " + str(Ts[i]) + " K"
                plt.scatter(x_unique, free_energies, label = lab)
                x0 = np.linspace(0, range_around_0, 100)
                x1 = np.linspace(range_around_1, 1, 100)
                plt.plot(x0, free_energy_fit_around_0(x0), "--")
                plt.plot(x1, free_energy_fit_around_1(x1), "--")

    plt.ylabel(r"$G_{mix}(meV/atom)$", fontsize = 25)
    plt.xlabel("GGI", fontsize = 25)
    plt.title("VCSGC ensemble", fontsize = 25)
    plt.grid()
    plt.tick_params(axis = "both", which = "major", labelsize = 20)
    plt.legend(fontsize = 10)
    # plt.title(r"VCSGC", fontsize = 25)
    plt.show()

    return T_free_energies

def analyze_free_energy_fits_over_T(T_free_energies, plot_or_no = True, critical_temperature = 150):
    T_GGIs = {}
    for T in T_free_energies.keys():
        if T > critical_temperature:
            T_GGIs[T] = find_maxwell_points(T_free_energies[T])     

            x = np.linspace(0, 1, 40)
            y = [T_free_energies[T].deriv()(i) for i in x]

        else:
            free_energy_derivative_around_0 = T_free_energies[T][0].deriv()
            free_energy_derivative_around_1 = T_free_energies[T][1].deriv()

            T_GGIs[T] = find_maxwell_points_2_polynoms(T_free_energies[T][0], 
                                                       T_free_energies[T][1])

            x0 = np.linspace(0, 0.1, 100)
            x1 = np.linspace(0.9, 1, 100)

            y0 = free_energy_derivative_around_0(x0)
            y1 = free_energy_derivative_around_1(x1)

            x = [*x0, *x1]
            y = [*y0, *y1]

        if plot_or_no:        
            leg = "T: " + str(T) + "K"
            plt.plot(x, y, linestyle = "--", label = leg)

    if plot_or_no:
        plt.title("Temperature integration")
        plt.legend()
        plt.grid()
        plt.show()

    return T_GGIs

def analyze_free_energy_derivative_fits_over_T(T_free_energy_derivatives, plot_or_no = True, critical_temperature = 150):
    T_free_energies = {}
    T_GGIs = {}
    for T in T_free_energy_derivatives.keys():
        T_free_energies[T] = T_free_energy_derivatives.integ()
        if T > critical_temperature:
            T_GGIs[T] = find_maxwell_points(T_free_energies[T])     

            x = np.linspace(0, 1, 40)
            y = [T_free_energies[T].deriv()(i) for i in x]

        else:
            free_energy_derivative_around_0 = T_free_energies[T][0].deriv()
            free_energy_derivative_around_1 = T_free_energies[T][1].deriv()

            T_GGIs[T] = find_maxwell_points_2_polynoms(T_free_energies[T][0], 
                                                       T_free_energies[T][1])

            x0 = np.linspace(0, 0.1, 100)
            x1 = np.linspace(0.9, 1, 100)

            y0 = free_energy_derivative_around_0(x0)
            y1 = free_energy_derivative_around_1(x1)

            x = [*x0, *x1]
            y = [*y0, *y1]

        if plot_or_no:        
            leg = "T: " + str(T) + "K"
            plt.scatter(x, y, label = leg)

    if plot_or_no:
        plt.title("Temperature integration")
        plt.legend()
        plt.grid()
        plt.show()

    return T_GGIs

def plot_phase_diagrams(T_GGIs_list, labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    for T_GGIs, lab in zip(T_GGIs_list, labels):
        phase_diagram_Ts  = []
        phase_diagram_GGIs = []

        for T in T_GGIs.keys():

            for _ in range(len(T_GGIs[T])):
                phase_diagram_Ts.append(T)

            for GGI in T_GGIs[T]:
                phase_diagram_GGIs.append(GGI)


        phase_diagram_GGIs = np.array(phase_diagram_GGIs)
        phase_diagram_Ts   = np.array(phase_diagram_Ts)
        indecies = phase_diagram_GGIs.argsort()
        phase_diagram_GGIs = phase_diagram_GGIs[indecies]
        phase_diagram_Ts   = phase_diagram_Ts[indecies]
        
        if lab == "SGC":
            plt.scatter(phase_diagram_GGIs, phase_diagram_Ts, label = lab, color = 'red')
        else:
            plt.plot   (phase_diagram_GGIs, phase_diagram_Ts, label = lab)

    plt.tick_params(axis = "both", which = "major", labelsize = 20)
    plt.title("Phase Diagrams", fontsize = 25)
    plt.ylabel("T(K)", fontsize = 25)
    plt.xlabel("GGI", fontsize = 25)
    plt.legend(fontsize = 25)    
    plt.grid()
    plt.show()

def phase_diagram_SGC(file_to_read, Ts, plot_or_no = True, critical_T_1 = 250, critical_T_2 = 325):
    T_GGIs = {}
    phase_diagram_T = []
    phase_diagram_GGI_right = []
    phase_diagram_GGI_left = []
    for T in Ts:

        GGI = []
        dmu = []
        E = []

        with open(file_to_read) as f:
            for line in f:
                data = line.split()

                if float(data[0]) == T:
                    GGI.append(float(data[1]))
                    E.append(float(data[2]))
                    dmu.append(float(data[3]))

        if not len(GGI):
            print(f"WARNING! TARGET TEMPERATURE {T} IS NOT FOUND")
            continue

        GGI = np.array(GGI)
        dmu = np.array(dmu)
        E = np.array(E)

        # Sort by chemical potential
        indices = np.argsort(dmu)

        dmu = dmu[indices]
        GGI = GGI[indices]
        E = E[indices]

        edges_mask = []

        if T <= critical_T_1:
            for i in range(len(GGI)):
                if GGI[i] < 0.05 or GGI[i] > 0.95:
                    edges_mask.append(i)

            dmu = dmu[edges_mask]
            GGI = GGI[edges_mask]
            E = E[edges_mask]
        elif T < critical_T_2:
            for i in range(len(GGI)):
                if GGI[i] < 0.3 or GGI[i] > 0.7:
                    edges_mask.append(i)

            dmu = dmu[edges_mask]
            GGI = GGI[edges_mask]
            E = E[edges_mask]

        delta_GGI = np.abs(np.diff(GGI))

        jump_index = np.argmax(delta_GGI)

        # Compositions before and after the jump
        GGI_left = min(GGI[jump_index], GGI[jump_index + 1])
        GGI_right = max(GGI[jump_index], GGI[jump_index + 1])

        phase_diagram_T.append(T)
        phase_diagram_GGI_left.append(GGI_left)
        phase_diagram_GGI_right.append(GGI_right)

        print(
            f"T = {T} K: "
            f"GGI_left = {GGI_left:.4f}, "
            f"GGI_right = {GGI_right:.4f}, "
            f"Jump = {delta_GGI[jump_index]:.4f}"
        )

        if plot_or_no:
            plt.scatter(GGI, dmu, label=f"T = {T} K")

    if plot_or_no:
        plt.ylabel("E derivative (eV)", fontsize = 25)
        plt.xlabel("GGI", fontsize = 25)
        plt.title("SGC composition vs chemical potential", fontsize = 25)
        plt.legend(fontsize = 25)
        plt.show()

    phase_diagram_T         = np.array(phase_diagram_T)
    phase_diagram_GGI_left  = np.array(phase_diagram_GGI_left)
    phase_diagram_GGI_right = np.array(phase_diagram_GGI_right)

    indices = np.argsort(phase_diagram_T)


    phase_diagram_T = phase_diagram_T[indices]
    phase_diagram_GGI_left  = phase_diagram_GGI_left[indices]
    phase_diagram_GGI_right = phase_diagram_GGI_right[indices]

    for i in range(len(phase_diagram_T)):
        T_GGIs[phase_diagram_T[i]] = [phase_diagram_GGI_left[i], phase_diagram_GGI_right[i]] 

    # if plot_or_no:
    #     plt.plot(
    #         phase_diagram_GGI_left,
    #         phase_diagram_T,
    #         'o-',
    #         label="Left boundary"
    #     )
    # if plot_or_no:
    #     plt.plot(
    #         phase_diagram_GGI_right,
    #         phase_diagram_T,
    #         'o-',
    #         label="Right boundary"
    #     )
    # if plot_or_no:
    #     plt.xlabel("GGI")
    #     plt.ylabel("Temperature (K)")
    #     plt.title("Phase diagram from SGC ensemble")
    #     plt.legend()
    #     plt.grid()
    #     plt.show()
    return T_GGIs