#!/usr/bin/python

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.special import comb
from glob import glob
import argparse
import bisect
import time

# Configuration des paramètres d'affichage de numpy
np.set_printoptions(threshold=np.inf)

"""
Implementation of the Stationary Structured Coalescent (NSSC)
"""

class SSC:
    """
    Stationary Structured Coalescent
    This class represents a Structured Coalescent
    Markov Chain for structured populations.
    """

    def __init__(self, model_params, lineages_are_dist=False):
        """
        Create a Structured Coalescent Markov Chain model.
        i.e.
        - a Q-matrix based on the input parameters
        - the sampling
        model_params: dictionary,
            samplingVector: list of integer, how many sequences to sample from
                            each deme
            'M': matrix of real, the migration rate from deme i to deme j
            'c': list of real, the size of each deme
        lineages_are_dist: indicates whether lineages are distinguishable
                           or not
        """
        sampling_vector = model_params['samplingVector']
        n = len(sampling_vector)
        if 2 in sampling_vector:
            l1a = sampling_vector.index(2)
            l1b = sampling_vector.index(2)
        else:
            lineages_pos = [i for i, x in enumerate(sampling_vector) if x == 1]
            l1a = lineages_pos[0]
            l1b = lineages_pos[1]
        if lineages_are_dist:
            initial_state_vect = np.zeros(n**2+1)
            initial_state_ix = l1a*n + l1b
            initial_state_vect[initial_state_ix] = 1
        else:
            initial_state_vect = np.zeros(int(0.5*n*(n+1) + 1))
            initial_state_ix = int(l1a*n - 0.5*l1a*(l1a-1) + (l1b-l1a))
            initial_state_vect[initial_state_ix] = 1
        Q = self.createQmatrix(np.array(model_params['M']), np.array(
                model_params['size']), lineages_are_dist)
      #  print("M",model_params['M'])
        self.Qmatrix = Q
      #  print("Q",self.Qmatrix)
        self.initial_state_vect = initial_state_vect
        self.initial_state_ix = initial_state_ix
        self.diagonalizedQ = self.diagonalize_Q(self.Qmatrix)

    def createQmatrix(self, migMatrix, demeSizeVector, lineagesAreDist=False):
        """
        Create a Q-matrix based on the migration Matrix and the size
        of each deme
        """
        migMatrix = 0.5 * np.array(migMatrix)
        n = migMatrix.shape[0]
        kIx = np.arange(0, n)
        if lineagesAreDist:
            Q = np.zeros((n**2 + 1, n**2 + 1), dtype = 'complex_')
            for i in range(n):
                for j in range(n):
                    iMigratesTo = kIx[kIx != i]
                    Q[n*i+j, n*iMigratesTo + j] = migMatrix[i, iMigratesTo]
                    jMigratesTo = kIx[kIx != j]
                    Q[n*i+j, n*i + jMigratesTo] = migMatrix[j, jMigratesTo]
                # Coalescence inside a deme
                Q[n*i+i, -1] = 1./demeSizeVector[i]
        else:
            sizeQ = int(0.5*n*(n+1) + 1)
            Q = np.zeros((sizeQ, sizeQ), dtype = 'complex_')
            d = {}
            k = 0
            for i in range(n):
                for j in range(i, n):
                    d[(i, j)] = k
                    k += 1
            for i in range(n):
                for j in range(i, n):
                    rowNumber = d[(i, j)]
                    iMigratesTo = kIx[kIx != i]
                    columnNumbers = [d[(min(l, j), max(l, j))]
                                     for l in iMigratesTo]
                    Q[rowNumber, columnNumbers] = migMatrix[i, iMigratesTo]
                    jMigratesTo = kIx[kIx != j]
                    columnNumbers = [d[(min(i, l), max(i, l))]
                                     for l in jMigratesTo]
                    Q[rowNumber, columnNumbers] = Q[rowNumber,
                                                    columnNumbers] +\
                        migMatrix[j, jMigratesTo]
                # Coalescence inside a deme
                Q[d[(i, i)], -1] = 1./demeSizeVector[i]
        # Add the values of the diagonal
        columNumbers = np.arange(Q.shape[0])
        for i in columNumbers:
            noDiagColumn = columNumbers[columNumbers != i]
            Q[i, i] = -sum(Q[i, noDiagColumn])
        return(Q)

    def diagonalize_Q(self, Q):
        # Compute the eigenvalues and vectors for the diagonalization
        eigenval, eigenvect = np.linalg.eig(Q)
        # Put the eigenvalues in a diagonal
        D = np.diag(eigenval)
        A = eigenvect
        Ainv = np.linalg.inv(A)
        return (A, D, Ainv)

    def exponential_Q(self, t, i=0):
        """
        Computes e^{tQ_i} for a given t.
        Note that we will use the stored values of the diagonal expression
        of Q_i. The value of i is between 0 and the index of the last
        demographic event (i.e. the last change in the migration rate).
        """
        (A, D, Ainv) = self.diagonalizedQ
        exp_tD = np.diag(np.exp(t * np.diag(D))) # cette ligne est responsable de la lenteur du programme je crois ...
        return(A.dot(exp_tD).dot(Ainv))

    def evaluate_Pt(self, t):
        """
        Evaluate the transition semigroup at t.
        Uses previously computed values to speed up the computation.
        """
        # Get the left of the time interval that contains t.
        P_deltaT = self.exponential_Q(t, 0)
        return(P_deltaT)

    def cdfT2(self, t):
        """
        Evaluates the cumulative distribution function of T2
        for the current model
        """
        Pt = self.evaluate_Pt(t)
        return(np.real(Pt[self.initial_state_ix, -1]))

    def pdfT2(self, t):
        """
        Evaluates the probability density function of T2
        for the current model
        """
        S0 = self.initial_state_ix
        # Get the time interval that contains t.
        P_delta_t = self.exponential_Q(t, 0)
        return(self.Qmatrix.dot(P_delta_t)[S0, -1])

    def evaluateIICR(self, t):
        """
        Evaluates the IICR at time t for the current model
        t: list of values to evaluate the IICR
        """
        # Sometimes, for numerical
        # errors, F_x and f_x get negative values
        # or "nan" or "inf"
        # In any of these cases, a default value is returned
        F_x = np.ones(len(t))
        f_x = np.ones(len(t))
        quotient_F_f = np.ones(len(t))
        eigvals = list(np.linalg.eigvals(self.Qmatrix))
        eigvals.remove(max(eigvals))
        plateau = (-1)/max(eigvals)
        prec_plateau = 1e-3
        F_x[0] = self.cdfT2(t[0])
        if not(0<=F_x[0]<=1): F_x[0]=1
        f_x[0] = self.pdfT2(t[0]) # avertissement de Complex Warning : Casting complex values to real discards the imaginary part, sur cette ligne
        if (f_x[0] < 1e-14) or (np.isinf(f_x[0])) or np.isnan(f_x[0]):
            f_x[0] = 1e-14
        quotient_F_f[0] = (1-F_x[0])/f_x[0]
        for i in range(1, len(t)):
            F_x[i] = self.cdfT2(t[i])
            f_x[i] = self.pdfT2(t[i]) # avertissement de Complex Warning : Casting complex values to real discards the imaginary part, sur cette ligne
            quot_F_f = (1-F_x[i])/f_x[i]
            if i>10 and ((abs(quot_F_f-plateau) < prec_plateau) or (np.allclose(quotient_F_f[i-10:i- 1],np.repeat(plateau,9)))): 
                quotient_F_f[i] = plateau # avertissement de Complex Warning : Casting complex values to real discards the imaginary part, sur cette ligne
            else:
                quotient_F_f[i] = quot_F_f
            if (f_x[i] < 1e-14) or (np.isinf(f_x[i])) or np.isnan(f_x[i]):
                f_x[i] = f_x[i-1]
        return(quotient_F_f)

    def compute_distance(self, x, y, Nref):
        """
        Compute the distance between the IICR of the model,
        scaled by Nref, and some psmc curve given by x, y
        """
        # We evaluate the theoretical IICR at points
        # (x[i] + x[i-1])*0.5/(2*Nref)
        points_to_evaluate = [(x[i] + x[i-1])*0.25/Nref
                             for i in range(1, len(x))]
        m_IICR = self.evaluateIICR(points_to_evaluate)
        distance = 0
        for i in range(len(m_IICR)):
            distance += (y[i] - Nref*m_IICR[i])**2
        return distance

"""
Fin de la classe SSC pour la génération des IICR - issu de https://github.com/aljounia/Theoretical_IICR_with_glue_matrices_generator/tree/main
"""

# Fonction pour générer la matrice M
def generate_matrix_M(x, y, M):
    n = x * y
    Mi = M / (n - 1)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = 0
            if (i + 1 == j) or (i + y == j) or (i - 1 == j) or (i - y == j):
                matrix[i, j] = Mi
            if j != n:
                if ((j + 1) % y == 0) and (j + 1 < n):
                    matrix[j, j + 1] = 0
                    matrix[j + 1, j] = 0
            if i == j:
                matrix[i, j] = 0
    return matrix.tolist()

# Fonction pour obtenir le schéma d'échantillon
def get_schema_echantillon(m, n):
    unique_points = set()
    unique_coords = []

    for i, j in product(range(m), range(n)):
        transformations = {
            (i, j),
            (j, m - 1 - i),
            (m - 1 - i, n - 1 - j),
            (n - 1 - j, i),
            (i, n - 1 - j),
            (j, i),
            (m - 1 - i, j),
            (n - 1 - j, m - 1 - i)
        }

        if (i, j) == min(transformations):
            unique_points.add(tuple(sorted(transformations)))
            unique_coords.append((i, j))

    unique_coords.sort(key=lambda x: (x[0] ** 2 + x[1] ** 2))

    vectors = []
    for x, y in unique_coords:
        vector = [0] * (m * n)
        vector[x * n + y] = 2
        vectors.append((vector, x + 1, y + 1))

    return vectors


"""
 Fonction pour générer les fichiers JSON
"""
def generate_json_files(output_dir, migration_rates, N0=100, taille=10, generation_time=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    case_number = 1
    json_files = []

    for i in range(1, taille + 1):
        for j in range(i, taille + 1):
            if not (j == 1 and i == 1):
                vecteur_echantillon = get_schema_echantillon(i, j)
                for M in migration_rates:
                    for scheme, x, y in vecteur_echantillon:
                        data = {
                            "theoretical_IICR_general": [
                                {
                                    "M": generate_matrix_M(i, j, M),
                                    "sampling": scheme,
                                    "size": [1] * (i * j),
                                    "label": f"Stepping-stone IICR n={i}x{j} sampling in deme ({x},{y})",
                                    "color": "b",
                                    "linestyle": "-",
                                    "linewidth": 1,
                                    "alpha": 0.8
                                }
                            ],
                            "piecewise_constant_functions": [],
                            "computation_parameters": {
                                "x_vector_type": "log",
                                "start": 0,
                                "end": 100,
                                "number_of_values": 64,
                                "pattern": "4*1+25*2+4+6"
                            },
                            "custom_x_vector": {
                                "set_custom_xvector": 0,
                                "x_vector": 0
                            },
                            "scale_params": {
                                "N0": N0,
                                "generation_time": generation_time
                            },
                            "plot_params": {
                                "plot_theor_IICR": 1,
                                "plot_limits": json.loads('[1e-1, 1e7, 0, 1e4]'),
                                "plot_xlabel": "time ",
                                "plot_ylabel": "IICR(t)",
                                "time_log": "True"
                            },
                            "save_theor_IICR_as_file": 1,
                            "vertical_lines": []
                        }

                        filename = os.path.join(output_dir, f"cas{format(case_number, '04d')}_{i}x{j}_M{int(10 * M)}_{x}{y}.json")
                        with open(filename, 'w') as file:
                            json.dump(data, file, indent=4)

                        json_files.append(filename)
                        case_number += 1

    print(f"Génération des fichiers JSON terminée. {len(json_files)} fichiers générés dans le dossier {output_dir}.")
    return json_files

# Fonction pour vérifier si un objet est de type array
def is_array_like(obj, string_is_array=False, tuple_is_array=True):
    result = hasattr(obj, "__len__") and hasattr(obj, '__getitem__')
    if result and not string_is_array and isinstance(obj, str):
        result = False
    if result and not tuple_is_array and isinstance(obj, tuple):
        result = False
    return result

"""
Fonctions pour calculer l'IICR général - récupérer de 'https://github.com/aljounia/Theoretical_IICR_with_glue_matrices_generator/tree/main' que l'on a coupé et factoriser pour ne garder que ce qu'il nous intéresse (la génération d'IICR stationnaire qui ne sont pas des n-islands
"""

def compute_IICR_general(t, params):
    M = params["M"]
    s = params["sampling"]
    c = params["size"]
    return compute_stationary_IICR_general(M, t, s, c)

# Fonction pour calculer l'IICR général stationnaire
def compute_stationary_IICR_general(M, t, s, c):
    #from model_ssc import SSC
    model_params = {"samplingVector": s, "M": M, "size": c}
    ssc = SSC(model_params)
    return ssc.evaluateIICR(t)

# Fonction pour traiter un fichier JSON et renvoyer un IICR
def process_json_file(json_file, output_dir):
    #pour compter le temps de calcul pour chaque IICR
    start_time = time.time()
    with open(json_file) as json_params:
        p = json.load(json_params)

#    fig = plt.figure()
#    ax = fig.add_subplot(1, 1, 1)

    N0 = p["scale_params"]["N0"]
    g_time = p["scale_params"]["generation_time"]

 #   for vl in p["vertical_lines"]:
 #       ax.axvline(2 * N0 * g_time * vl, color='k', ls='--')

    if p["plot_params"]["plot_theor_IICR"]:
        theoretical_IICR_general_list = []
        if p["plot_params"]["time_log"]:
            T_max = np.log10(p["plot_params"]["plot_limits"][1])
            t_k = np.logspace(-1, T_max, 1000)
            t_k = np.true_divide(t_k, 2 * N0 * g_time)
        else:
            T_max = p["plot_params"]["plot_limits"][1]
            t_k = np.linspace(0, T_max, 1000)
            t_k = np.true_divide(t_k, 2 * N0 * g_time)
        if "theoretical_IICR_general" in p:
            for i in range(len(p["theoretical_IICR_general"])):
                params = p["theoretical_IICR_general"][i]
                theoretical_IICR_general_list.append(compute_IICR_general(t_k, params))

        if "save_theor_IICR_as_file" in p:
            if p["save_theor_IICR_as_file"]:
                for i in range(len(theoretical_IICR_general_list)):
                    (time_k, theor_iicr) = (t_k, theoretical_IICR_general_list[i])
                    iicr_filename = os.path.join(output_dir, f"sim_{os.path.basename(json_file)[3:7]}.iicr")
                    with open(iicr_filename, "w") as f:
                        x2write = [str(2 * N0 * g_time * value) for value in t_k]
                        IICR2write = [str(N0 * value) for value in theor_iicr]
                        f.write("{}\n".format(" ".join(x2write)))
                        f.write("{}\n".format(" ".join(IICR2write)))
                    time_end = time.time()
                    print(f"IICR {iicr_filename} from {json_file} computed in {format(time_end-start_time, '.2f')} sec")
                    return iicr_filename

# directement récupérer du code iicr2psmc.py inclus dans le package SNIF
def convert_iicr_to_psmc(iicr_output_dir, psmc_output_dir, generation_time, mu):
    if not os.path.exists(psmc_output_dir):
        os.makedirs(psmc_output_dir)

    filenames = glob(os.path.join(iicr_output_dir, '*.iicr'))
    (theta, rho, s) = (0.05, 0.01, 100)
    N0 = theta / (4 * mu * s)

    psmc_files = []

    for fn in filenames:
        with open(fn, 'r') as f:
            lines = f.readlines()

        time = [float(years) / (2 * N0 * generation_time) for years in lines[0].strip().split(' ')]
        iicr = [float(psmc) / N0 for psmc in lines[1].strip().split(' ')]

        if len(time) != len(iicr):
            raise Exception('invalid .IICR file')

        lines = ['//\n']
        lines.append(f'TR\t{theta}\t{rho}\n')
        for i, (x, y) in enumerate(zip(time, iicr)):
            lines.append(f'RS\t{i}\t{x}\t{y}\n')
        lines.append('//\n')

        psmc_filename = os.path.join(psmc_output_dir, os.path.basename(fn).replace('.iicr', '.psmc'))
        with open(psmc_filename, 'w') as f:
            f.writelines(lines)

        psmc_files.append(psmc_filename)

    print(f"Conversion des fichiers IICR en PSMC terminée. {len(psmc_files)} fichiers générés dans le dossier {psmc_output_dir}")
    return psmc_files

# Fonction pour générer les fichiers SNIF
def generate_snif_files(psmc_output_dir, snif_output_dir, omega, mu, generation_time):
    if not os.path.exists(snif_output_dir):
        os.makedirs(snif_output_dir)

    distance_parameters = omega
    sim_files = glob(os.path.join(psmc_output_dir, '*.psmc'))

#dans template les 5 valeurs que l'on modifie sont entre crochets et leur valeur est détaillé dans la ligne file_content=
    template = """#!/usr/bin/python

from snif import *
from snif2tex import *

h_inference_parameters = InferenceParameters(
    data_source = '{data_source}',
    source_type = SourceType.PSMC,
    IICR_type = IICRType.Exact,
    ms_reference_size = None,
    ms_simulations = None,
    psmc_mutation_rate ={psmc_mutation_rate} ,
    psmc_number_of_sequences = None,
    psmc_length_of_sequences = None,
    infer_scale = True,
    data_cutoff_bounds = (1, 2e7),
    data_time_intervals = 1000,
    distance_function = ErrorFunction.ApproximatePDF,
    distance_parameter = {distance_parameter},
    distance_max_allowed = 7e3,
    distance_computation_interval = (1, 2e7),
    rounds_per_test_bounds = (5, 5),
    repetitions_per_test = 5,
    number_of_components = 1,
    bounds_islands = (2, 200),
    bounds_migrations_rates = (0.05, 20),
    bounds_deme_sizes = (1, 1),
    bounds_event_times = (1, 2e7),
    bounds_effective_size = (10, 10000)
)

h_settings = Settings(
    static_library_location = './libs/libsnif.so',
    custom_filename_tag = '{cas}',
    output_directory = './SNIF_results/{cas}',
    default_output_dirname = './SNIF_results/{cas}'
)

basename = infer(
    inf = h_inference_parameters,
    settings = h_settings
)

config = Configuration(
        SNIF_basename = basename,
        plot_width_cm = 13,
        plot_height_cm = 6,
        IICR_plots_style  = OutputStyle.Full,
        PDF_plots_style = OutputStyle.Excluded,
        CDF_plots_style = OutputStyle.Excluded,
        islands_plot_style = OutputStyle.Excluded,
        Nref_plot_style = OutputStyle.Excluded,
        test_numbers = "all",
        one_file_per_test = False,
        versus_plot_style = OutputStyle.Excluded,
        CG_style = OutputStyle.Excluded,
        CG_size_history = False,
        CG_source = '',
        CG_source_legend = "",
        Nref_histograms_bins = 100,
        islands_histograms_bins = 100,
        time_histograms_bins = 100,
        migration_histograms_bins = 100,
        size_histograms_bins = 100,
        scaling_units = TimeScale.Years,
        generation_time = {gen_time} 
)
TeXify(config)
"""
    file_counter = 1
    snif_files = []

    for sim_file in sorted(sim_files):
        for distance_parameter in distance_parameters:
            file_content = template.format(data_source=sim_file, distance_parameter=distance_parameter, cas=format(file_counter, '04d'), gen_time = generation_time, psmc_mutation_rate = mu)
            output_filename = os.path.join(snif_output_dir, f"SNIF_sim_{format(file_counter, '04d')}.py")
            with open(output_filename, 'w') as f:
                f.write(file_content)
            snif_files.append(output_filename)
            file_counter += 1

    print(f"Génération des fichiers SNIF terminée. {len(snif_files)} fichiers générés dans le dossier {snif_output_dir}.")
    return snif_files

# Fonction principale
def main(json_output_dir, iicr_output_dir, psmc_output_dir, snif_output_dir, migration_rates, N0, taille, generation_time, mu, omega):

    # Création des dossiers de sortie s'ils n'existent pas
    if not os.path.exists(json_output_dir):
        os.makedirs(json_output_dir)
    if not os.path.exists(iicr_output_dir):
        os.makedirs(iicr_output_dir)
    if not os.path.exists(psmc_output_dir):
        os.makedirs(psmc_output_dir)
    if not os.path.exists(snif_output_dir):
        os.makedirs(snif_output_dir)
    
    # Génération des fichiers JSON
    start_json = time.time()
    json_files = generate_json_files(json_output_dir, migration_rates, N0, taille, generation_time)
    end_json = time.time()
    print(f"Fichiers JSON computed in {format(end_json-start_json, '.2f')} sec")
    # Génération des fichiers IICR
    start_iicr = time.time()
    iicr_files = []
    for json_file in sorted(json_files):
        iicr_file = process_json_file(json_file, iicr_output_dir)
        iicr_files.append(iicr_file)
    end_iicr = time.time()
    print(f"Génération des fichiers IICR terminée. {len(iicr_files)} fichiers générés dans le dossier {iicr_output_dir} en {format(end_iicr-start_iicr, '.2f')} sec")

    # Conversion des fichiers IICR en PSMC
    start_psmc = time.time()
    psmc_files = convert_iicr_to_psmc(iicr_output_dir, psmc_output_dir, generation_time, mu)
    end_psmc = time.time()
    print(f"Fichiers PSMC computed in {format(end_psmc-start_psmc, '.2f')} sec")
    # Génération des fichiers SNIF
    start_snif = time.time()
    snif_files = generate_snif_files(psmc_output_dir, snif_output_dir, omega, mu, generation_time)
    end_snif = time.time()
    print(f"Fichiers SNIF computed in {format(end_psmc-start_psmc, '.2f')} sec")
    print(f"Temps total d'execution : {format(end_snif-start_json, '.2f')} sec")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Génération de fichiers JSON, IICR, PSMC et SNIF à partir de paramètres donnés en entrer. Le programme va générer les fichiers pour chaque possibilités selon les paramètres entrés.')
    parser.add_argument('--json_output_dir', type=str, default='./Generate_JSON', help='Dossier de sortie pour les fichiers JSON, par défaut "./Generate_JSON"')
    parser.add_argument('--iicr_output_dir', type=str, default='./Generate_IICR', help='Dossier de sortie pour les fichiers IICR, par défaut "./Generate_IICR"')
    parser.add_argument('--psmc_output_dir', type=str, default='./Generate_PSMC', help='Dossier de sortie pour les fichiers PSMC, par défaut "./Generate_PSMC"')
    parser.add_argument('--snif_output_dir', type=str, default='.', help='Dossier de sortie pour les fichiers SNIF, par défaut "." - répertoire actuel')
    parser.add_argument('--migration_rates', type=float, nargs='+', default=[0.1, 0.5, 1, 5, 10], help='Taux de migration, par défaut [0.1, 0.5, 1, 5, 10]')
    parser.add_argument('--N0', type=int, default=100, help='Taille de la population de référence, par défaut 100')
    parser.add_argument('--taille', type=int, default=10, help='Longueur maximale des matrices, par défaut 10 (donc taille max = 10*10)')
    parser.add_argument('--generation_time', type=int, default=1, help='Temps de génération en années, par défaut 1')
    parser.add_argument('--mu', type=float, default=1e-8, help='Taux de mutation par site et par génération, par défaut 1e-8')
    parser.add_argument('--omega', type=float, nargs='+', default=[0.3, 0.5, 1], help='Distance parameter pour SNIF, par défaut [0.3, 0.5, 1]') 
    args = parser.parse_args()

    main(args.json_output_dir, args.iicr_output_dir, args.psmc_output_dir, args.snif_output_dir, args.migration_rates, args.N0, args.taille, args.generation_time, args.mu, args.omega)

