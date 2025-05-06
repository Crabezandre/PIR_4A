import os
import csv
import pandas as pd

# Chemin vers le répertoire initial
repertoire_initial = '.'

# Chemin vers le fichier de sortie
fichier_sortie = os.path.join(repertoire_initial, 'resultats_generes.csv')

# Liste pour stocker les résultats
resultats = []

# Parcourir chaque dossier de résultats
for dossier_num in range(1, 2686):
    dossier_path = os.path.join(repertoire_initial, 'SNIF_results', f'{dossier_num:04d}')

    # Vérifier si le dossier existe
    if not os.path.isdir(dossier_path):
        continue

    # Chemin vers le fichier exemple.csv
    fichier_csv = None
    for fichier in os.listdir(dossier_path):
        if fichier.endswith('.csv') and not fichier.endswith('_curves.csv'):
            fichier_csv = os.path.join(dossier_path, fichier)
            break

    if fichier_csv is None:
        continue

    # Lire le fichier exemple.csv
    with open(fichier_csv, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            individu = int(row['id'])+5*(dossier_num-1)
            inf_n = float(row['inf. n'])
            inf_m = float(row['inf. M'])
            inf_n_ref = float(row['inf. N_ref'])

            # Ajouter les résultats à la liste
            resultats.append([individu, inf_n, inf_m, inf_n_ref])

# Convertir la liste des résultats en DataFrame
df_resultats = pd.DataFrame(resultats, columns=['Individus', 'inf. n', 'inf. M', 'inf. N_ref'])
print(df_resultats)

# Lire le fichier listing_resultat_brut.ods pour obtenir les valeurs d'entrée
fichier_ods = os.path.join(repertoire_initial, 'listing_resultat_brut.ods')
df_entree = pd.read_excel(fichier_ods, engine='odf')

# Fusionner les DataFrames sur la colonne 'Individu'
df_final = pd.merge(df_entree, df_resultats, left_on='Individus', right_on='Individus', how='left')

# Sauvegarder le DataFrame final en fichier CSV
df_final.to_csv(fichier_sortie, index=False)

print(f"Fichier généré avec succès : {fichier_sortie}")

