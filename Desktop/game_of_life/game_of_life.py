import numpy as np
import time

# configuration de la frame
'''frame = np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]])'''

import numpy as np

# Demander à l'utilisateur les dimensions de la frame
rows = int(input("Entrez le nombre de lignes (rows) de la frame : "))
cols = int(input("Entrez le nombre de colonnes (cols) de la frame : "))

# Créer une frame de zéros avec la taille spécifiée
frame = np.zeros((rows, cols), dtype=int)

# Menu pour choisir un motif
print("\nChoisissez un motif à ajouter à la frame :")
print("1. Ligne de '1' au centre")
print("2. Carré de '1' au centre")
print("3. Fer à cheval")
print("4. Aucun motif (frame vide)")

# Demander à l'utilisateur de choisir un motif
motif_choice = int(input("Entrez le numéro du motif choisi (1-4) : "))

# Appliquer le motif choisi
if motif_choice == 1:  # Ligne de '1' au centre
    if rows >= 3 and cols >= 3:
        center_row = rows // 2
        start_col = (cols // 2) - 1
        frame[center_row, start_col:start_col + 3] = 1
elif motif_choice == 2:  # Carré de '1' au centre
    if rows >= 3 and cols >= 3:
        center_row = rows // 2
        center_col = cols // 2
        frame[center_row - 1:center_row + 2, center_col - 1:center_col + 2] = 1
elif motif_choice == 3:  # Fer à cheval
    if rows >= 5 and cols >= 5:
        center_row = rows // 2
        center_col = cols // 2
        frame[center_row - 2:center_row + 3, center_col - 2:center_col + 3] = 0
        frame[center_row - 2, center_col] = 1
        frame[center_row - 1:center_row + 2, center_col - 2:center_col + 3] = 1
elif motif_choice == 4:  # Aucun motif
    pass
else:
    print("Choix invalide. Aucune modification appliquée.")

# Afficher la frame résultante
print("\nVoici la frame créée :")
print(frame)


### Index de la frame:

'''Frame actuelle :
    0 1 2 3 4 5 6
0 | 0 0 0 0 0 0 0
1 | 0 0 0 0 0 0 0
2 | 0 0 0 0 0 0 0
3 | 0 0 1 1 1 0 0
4 | 0 0 0 0 0 0 0
5 | 0 0 0 0 0 0 0
6 | 0 0 0 0 0 0 0'''

# fonction pour calculer le nombre de voisins
def compute_number_neighbors(paded_frame, index_row, index_col):
    """
    Cette fonction prend en entrée la matrice avec bordure et
    renvoie le nombre de cellules voisines vivantes
    """
    # Initialiser le nombre de voisins à 0
    number_neighbors = 0
    # on cherche les cellules voisines de la cellule actuelle (contenant les 1) ---> on parcourt les cellules voisines
     # on ne doit pas inclure la cellule actuelle

        # position de la cellule actuelle : index_row, index_col
        # position des cellules voisines : (index_row - 1, index_col - 1), (index_row - 1, index_col), (index_row - 1, index_col + 1)
    # vérifier si la cellule est vivante et différente de la cellule actuelle
    # if not (row == index_row and col == index_col): # on ne vérifie pas la cellule actuelle
    # ajouter la valeur de la cellule à number_neighbors

    
    
    for row in range(index_row - 1, index_row + 2): #attention à l'exclusion de la valeur de fin / pas index_row + 1 mais index_row + 2
        # parcourir les colonnes voisines
        for col in range(index_col - 1, index_col + 2): #attention à l'exclusion de la valeur de fin / pas index_col + 1 mais index_col + 2

            # vérifier si la cellule est vivante et différente de la cellule actuelle
            # La boucle parcourt toutes les cases autour de la cellule centrale, y compris la cellule centrale elle-même

            # On doit ajouter if not pour ignorer la cellule centrale.
            if not (row == index_row and col == index_col):
                # ajouter la valeur de la cellule à number_neighbors
                number_neighbors += paded_frame[row, col]
                
    return number_neighbors


# fonction pour calculer la frame suivante
def compute_next_frame(frame):
    """
    Cette fonction prend en entrée une frame et calcule la frame suivante
    à partir des règles du jeu de la vie
    """
    paded_frame = np.pad(frame, 1, mode="constant")  # ajoute une bordure de zéros autour de la matrice
    new_frame = np.copy(frame)  # crée une copie de la frame pour stocker les nouveaux états
    
    # Parcourir les cellules de la frame
    for row_next in range(1, paded_frame.shape[0] - 1):
        # Parcourir les colonnes de la frame
        for col_next in range(1, paded_frame.shape[1] - 1):
            # Calculer le nombre de voisins vivants
            num_neighbors = compute_number_neighbors(paded_frame, row_next, col_next)
            # Vérifier si la cellule est vivante
            if paded_frame[row_next, col_next] == 1:
                # Règles pour les cellules vivantes: si le nombre de voisins est inférieur à 2 ou supérieur à 3, la cellule meurt
                if num_neighbors < 2 or num_neighbors > 3:
                    # La cellule meurt
                    new_frame[row_next-1, col_next-1] = 0  # Meurt par solitude ou surpopulation
            else:
                # Règle pour les cellules morte: si le nombre de voisins est égal à 3, la cellule devient vivante
                if num_neighbors == 3:
                    # La cellule devient vivante
                    new_frame[row_next-1, col_next-1] = 1  # Naît par reproduction
    
    return new_frame

# Afficher la frame initiale
while True:
    print(frame)
    frame = compute_next_frame(frame)
    time.sleep(1)  # Ajouter une pause de 1 seconde entre les itérations
