import numpy as np
import time

frame = np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]])

# fonction pour calculer le nombre de voisins
def compute_number_neighbors(paded_frame, index_row, index_col):
    """
    Cette fonction prend en entrée la matrice avec bordure et
    renvoie le nombre de cellules voisines vivantes
    """
    # Initialiser le nombre de voisins à 0
    number_neighbors = 0
    # parcourir les cellules voisines
    for row in range(index_row - 1, index_row + 2):
        # parcourir les colonnes voisines
        for col in range(index_col - 1, index_col + 2):
            # vérifier si la cellule est vivante et différente de la cellule actuelle
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