###Le Mastermind ou Master Mind est un jeu de société pour deux joueurs dont le but est de trouver un code 
# (couleur et position de 4 éléments) en 10 coups

# Importation de la bibliothèque random
import random

# Affiche un message d'accueil avec les règles du jeu
def show_game_instructions():
    print("Mastermind")
    print("Tu dois deviner une combinaison de 4 couleurs")
    print("Les couleurs possibles sont : Rouge, Bleu, Vert, Jaune, Orange, Violet")

# Générer un code secret aléatoire

def generate_secret_code():

    # liste des couleurs possibles
    color_options = ["Rouge", "Bleu", "Vert", "Jaune", "Orange", "Violet"]
    # création d'une liste vide pour stocker le code secret
    generated_code = []
    # Boucle pour générer 4 couleurs
    # for_ pour indiquer que la variable n'est pas utilisée
    for _ in range(4): #génère une séquence de 4 nombres
        # Ajout d'une couleur aléatoire à la liste
        generated_code.append(random.choice(color_options))
        # Retourne le code secret
    return generated_code

# combinaison du joueur / input
def get_player_combination():
    # combinaison du joueur
    user_input = input("Tapez votre combinaison (couleurs séparées par des espaces, attention à la casse) : ")
    # split() pour séparer les couleurs et les stocker dans une liste
    player_combination = user_input.split()
    # retourne la combinaison du joueur
    return player_combination

# validité de la combinaison
# --> Vérifie si la combinaison est valide

def verify_color_selection(player_combinaison, color_options):
    # Vérifie si la combinaison contient exactement 4 couleurs
    if len(player_combinaison) != 4:
        print("Tu dois entrer exactement 4 couleurs")
    
        return False
    
    # Vérifie si chaque couleur de la combinaison est valide
    for couleur in player_combinaison:  # Boucle pour vérifier chaque couleur
        # Si la couleur n'est pas dans la liste des couleurs possibles
        if couleur not in color_options:
            print(f"'{couleur}' n'est pas une couleur valide")
            return False
    return True

# Compare la combinaison du joueur avec le code secret
def compare_combinations(player_combinaison, generated_code):
    # Initialisation des variables
    nombre_bien_places = 0
    nombre_mal_places = 0

    # Boucle pour parcourir les positions
    
    for position_couleur in range(4):
        # Si la couleur est bien placée
        if player_combinaison[position_couleur] == generated_code[position_couleur]:
            nombre_bien_places += 1 #incrémente le nombre de couleurs bien placées
            #si la couleur est mal placée
        elif player_combinaison[position_couleur] in generated_code:
            nombre_mal_places += 1 #incrémente le nombre de couleurs mal placées

    return nombre_bien_places, nombre_mal_places