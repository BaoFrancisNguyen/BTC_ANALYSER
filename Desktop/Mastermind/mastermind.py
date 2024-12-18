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

def verify_color_selection(combinaison, liste_couleurs_possibles):
    # Vérifie si la combinaison contient exactement 4 couleurs
    if len(combinaison) != 4:
        print("Tu dois entrer exactement 4 couleurs")
    
        return False
    
    # Vérifie si chaque couleur de la combinaison est valide
    for couleur in combinaison:  # Boucle pour vérifier chaque couleur
        # Si la couleur n'est pas dans la liste des couleurs possibles
        if couleur not in liste_couleurs_possibles:
            print(f"'{couleur}' n'est pas une couleur valide")
            return False
    return True