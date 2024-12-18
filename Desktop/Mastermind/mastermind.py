###Le Mastermind ou Master Mind est un jeu de société pour deux joueurs dont le but est de trouver un code 
# (couleur et position de 4 éléments) en 10 coups

# Importation de la bibliothèque random
import random

# Affiche un message d'accueil avec les règles du jeu
def show_game_instructions():
    print("Bienvenue dans Mastermind")
    print("Tu dois deviner une combinaison de 4 couleurs")
    print("Les couleurs possibles sont : Rouge, Bleu, Vert, Jaune, Orange, Violet")
    print("Tu as 10 essais pour réussir. Bonne chance\n")

# Générer un code secret aléatoire

def generate_secret_code():

    # liste des couleurs possibles
    couleurs = ["Rouge", "Bleu", "Vert", "Jaune", "Orange", "Violet"]
    # création d'une liste vide pour stocker le code secret
    code_secret = []
    # Boucle pour générer 4 couleurs
    # for_ pour indiquer que la variable n'est pas utilisée
    for _ in range(4): #génère une séquence de 4 nombres
        # Ajout d'une couleur aléatoire à la liste
        code_secret.append(random.choice(couleurs))
        # Retourne le code secret
    return code_secret