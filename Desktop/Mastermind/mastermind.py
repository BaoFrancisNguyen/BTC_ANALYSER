###Le Mastermind ou Master Mind est un jeu de société pour deux joueurs dont le but est de trouver un code 
# (couleur et position de 4 éléments) en 10 coups

# Importation de la bibliothèque random
import random

# Affiche un message d'accueil avec les règles du jeu
def afficher_regles():
    print("Bienvenue dans Mastermind")
    print("Tu dois deviner une combinaison de 4 couleurs")
    print("Les couleurs possibles sont : Rouge, Bleu, Vert, Jaune, Orange, Violet")
    print("Tu as 10 essais pour réussir. Bonne chance\n")