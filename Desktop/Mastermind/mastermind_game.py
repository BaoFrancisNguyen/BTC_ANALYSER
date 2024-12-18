###Le Mastermind ou Master Mind est un jeu de société pour deux joueurs dont le but est de trouver un code 
# (couleur et position de 4 éléments) en 10 coups


# Fonction principale du jeu

# importation des fonctions du fichier mastermind.py
from mastermind import show_game_instructions, generate_secret_code, get_player_combination, verify_color_selection, compare_combinations

def mastermind():
    show_game_instructions()
    color_options = ["Rouge", "Bleu", "Vert", "Jaune", "Orange", "Violet"]
    generated_code = generate_secret_code()

    # Boucle pour gérer les 10 essais
    for essai_actuel in range(10): 
        essais_restants = 10 - essai_actuel 
        print(f"Il reste {essais_restants} essai(s)")
        # redemande une combinaison au joueur
        player_combinaison = get_player_combination()

        # vérifie si la combinaison est correcte
        valid_code = verify_color_selection(player_combinaison, color_options)
        # Si la combinaison n'est pas valide, on redemande une combinaison
        if not valid_code:
            
            continue

        # Compare la combinaison
        nombre_bien_places, nombre_mal_places = compare_combinations(player_combinaison, generated_code)
        print(f"Couleurs bien placées : {nombre_bien_places}, Couleurs correctes mal placées : {nombre_mal_places}")

        # Si le joueur a trouvé le code secret
        if nombre_bien_places == 4:
            print("Tu as trouvé le code secret")
            # Sortie de la boucle
            return

    # Si les essais sont épuisés
    print("GAME OVER. Tu as utilisé tous tes essais")
    print("Le code secret était :", " ".join(generated_code))

# Lancement du jeu
mastermind()