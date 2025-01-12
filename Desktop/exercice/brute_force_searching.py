liste = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
target = 8

# searching i + j = target
# return index_i, index_j

# méthode brute force 
# on créera une fonction qui prend en paramètres la liste et le résultat cible
def searching_target(liste, target):
    # on parcourt la liste avec 1 élément i
    for index_i in range(len(liste)):
        # on parcourt la liste avec 1 élément j / on commence à index_i + 1 pour ne pas avoir de doublons
        for index_j in range(index_i + 1, len(liste)):
            # si la somme de i et j est égale à la cible / on retourne les index de i et j
            if liste[index_i] + liste[index_j] == target:
                return[index_i, index_j]
    # si on ne trouve pas de solution, on retourne None        
    else:
        return None
    
print(searching_target(liste, target))

