from pytrends.request import TrendReq
import pandas as pd
import matplotlib.pyplot as plt
import time

# Connexion à Google Trends et récupération des données
# Liste de proxies (remplacer par les proxies valides)

proxies = ["http://123.45.67.89:8080"]  # Exemple de proxy
pytrends = TrendReq(hl='fr-FR', tz=360)

keywords = ["Bitcoin", "BTC"]  # Mots-clés à rechercher / le nombre de mots-clés est limité à 5 / attention au nombre de requêtes qui est limité à 5 par minute
for keyword in keywords:
    try:
        pytrends.build_payload([keyword], timeframe='2024-09-01 2024-10-28', geo='FR')
        data = pytrends.interest_over_time()
        print(data)
        time.sleep(60)  # Pause de 60 secondes

    except Exception as e:
        print(f"Erreur pour {keyword}: {e}")
        time.sleep(120)  # Pause plus longue en cas d'erreur


# Vérifier et sauvegarder les données dans un fichier CSV

if not data.empty:
    data.to_csv('search_trends_bitcoin.csv')
    #test
    print("Données enregistrées dans 'search_trends_bitcoin.csv'.")

# Afficher un graphique des résultats

data.plot(title="Volume de recherche pour Bitcoin et BTC")
plt.xlabel("Date")
plt.ylabel("Volume de recherche")
plt.show()
