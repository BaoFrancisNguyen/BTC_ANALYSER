import requests
import json
from coinAPI_service import BASE_URL
from api_config import API_KEY
from fonctions_app import coinAPI_service_get_all_assets, coinAPI_get_exchange_rates, get_json_rates, save_json_rates

from datetime import date, timedelta

### affichier la date du jour et la date -10 jours

# date d'aujourd'hui
today = date.today()

# convertir la date en string
#date d'aujourd'hui
today_str = today.strftime("%Y-%m-%d")

# date d'y a 10 jours
delta_10 = today - timedelta(days=10)

# date d'y a 10 jours en string
diff_str = delta_10.strftime("%Y-%m-%d")

### limite de l'appel API en mode gratuit : max 100 appels par jour, max: historique de 100 jours

# générer les blocs de 100 jours et les stocker dans un fichier json
# pour chaque bloc de 100 jours, on fait un appel API pour récupérer les taux de change BTC/EUR

### pour l'année 2024 ###
# bloc 1 : 2024-01-01 -> 2024-04-09
# bloc 2 : 2024-04-10 -> 2024-07-18
# bloc 3 : 2024-07-19 -> 2024-10-26
# bloc 4 : 2024-10-27 -> date du jour

# faire un algorithme pour générer les dates de début et de fin pour chaque bloc
# consolider les données dans un fichier json

asset_id_base = 'BTC'
asset_id_quote = 'EUR'

rates = coinAPI_get_exchange_rates(asset_id_base='BTC', asset_id_quote='EUR', time_start=diff_str, time_end= today_str, period_id='1DAY')
filename = asset_id_base + '_' + asset_id_quote + ".json"

if rates:
    json_rates = get_json_rates(rates)
    print(json_rates)
    save_json_rates(rates, filename)
    print(f"Les taux de change ont été sauvegardés dans le fichier {filename}")
else:
    print("Erreur lors de la récupération des taux de change")


