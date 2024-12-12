import requests
import json
from coinAPI_service import BASE_URL
from api_config import API_KEY
from fonctions_app import coinAPI_service_get_all_assets, coinAPI_get_exchange_rates, get_json_rates, save_json_rates, find_missing_dates, load_json_data_from_file, get_dates_interval
from datetime import date, timedelta, datetime
from os import path
from datetime import date, timedelta, datetime


# Configurations
asset_id_base = 'BTC'
asset_id_quote = 'EUR'
today = date.today()
today_str = today.strftime("%Y-%m-%d")
delta_100 = today - timedelta(days=100)
diff_str = delta_100.strftime("%Y-%m-%d")
filename = f"{asset_id_base}_{asset_id_quote}.json"

# Charger ou créer le fichier JSON
if path.exists(filename):
    #le fichier json existe
    print(f"Chargement des données depuis {filename}...")
    json_rates = load_json_data_from_file(filename)
    rates = json.loads(json_rates)
    if rates:
        save_data_date_start = rates[0]['date']
        save_data_date_end = rates[-1]['date']
        print(f"Les données existantes couvrent de {save_data_date_start} à {save_data_date_end}.")
        # Vérifier les dates manquantes
        start_date = datetime.strptime(save_data_date_start, "%Y-%m-%d")
        end_date = datetime.strptime(save_data_date_end, "%Y-%m-%d")
        missing_dates = find_missing_dates(start_date, today, rates)
        if missing_dates:
            print(f"Dates manquantes : {missing_dates}")
    else:
        print("Aucune donnée trouvée dans le fichier.")
#si le fichier json n'existe pas
else:
    print(f"Aucun fichier trouvé. Création de {filename}...")
    rates = coinAPI_get_exchange_rates(asset_id_base, asset_id_quote, diff_str, today_str, period_id='1DAY')
    if rates:
        save_json_rates(rates, filename)
        print(f"Données sauvegardées dans {filename}.")
    else:
        print("Erreur : Impossible de récupérer les données via l'API.")



# Les dates au format str
date_start_str = "2021-01-01"
date_end_str = "2021-01-10"

# Conversion en objets datetime.date
date_start = datetime.strptime(date_start_str, "%Y-%m-%d").date()
date_end = datetime.strptime(date_end_str, "%Y-%m-%d").date()

# Appel de la fonction
intervals = get_dates_interval(date_start, date_end)
print(intervals)

# si le fichier json existe:
# comparer le date_star avec le saved_data_date_start
# comparer le date_end avec le saved_data_date_end
# [date_start / saved_data_date_start] [date_end / saved_data_date_end]

#si le fichier json n'existe pas:
# créer le fichier json
# [date_start / date_end]

