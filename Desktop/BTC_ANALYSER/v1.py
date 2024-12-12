import requests
import json
from coinAPI_service import BASE_URL
from api_config import API_KEY
from fonctions_app import (
    coinAPI_service_get_all_assets, 
    coinAPI_get_exchange_rates, 
    get_json_rates, 
    save_json_rates, 
    find_missing_dates, 
    load_json_data_from_file, 
    get_dates_interval
)
from datetime import date, timedelta, datetime
from os import path

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
    print(f"Chargement des données depuis {filename}...")
    json_rates = load_json_data_from_file(filename)
    try:
        rates = json.loads(json_rates) if json_rates else []
    except json.JSONDecodeError:
        print(f"Erreur : Fichier {filename} corrompu. Initialisation avec une liste vide.")
        rates = []
    if rates:
        save_data_date_start = rates[0]['date']
        save_data_date_end = rates[-1]['date']
        print(f"Les données existantes couvrent de {save_data_date_start} à {save_data_date_end}.")
        start_date = datetime.strptime(save_data_date_start, "%Y-%m-%d")
        end_date = datetime.strptime(save_data_date_end, "%Y-%m-%d")
        missing_dates = find_missing_dates(start_date, today, rates)
        if missing_dates:
            print(f"Dates manquantes : {missing_dates}")
    else:
        print("Aucune donnée trouvée dans le fichier.")
else:
    print(f"Aucun fichier trouvé. Création de {filename}...")
    rates = coinAPI_get_exchange_rates(asset_id_base, asset_id_quote, diff_str, today_str, period_id='1DAY')
    if rates:
        save_json_rates(rates, filename)
        print(f"Données sauvegardées dans {filename}.")
    else:
        print("Erreur : Impossible de récupérer les données via l'API.")

# Calcul des intervalles de dates
date_start_str = "2021-01-01"
date_end_str = "2021-01-10"
try:
    date_start = datetime.strptime(date_start_str, "%Y-%m-%d").date()
    date_end = datetime.strptime(date_end_str, "%Y-%m-%d").date()
    intervals = get_dates_interval(date_start, date_end)
    if not intervals:
        print("Aucun intervalle valide trouvé. Vérifiez les dates.")
except Exception as e:
    print(f"Erreur lors de la génération des intervalles : {e}")
    intervals = []

# Charger les données existantes
existing_data = load_json_data_from_file(filename)
if existing_data:
    try:
        existing_rates = json.loads(existing_data)
    except json.JSONDecodeError:
        print(f"Erreur de décodage JSON dans {filename}. Initialisation avec une liste vide.")
        existing_rates = []
else:
    print(f"Le fichier {filename} est vide. Initialisation avec une liste vide.")
    existing_rates = []

# Récupérer et ajouter les données manquantes
# Charger ou créer le fichier JSON

if path.exists(filename):
    print(f"Chargement des données depuis {filename}...")
    json_rates = load_json_data_from_file(filename)
    try:
        rates = json.loads(json_rates) if json_rates else []
    except json.JSONDecodeError:
        print(f"Erreur : Fichier {filename} corrompu. Initialisation avec une liste vide.")
        rates = []
    if rates:
        save_data_date_start = rates[0]['date']
        save_data_date_end = rates[-1]['date']
        print(f"Les données existantes couvrent de {save_data_date_start} à {save_data_date_end}.")
        start_date = datetime.strptime(save_data_date_start, "%Y-%m-%d")
        end_date = datetime.strptime(save_data_date_end, "%Y-%m-%d")
        missing_dates = find_missing_dates(start_date, today, rates)
        if missing_dates:
            print(f"Dates manquantes : {missing_dates}")
        else:
            print("Toutes les dates sont déjà couvertes.")
    else:
        print("Aucune donnée trouvée dans le fichier. Force une récupération complète.")
        rates = coinAPI_get_exchange_rates(asset_id_base, asset_id_quote, diff_str, today_str, period_id='1DAY')
        if rates:
            save_json_rates(rates, filename)
            print(f"Données initiales sauvegardées dans {filename}.")
        else:
            print("Erreur : Impossible de récupérer les données via l'API.")
else:
    print(f"Aucun fichier trouvé. Création de {filename}...")
    rates = coinAPI_get_exchange_rates(asset_id_base, asset_id_quote, diff_str, today_str, period_id='1DAY')
    if rates:
        save_json_rates(rates, filename)
        print(f"Données sauvegardées dans {filename}.")
    else:
        print("Erreur : Impossible de récupérer les données via l'API.")

# Récupérer les données manquantes par intervalles
if intervals:
    for interval in intervals:
        start_date = interval[0].strftime("%Y-%m-%d")
        end_date = interval[1].strftime("%Y-%m-%d")
        print(f"Récupération des données pour l'intervalle {start_date} à {end_date}...")
        rates = coinAPI_get_exchange_rates(asset_id_base, asset_id_quote, start_date, end_date, "1DAY")
        if rates:
            print(f"Données récupérées : {len(rates)} entrées.")
            existing_rates.extend(rates)
        else:
            print(f"Pas de données pour l'intervalle {start_date} - {end_date}")
else:
    print("Aucun intervalle à traiter.")

# Vérification des données avant sauvegarde

if existing_rates:
    print(f"Données avant suppression des doublons : {len(existing_rates)}")
    unique_rates = {item['date']: item for item in existing_rates}.values()
    unique_rates_list = sorted(unique_rates, key=lambda x: x['date'])  # Trier par date
    print(f"Données après suppression des doublons : {len(unique_rates_list)}")
    save_json_rates(list(unique_rates_list), filename)
    print(f"Données consolidées sauvegardées dans {filename}.")
else:
    print("Aucune donnée à sauvegarder.")


