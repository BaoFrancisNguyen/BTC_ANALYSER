import requests
import json
from coinAPI_service import BASE_URL
from api_config import API_KEY
from datetime import date, timedelta, datetime

    ####Fonctions####

def coinAPI_service_get_all_assets():
    
    url = BASE_URL + "v1/assets"
    payload = {}
    headers = {
      'Accept': 'text/plain',
      'X-CoinAPI-Key': API_KEY
    }
    response = requests.request("GET", url, headers=headers, data=payload)
    return response

def coinAPI_get_exchange_rates(asset_id_base, asset_id_quote, time_start, time_end, period_id):
    url = BASE_URL + "v1/exchangerate/" + asset_id_base + "/" + asset_id_quote + "/history"
    params = {
        "period_id": period_id,
        "time_start": time_start,
        "time_end": time_end
    }
    headers = {
        'Accept': 'application/json',
        'X-CoinAPI-Key': API_KEY
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        print(f"Analyse sur le taux de change {asset_id_base}/{asset_id_quote} entre {time_start} et {time_end}")
        if response.status_code == 200:
            data = response.json()
            # Optionnel : Affichage des données pour vérification
            for item in data:
                print(f"Date : {item['time_period_start'][:10]}, Taux : {item['rate_close']}")
            return data
        else:
            print(f"Erreur {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Erreur lors de l'appel API : {e}")
        return None




def get_json_rates(rates_data):
    rate_json = []
    for item in rates_data:
        rate_json.append({"date": item['time_period_start'][:10], "value": item['rate_close']})
    return json.dumps(rate_json, indent=4)  # Indenté pour une meilleure lisibilité

    

def save_json_rates(rates_data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(get_json_rates(rates_data))

def load_json_data_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        json = f.read()
        f.close()
        return json

def find_missing_dates(time_start, time_end, rates):
    # Convertir time_start et time_end en date si nécessaire
    if isinstance(time_start, datetime):
        time_start = time_start.date()
    if isinstance(time_end, datetime):
        time_end = time_end.date()

    # Générer toutes les dates dans l'intervalle
    all_dates = set(
        (time_start + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((time_end - time_start).days + 1)
    )
    # Obtenir les dates présentes dans les données
    returned_dates = set(item['date'] for item in rates)
    # Trouver les dates manquantes
    missing_dates = all_dates - returned_dates
    return sorted(missing_dates) 


# date_start / date_end : object
# max_days : int
# return : list
# start : 2021-01-01
# end : 2021-05-01
# max_days : 100
# [2021-01-01 / 2021-04-11] [2021-04-12 / 2021-05-01]


def get_dates_interval(date_start, date_end, max_days=100):
    # Vérifier que les entrées sont du bon type
    if not isinstance(date_start, (date, datetime)) or not isinstance(date_end, (date, datetime)):
        raise ValueError("date_start et date_end doivent être de type datetime.date ou datetime.datetime")
    
    # Convertir datetime.datetime en datetime.date si nécessaire
    if isinstance(date_start, datetime):
        date_start = date_start.date()
    if isinstance(date_end, datetime):
        date_end = date_end.date()
    
    # Vérifier si les dates sont valides
    if date_start > date_end:
        raise ValueError("date_start doit être antérieure ou égale à date_end")
    
    if max_days <= 0:
        raise ValueError("max_days doit être un entier positif")

    # Calculer les intervalles
    diff = date_end - date_start
    diff_days = diff.days
    dates_interval = []
    while diff_days > 0:
        if diff_days > max_days:
            new_date_end = date_start + timedelta(days=max_days)
            dates_interval.append((date_start, new_date_end))
            date_start = new_date_end + timedelta(days=1)
            diff_days -= max_days
        else:
            dates_interval.append((date_start, date_end))
            diff_days = 0
    return dates_interval



       