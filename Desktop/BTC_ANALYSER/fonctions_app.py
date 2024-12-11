import requests
import json
from coinAPI_service import BASE_URL
from api_config import API_KEY
from datetime import date, timedelta

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

def find_missing_dates(time_start, time_end, rates):
    all_dates = set(
        (time_start + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((time_end - time_start).days + 1)
    )
    returned_dates = set(item['time_period_start'][:10] for item in rates)
    missing_dates = all_dates - returned_dates
    return sorted(missing_dates)       