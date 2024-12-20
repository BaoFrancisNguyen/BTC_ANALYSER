import requests
import json
from coinAPI_service import BASE_URL
from api_config import API_KEY
from datetime import date, timedelta, datetime
import os

#### Fonctions ####

def coinAPI_service_get_all_assets():
    url = BASE_URL + "v1/assets"
    headers = {
        'Accept': 'text/plain',
        'X-CoinAPI-Key': API_KEY
    }
    response = requests.get(url, headers=headers)
    return response

def coinAPI_get_exchange_rates(asset_id_base, asset_id_quote, time_start, time_end, period_id):
    url = BASE_URL + f"v1/exchangerate/{asset_id_base}/{asset_id_quote}/history"
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
        if 'time_period_start' in item and 'rate_close' in item:
            rate_json.append({
                "date": item['time_period_start'][:10],
                "value": item['rate_close']
            })
        else:
            print(f"Donnée ignorée : {item}")
    return json.dumps(rate_json, indent=4)

def save_json_rates(rates_data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(get_json_rates(rates_data))

def load_json_data_from_file(filename):
    if not os.path.exists(filename):
        print(f"Fichier {filename} introuvable. Création d'un nouveau fichier.")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("[]")
        return "[]"
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        if not content.strip():
            print(f"Le fichier {filename} est vide. Initialisation avec une liste vide.")
            return "[]"
        return content

def find_missing_dates(time_start, time_end, rates):
    all_dates = set(
        (time_start + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((time_end - time_start).days + 1)
    )
    returned_dates = set(item['date'] for item in rates)
    missing_dates = all_dates - returned_dates
    return sorted(missing_dates)

def get_dates_interval(date_start, date_end, max_days=100):
    if not isinstance(date_start, (date, datetime)) or not isinstance(date_end, (date, datetime)):
        raise ValueError("date_start et date_end doivent être de type datetime.date ou datetime.datetime")
    if isinstance(date_start, datetime):
        date_start = date_start.date()
    if isinstance(date_end, datetime):
        date_end = date_end.date()
    if date_start > date_end:
        raise ValueError("date_start doit être antérieure")



def get_social_mentions(keyword, start_date, end_date, platform='twitter'):
    """Récupère les mentions d'un mot-clé sur une plateforme sociale."""
    url = f"https://api.socialmediaapi.com/{platform}/mentions"
    headers = {
        'Authorization': 'Bearer YOUR_API_KEY',  # Remplacer par votre clé API
        'Content-Type': 'application/json'
    }
    params = {
        'q': keyword,
        'from': start_date,
        'to': end_date
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            mentions = [
                {"date": item["created_at"][:10], "count": 1}
                for item in data.get("results", [])
            ]
            return mentions
        else:
            print(f"Erreur {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Erreur lors de l'appel API : {e}")
        return None

def save_social_data(data, filename="social_mentions.json"):
    """Sauvegarde les mentions sociales dans un fichier JSON."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# Exemple d'utilisation
start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')
mentions = get_social_mentions("bitcoin", start_date, end_date, platform='twitter')
if mentions:
    save_social_data(mentions)


       