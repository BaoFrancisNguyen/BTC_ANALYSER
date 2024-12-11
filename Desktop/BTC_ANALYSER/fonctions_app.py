import requests
import json
from coinAPI_service import BASE_URL
from api_config import API_KEY

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

def coinAPI_get_exchange_rates():
    url = BASE_URL + "v1/exchangerate/BTC/EUR/history?period_id=1DAY&time_start=2024-01-01T00:00:00&time_end=2024-01-10T00:00:00"
    payload = {}
    headers = {
        'Accept': 'application/json',
        'X-CoinAPI-Key': API_KEY
    }
    try:
        response = requests.request("GET", url, headers=headers, data=payload)
        if response.status_code == 200:
            data = json.loads(response.text)
            # Vérifiez et affichez les données
            for item in data:
                print(f"Date : {item['time_period_start']}, Taux : {item['rate_close']}")
        else:
            print(f"Erreur {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Erreur lors de l'appel API : {e}")
