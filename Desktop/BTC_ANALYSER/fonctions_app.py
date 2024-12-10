import requests
import json

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
    url = BASE_URL + "v1/exchangerate/BTC/EUR?period_id=1DAY&date_start=2024-01-01T00:00:00&date_end=2024-01-10T00:00:00"
    payload = {}
    headers = {
        'Accept': 'text/plain',
        'X-CoinAPI-Key': API_KEY
    }
    try:
        response = requests.request("GET", url, headers=headers, data=payload)
        if response.status_code == 200:

            # Désérialiser le JSON
            data = json.loads(response.text)

            # Afficher les taux de change pour chaque jour

            for item in data.get('rates', []):  # Utilisez data.get pour éviter KeyError
                print(f"{item['time_period_start']} : {item['rates']}")
        else:
            print(f"Erreur {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Erreur lors de l'appel API : {e}")