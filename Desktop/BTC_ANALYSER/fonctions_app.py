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

def coinAPI_get_exchange_rates(asset_id_base, asset_id_quote, time_start, time_end, period_id):
    url = BASE_URL + "v1/exchangerate/"+ asset_id_base + "/" + asset_id_quote + "/history?period_id=" + period_id + "&time_start="+ time_start + "&time_end=" + time_end
    payload = {}
    headers = {
        'Accept': 'application/json',
        'X-CoinAPI-Key': API_KEY
    }
    try:
        response = requests.request("GET", url, headers=headers, data=payload)
        print('analyse sur le taux de change ' + asset_id_base + '/' + asset_id_quote + 'sur les 10 premiers jours de 2024')
        if response.status_code == 200:
            data = json.loads(response.text)
            # affichez les données

            for item in data:
                print(f"Date : {item['time_period_start'][:10]} Taux : {item['rate_close']}") #taux à la fermeture / fin de journée, on enlève les heures   
        else:
            print(f"Erreur {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Erreur lors de l'appel API : {e}")
