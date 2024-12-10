import requests
import json
from api_config import API_KEY
BASE_URL = "https://rest.coinapi.io/"
url = BASE_URL + "v1/assets"


payload = {}
headers = {
  'Accept': 'text/plain',
  'X-CoinAPI-Key': API_KEY
}

response = requests.request("GET", url, headers=headers, data=payload)

## APPEL A L'API
# 1. Afficher le status_code
# 2. Si le status_code est 200, afficher les données
# 3. Si le status_code est différent de 200, afficher une erreur

if response.status_code == 200:

  # Afficher le contenu de la réponse
  
  data = json.loads(response.text)

  # Afficher le nombre d'assets
  nb_assets = len(data)
  print('Nombre d\'assets:', nb_assets)

  if nb_assets > 0:

    # Afficher les 10 premiers assets
    for i in range(10):
      print(data[i]['name'])
    
  print('l\'appel à l\'API a fonctionné')

  # Afficher le quota restant

  
  #print('Quota restant:', response.headers['x-ratelimit-remaining'])
  #print('Quota total:', response.headers['x-ratelimit-limit'])

else:
  # Afficher une erreur d'appel à l'API
  print('Erreur',response.status_code, ', l\'appel à l\'API a retourné une erreur')


#Créer une fonction coinAPI_service_get_all_assets() qui retourne la réponse de l'API:

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
                print(f"{item['time_period_start']} : {item['rate']}")
        else:
            print(f"Erreur {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Erreur lors de l'appel API : {e}")





   
