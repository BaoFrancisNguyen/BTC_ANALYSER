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

