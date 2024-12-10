

import requests
from api_config import API_KEY
BASE_URL = "https://rest.coinapi.io/"
url = BASE_URL + "v1/assets"


payload = {}
headers = {
  'Accept': 'text/plain',
  'X-CoinAPI-Key': API_KEY
}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)