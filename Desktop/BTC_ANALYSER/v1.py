import requests
import json
from coinAPI_service import BASE_URL
from api_config import API_KEY
from fonctions_app import coinAPI_service_get_all_assets, coinAPI_get_exchange_rates

from datetime import date, timedelta

### affichier la date du jour et la date -10 jours

# date d'aujourd'hui
today = date.today()

# convertir la date en string
#date d'aujourd'hui
today_str = today.strftime("%Y-%m-%d")

# date d'y a 10 jours
delta_10 = today - timedelta(days=10)

# date d'y a 10 jours en string
diff_str = delta_10.strftime("%Y-%m-%d")


coinAPI_get_exchange_rates(asset_id_base='BTC', asset_id_quote='EUR', time_start=diff_str, time_end= today_str, period_id='1DAY')
