import requests
import json
from coinAPI_service import BASE_URL
from api_config import API_KEY
from fonctions_app import coinAPI_service_get_all_assets, coinAPI_get_exchange_rates


coinAPI_get_exchange_rates(asset_id_base='BTC', asset_id_quote='EUR', time_start='2024-01-01T00:00:00', time_end='2024-01-10T00:00:00', period_id='1DAY')
