from datetime import datetime, timedelta
from fonctions_app import get_social_mentions, save_social_data


start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')
mentions = get_social_mentions("bitcoin", start_date, end_date, platform='twitter')
if mentions:
    save_social_data(mentions)