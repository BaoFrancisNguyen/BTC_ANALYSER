import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime

def scrape_reddit_mentions(keyword, subreddit="cryptocurrency", max_posts=50):
    base_url = f"https://www.reddit.com/r/{subreddit}/search"
    params = {
        "q": keyword,
        "restrict_sr": 1,
        "sort": "new",
        "limit": max_posts
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.62 Safari/537.36"
    }

    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code != 200:
        print(f"Erreur lors de la récupération des données : {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    posts = soup.find_all("a", {"data-testid": "post-title"})  # Sélecteur pour les titres

    mentions = []
    for post in posts[:max_posts]:
        title = post.get("aria-label", "Titre non disponible")
        url = f"https://www.reddit.com{post['href']}" if post.has_attr("href") else ""
        timestamp = datetime.now().strftime("%Y-%m-%d")
        mentions.append({
            "date": timestamp,
            "title": title,
            "url": url
        })
        print(f"Titre trouvé : {title}")  # Debugging

    return mentions

#utilisation

mentions = scrape_reddit_mentions("bitcoin", "cryptocurrency", max_posts=100)
with open("reddit_mentions.json", "w", encoding="utf-8") as f:
    json.dump(mentions, f, indent=4)

print(f"Mentions sauvegardées dans reddit_mentions.json : {len(mentions)} posts trouvés.")

