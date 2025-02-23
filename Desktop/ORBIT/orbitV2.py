import requests
from bs4 import BeautifulSoup
from notion_client import Client
from datetime import datetime
import ollama
from googlesearch import search
from config import NOTION_TOKEN, DATABASE_ID
from tqdm import tqdm  # Barre de progression
import json

# Config Notion
#NOTION_TOKEN = "ton_token_secret"
#DATABASE_ID = "ton_database_id"
notion = Client(auth=NOTION_TOKEN)
print("NOTION_TOKEN:", NOTION_TOKEN)


#  Dictionnaire pour suivre les scores des sources
# score de fiabilité de 1 à 10 (10 étant le plus fiable) prédéfini

historique_fiabilite = {
    "lemonde.fr": 9,
    "numerama.com": 8,
    "bbc.com": 9,
    "france24.com": 8,
    "medium.com": 6
}

# Recherche Google News pour IA

# Charger les préférences
with open("preferences.json", "r") as f:
    preferences = json.load(f)

def rechercher_articles_ia():
    requete = f"{' OR '.join(preferences['themes'])} actualités"
    liens = list(search(requete, num_results=20))
    
    articles = []
    for url in liens:
        # Récupération des articles comme avant...
        pass

    return articles

# Fonction évolutive de scoring
def score_pertinence(article):
    """ Attribuer un score en fonction de plusieurs critères """
    score = 0
    domaine = article["url"].split("/")[2]
    score += historique_fiabilite.get(domaine, 5) * 2  # Plus fiable = plus de points

    # Ajouter d'autres critères (ex: longueur du texte si récupérable)
    if len(article["title"]) > 50:  # Un titre plus long peut indiquer un article détaillé
        score += 2

    return score

# Trier les articles avant de les traiter
articles = sorted(rechercher_articles_ia(), key=score_pertinence, reverse=True)


# Fonction pour générer un résumé avec Ollama
def generer_resume(article_url):
    prompt = f"Lis cet article {article_url} et résume-le en 3 phrases en reprenant les points clés et les idées principales."
    reponse = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return reponse["message"]["content"]

# Ajouter à Notion
def ajouter_dans_notion(title, summary, source_url, reliability):
    notion.pages.create(
        parent={"database_id": DATABASE_ID},
        properties={
            "Title": {"title": [{"text": {"content": title}}]},
            "Summary": {"rich_text": [{"text": {"content": summary}}]},
            "Date": {"date": {"start": datetime.today().isoformat()}},
            "Source URL": {"url": source_url},
            "Reliability Score": {"number": reliability},
        },
    )
    print(f"✅ Article ajouté à Notion : {title}")

# Exécution de l'agent

print(" Recherche d'articles sur Google News...")
articles = rechercher_articles_ia()

for article in articles:
    try:
        print(f"Résumé de : {article['title']}")
        resume = generer_resume(article["url"])
        score = score_fiabilite(article["url"])
        ajouter_dans_notion(article["title"], resume, article["url"], score)
    except Exception as e:
        print(f"❌ Erreur pour {article['title']}: {e}")

print("Processus terminé avec mise à jour du scoring !")
