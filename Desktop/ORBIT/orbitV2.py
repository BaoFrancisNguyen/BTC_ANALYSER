import requests
from bs4 import BeautifulSoup
from notion_client import Client
from datetime import datetime
import ollama
from googlesearch import search
from config import NOTION_TOKEN, DATABASE_ID
from tqdm import tqdm  # Barre de progression

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

def rechercher_articles_ia():
    requete = "intelligence artificielle actualités"
    liens = list(search(requete, num_results=20))

    articles = []
    print("\n Recherche d'articles en cours...")
    for url in tqdm(liens, desc="Recherche d'articles", unit="article"):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")

            # Trouver le titre de l'article
            titre = soup.find("h1") or soup.find("h2")
            if titre:
                titre = titre.text.strip()
                articles.append({"title": titre, "url": url})
        except Exception as e:
            print(f" Erreur en récupérant {url} : {e}")

    return articles

# Fonction évolutive de scoring
def score_fiabilite(source_url):
    domaine = source_url.split("/")[2]  # Récupérer le domaine du site
    score = historique_fiabilite.get(domaine, 5)  # Score par défaut à 5 si inconnu

    # Critères d'évolution
    if score >= 8:
        score += 0.5  # Boost si site déjà fiable
    elif score <= 5:
        score -= 1  # Réduction pour sites peu connus

    # Mise à jour du dictionnaire (on pourrait stocker dans Notion aussi)
    historique_fiabilite[domaine] = max(3, min(10, score))  # Score entre 3 et 10
    return historique_fiabilite[domaine]

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
