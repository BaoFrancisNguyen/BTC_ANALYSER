import requests
from bs4 import BeautifulSoup
from notion_client import Client
from datetime import datetime
import ollama
from googlesearch import search
from config import NOTION_TOKEN, DATABASE_ID
from tqdm import tqdm  # Barre de progression
import json
import schedule
import time

# ✅ Configuration Notion
notion = Client(auth=NOTION_TOKEN)

# ✅ Charger les préférences utilisateur
try:
    with open("preferences.json", "r") as f:
        preferences = json.load(f)
except FileNotFoundError:
    preferences = {"themes": ["intelligence artificielle"], "min_reliability": 5, "language": "fr"}

# ✅ Dictionnaire de fiabilité des sources
historique_fiabilite = {
    "lemonde.fr": 9,
    "numerama.com": 8,
    "bbc.com": 9,
    "france24.com": 8,
    "medium.com": 6
}

# 🔍 **1. Rechercher des articles**
def rechercher_articles_ia():
    requete = f"{' OR '.join(preferences['themes'])} actualités"
    print(f"\n🔎 Recherche Google : {requete}")

    liens = list(search(requete, num_results=10))  # Limité à 10 pour test
    articles = []

    for url in tqdm(liens, desc="Recherche d'articles", unit="article"):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")

            titre = soup.find("h1") or soup.find("h2")
            if titre:
                titre = titre.text.strip()
                articles.append({"title": titre, "url": url})
        except Exception as e:
            print(f"⚠️ Erreur récupération {url} : {e}")

    print(f"🔍 {len(articles)} articles trouvés.")
    return articles

# 📊 **2. Évaluer la fiabilité d'une source**
def score_fiabilite(source_url):
    domaine = source_url.split("/")[2]
    score = historique_fiabilite.get(domaine, 7)  # Score par défaut 7

    # Ajustements
    if score >= 8:
        score += 0.5
    elif score <= 5:
        score -= 1

    historique_fiabilite[domaine] = max(3, min(10, score))
    return historique_fiabilite[domaine]

# 🤖 **3. Générer un résumé avec Ollama (Mistral)**
def generer_resume(article_url):
    prompt = f"Lis cet article {article_url} et résume-le en 3 phrases clés."
    try:
        reponse = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
        return reponse["message"]["content"]
    except Exception as e:
        print(f"⚠️ Erreur avec Ollama : {e}")
        return "Résumé indisponible."

# 📝 **4. Ajouter un article dans Notion**
def ajouter_dans_notion(title, summary, source_url, reliability):
    try:
        print(f"🚀 Ajout dans Notion : {title}")
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
        print(f"✅ Ajout réussi : {title}")
    except Exception as e:
        print(f"❌ Erreur d'ajout dans Notion : {e}")

# 🚀 **Exécution principale**
print("🔄 Démarrage de l'agent IA...")
articles = rechercher_articles_ia()

for article in articles:
    print(f"\n📄 Traitement : {article['title']} - {article['url']}")
    try:
        resume = generer_resume(article["url"])
        score = score_fiabilite(article["url"])
        
        if score >= preferences["min_reliability"]:
            ajouter_dans_notion(article["title"], resume, article["url"], score)
        else:
            print(f"⏭ Article ignoré (fiabilité trop basse : {score})")

    except Exception as e:
        print(f"❌ Erreur pour {article['title']}: {e}")

print("🎯 Processus terminé avec mise à jour du scoring !")



def lancer_agent():
    print("\n🔄 Exécution automatique de l'agent IA...")
    articles = rechercher_articles_ia()
    
    for article in articles:
        print(f"\n📄 Traitement : {article['title']} - {article['url']}")
        try:
            resume = generer_resume(article["url"])
            score = score_fiabilite(article["url"])
            
            if score >= preferences["min_reliability"]:
                ajouter_dans_notion(article["title"], resume, article["url"], score)
            else:
                print(f"⏭ Article ignoré (fiabilité trop basse : {score})")
        except Exception as e:
            print(f"❌ Erreur pour {article['title']}: {e}")

    print("\nProcessus terminé avec mise à jour du scoring !")

# 🔥 Planifier l'exécution tous les jours à 8h du matin
schedule.every().day.at("08:00").do(lancer_agent)

print("⏳ L'agent est en attente d'exécution... (Ctrl+C pour quitter)")

# Boucle infinie pour exécuter l'agent chaque jour
while True:
    schedule.run_pending()
    time.sleep(60)  # Vérification toutes les minutes
