import requests
from bs4 import BeautifulSoup
from notion_client import Client
from datetime import datetime
import ollama
from googlesearch import search
from config import NOTION_TOKEN, DATABASE_ID
from tqdm import tqdm
import json
import schedule # Bibliothèque pour la planification des tâches / agent
import time
import os
from textblob import TextBlob  # Bibliothèque pour l'analyse de sentiment

#  Configuration Notion
notion = Client(auth=NOTION_TOKEN)

# Charger les préférences utilisateur
try:
    with open("preferences.json", "r") as f:
        preferences = json.load(f)
except FileNotFoundError:
    preferences = {"themes": ["intelligence artificielle"], "min_reliability": 5, "language": "fr"}

# Vérification de l'existence du fichier JSONL
if not os.path.exists("mistral_training_data.jsonl"):
    with open("mistral_training_data.jsonl", "w", encoding="utf-8") as f:
        pass  # Création du fichier s'il n'existe pas

# Dictionnaire de fiabilité des sources
historique_fiabilite = {
    "lemonde.fr": 9,
    "numerama.com": 8,
    "bbc.com": 9,
    "france24.com": 8,
    "medium.com": 6
}

#  **1. Rechercher des articles**


def rechercher_articles_ia():
    requete = f"{' OR '.join(preferences['themes'])} actualités"
    print(f"\n Recherche Google : {requete}")

    liens = list(search(requete, num_results=10))
    articles = []

    for url in tqdm(liens, desc="Recherche d'articles", unit="article"):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")

            #  Nettoyage : Ne garder que le texte pertinent
            titre = soup.find("h1") or soup.find("h2")
            contenu = "\n".join([p.text.strip() for p in soup.find_all("p")])  # Extraire seulement les paragraphes

            if titre and contenu:
                titre = titre.text.strip()
                contenu = " ".join(contenu.split()[:300])  #  Limiter à 300 mots max
                articles.append({"title": titre, "url": url, "content": contenu})
        except Exception as e:
            print(f" Erreur récupération {url} : {e}")

    print(f"🔍 {len(articles)} articles trouvés.")
    return articles


#  **2. Évaluer la fiabilité d'une source**
def score_fiabilite(source_url):
    domaine = source_url.split("/")[2]
    return historique_fiabilite.get(domaine, 7)  # Score par défaut 7

#  **3. Générer un résumé avec Ollama (Mistral)**
def generer_resume(article_url):
    prompt = f"Lis cet article {article_url} et résume-le en 3 phrases clés."
    try:
        reponse = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
        return reponse["message"]["content"]
    except Exception as e:
        print(f" Erreur avec Ollama : {e}")
        return "Résumé indisponible."



def analyser_sentiment(texte):
    """Analyse le sentiment du texte et retourne 'Positif', 'Neutre' ou 'Négatif'."""
    analyse = TextBlob(texte)
    score = analyse.sentiment.polarity  # Score entre -1 (négatif) et +1 (positif)

    if score > 0.1:
        return "Positif"
    elif score < -0.1:
        return "Négatif"
    else:
        return "Neutre"


# **4. Sauvegarde des articles pour l'entraînement de Mistral**
def sauvegarder_pour_mistral(article, summary, sentiment):
    data = {
        "input": article["title"] + "\n" + article["url"] + "\n" + article["content"],
        "output": summary,
        "sentiment": sentiment  # ✅ Ajout du sentiment dans le dataset
    }

    with open("mistral_training_data.jsonl", "a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")

    print(f"📝 Article sauvegardé pour l'entraînement (Sentiment : {sentiment}) : {article['title']}")


# **5. Exécution principale**
def lancer_agent():
    print("\n🔄 Exécution automatique de l'agent IA...")
    articles = rechercher_articles_ia()
    
    for article in articles:
        print(f"\n📄 Traitement : {article['title']} - {article['url']}")
        try:
            resume = generer_resume(article["url"])  # ✅ Génération du résumé
            sentiment = analyser_sentiment(resume)  # ✅ Analyse du sentiment du résumé
            score = score_fiabilite(article["url"])

            if score >= preferences["min_reliability"]:
                ajouter_dans_notion(article["title"], resume, article["url"], score, sentiment)
                sauvegarder_pour_mistral(article, resume, sentiment)  # ✅ Ajout pour l'entraînement
            else:
                print(f"⏭ Article ignoré (fiabilité trop basse : {score})")

        except Exception as e:
            print(f"❌ Erreur pour {article['title']}: {e}")

def ajouter_dans_notion(title, summary, source_url, reliability, sentiment):
    try:
        print(f"🚀 Ajout dans Notion : {title} (Sentiment : {sentiment})")
        notion.pages.create(
            parent={"database_id": DATABASE_ID},
            properties={
                "Title": {"title": [{"text": {"content": title}}]},
                "Summary": {"rich_text": [{"text": {"content": summary}}]},
                "Date": {"date": {"start": datetime.today().isoformat()}},
                "Source URL": {"url": source_url},
                "Reliability Score": {"number": reliability},
                "Sentiment": {"rich_text": [{"text": {"content": sentiment}}]}  # ✅ Correction ici
            },
        )
        print(f"✅ Ajout réussi : {title}")
    except Exception as e:
        print(f"❌ Erreur d'ajout dans Notion : {e}")




# Planifier l'exécution tous les jours à 8h du matin
schedule.every().day.at("23:32:00").do(lancer_agent)

print(" L'agent est en attente d'exécution... (Ctrl+C pour quitter)")

# Boucle infinie pour exécuter l'agent chaque jour
while True:
    schedule.run_pending()
    time.sleep(60)  # Vérification toutes les minutes






