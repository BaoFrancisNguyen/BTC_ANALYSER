import requests
import ollama
from bs4 import BeautifulSoup
from notion_client import Client
from datetime import datetime
from config import NOTION_API_KEY, DATABASE_ID

# Configuration

NOTION_API_KEY = "notion api key"
DATABASE_ID = "database id de notion"

# URL de l'article à résumer / il faudra donner la tâche à un agent de chercher des articles traitant des thématiques désirées
ARTICLE_URL = "https://www.lemonde.fr/intelligence-artificielle/"  # Remplace par l'URL exacte de l'article

# Étape 1 : scrapping / extraction du contenu de l'article

def extract_article(url):

    response = requests.get(url)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find('h1').get_text()
    paragraphs = soup.find_all('p')
    content = "\n".join([p.get_text() for p in paragraphs])

    return title, content

# Étape 2 : Résumer avec Ollama (Mistral)

def summarize_with_ollama(text):

    prompt = f"Résumé cet article en 5 phrases :\n\n{text}"
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# Étape 3 : Envoyer dans la base de données crée dans Notion

def send_to_notion(title, summary, url):

    notion = Client(auth=NOTION_API_KEY)

    page_data = {
        "parent": {"database_id": DATABASE_ID},
        "properties": {
            "Title": {"title": [{"text": {"content": title}}]},
            "Summary": {"rich_text": [{"text": {"content": summary}}]},
            "Date": {"date": {"start": datetime.today().isoformat()}},
            "Source URL": {"url": url}
        }
    }

    notion.pages.create(**page_data)
    print("✅ Résumé envoyé à Notion !")

# 🔹 Exécution du processus

article_title, article_text = extract_article(ARTICLE_URL)

if article_text:
    summary = summarize_with_ollama(article_text[:2000])  # Limite à 2000 caractères
    send_to_notion(article_title, summary, ARTICLE_URL)
else:
    print("Erreur lors de l'extraction de l'article.")
