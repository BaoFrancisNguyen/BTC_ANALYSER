import json
import matplotlib.pyplot as plt
from datetime import datetime

def load_data(filename):
    """Charge les données depuis un fichier JSON."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print(f"Le fichier {filename} est introuvable.")
        return []
    except json.JSONDecodeError:
        print(f"Erreur lors du décodage du fichier {filename}.")
        return []

def plot_exchange_rates(data, title="Taux d'échange BTC/EUR"):
    """Affiche un graphique des taux d'échange."""
    if not data:
        print("Aucune donnée disponible pour le graphique.")
        return

    # Extraire les dates et les valeurs
    dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in data]
    values = [item['value'] for item in data]

    # Création du graphique
    plt.figure(figsize=(12, 6))
    plt.plot(dates, values, marker='o', linestyle='-', label="BTC/EUR")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Taux de change")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    filename = "BTC_EUR.json"
    data = load_data(filename)
    plot_exchange_rates(data)
