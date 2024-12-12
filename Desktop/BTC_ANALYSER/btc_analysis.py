import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import mplcursors

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
    plt.plot(dates, values, marker='o', linestyle='-', color='blue', label="BTC/EUR")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Taux de change")
    plt.grid(True)
    plt.legend()

    # Ajouter un curseur interactif
    cursor = mplcursors.cursor(hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"{sel.artist.get_label()}\nDate: {dates[int(sel.index)].strftime('%Y-%m-%d')}\nTaux: {values[int(sel.index)]:.2f}"
    ))

    plt.show()

def predict_future_rates(data, days_to_predict=30):
    """Prédit les futurs taux d'échange."""
    if not data:
        print("Aucune donnée disponible pour effectuer une prédiction.")
        return []

    # Préparer les données
    dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in data]
    values = [item['value'] for item in data]
    X = np.array([(date - dates[0]).days for date in dates]).reshape(-1, 1)  # Convertir les dates en jours
    y = np.array(values).reshape(-1, 1)

    # Entraîner un modèle de régression linéaire
    model = LinearRegression()
    model.fit(X, y)

    # Générer des prédictions
    future_dates = [(dates[-1] - dates[0]).days + i for i in range(1, days_to_predict + 1)]
    future_values = model.predict(np.array(future_dates).reshape(-1, 1))

    # Convertir les résultats en un format lisible
    predicted_data = [
        {
            "date": (dates[-1] + timedelta(days=i)).strftime('%Y-%m-%d'),
            "value": float(value)
        }
        for i, value in enumerate(future_values)
    ]
    return predicted_data

def display_predictions_table(predictions):
    """Affiche les prédictions dans un tableau."""
    if not predictions:
        print("Aucune prédiction à afficher.")
        return

    df = pd.DataFrame(predictions)
    print("\nPrédictions :")
    print(df.to_string(index=False))

def plot_with_predictions(data, predictions, title="Taux d'échange BTC/EUR avec prédictions"):
    """Affiche un graphique des taux d'échange avec les prédictions."""
    if not data:
        print("Aucune donnée disponible pour le graphique.")
        return

    # Extraire les données historiques
    dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in data]
    values = [item['value'] for item in data]

    # Extraire les données prédites
    pred_dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in predictions]
    pred_values = [item['value'] for item in predictions]

    # Création du graphique
    plt.figure(figsize=(12, 6))
    plt.plot(dates, values, marker='o', linestyle='-', color='blue', label="Données historiques (BTC/EUR)")
    plt.plot(pred_dates, pred_values, marker='x', linestyle='--', color='green', label="Prédictions")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Taux de change")
    plt.grid(True)
    plt.legend()

    # Ajouter un curseur interactif
    cursor = mplcursors.cursor(hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"{sel.artist.get_label()}\nDate: {pred_dates[int(sel.index - len(values))].strftime('%Y-%m-%d') if sel.index >= len(values) else dates[int(sel.index)].strftime('%Y-%m-%d')}\nTaux: {sel.target[1]:.2f}"
    ))

    plt.show()

if __name__ == "__main__":
    filename = "BTC_EUR.json"
    data = load_data(filename)

    # Affichage des données historiques
    plot_exchange_rates(data)

    # Prédiction
    future_data = predict_future_rates(data, days_to_predict=30)

    # Afficher les prédictions dans un tableau
    display_predictions_table(future_data)

    # Affichage avec prédictions
    plot_with_predictions(data, future_data)





