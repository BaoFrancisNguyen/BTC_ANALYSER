import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np

def load_data_from_csv(filename):
    """Charge les données depuis un fichier CSV et ajuste les colonnes si nécessaire."""
    try:
        # Charger les données
        data = pd.read_csv(filename)
        print("Colonnes disponibles après chargement initial :", data.columns.tolist())
        
        # Identifier et renommer les colonnes
        if 'Price' in data.columns:
            data.rename(columns={'Price': 'Date'}, inplace=True)
        elif 'Date' not in data.columns:
            raise KeyError("La colonne 'Price' ou 'Date' est introuvable.")

        if 'Close' not in data.columns:
            raise KeyError("La colonne 'Close' est introuvable.")

        # Convertir la colonne Date en datetime
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        
        # Supprimer les lignes avec des données manquantes
        data = data.dropna(subset=['Date', 'Close'])
        
        print("Colonnes disponibles après traitement :", data.columns.tolist())
        print("Aperçu des données :", data.head())
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Le fichier {filename} est introuvable.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Le fichier {filename} est vide ou invalide.")
    except Exception as e:
        raise e

def plot_exchange_rates(data, title="Taux d'échange BTC/EUR"):
    """Génère un graphique des taux d'échange."""
    if data is None or data.empty:
        raise ValueError("Aucune donnée disponible pour le graphique.")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Date'], data['Close'], marker='o', linestyle='-', color='blue', label="BTC/EUR")
    ax.set_title(title, color='black')
    ax.set_xlabel("Date", color='black')
    ax.set_ylabel("Taux de change", color='black')
    ax.grid(True, color='gray')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig

def predict_future_rates(data, days_to_predict=30):
    """Prédit les futurs taux d'échange."""
    if data is None or data.empty:
        raise ValueError("Aucune donnée disponible pour effectuer une prédiction.")
    data = data.sort_values(by='Date')
    dates = (data['Date'] - data['Date'].min()).dt.days.values.reshape(-1, 1)
    values = data['Close'].values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(dates, values)
    future_dates = np.arange(dates[-1] + 1, dates[-1] + days_to_predict + 1).reshape(-1, 1)
    future_values = model.predict(future_dates)
    future_dates = [data['Date'].max() + timedelta(days=int(i)) for i in range(1, days_to_predict + 1)]
    predicted_data = pd.DataFrame({
        "Date": future_dates,
        "Close": future_values.flatten()
    })
    return predicted_data

def plot_with_predictions(data, predictions, title="Taux d'échange BTC/EUR avec prédictions"):
    """Affiche un graphique des taux d'échange avec les prédictions."""
    if data is None or data.empty or predictions is None or predictions.empty:
        raise ValueError("Aucune donnée disponible pour le graphique.")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Date'], data['Close'], marker='o', linestyle='-', color='blue', label="Données historiques (BTC/EUR)")
    ax.plot(predictions['Date'], predictions['Close'], marker='x', linestyle='--', color='green', label="Prédictions")
    ax.set_title(title, color='black')
    ax.set_xlabel("Date", color='black')
    ax.set_ylabel("Taux de change", color='black')
    ax.grid(True, color='gray')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig





