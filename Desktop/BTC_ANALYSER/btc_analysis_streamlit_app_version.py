import streamlit as st
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def load_data(filename):
    """Charge les données depuis un fichier JSON."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        st.error(f"Le fichier {filename} est introuvable.")
        return []
    except json.JSONDecodeError:
        st.error(f"Erreur lors du décodage du fichier {filename}.")
        return []

def plot_exchange_rates(data, title="Taux d'échange BTC/EUR"):
    """Génère un graphique des taux d'échange."""
    if not data:
        st.warning("Aucune donnée disponible pour le graphique.")
        return None

    # Créer une figure et un axe
    fig, ax = plt.subplots(figsize=(12, 6))
    dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in data]
    values = [item['value'] for item in data]
    ax.plot(dates, values, marker='o', linestyle='-', color='blue', label="BTC/EUR")
    ax.set_title(title, color='white')
    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Taux de change", color='white')
    ax.grid(True, color='gray')
    ax.legend()
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    fig.patch.set_facecolor('#2b2b2b')
    ax.set_facecolor('#2b2b2b')
    return fig

def predict_future_rates(data, days_to_predict=30):
    """Prédit les futurs taux d'échange."""
    if not data:
        st.warning("Aucune donnée disponible pour effectuer une prédiction.")
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
    """Crée un tableau des prédictions (sans affichage)."""
    if not predictions:
        return None

    df = pd.DataFrame(predictions)
    return df

def plot_with_predictions(data, predictions, title="Taux d'échange BTC/EUR avec prédictions"):
    """Affiche un graphique des taux d'échange avec les prédictions."""
    if not data:
        st.warning("Aucune donnée disponible pour le graphique.")
        return None

    # Extraire les données historiques
    dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in data]
    values = [item['value'] for item in data]

    # Extraire les données prédites
    pred_dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in predictions]
    pred_values = [item['value'] for item in predictions]

    # Créer une figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, values, marker='o', linestyle='-', color='blue', label="Données historiques (BTC/EUR)")
    ax.plot(pred_dates, pred_values, marker='x', linestyle='--', color='green', label="Prédictions")
    ax.set_title(title, color='white')
    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Taux de change", color='white')
    ax.grid(True, color='gray')
    ax.legend()
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    fig.patch.set_facecolor('#2b2b2b')
    ax.set_facecolor('#2b2b2b')
    return fig

# Interface Streamlit
st.title("Analyse et Prévisions du BTC/EUR")
st.markdown(
    """
    <style>
    .stApp {
        background: url("https://img.freepik.com/free-vector/blue-curve-frame-template_53876-99024.jpg?t=st=1734134137~exp=1734137737~hmac=10113e43ef001b961bab0745e15c40e06036fc9a9704d58fe29a64c1ad0feff1&w=996"); /* Chemin relatif à l'image dans le dossier racine */
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        color: grey;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Charger les données
data_file = "BTC_EUR.json"
data = load_data(data_file)

# Affichage des données historiques
st.header("Données Historiques")
if data:
    fig = plot_exchange_rates(data)
    if fig:
        st.pyplot(fig)
else:
    st.error("Aucune donnée disponible.")

# Prédictions des taux futurs
st.header("Prédictions")
days_to_predict = st.slider("Nombre de jours à prédire", min_value=1, max_value=365, value=30)
if data:
    predictions = predict_future_rates(data, days_to_predict)
    if predictions:
        # Crée le tableau sans l'afficher
        df_predictions = display_predictions_table(predictions)
        fig_with_predictions = plot_with_predictions(data, predictions)
        if fig_with_predictions:
            st.pyplot(fig_with_predictions)
else:
    st.error("Aucune donnée disponible pour les prédictions.")








