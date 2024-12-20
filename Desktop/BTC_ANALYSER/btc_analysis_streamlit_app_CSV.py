import streamlit as st
from fonctions_CSV import load_data_from_csv, predict_future_rates
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.cm as cm

# Dates des halvings
halvings = [
    {"date": "2012-11-28", "label": "Halving 2012"},
    {"date": "2016-07-09", "label": "Halving 2016"},
    {"date": "2020-05-11", "label": "Halving 2020"},
    {"date": "2024-03-31", "label": "Halving 2024 (estimé)"}
]

# Interface Streamlit
st.title("Analyse et Prévisions du BTC/EUR avec Halvings")

# Charger les données BTC/EUR depuis le fichier CSV
data_file = "btc_eur_data.csv"
data = load_data_from_csv(data_file)

# Nettoyer et convertir les valeurs de 'Close' en entiers après arrondi
if data is not None:
    try:
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')  # Convertir en numérique
        data = data.dropna(subset=['Close'])  # Supprimer les valeurs non numériques
        data['Close'] = data['Close'].round(0).astype(int)  # Convertir en entier après arrondi
    except Exception as e:
        st.error(f"Erreur lors de la conversion des données : {e}")

# Ajouter une colonne pour indiquer le cycle de halving
if data is not None:
    try:
        data['Halving Cycle'] = None
        for i in range(len(halvings) - 1):
            start_date = pd.to_datetime(halvings[i]["date"])
            end_date = pd.to_datetime(halvings[i + 1]["date"])
            mask = (data['Date'] >= start_date) & (data['Date'] < end_date)
            data.loc[mask, 'Halving Cycle'] = f"Cycle {i + 1}"

        # Enrichir les données avec des caractéristiques par cycle
        cycle_features = []
        for i in range(len(halvings) - 1):
            start_date = pd.to_datetime(halvings[i]["date"])
            end_date = pd.to_datetime(halvings[i + 1]["date"])
            cycle_data = data[(data['Date'] >= start_date) & (data['Date'] < end_date)]
            if not cycle_data.empty:
                start_price = cycle_data.iloc[0]['Close']
                end_price = cycle_data.iloc[-1]['Close']
                max_price = cycle_data['Close'].max()
                min_price = cycle_data['Close'].min()
                duration = (end_date - start_date).days
                avg_growth = (end_price - start_price) / duration  # Croissance moyenne par jour
                growth_ratio = end_price / start_price
                volatility_ratio = max_price / min_price

                cycle_features.append({
                    'Cycle': i + 1,
                    'Start_Price': start_price,
                    'End_Price': end_price,
                    'Duration': duration,
                    'Avg_Growth': avg_growth,
                    'Growth_Ratio': growth_ratio,
                    'Volatility_Ratio': volatility_ratio
                })

        # Convertir en DataFrame
        cycle_features_df = pd.DataFrame(cycle_features)
        st.write("Caractéristiques des cycles de Halving :")
        st.dataframe(cycle_features_df)

        # Préparer les données pour la régression
        X = cycle_features_df[['Cycle', 'Start_Price', 'Duration', 'Volatility_Ratio']]  # Variables explicatives
        y = cycle_features_df['Avg_Growth']  # Variable cible

        # Entraîner le modèle de régression
        model = LinearRegression()
        model.fit(X, y)

        # Ajouter un contrôle interactif pour la durée des prédictions
        days_to_predict = st.slider("Nombre de jours à prédire pour le cycle 2024-2028", min_value=1, max_value=1460, value=365)

        # Prédire pour le cycle 2024-2028
        last_close_price = data.iloc[-1]['Close']  # Dernier prix de clôture connu
        next_cycle_features = {
            'Cycle': len(halvings),
            'Start_Price': last_close_price,
            'Duration': 1460,  # Durée estimée d'un cycle (4 ans)
            'Volatility_Ratio': cycle_features_df['Volatility_Ratio'].mean()  # Moyenne historique
        }

        next_cycle_X = pd.DataFrame([next_cycle_features])
        predicted_growth = model.predict(next_cycle_X)[0]

        # Générer les prédictions de prix
        predicted_prices = [last_close_price + i * predicted_growth for i in range(days_to_predict)]
        predictions = pd.DataFrame({
            'Date': pd.date_range(start=pd.to_datetime(halvings[-1]["date"]), periods=days_to_predict),
            'Close': predicted_prices
        })

        st.write("Tableau des prédictions pour le cycle 2024-2028 :")
        st.dataframe(predictions)

        # Afficher le graphique des cycles et des prédictions
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = cm.rainbow(np.linspace(0, 1, len(halvings)))

        for i in range(len(halvings) - 1):
            cycle_data = data[data['Halving Cycle'] == f"Cycle {i + 1}"]
            if not cycle_data.empty:
                ax.plot(cycle_data['Date'], cycle_data['Close'], marker='o', linestyle='-', color=colors[i], label=f"Cycle {i + 1}")

        ax.plot(predictions['Date'], predictions['Close'], marker='x', linestyle='--', color='green', label="Prédictions (2024-2028)")
        ax.set_title("Prédictions pour le cycle 2024-2028 avec régression multiple")
        ax.set_xlabel("Date")
        ax.set_ylabel("Prix de clôture (EUR)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur lors de la prédiction pour le cycle 2024-2028 : {e}")
else:
    st.error("Aucune donnée BTC/EUR disponible pour les prédictions.")


















