import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os

# === Configuration de la page ===
st.set_page_config(page_title="Qualité de Vie au Travail", layout="wide")

col1, col2, col3 = st.columns([1, 2, 1]) 

# === Ajout de style CSS ===
with col2:
    st.image("stressless.png", width=500)

st.markdown("""
    <style>
        html, body, [class*="st"] {
            background-color: #121212;
            color: #e0e0e0;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #BB86FC;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            padding: 10px;
        }
        .stMetric {
            font-size: 20px;
            color: #BB86FC;
        }
        .stAlert {
            background-color: #333333;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)


# === Disposition en colonnes ===
col1, col2 = st.columns([2, 1])

# === 1. Création du Questionnaire Bien-Être ===
st.subheader("Questionnaire de Bien-Être (25 Questions)")
st.write("Répondez aux questions pour évaluer votre bien-être au travail.")

questions = [
    # Équilibre vie pro/perso
    "Je ressens un bon équilibre entre mon travail et ma vie personnelle.",
    "Je peux facilement déconnecter après ma journée de travail.",
    "Mon travail me permet d’avoir du temps pour mes loisirs et ma famille.",
    
    # Stress et charge de travail
    "Je ressens peu de stress dans mon travail quotidien.",
    "Ma charge de travail est adaptée à mes capacités.",
    "Je me sens soutenu(e) par mes collègues et supérieurs en cas de difficulté.",
    
    # Reconnaissance et motivation
    "Je me sens reconnu(e) et valorisé(e) pour mon travail.",
    "Mon travail est stimulant et me motive chaque jour.",
    "Je me sens impliqué(e) dans les décisions liées à mon travail.",
    
    # Santé et bien-être
    "Je dors suffisamment et me sens reposé(e) pour travailler efficacement.",
    "Je prends régulièrement des pauses pour préserver mon bien-être.",
    "Mon environnement de travail est agréable et confortable.",
]

# Réponses utilisateur
responses = []
for q in questions:
    responses.append(st.slider(q, 0, 10, 5))

# === 2. Calcul des Indicateurs de Bien-Être ===
st.subheader("Analyse de votre Bien-Être")
current_wellbeing_score = np.mean(responses) * 10  # Score bien-être sur 100
stress_score = (10 - np.mean(responses[3:6])) * 10  # Stress basé sur questions 4-6

st.success(f"Votre score de bien-être global est : **{current_wellbeing_score:.2f}/100**")
st.warning(f"Votre niveau de stress est estimé à **{stress_score:.2f}/100**")

# === 3. Suivi et sauvegarde des scores ===
csv_file = "bien_etre_scores.csv"

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    df = pd.DataFrame(columns=["Semaine", "Score Bien-Être", "Score Stress"])

if st.button("Sauvegarder mes scores"):
    new_entry = pd.DataFrame({"Semaine": [len(df) + 1], "Score Bien-Être": [current_wellbeing_score], "Score Stress": [stress_score]})
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(csv_file, index=False)
    st.success("Scores sauvegardés avec succès !")

fig = px.line(df, x="Semaine", y=["Score Bien-Être", "Score Stress"], title="Évolution du Bien-Être et du Stress")
# Centrage du titre
fig.update_layout(title_x=0.5)
st.plotly_chart(fig)

# === 4. Prédiction du Bien-Être Futur ===
st.subheader("Prédiction de votre bien-être futur")
if len(df) > 1:
    model = LinearRegression()
    model.fit(df[["Semaine"]], df["Score Bien-Être"])
    future_week = len(df) + 1
    predicted_wellbeing = model.predict([[future_week]])[0]
    st.write(f"Estimation du Score Bien-Être pour la semaine {future_week} : **{predicted_wellbeing:.2f}**")
    
    if predicted_wellbeing < 70:
        st.warning("Alerte ! Votre bien-être pourrait être en baisse. Pensez à discuter avec votre tuteur ou RH.")

# === 5. Recommandations personnalisées ===
st.subheader("Conseils pour améliorer votre bien-être")
if current_wellbeing_score < 70:
    st.warning("Votre score de bien-être est faible. Voici quelques conseils pour vous aider :")
    st.write("Prenez des pauses régulières et pratiquez la respiration profonde.")
elif current_wellbeing_score < 50:
    st.write("Parlez de vos préoccupations avec un manager ou un collègue de confiance.")
    st.write("Fixez des limites claires entre votre travail et votre vie personnelle.")
    st.write("Faites une activité qui vous détend après le travail (sport, lecture, méditation, etc.).")
elif current_wellbeing_score > 70:
    st.success("Votre bien-être est excellent ! Continuez sur cette voie en maintenant vos bonnes habitudes.")
