import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os

# === Configuration Streamlit ===
st.title("Amélioration de la QVT des Alternants avec Belbin")

# === 1. Définition des 9 Profils Belbin ===
roles = [
    "Propulseur", "Réalisateur", "Perfectionniste",
    "Créatif", "Évaluateur", "Spécialiste",
    "Coordinateur", "Soutien", "Chercheur de ressources"
]

# === 2. Création du Questionnaire (20 questions) ===
st.subheader("Questionnaire Belbin (20 Questions)")
st.write("Répondez aux questions pour déterminer votre rôle dominant.")

questions = [
    "J’aime challenger les idées et pousser mon équipe à aller plus loin.",
    "Je suis très organisé et je mets en place des processus pour être efficace.",
    "Je fais attention aux détails et je n’aime pas les erreurs.",
    "J’ai toujours de nouvelles idées innovantes.",
    "J’analyse en profondeur les problèmes avant de prendre une décision.",
    "Je suis passionné par un domaine et j’aime partager mon expertise.",
    "J’aime organiser et répartir le travail au sein de mon équipe.",
    "Je suis un bon médiateur et j’évite les conflits.",
    "J’aime découvrir de nouvelles opportunités et créer des connexions.",
    "Je prends des décisions rapidement et j’aime l’action.",
    "Je suis méthodique et pragmatique.",
    "Je vérifie toujours mon travail plusieurs fois avant de le soumettre.",
    "J’aime expérimenter de nouvelles idées.",
    "Je suis critique et je prends du recul avant d’agir.",
    "J’aime approfondir mes connaissances et me spécialiser.",
    "Je suis diplomate et je valorise l’esprit d’équipe.",
    "Je motive mon équipe en donnant une vision claire des objectifs.",
    "J’aime travailler en groupe et je favorise la cohésion.",
    "Je suis toujours à la recherche de nouveaux contacts pour mon entreprise.",
    "J’aime résoudre les problèmes complexes."
]

# Réponses de l'utilisateur
responses = []
for q in questions:
    responses.append(st.slider(q, 0, 10, 5))

# === 3. Entraînement d'un modèle K-Means ===
st.subheader("Analyse de votre profil")

np.random.seed(42)
data_train = np.random.randint(0, 11, size=(200, len(questions)))
scaler = StandardScaler()
scaled_data_train = scaler.fit_transform(data_train)

kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
kmeans.fit(scaled_data_train)

scaled_user_data = scaler.transform(np.array(responses).reshape(1, -1))
predicted_role = kmeans.predict(scaled_user_data)

st.success(f"Votre rôle principal est : **{roles[predicted_role[0]]}**")

# === 4. Suivi et sauvegarde des scores QVT ===
st.subheader("Suivi de votre QVT")

csv_file = "qvt_scores.csv"
current_qvt_score = np.mean(responses) * 10  # Normalisation du score sur 100


if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    df = pd.DataFrame(columns=["Semaine", "Score QVT"])

# Ajout d'un bouton pour enregistrer manuellement
if st.button("Sauvegarder mon score QVT"):
    new_entry = pd.DataFrame({"Semaine": [len(df) + 1], "Score QVT": [current_qvt_score]})
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(csv_file, index=False)
    st.success("Score QVT sauvegardé avec succès !")

fig = px.line(df, x="Semaine", y="Score QVT", title="Évolution de la qualité de vie au travail")
st.plotly_chart(fig)

# === 5. Modèle prédictif du QVT ===
st.subheader("Prédiction du futur QVT")

if len(df) > 1:
    model = LinearRegression()
    model.fit(df[["Semaine"]], df["Score QVT"])
    future_week = len(df) + 1
    predicted_qvt = model.predict([[future_week]])[0]
    
    st.write(f"Estimation du Score QVT pour la semaine {future_week} : **{predicted_qvt:.2f}**")

    # === 6. Alerte pour le tuteur ===
    alert_threshold = 60
    if predicted_qvt < alert_threshold:
        st.warning("Alerte ! Le score QVT prédit est trop bas. Contactez le tuteur pour discuter des améliorations possibles.")

# === 7. Recommandations personnalisées ===
st.subheader("Recommandations pour améliorer votre bien-être")

reco_dict = {
    "Propulseur": [" Canalisez votre énergie pour éviter les conflits.", " Apprenez à écouter les autres avant d'imposer vos idées."],
    "Réalisateur": [" Déléguez plus pour éviter la surcharge de travail.", " Expérimentez de nouvelles méthodes de travail."],
    "Perfectionniste": ["Fixez-vous une limite de révision pour éviter la paralysie.", " Acceptez que la perfection n'est pas toujours nécessaire."],
    "Créatif": ["Travaillez avec des personnes structurées pour concrétiser vos idées.", " Testez vos idées avant de les imposer."],
    "Évaluateur": [" Ne tombez pas dans l'excès d'analyse.", " Faites confiance à votre intuition parfois."],
    "Spécialiste": [" Travaillez sur votre communication pour mieux partager votre savoir.", " Évitez de trop vous isoler."],
    "Coordinateur": [" Donnez plus d’autonomie aux autres.", " Évitez d’être trop centré sur l’organisation."],
    "Soutien": [" Apprenez à dire non pour éviter la surcharge.", " Valorisez aussi vos propres compétences."],
    "Chercheur de ressources": [" Apprenez à rester focus sur un projet.", " Suivez vos idées jusqu’à leur aboutissement."]
}

st.write("🔹 " + "\n🔹 ".join(reco_dict[roles[predicted_role[0]]]))

