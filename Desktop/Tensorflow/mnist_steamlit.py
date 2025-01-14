import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# Interface Streamlit
st.title("Modèle de Classification avec TensorFlow")

# Chargement des données ou d'une image
upload_choice = st.radio("Choisissez le type de données à charger :", ("Charger un fichier CSV", "Charger une image"))

dataset_loaded = False
uploaded_file = None

if upload_choice == "Charger un fichier CSV":
    uploaded_file = st.file_uploader("Téléchargez un fichier CSV contenant les données d'entraînement", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Aperçu des données :", data.head())
        dataset_loaded = True
        st.success("Jeu de données chargé avec succès.")

elif upload_choice == "Charger une image":
    uploaded_file = st.file_uploader("Téléchargez une image (28x28 pixels, en niveaux de gris)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L").resize((28, 28))
        image_array = np.array(image) / 255.0
        st.image(image, caption='Image chargée', use_column_width=True)
        st.success("Image chargée avec succès.")

# Paramètres ajustables
optimizer_choice = st.selectbox("Choisissez l'optimiseur", ["Adam", "RMSprop", "SGD"])
learning_rate = st.slider("Taux d'apprentissage", 0.0001, 0.01, 0.001, step=0.0001)
epochs = st.slider("Nombre d'époques", 1, 50, 10)
batch_size = st.slider("Taille du batch", 16, 128, 32, step=16)

# Fonction pour choisir l'optimiseur
def get_optimizer(name, lr):
    if name == "Adam":
        return Adam(learning_rate=lr)
    elif name == "RMSprop":
        return RMSprop(learning_rate=lr)
    else:
        return SGD(learning_rate=lr)

# Construire le modèle
if st.button("Lancer l'entraînement") and uploaded_file is not None:
    model = Sequential([
        Dense(128, activation='relu', input_shape=(28*28,)),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer=get_optimizer(optimizer_choice, learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    if dataset_loaded:
        # Simulation d'entraînement sur des données CSV avec des features et labels
        features = data.iloc[:, :-1].values / 255.0
        labels = data.iloc[:, -1].values

        history = model.fit(features,
                            labels,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.2,
                            verbose=2)
        st.success("Entraînement terminé !")

        # Visualisation des courbes d'entraînement
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(history.history['loss'], label='Perte Entraînement')
        ax1.plot(history.history['val_loss'], label='Perte Validation')
        ax1.set_title('Courbe de Perte')
        ax1.set_xlabel('Époques')
        ax1.set_ylabel('Perte')
        ax1.legend()

        ax2.plot(history.history['accuracy'], label='Précision Entraînement')
        ax2.plot(history.history['val_accuracy'], label='Précision Validation')
        ax2.set_title('Courbe de Précision')
        ax2.set_xlabel('Époques')
        ax2.set_ylabel('Précision')
        ax2.legend()
        st.pyplot(fig)

    elif upload_choice == "Charger une image":
        prediction = model.predict(image_array.reshape(1, 28*28))
        predicted_class = np.argmax(prediction)
        st.write(f"**Prédiction : {predicted_class}**")
else:
    st.warning("Veuillez charger un fichier CSV ou une image avant de lancer l'entraînement.")

