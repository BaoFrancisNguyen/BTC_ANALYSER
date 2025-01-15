import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Interface Streamlit
st.title("Modèle de Classification d'Images Médicales avec TensorFlow")

# Chargement des données ou d'une image
upload_choice = st.radio("Choisissez le type de données à charger :", ("Charger un dossier d'images", "Charger une image pour prédiction"))

dataset_loaded = False
uploaded_file = None

if upload_choice == "Charger un dossier d'images":
    uploaded_file = st.file_uploader("Téléchargez un fichier ZIP contenant les images d'entraînement (organisé par dossier de classe)", type=["zip"])
    if uploaded_file is not None:
        import zipfile
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall("dataset")
        st.success("Images extraites avec succès.")
        dataset_loaded = True

elif upload_choice == "Charger une image pour prédiction":
    uploaded_file = st.file_uploader("Téléchargez une image médicale", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L").resize((128, 128))
        image_array = np.array(image) / 255.0
        st.image(image, caption='Image chargée', use_container_width=True)
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

# Construire le modèle CNN pour les images
def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(nombre_de_classes, activation='softmax')  # Multi-classes
    ])
    return model

if st.button("Lancer l'entraînement") and dataset_loaded:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Générer les données depuis le dossier extrait
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
    "dataset",
    target_size=(128, 128),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
    )

val_generator = datagen.flow_from_directory(
    "dataset",
    target_size=(128, 128),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
    )


    model = build_cnn_model()
    model.compile(optimizer=get_optimizer(optimizer_choice, learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)
    st.success("Entraînement terminé !")

    # Afficher les résultats
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

elif upload_choice == "Charger une image pour prédiction" and uploaded_file is not None:
    model = build_cnn_model()
    flat_image = image_array.reshape(1, 128, 128, 1)
    prediction = model.predict(flat_image)
    predicted_class = np.argmax(prediction[0])
    st.write(f"**Prédiction : {predicted_class}**")


