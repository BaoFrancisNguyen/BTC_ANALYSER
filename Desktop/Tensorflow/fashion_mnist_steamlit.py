import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import numpy as np
import matplotlib.pyplot as plt

# Chargement des données
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Interface Streamlit
st.title("Modèle Fashion MNIST avec TensorFlow")

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
if st.button("Lancer l'entraînement"):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer=get_optimizer(optimizer_choice, learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_images.reshape(-1, 28, 28, 1), train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        verbose=2)
    st.success("Entraînement terminé !")

    # Afficher les résultats
    loss, accuracy = model.evaluate(test_images.reshape(-1, 28, 28, 1), test_labels, verbose=0)
    st.write(f"**Précision sur les données de test : {accuracy * 100:.2f}%**")

    # Visualisation des courbes d'entraînement
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Courbe de perte
    ax1.plot(history.history['loss'], label='Perte Entraînement')
    ax1.plot(history.history['val_loss'], label='Perte Validation')
    ax1.set_title('Courbe de Perte')
    ax1.set_xlabel('Époques')
    ax1.set_ylabel('Perte')
    ax1.legend()

    # Courbe de précision
    ax2.plot(history.history['accuracy'], label='Précision Entraînement')
    ax2.plot(history.history['val_accuracy'], label='Précision Validation')
    ax2.set_title('Courbe de Précision')
    ax2.set_xlabel('Époques')
    ax2.set_ylabel('Précision')
    ax2.legend()

    st.pyplot(fig)
