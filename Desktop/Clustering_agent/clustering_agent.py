import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer  # Importer SimpleImputer
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le dataset
data = pd.read_csv("Sad.csv")  # Remplace par ton fichier CSV

# Vérifier si les données sont bien chargées et s'il y a des NaN
print("Premières lignes du dataset :")
print(data.head())

# Remplir les valeurs NaN par la moyenne dans les colonnes numériques
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Academic Pressure'].fillna(data['Academic Pressure'].mean(), inplace=True)
data['Work Pressure'].fillna(data['Work Pressure'].mean(), inplace=True)
data['CGPA'].fillna(data['CGPA'].mean(), inplace=True)

# Vérifier la forme après nettoyage
print(f"Shape après nettoyage : {data.shape}")

# Sélectionner les colonnes numériques
numerical_data = data[['Age', 'Academic Pressure', 'Work Pressure', 'CGPA']]

# Imputer les valeurs manquantes avec la moyenne des colonnes
imputer = SimpleImputer(strategy="mean")
numerical_data_imputed = imputer.fit_transform(numerical_data)

# Appliquer la standardisation
scaler = StandardScaler()
numerical_data_scaled = scaler.fit_transform(numerical_data_imputed)

# Appliquer KMeans pour le clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(numerical_data_scaled)

# Ajouter les clusters au DataFrame
data['Cluster'] = clusters

# Afficher les résultats des clusters
print("Données avec clusters :")
print(data.head())

# Calculer le score de silhouette pour évaluer la qualité du clustering
silhouette_avg = silhouette_score(numerical_data_scaled, clusters)
print(f"Silhouette Score: {silhouette_avg}")

# Recommandations avec Deepseek
model_name = "C:\\Users\\Francis\\Desktop\\Clustering_agent\\deepseek-coder-1.3b-base"  # Remplace par ton chemin local vers le modèle .gguf

# Charger le modèle et le tokenizer
try:
    # Chargement du modèle et du tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")

# Fonction de génération de texte pour faire des recommandations
def generate_recommendations(cluster_id):
    # Exemple de prompt de recommandation basé sur le cluster
    prompt = f"Fournir des recommandations pour les individus du cluster {cluster_id}. Ces personnes ont les caractéristiques suivantes : "
    
    # Tokenize le prompt avec gestion de l'attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Générer la réponse avec le modèle
    outputs = model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'],  # Ajout de l'attention mask pour améliorer la génération
        max_length=300,  # Limite de longueur de la réponse
        num_beams=5,     # Utiliser un beam search pour une meilleure qualité
        no_repeat_ngram_size=2,  # Eviter les répétitions
        temperature=0.7, # Contrôle la créativité de la génération
        pad_token_id=tokenizer.eos_token_id  # Indique la fin de la séquence
    )
    
    # Décoder la réponse générée
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Afficher les recommandations pour tous les clusters
for cluster_id in range(n_clusters):
    recommendations = generate_recommendations(cluster_id)
    print(f"Recommandations pour le cluster {cluster_id} :")
    print(recommendations)
    print("\n")

# Réduire les dimensions avec PCA pour visualiser en 2D
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(numerical_data_imputed)  # Appliquer PCA sur les données imputées

# Clustering (KMeans par exemple)
kmeans = KMeans(n_clusters=3, random_state=50)  # Choisir le nombre de clusters
data['Cluster'] = kmeans.fit_predict(data_reduced)

# Visualiser avec Matplotlib
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_reduced[:, 0], y=data_reduced[:, 1], hue=data['Cluster'], palette="Set2", s=100, edgecolor="black")
plt.title("Visualisation des Clusters (PCA)", fontsize=16)
plt.xlabel("PCA Composant 1")
plt.ylabel("PCA Composant 2")
plt.legend(title="Cluster")
plt.show()

















