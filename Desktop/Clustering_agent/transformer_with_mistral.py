import os
import pandas as pd
from data_transformer import DataTransformer  # Importe la classe que nous avons créée précédemment

def main():
    """
    Fonction principale démontrant l'utilisation complète du transformateur
    avec définition claire des entrées, sorties et utilisation du modèle Mistral.
    """
    print("=== Agent de Transformation et d'Analyse de Données ===")
    
    # 1. DÉFINITION DES ENTRÉES
    # -------------------------
    # Chemin vers le fichier CSV à transformer
    input_file = "data_source.csv"
    
    # Chemin vers le fichier CSV transformé (sortie)
    output_file = "data_transformed.csv"
    
    # Chemin vers le modèle Mistral (tel que défini dans les messages précédents)
    mistral_model_path = "C:\\Users\\Francis\\Desktop\\ORBIT\\mistral-7b-openorca.Q4_K_M.gguf"
    
    # Taille du contexte pour Mistral (512 tokens comme mentionné précédemment)
    context_size = 512
    
    # 2. INITIALISATION DU TRANSFORMATEUR
    # -----------------------------------
    print(f"\nInitialisation du transformateur avec modèle: {os.path.basename(mistral_model_path)}")
    transformer = DataTransformer(
        model_path=mistral_model_path,
        context_size=context_size
    )
    
    # 3. VÉRIFICATION DE L'EXISTENCE DU FICHIER D'ENTRÉE
    # -------------------------------------------------
    if not os.path.exists(input_file):
        print(f"Erreur: Le fichier d'entrée '{input_file}' n'existe pas.")
        return
    
    print(f"Fichier source: {input_file}")
    
    # 4. TRANSFORMATION DU DATASET
    # ---------------------------
    print("\nDébut de la transformation...")
    
    # Options de transformation
    # Vous pouvez choisir une transformation spécifique ou laisser la détection automatique
    transformations_options = [
        None,  # Détection automatique 
        ["missing_values", "normalize", "encode"],  # Transformation standard
        ["missing_values", "normalize", "encode", "fusion"],  # Avec fusion forcée
    ]
    
    # Choisir une option (0=auto, 1=standard, 2=avec fusion)
    transformations = transformations_options[0]  # Détection automatique par défaut
    
    # Lecture directe du DataFrame pour plus de contrôle
    try:
        df = pd.read_csv(input_file)
        print(f"Dataset chargé: {len(df)} lignes, {len(df.columns)} colonnes")
        
        # Affichage des premières lignes pour vérification
        print("\nAperçu des données originales:")
        print(df.head(2))
        
        # Examinons les paires de colonnes candidates pour la fusion
        pairs = transformer._identify_column_pairs(df)
        if pairs:
            print("\nPaires de colonnes candidates pour fusion:")
            for i, (col1, col2, score) in enumerate(pairs[:5], 1):
                print(f"  {i}. '{col1}' + '{col2}' (score: {score:.2f})")
            
            # Si vous voulez forcer la fusion d'une paire spécifique, 
            # vous pouvez le faire ici en définissant transformations explicitement
            if len(pairs) > 0 and transformations is None:
                # Commentez cette ligne pour revenir à la détection automatique
                # transformations = ["missing_values", "normalize", "encode", "fusion"]
                pass
        
        # Appliquer la transformation
        df_transformed, metadata = transformer.transform(df, transformations)
        
        # Enregistrer le résultat
        df_transformed.to_csv(output_file, index=False)
        print(f"Dataset transformé enregistré dans: {output_file}")
        
        # Afficher un aperçu du résultat
        print("\nAperçu des données transformées:")
        print(df_transformed.head(2))
        
    except Exception as e:
        print(f"Erreur lors de la transformation: {e}")
        return
    
    # 5. ANALYSE DES RÉSULTATS
    # -----------------------
    print("\n=== RÉSULTATS DE LA TRANSFORMATION ===")
    print(f"Dimensions: {metadata['original_shape']} → {metadata['transformed_shape']}")
    
    # Afficher les transformations appliquées
    if metadata['transformations']:
        print("\nTransformations appliquées:")
        for i, t in enumerate(metadata['transformations'], 1):
            print(f"  {i}. {t['details']}")
    else:
        print("\nAucune transformation n'a été appliquée.")
    
    # Afficher les nouvelles colonnes créées
    if metadata['new_columns']:
        print("\nNouvelles colonnes créées:")
        for col in metadata['new_columns']:
            print(f"  - {col}")
    
    # Afficher l'analyse Mistral si disponible
    if "analysis" in metadata and metadata["analysis"]:
        print("\n=== ANALYSE MISTRAL ===")
        print(metadata["analysis"])
        
        # Enregistrer l'analyse dans un fichier texte
        analysis_file = output_file.replace(".csv", "_analyse.txt")
        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write("ANALYSE DU DATASET PAR MISTRAL\n")
            f.write("============================\n\n")
            f.write(metadata["analysis"])
        
        print(f"\nL'analyse complète a été enregistrée dans: {analysis_file}")
    else:
        print("\nL'analyse Mistral n'est pas disponible.")
    
    # 6. CONCLUSION
    # ------------
    print("\nTraitement terminé avec succès.")
    print(f"Fichier transformé: {output_file}")

if __name__ == "__main__":
    main()
