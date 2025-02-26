import os
import pandas as pd
import numpy as np
import re
from llama_cpp import Llama
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DataTransformer:
    """
    Classe d'agent de transformation intégrable dans une application plus large.
    Conçue pour être utilisée comme un composant d'une application plus complète.
    """
    
    def __init__(self, model_path=None, context_size=512, log_level=logging.INFO):
        """
        Initialise l'agent de transformation.
        
        Args:
            model_path: Chemin vers le modèle LLM (ou None pour mode sans modèle)
            context_size: Taille du contexte du modèle
            log_level: Niveau de détail des logs
        """
        self.model_path = model_path
        self.context_size = context_size
        self.llm = None
        
        # Configurer le logger
        self.logger = logging.getLogger(f"{__name__}.transformer")
        self.logger.setLevel(log_level)
        
        # Charger le modèle s'il est spécifié
        if model_path and os.path.exists(model_path):
            try:
                self.llm = Llama(model_path=model_path, n_ctx=context_size)
                self.logger.info(f"Modèle chargé: {model_path}")
            except Exception as e:
                self.logger.error(f"Erreur lors du chargement du modèle: {e}")
    
    def transform(self, df, transformations=None):
        """
        Applique des transformations à un DataFrame.
        Point d'entrée principal pour l'intégration dans une application.
        
        Args:
            df: DataFrame à transformer
            transformations: Liste des transformations à appliquer 
                             (ou None pour détection automatique)
        
        Returns:
            DataFrame: DataFrame transformé
            dict: Métadonnées sur les transformations appliquées
        """
        self.logger.info(f"Transformation d'un DataFrame: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        # Copier le DataFrame pour éviter de modifier l'original
        df_transformed = df.copy()
        
        # Analyser le dataset pour obtenir des informations
        dataset_info = self._analyze_dataset(df)
        
        # Déterminer les transformations à appliquer
        if transformations is None:
            transformations = self._detect_transformations(df, dataset_info)
            self.logger.info(f"Transformations détectées: {transformations}")
        
        # Appliquer les transformations
        transformations_applied = []
        
        # Appliquer d'abord les transformations qui ne sont pas des fusions
        for transform in transformations:
            if transform != "fusion":
                if transform == "missing_values":
                    df_transformed, success, method = self._clean_missing_values(df_transformed)
                    if success:
                        transformations_applied.append({
                            "type": "missing_values",
                            "method": method,
                            "details": f"Nettoyage des valeurs manquantes avec méthode: {method}"
                        })
                
                elif transform == "normalize":
                    df_transformed, success, cols = self._normalize_numeric_columns(df_transformed)
                    if success:
                        transformations_applied.append({
                            "type": "normalize",
                            "columns": cols,
                            "details": f"Normalisation de {len(cols)} colonnes numériques"
                        })
                
                elif transform == "encode":
                    df_transformed, success, method, cols = self._encode_categorical_columns(df_transformed)
                    if success:
                        transformations_applied.append({
                            "type": "encode",
                            "method": method,
                            "columns": cols,
                            "details": f"Encodage de {len(cols)} colonnes avec méthode: {method}"
                        })
        
        # Appliquer la fusion si nécessaire (toutes les paires éligibles)
        if "fusion" in transformations:
            # Identifier les paires de colonnes
            pairs = self._identify_column_pairs(df_transformed)
            
            # Filtrer les paires selon le seuil
            fusion_threshold = 0.3  # Seuil de fusion
            eligible_pairs = [p for p in pairs if p[2] >= fusion_threshold]
            
            if eligible_pairs:
                self.logger.info(f"Fusion de {len(eligible_pairs)} paires de colonnes")
                
                # Appliquer la fusion pour chaque paire éligible
                for col1, col2, score in eligible_pairs:
                    # Vérifier que les colonnes existent encore (pas déjà fusionnées)
                    if col1 in df_transformed.columns and col2 in df_transformed.columns:
                        df_result, success, new_col = self._fuse_columns(df_transformed, col1, col2)
                        if success:
                            df_transformed = df_result  # Mettre à jour le DataFrame
                            transformations_applied.append({
                                "type": "fusion",
                                "columns": [col1, col2],
                                "new_column": new_col,
                                "score": score,
                                "details": f"Fusion de '{col1}' et '{col2}' → '{new_col}' (score: {score:.2f})"
                            })
        
        # Créer les métadonnées de résultat
        metadata = {
            "original_shape": df.shape,
            "transformed_shape": df_transformed.shape,
            "transformations": transformations_applied,
            "new_columns": [col for col in df_transformed.columns if col not in df.columns],
            "removed_columns": [col for col in df.columns if col not in df_transformed.columns],
            "missing_values": {
                "before": df.isna().sum().sum(),
                "after": df_transformed.isna().sum().sum()
            }
        }
        
        # Ajouter une analyse avec Mistral si disponible
        if self.llm is not None:
            analysis = self.generate_dataset_analysis(df_transformed)
            if analysis:
                metadata["analysis"] = analysis
        
        return df_transformed, metadata
    
    def generate_dataset_analysis(self, df):
        """
        Génère une analyse simple du dataset avec Mistral.
        
        Args:
            df: DataFrame à analyser
            
        Returns:
            str: Analyse textuelle du dataset ou None en cas d'échec
        """
        if self.llm is None:
            return None
        
        self.logger.info("Génération d'une analyse du dataset avec Mistral")
        
        # Créer un prompt très simple pour éviter les problèmes de contexte
        prompt = (
            f"Analyse ce dataset de {len(df)} lignes et {len(df.columns)} colonnes.\n"
            f"Colonnes numériques: {len(df.select_dtypes(include=['number']).columns)}\n"
            f"Colonnes catégorielles: {len(df.select_dtypes(exclude=['number']).columns)}\n"
            f"Valeurs manquantes: {df.isna().sum().sum()}\n\n"
            f"Donne une analyse très courte en 3 phrases maximum sans mettre de titre."
        )
        
        try:
            # Utiliser max_tokens réduit
            result = self.llm(prompt, max_tokens=200)
            analysis = result["choices"][0]["text"].strip()
            self.logger.info("Analyse générée avec succès")
            return analysis
        except Exception as e:
            self.logger.error(f"Erreur d'analyse: {e}")
            return None
    
    def transform_file(self, input_path, output_path=None):
        """
        Transforme un fichier CSV et sauvegarde le résultat.
        
        Args:
            input_path: Chemin du fichier CSV à transformer
            output_path: Chemin de sortie (ou None pour générer automatiquement)
            
        Returns:
            tuple: (Succès de l'opération, Métadonnées des transformations)
        """
        # Générer un nom de fichier de sortie si non spécifié
        if output_path is None:
            base_name = os.path.basename(input_path)
            dir_name = os.path.dirname(input_path) or os.getcwd()
            output_path = os.path.join(dir_name, f"transformed_{base_name}")
        
        # Charger le dataset
        try:
            df = pd.read_csv(input_path)
            self.logger.info(f"Dataset chargé: {len(df)} lignes, {len(df.columns)} colonnes")
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du CSV: {e}")
            return False, None
        
        # Transformer le dataset
        df_transformed, metadata = self.transform(df)
        
        # Sauvegarder le résultat
        try:
            df_transformed.to_csv(output_path, index=False)
            self.logger.info(f"Dataset transformé sauvegardé: {output_path}")
            
            # Sauvegarder l'analyse dans un fichier texte si disponible
            if "analysis" in metadata and metadata["analysis"]:
                analysis_path = output_path.replace(".csv", "_analyse.txt")
                with open(analysis_path, "w", encoding="utf-8") as f:
                    f.write("ANALYSE DU DATASET PAR MISTRAL\n")
                    f.write("=============================\n\n")
                    f.write(metadata["analysis"])
                self.logger.info(f"Analyse sauvegardée: {analysis_path}")
            
            return True, metadata
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde: {e}")
            return False, metadata
    
    def _analyze_dataset(self, df):
        """Analyse le dataset et retourne des informations utiles."""
        info = {
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(exclude=['number']).columns.tolist(),
            "missing_values": df.isna().sum().to_dict(),
            "total_missing": df.isna().sum().sum()
        }
        
        # Ajouter des statistiques pour les colonnes numériques
        if info["numeric_columns"]:
            info["numeric_stats"] = {}
            for col in info["numeric_columns"]:
                try:
                    info["numeric_stats"][col] = {
                        "mean": df[col].mean(),
                        "std": df[col].std(),
                        "min": df[col].min(),
                        "max": df[col].max(),
                        "skew": df[col].skew(),
                        "needs_normalization": df[col].std() > 0 and (df[col].max() - df[col].min()) > 10
                    }
                except:
                    # Ignorer les erreurs pour cette colonne
                    pass
        
        # Ajouter des statistiques pour les colonnes catégorielles
        if info["categorical_columns"]:
            info["categorical_stats"] = {}
            for col in info["categorical_columns"]:
                try:
                    top_value = df[col].mode()[0] if not df[col].mode().empty else None
                    info["categorical_stats"][col] = {
                        "unique_values": df[col].nunique(),
                        "top_value": top_value,
                        "suitable_for_onehot": df[col].nunique() < 10
                    }
                except:
                    # Ignorer les erreurs pour cette colonne
                    pass
        
        return info
    
    def _detect_transformations(self, df, info):
        """Détecte automatiquement les transformations nécessaires."""
        transformations = []
        
        # Valeurs manquantes
        if info["total_missing"] > 0:
            transformations.append("missing_values")
        
        # Normalisation
        if info["numeric_columns"] and "numeric_stats" in info:
            needs_normalization = any(
                info["numeric_stats"].get(col, {}).get("needs_normalization", False)
                for col in info["numeric_columns"]
            )
            if needs_normalization:
                transformations.append("normalize")
        
        # Encodage
        if info["categorical_columns"]:
            transformations.append("encode")
        
        # Fusion de colonnes
        pairs = self._identify_column_pairs(df)
        if pairs:
            # Seuil réduit à 0.3 pour faciliter la fusion de colonnes
            best_score = pairs[0][2]
            if best_score > 0.3:  # Moins strict qu'avant (0.7)
                transformations.append("fusion")
        
        return transformations
    
    def _clean_missing_values(self, df):
        """
        Nettoie les valeurs manquantes.
        
        Returns:
            tuple: (DataFrame nettoyé, succès, méthode utilisée)
        """
        df_clean = df.copy()
        na_counts = df.isna().sum()
        cols_with_na = na_counts[na_counts > 0].index
        
        if len(cols_with_na) == 0:
            return df_clean, False, None
        
        # Déterminer la méthode à utiliser (moyenne, médiane ou mode)
        # Basé sur l'asymétrie des données
        numeric_cols = [col for col in cols_with_na if pd.api.types.is_numeric_dtype(df[col])]
        if numeric_cols:
            skew_values = [abs(df[col].skew()) for col in numeric_cols if df[col].nunique() > 1]
            avg_skew = np.mean(skew_values) if skew_values else 0
            method = "médiane" if avg_skew > 1 else "moyenne"
        else:
            method = "mode"
        
        # Appliquer la méthode choisie
        for col in cols_with_na:
            if pd.api.types.is_numeric_dtype(df[col]):
                if method == "moyenne":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                elif method == "médiane":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:  # mode
                    if not df_clean[col].mode().empty:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                    else:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            else:
                # Colonnes catégorielles: toujours utiliser le mode
                if not df_clean[col].mode().empty:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                else:
                    df_clean[col] = df_clean[col].fillna("Unknown")
        
        return df_clean, True, method
    
    def _normalize_numeric_columns(self, df):
        """
        Normalise les colonnes numériques.
        
        Returns:
            tuple: (DataFrame normalisé, succès, liste des colonnes normalisées)
        """
        df_norm = df.copy()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return df_norm, False, []
        
        normalized_cols = []
        for col in numeric_cols:
            # Vérifier si la normalisation est nécessaire
            if df[col].std() > 0 and df[col].nunique() > 1:
                # Z-score normalization
                df_norm[col] = (df[col] - df[col].mean()) / df[col].std()
                normalized_cols.append(col)
        
        return df_norm, len(normalized_cols) > 0, normalized_cols
    
    def _encode_categorical_columns(self, df):
        """
        Encode les colonnes catégorielles.
        
        Returns:
            tuple: (DataFrame encodé, succès, méthode utilisée, liste des colonnes encodées)
        """
        df_encoded = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if not categorical_cols:
            return df_encoded, False, None, []
        
        # Déterminer si one-hot est approprié
        can_use_onehot = True
        total_unique_values = 0
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            total_unique_values += unique_count
            
            if unique_count > 10:
                can_use_onehot = False
                break
        
        # Si le total des valeurs uniques est trop grand, éviter one-hot
        if total_unique_values > 100:
            can_use_onehot = False
        
        # Appliquer l'encodage approprié
        encoded_cols = []
        if can_use_onehot:
            method = "onehot"
            for col in categorical_cols:
                try:
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    df_encoded = df_encoded.drop(col, axis=1)
                    encoded_cols.append(col)
                except:
                    # Fallback à label encoding en cas d'erreur
                    df_encoded[col] = pd.factorize(df[col])[0]
                    encoded_cols.append(col)
        else:
            method = "label"
            for col in categorical_cols:
                df_encoded[col] = pd.factorize(df[col])[0]
                encoded_cols.append(col)
        
        return df_encoded, True, method, encoded_cols
    
    def _identify_column_pairs(self, df):
        """
        Identifie des paires de colonnes qui pourraient être fusionnées.
        
        Returns:
            list: Liste de tuples (col1, col2, score)
        """
        pairs = []
        
        # 1. Corrélations entre colonnes numériques
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr()
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        if i < j and abs(corr_matrix.loc[col1, col2]) > 0.5:
                            pairs.append((col1, col2, abs(corr_matrix.loc[col1, col2])))
            except:
                pass
        
        # 2. Similarité des noms de colonnes
        columns = df.columns
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i < j:
                    # Calculer la similarité de nom
                    words1 = set(re.findall(r'\w+', col1.lower()))
                    words2 = set(re.findall(r'\w+', col2.lower()))
                    
                    if words1 and words2:  # Éviter les divisions par zéro
                        common_words = words1.intersection(words2)
                        if common_words:
                            sim_score = len(common_words) / max(len(words1), len(words2))
                            if sim_score > 0.3:
                                pairs.append((col1, col2, sim_score))
        
        # Trier par score et éliminer les duplications
        return sorted(pairs, key=lambda x: x[2], reverse=True)
    
    def _fuse_columns(self, df, col1, col2):
        """
        Fusionne deux colonnes.
        
        Returns:
            tuple: (DataFrame avec la fusion, succès, nom de la nouvelle colonne)
        """
        df_result = df.copy()
        
        # Créer un nom pour la nouvelle colonne
        new_col_name = f"{col1}_{col2}_Combined"
        
        # Déterminer les types
        is_col1_numeric = pd.api.types.is_numeric_dtype(df[col1])
        is_col2_numeric = pd.api.types.is_numeric_dtype(df[col2])
        
        try:
            # Deux colonnes numériques
            if is_col1_numeric and is_col2_numeric:
                df_result[new_col_name] = (df[col1] + df[col2]) / 2
            
            # Deux colonnes catégorielles
            elif not is_col1_numeric and not is_col2_numeric:
                df_result[new_col_name] = df[col1].astype(str) + " | " + df[col2].astype(str)
            
            # Une colonne numérique et une catégorielle
            else:
                if is_col1_numeric:
                    num_col, cat_col = col1, col2
                else:
                    num_col, cat_col = col2, col1
                
                # Créer des bins pour la partie numérique
                try:
                    bins = pd.qcut(df[num_col], 3, labels=['Faible', 'Moyen', 'Élevé'])
                except:
                    bins = pd.cut(df[num_col], 3, labels=['Faible', 'Moyen', 'Élevé'])
                
                df_result[new_col_name] = bins.astype(str) + " " + df[cat_col].astype(str)
            
            return df_result, True, new_col_name
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la fusion: {e}")
            return df, False, None


# Exemple d'utilisation dans une application
if __name__ == "__main__":
    # Ce bloc ne s'exécute que lorsque le fichier est exécuté directement
    # Pas lorsqu'il est importé comme module
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        model_path = sys.argv[3] if len(sys.argv) > 3 else None
        
        # Créer et utiliser le transformateur
        transformer = DataTransformer(model_path=model_path)
        success, metadata = transformer.transform_file(input_file, output_file)
        
        if success:
            print(f"Transformation réussie.")
            if metadata and metadata['transformations']:
                print("Transformations appliquées:")
                for i, t in enumerate(metadata['transformations'], 1):
                    print(f"  {i}. {t['details']}")
            
            if "analysis" in metadata and metadata["analysis"]:
                print("\nAnalyse du dataset:")
                print(metadata["analysis"])
    else:
        print("Usage: python transformer.py input.csv [output.csv] [model_path]")
