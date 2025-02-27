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
    
    def transform(self, df, transformations=None, context=None):
        """
        Applique des transformations à un DataFrame.
        Point d'entrée principal pour l'intégration dans une application.
        
        Args:
            df: DataFrame à transformer
            transformations: Liste des transformations à appliquer 
                             (ou None pour détection automatique)
            context: Contexte optionnel pour guider l'analyse Mistral
        
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
            analysis = self.generate_dataset_analysis(df_transformed, context)
            if analysis:
                metadata["analysis"] = analysis
        
        return df_transformed, metadata
    
    def generate_dataset_analysis(self, df, context=None):
        """
        Génère une analyse orientée insights du dataset avec Mistral.
        
        Args:
            df: DataFrame à analyser
            context: Contexte optionnel fourni par l'utilisateur pour guider l'analyse
                
        Returns:
            str: Analyse textuelle du dataset ou None en cas d'échec
        """
        if self.llm is None:
            return None
        
        self.logger.info("Génération d'une analyse du dataset avec Mistral")
        
        # Créer un prompt personnalisé pour obtenir des insights
        # Préparer des informations générales sur le dataset
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(exclude=['number']).columns
        
        # Produire des statistiques clés sur les corrélations et distributions
        correlations = {}
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr()
                # Trouver les 3 paires les plus corrélées
                corr_pairs = []
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        if i < j:
                            corr_pairs.append((col1, col2, abs(corr_matrix.loc[col1, col2])))
                
                # Trier et prendre les 3 plus fortes corrélations
                corr_pairs.sort(key=lambda x: x[2], reverse=True)
                for col1, col2, corr_val in corr_pairs[:3]:
                    correlations[f"{col1}-{col2}"] = corr_val
            except:
                pass
        
        # Détecter les colonnes avec distributions intéressantes
        skewed_cols = []
        for col in numeric_cols:
            try:
                skew = df[col].skew()
                if abs(skew) > 1.5:
                    skewed_cols.append((col, skew))
            except:
                continue
        
        # Créer un prompt détaillé qui oriente vers les insights
        base_prompt = (
            f"Tu es un analyste de données expert. J'ai un dataset avec {len(df)} lignes et {len(df.columns)} colonnes. "
            f"Je veux que tu identifies 2-3 insights intéressants et pertinents, pas une simple description.\n\n"
            f"Voici quelques informations:\n"
            f"- {len(numeric_cols)} colonnes numériques et {len(categorical_cols)} colonnes catégorielles\n"
            f"- Valeurs manquantes totales: {df.isna().sum().sum()}\n"
        )
        
        # Ajouter des informations sur les corrélations
        if correlations:
            base_prompt += "- Corrélations notables:\n"
            for pair, corr in correlations.items():
                base_prompt += f"  * {pair}: {corr:.2f}\n"
        
        # Ajouter des informations sur les distributions asymétriques
        if skewed_cols:
            base_prompt += "- Colonnes avec distribution asymétrique:\n"
            for col, skew in skewed_cols:
                direction = "droite" if skew > 0 else "gauche"
                base_prompt += f"  * {col}: forte asymétrie vers la {direction}\n"
        
        # Ajouter le contexte spécifique si fourni
        if context:
            prompt = (
                f"{base_prompt}\n"
                f"CONTEXTE SPÉCIFIQUE: {context}\n\n"
                f"Identifie les insights les plus importants dans ce dataset, en tenant compte du contexte fourni. "
                f"Concentre-toi sur ce qui est surprenant, contre-intuitif ou actionnable. "
                f"Ne te contente pas de décrire les colonnes ou les statistiques. "
                f"Sois concis (max. 3-4 phrases par insight)."
            )
        else:
            prompt = (
                f"{base_prompt}\n"
                f"Identifie les 2-3 insights les plus importants dans ce dataset. "
                f"Priorise ce qui est surprenant, contre-intuitif ou actionnable. "
                f"Évite la description simple. Sois concis (max. 3-4 phrases par insight)."
            )
        
        try:
            # Limiter à 300 tokens pour garder l'analyse concise
            result = self.llm(prompt, max_tokens=300)
            analysis = result["choices"][0]["text"].strip()
            self.logger.info("Analyse générée avec succès")
            return analysis
        except Exception as e:
            self.logger.error(f"Erreur d'analyse: {e}")
            return None
    
    def transform_file(self, input_path, output_path=None, context=None):
        """
        Transforme un fichier CSV et sauvegarde le résultat.
        
        Args:
            input_path: Chemin du fichier CSV à transformer
            output_path: Chemin de sortie (ou None pour générer automatiquement)
            context: Contexte optionnel pour l'analyse
            
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
        df_transformed, metadata = self.transform(df, context=context)
        
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

    # Les autres méthodes (_analyze_dataset, _detect_transformations, etc.) restent identiques
    # Je les ai omises ici pour des raisons de concision
    
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
                    words1 = set(re.findall(r'\w+', str(col1).lower()))
                    words2 = set(re.findall(r'\w+', str(col2).lower()))
                    
                    if words1 and words2:  # Éviter les divisions par zéro
                        common_words = words1.intersection(words2)
                        if common_words:
                            sim_score = len(common_words) / max(len(words1), len(words2))
                            if sim_score > 0.3:
                                pairs.append((col1, col2, sim_score))
        
        # Trier par score et éliminer les duplications
        return sorted(pairs, key=lambda x: x[2], reverse=True)
    
    def _detect_transformations(self, df, dataset_info):
        """
        Détecte automatiquement les transformations à appliquer 
        selon les caractéristiques du dataset.
        
        Args:
            df: DataFrame à analyser
            dataset_info: Informations d'analyse du dataset
        
        Returns:
            list: Liste des transformations recommandées
        """
        transformations = []
    
        # 1. Détection des valeurs manquantes
        if dataset_info["missing_values"]["total_missing"] > 0:
            transformations.append("missing_values")
        
        # 2. Détection du besoin de normalisation
        numeric_cols = list(dataset_info["numeric_columns"].keys())
        needs_normalization = False
        
        for col, info in dataset_info["numeric_columns"].items():
            if info.get("needs_normalization", False):
                needs_normalization = True
                break
        
        if needs_normalization and numeric_cols:
            transformations.append("normalize")
        
        # 3. Détection du besoin d'encodage
        categorical_cols = list(dataset_info["categorical_columns"].keys())
        if categorical_cols:
            transformations.append("encode")
        
        # 4. Détection du besoin de fusion de colonnes
        pairs = self._identify_column_pairs(df)
        # Filtrer les paires selon le seuil
        fusion_threshold = 0.3
        eligible_pairs = [p for p in pairs if p[2] >= fusion_threshold]
        
        if eligible_pairs:
            transformations.append("fusion")
        
        return transformations

    def _clean_missing_values(self, df):
        """
        Nettoie les valeurs manquantes dans le DataFrame.
        Ne supprime pas les colonnes avec beaucoup de valeurs manquantes,
        car elles peuvent contenir des informations complémentaires.
        
        Args:
            df: DataFrame à nettoyer
        
        Returns:
            tuple: (DataFrame nettoyé, succès, méthode utilisée)
        """
            # Copier le DataFrame pour éviter de modifier l'original
        df_clean = df.copy()
            
            # Compter les valeurs manquantes par colonne
        missing_counts = df_clean.isna().sum()
        total_rows = len(df_clean)
            
            # Méthode utilisée (par défaut)
        method = "combined"
            
            # Parcourir les colonnes avec des valeurs manquantes
        for col in missing_counts[missing_counts > 0].index:
                    # Pourcentage de valeurs manquantes
            missing_percent = missing_counts[col] / total_rows
                    
                    # MODIFICATION: Ne pas supprimer les colonnes avec beaucoup de valeurs manquantes
                    # Imputer selon le type de données
            if df_clean[col].dtype.kind in 'ifc':  # Numérique
                        # Si plus de 30% manquant, utiliser la médiane, sinon la moyenne
                if missing_percent > 0.3:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    self.logger.info(f"Colonne '{col}': imputation médiane (manquant: {missing_percent:.1%})")
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                    self.logger.info(f"Colonne '{col}': imputation moyenne (manquant: {missing_percent:.1%})")
            else:  # Catégoriel
                        # Utiliser le mode (valeur la plus fréquente)
                mode_value = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else None
                if mode_value is not None:
                    df_clean[col] = df_clean[col].fillna(mode_value)
                    self.logger.info(f"Colonne '{col}': imputation mode (manquant: {missing_percent:.1%})")
                else:
                            # Créer une catégorie "Unknown" si pas de mode
                    df_clean[col] = df_clean[col].fillna("Unknown")
                    self.logger.info(f"Colonne '{col}': imputation 'Unknown' (manquant: {missing_percent:.1%})")
                
                # Vérifier s'il reste des valeurs manquantes
        remaining_missing = df_clean.isna().sum().sum()
                
        if remaining_missing > 0:
                    # Avertissement mais pas de suppression automatique de lignes
            self.logger.warning(f"Il reste encore {remaining_missing} valeurs manquantes")
                
        return df_clean, True, method

    def _normalize_numeric_columns(self, df):
        """
        Normalise les colonnes numériques.
        
        Args:
            df: DataFrame à normaliser
        
        Returns:
            tuple: (DataFrame normalisé, succès, colonnes normalisées)
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        import numpy as np
        
        # Copier le DataFrame
        df_norm = df.copy()
        
        # Identifier les colonnes numériques
        numeric_cols = df_norm.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return df_norm, False, []
        
        # Colonnes effectivement normalisées
        normalized_cols = []
        
        for col in numeric_cols:
            # Vérifier si la normalisation est nécessaire
            col_data = df_norm[col].dropna()  # Ignorer les NaN pour le calcul
            
            if len(col_data) == 0:
                continue  # Colonne vide
            
            # Calculer plage et écart-type
            data_range = col_data.max() - col_data.min()
            std_dev = col_data.std()
            
            # Si la plage est grande ou l'écart-type significatif
            if data_range > 10 or std_dev > 1:
                # Préserver les valeurs NaN
                mask_nan = df_norm[col].isna()
                
                # Sélectionner la méthode de normalisation
                # StandardScaler pour distributions normales, MinMaxScaler pour les autres
                try:
                    if -0.5 < col_data.skew() < 0.5:  # Distribution approximativement normale
                        scaler = StandardScaler()
                        # Transformer uniquement les valeurs non-NaN
                        values = np.array(col_data).reshape(-1, 1)
                        normalized = scaler.fit_transform(values).flatten()
                        
                        # Mettre à jour le DataFrame avec les valeurs normalisées
                        df_norm.loc[~mask_nan, col] = normalized
                        self.logger.info(f"Colonne '{col}': normalisation StandardScaler")
                    else:
                        scaler = MinMaxScaler()
                        # Transformer uniquement les valeurs non-NaN
                        values = np.array(col_data).reshape(-1, 1)
                        normalized = scaler.fit_transform(values).flatten()
                        
                        # Mettre à jour le DataFrame avec les valeurs normalisées
                        df_norm.loc[~mask_nan, col] = normalized
                        self.logger.info(f"Colonne '{col}': normalisation MinMaxScaler")
                    
                    normalized_cols.append(col)
                except Exception as e:
                    self.logger.warning(f"Échec normalisation colonne '{col}': {e}")
        
        return df_norm, len(normalized_cols) > 0, normalized_cols

    def _encode_categorical_columns(self, df):
        """
        Encode les colonnes catégorielles.
        
        Args:
            df: DataFrame à encoder
        
        Returns:
            tuple: (DataFrame encodé, succès, méthode utilisée, colonnes encodées)
        """
        from sklearn.preprocessing import OneHotEncoder, LabelEncoder
        import pandas as pd
        
        # Copier le DataFrame
        df_encoded = df.copy()
        
        # Identifier les colonnes catégorielles
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            return df_encoded, False, "none", []
        
        # Colonnes effectivement encodées
        encoded_cols = []
        methods_used = set()
        
        for col in categorical_cols:
            # Vérifier le nombre de catégories uniques
            n_unique = df_encoded[col].nunique()
            
            # Si trop de catégories uniques ou colonne presque vide, ignorer
            if n_unique > len(df_encoded) * 0.5 or n_unique <= 1:
                continue
            
            # Vérifier si la colonne est binaire ou ordinale
            is_binary = n_unique == 2
            is_ordinal = self._check_ordinal_column(df_encoded[col])
            
            try:
                # 1. Pour les colonnes binaires ou ordinales: LabelEncoder
                if is_binary or is_ordinal:
                    encoder = LabelEncoder()
                    # Tenir compte des valeurs NaN
                    non_nan_mask = ~df_encoded[col].isna()
                    values = df_encoded.loc[non_nan_mask, col]
                    
                    encoded_values = encoder.fit_transform(values)
                    df_encoded.loc[non_nan_mask, col] = encoded_values
                    
                    methods_used.add("label")
                    encoded_cols.append(col)
                    self.logger.info(f"Colonne '{col}': encodage LabelEncoder")
                
                # 2. Pour les autres colonnes catégorielles: OneHotEncoder si peu de catégories
                elif n_unique <= 15:  # Limiter pour éviter l'explosion dimensionnelle
                    encoder = OneHotEncoder(sparse_output=False, drop='first')
                    values = df_encoded[col].fillna('Unknown').values.reshape(-1, 1)
                    
                    encoded_values = encoder.fit_transform(values)
                    categories = encoder.categories_[0][1:]  # Skip la première (dropped)
                    
                    # Créer les nouvelles colonnes
                    for i, category in enumerate(categories):
                        new_col = f"{col}_{category}"
                        df_encoded[new_col] = encoded_values[:, i]
                    
                    # Supprimer la colonne originale
                    df_encoded = df_encoded.drop(columns=[col])
                    
                    methods_used.add("onehot")
                    encoded_cols.append(col)
                    self.logger.info(f"Colonne '{col}': encodage OneHotEncoder ({n_unique} catégories)")
                
                # 3. Pour trop de catégories: frequency encoding
                else:
                    # Calculer la fréquence de chaque catégorie
                    freq_map = df_encoded[col].value_counts(normalize=True).to_dict()
                    
                    # Appliquer l'encodage en tenant compte des NaN
                    # Pour les NaN, utiliser 0 comme fréquence
                    df_encoded[col] = df_encoded[col].map(freq_map).fillna(0)
                    
                    methods_used.add("frequency")
                    encoded_cols.append(col)
                    self.logger.info(f"Colonne '{col}': encodage fréquence ({n_unique} catégories)")
                    
            except Exception as e:
                self.logger.warning(f"Échec encodage colonne '{col}': {e}")
        
        # Déterminer la méthode principale utilisée
        if len(methods_used) == 0:
            method = "none"
        elif len(methods_used) == 1:
            method = methods_used.pop()
        else:
            method = "mixed"
        
        return df_encoded, len(encoded_cols) > 0, method, encoded_cols

    def _fuse_columns(self, df, col1, col2):
        """
        Fusionne deux colonnes en une seule.
        
        Args:
            df: DataFrame contenant les colonnes
            col1: Première colonne à fusionner
            col2: Deuxième colonne à fusionner
        
        Returns:
            tuple: (DataFrame résultant, succès, nom de la nouvelle colonne)
        """
        # Copier le DataFrame
        df_fused = df.copy()
        
        # Vérifier que les colonnes existent
        if col1 not in df_fused.columns or col2 not in df_fused.columns:
            return df_fused, False, None
        
        # Déterminer le type des colonnes
        col1_numeric = pd.api.types.is_numeric_dtype(df_fused[col1])
        col2_numeric = pd.api.types.is_numeric_dtype(df_fused[col2])
        
        # Créer un nouveau nom pour la colonne fusionnée
        # Extraire les racines des noms de colonnes pour un meilleur nom
        words1 = set(re.findall(r'\w+', str(col1).lower()))
        words2 = set(re.findall(r'\w+', str(col2).lower()))
        common_words = words1.intersection(words2)
        
        if common_words:
            prefix = list(common_words)[0]
            new_col = f"{prefix}_combined"
        else:
            new_col = f"{col1}_{col2}_combined"
        
        # 1. Si les deux colonnes sont numériques
        if col1_numeric and col2_numeric:
            # Normaliser si nécessaire (pour éviter qu'une colonne domine)
            mean1, std1 = df_fused[col1].mean(), df_fused[col1].std()
            mean2, std2 = df_fused[col2].mean(), df_fused[col2].std()
            
            if std1 > 0 and std2 > 0:
                norm_col1 = (df_fused[col1] - mean1) / std1
                norm_col2 = (df_fused[col2] - mean2) / std2
                
                # Calculer la colonne fusionnée (moyenne des valeurs normalisées)
                df_fused[new_col] = (norm_col1 + norm_col2) / 2
            else:
                # Si écart-type nul, utiliser une moyenne simple
                df_fused[new_col] = (df_fused[col1] + df_fused[col2]) / 2
            
            self.logger.info(f"Fusion numérique: '{col1}' + '{col2}' -> '{new_col}'")
        
        # 2. Si les deux colonnes sont catégorielles
        elif not col1_numeric and not col2_numeric:
            # Combiner les valeurs des deux colonnes
            df_fused[new_col] = df_fused[col1].astype(str) + "_" + df_fused[col2].astype(str)
            self.logger.info(f"Fusion catégorielle: '{col1}' + '{col2}' -> '{new_col}'")
        
        # 3. Si les types sont différents (une numérique, une catégorielle)
        else:
            # Déterminer quelle colonne est numérique
            num_col = col1 if col1_numeric else col2
            cat_col = col2 if col1_numeric else col1
            
            # Calculer la moyenne par catégorie
            category_means = df_fused.groupby(cat_col)[num_col].mean()
            
            # Créer un nouvel attribut: différence entre valeur et moyenne de catégorie
            df_fused[new_col] = df_fused.apply(
                lambda row: row[num_col] - category_means.get(row[cat_col], 0)
                if pd.notna(row[cat_col]) and pd.notna(row[num_col])
                else 0,
                axis=1
            )
            
            self.logger.info(f"Fusion mixte: '{col1}' + '{col2}' -> '{new_col}'")
        
        # Option: supprimer les colonnes originales
        # Commenter ces lignes pour conserver les colonnes originales
        df_fused = df_fused.drop(columns=[col1, col2])
        
        return df_fused, True, new_col

    def _check_ordinal_column(self, series):
        """
        Vérifie si une colonne semble être ordinale.
        
        Args:
            series: Série pandas à analyser
        
        Returns:
            bool: True si la colonne est probablement ordinale
        """
        # Convertir en liste les valeurs non-NaN
        values = series.dropna().unique().tolist()
        
        # Moins de 2 valeurs uniques ne peut pas être ordinal
        if len(values) < 2:
            return False
        
        # Vérifier les préfixes ordinaux communs
        ordinal_prefixes = ['low', 'medium', 'high', 'very', 'extremely',
                            'small', 'large', 'petit', 'grand', 'moyen',
                            '1st', '2nd', '3rd', '4th', '5th',
                            'i', 'ii', 'iii', 'iv', 'v',
                            'faible', 'modéré', 'élevé', 'très', 'extrêmement',
                            'primaire', 'secondaire', 'tertiaire']
        
        # Vérifier si toutes les valeurs sont numériques
        try:
            # Si tous les éléments peuvent être convertis en nombres
            numeric_values = [float(x) for x in values if str(x).strip()]
            if len(numeric_values) > 0 and len(numeric_values) == len(values):
                return True
        except (ValueError, TypeError):
            pass
        
        # Vérifier les préfixes ordinaux
        for prefix in ordinal_prefixes:
            matches = [str(x).lower().startswith(prefix) for x in values]
            if sum(matches) > 1:  # Au moins 2 valeurs avec ce préfixe
                return True
        
        # Rechercher des mots ordinaux dans les valeurs
        ordinal_words = ['first', 'second', 'third', 'fourth', 'fifth',
                        'premier', 'deuxième', 'troisième', 'quatrième',
                        'basic', 'intermediate', 'advanced', 'expert',
                        'débutant', 'intermédiaire', 'avancé', 'expert']
        
        # Si au moins deux valeurs contiennent des mots ordinaux
        contained_ordinals = sum(1 for x in values if any(ord_word in str(x).lower() for ord_word in ordinal_words))
        if contained_ordinals > 1:
            return True
        
        return False

    def _identify_distribution(self, series):
        """
        Identifie le type de distribution d'une série.
        
        Args:
            series: Série pandas à analyser
        
        Returns:
            str: Type de distribution identifié
        """
        import scipy.stats as stats
        
        # Ignorer les valeurs manquantes
        data = series.dropna()
        
        # Si trop peu de données, impossible de déterminer
        if len(data) < 10:
            return "insufficient_data"
        
        # Calculer les statistiques descriptives
        skewness = data.skew()
        kurtosis = data.kurtosis()
        
        # Tests statistiques pour les distributions communes
        try:
            # Test de normalité (Shapiro-Wilk)
            # Limiter à 5000 échantillons pour la performance
            sample = data.sample(min(5000, len(data))) if len(data) > 5000 else data
            shapiro_stat, shapiro_p = stats.shapiro(sample)
            
            # Test de Poisson (moyenne ≈ variance)
            mean, var = data.mean(), data.var()
            poisson_ratio = abs(mean - var) / mean if mean != 0 else float('inf')
            
            # Test d'uniformité (Kolmogorov-Smirnov)
            a, b = data.min(), data.max()
            if a != b:
                ks_stat, ks_p = stats.kstest(data, 'uniform', args=(a, b - a))
            else:
                ks_p = 0
        except Exception:
            return "unknown"
        
        # Décision basée sur les tests et statistiques
        if shapiro_p > 0.05:
            # Distribution normale
            return "normal"
        elif poisson_ratio < 0.2 and data.min() >= 0 and all(x.is_integer() for x in data):
            # Distribution de Poisson
            return "poisson"
        elif ks_p > 0.05:
            # Distribution uniforme
            return "uniform"
        elif skewness > 1:
            # Distribution à queue droite
            return "right_skewed"
        elif skewness < -1:
            # Distribution à queue gauche
            return "left_skewed"
        elif kurtosis > 1:
            # Distribution leptokurtique (queue lourde)
            return "heavy_tailed"
        elif kurtosis < -1:
            # Distribution platykurtique (queue légère)
            return "light_tailed"
        else:
            # Distribution non identifiée
            return "unknown"
    
    def _analyze_dataset(self, df):
        """
        Analyse approfondie et détaillée du dataset.
        
        Args:
            df (pandas.DataFrame): DataFrame à analyser
        
        Returns:
            dict: Informations détaillées et structurées sur le dataset
        """
        # Informations générales
        info = {
            "dataset_info": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "data_types": dict(df.dtypes)
            },
            "missing_values": {
                "total_missing": df.isna().sum().sum(),
                "missing_per_column": df.isna().sum().to_dict()
            },
            "numeric_columns": {},
            "categorical_columns": {},
            "distribution_insights": {}
        }
        
        # Analyse des colonnes numériques
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            try:
                col_data = df[col]
                info["numeric_columns"][col] = {
                    "mean": col_data.mean(),
                    "median": col_data.median(),
                    "std_dev": col_data.std(),
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "skewness": col_data.skew(),
                    "kurtosis": col_data.kurtosis(),
                    "quartiles": {
                        "Q1": col_data.quantile(0.25),
                        "Q2": col_data.median(),
                        "Q3": col_data.quantile(0.75)
                    },
                    "iqr": col_data.quantile(0.75) - col_data.quantile(0.25),
                    "outliers": {
                        "lower_bound": col_data.quantile(0.25) - 1.5 * (col_data.quantile(0.75) - col_data.quantile(0.25)),
                        "upper_bound": col_data.quantile(0.75) + 1.5 * (col_data.quantile(0.75) - col_data.quantile(0.25))
                    },
                    "needs_normalization": (col_data.std() > 0 and (col_data.max() - col_data.min()) > 10),
                    "variance_coefficient": (col_data.std() / col_data.mean() * 100) if col_data.mean() != 0 else None
                }
            except Exception as e:
                self.logger.warning(f"Erreur analyse colonne numérique {col}: {e}")
        
        # Analyse des colonnes catégorielles
        categorical_cols = df.select_dtypes(exclude=['number']).columns
        for col in categorical_cols:
            try:
                col_data = df[col]
                value_counts = col_data.value_counts()
                mode = col_data.mode()[0] if not col_data.mode().empty else None
                
                info["categorical_columns"][col] = {
                    "unique_values": col_data.nunique(),
                    "mode": mode,
                    "top_5_values": value_counts.head().to_dict(),
                    "bottom_5_values": value_counts.tail().to_dict(),
                    "most_frequent_percentage": (value_counts.iloc[0] / len(col_data) * 100) if len(value_counts) > 0 else 0,
                    "is_binary": col_data.nunique() == 2,
                    "is_ordinal": self._check_ordinal_column(col_data)
                }
            except Exception as e:
                self.logger.warning(f"Erreur analyse colonne catégorielle {col}: {e}")
        
        # Analyse des distributions
        try:
            for col in numeric_cols:
                dist_type = self._identify_distribution(df[col])
                info["distribution_insights"][col] = dist_type
        except Exception as e:
            self.logger.warning(f"Erreur analyse distribution: {e}")
        
        return info

if __name__ == "__main__":
    # Exemple d'utilisation en script autonome
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        model_path = sys.argv[3] if len(sys.argv) > 3 else None
        context = sys.argv[4] if len(sys.argv) > 4 else None
        
        # Créer et utiliser le transformateur
        transformer = DataTransformer(model_path=model_path)
        success, metadata = transformer.transform_file(input_file, output_file, context)
        
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
        print("Usage: python transformer.py input.csv [output.csv] [model_path] [context]")
