import os
import json
import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

class AnalysisHistory:
    """
    Classe qui gère l'historique des analyses pour maintenir un contexte 
    entre différentes sessions d'analyse de données.
    """
    
    def __init__(self, storage_dir: str = "analysis_history"):
        """
        Initialise le gestionnaire d'historique d'analyse
        
        Args:
            storage_dir: Répertoire où seront stockés les fichiers d'historique
        """
        self.storage_dir = storage_dir
        print(f"Initialisation de l'historique d'analyse, répertoire: {storage_dir}")
        
        # Créer le répertoire de stockage s'il n'existe pas
        if not os.path.exists(storage_dir):
            try:
                os.makedirs(storage_dir)
                print(f"Répertoire créé: {storage_dir}")
            except Exception as e:
                print(f"Erreur lors de la création du répertoire: {e}")
        else:
            print(f"Répertoire existant: {storage_dir}")
        
        # Fichier qui contient l'index de tous les documents d'analyse
        self.index_file = os.path.normpath(os.path.join(storage_dir, "analysis_index.json"))
        
        # Charger l'index existant ou en créer un nouveau
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.index = json.load(f)
                print(f"Index chargé: {len(self.index.get('analyses', []))} analyses trouvées")
            except Exception as e:
                print(f"Erreur lors du chargement de l'index: {e}")
                self.index = {
                    "analyses": [],
                    "datasets": {},
                    "last_updated": datetime.datetime.now().isoformat()
                }
        else:
            self.index = {
                "analyses": [],
                "datasets": {},
                "last_updated": datetime.datetime.now().isoformat()
            }
            self._save_index()
    
    def _save_index(self):
        """Sauvegarde l'index dans le fichier"""
        try:
            self.index["last_updated"] = datetime.datetime.now().isoformat()
            index_file = os.path.normpath(self.index_file)
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, ensure_ascii=False, indent=2)
            print(f"Index sauvegardé dans: {index_file}")
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'index: {e}")
            return False
    
    def add_analysis(self, dataset_name: str, dataset_description: str, analysis_text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Ajoute une nouvelle analyse à l'historique
        
        Args:
            dataset_name: Nom du jeu de données analysé
            dataset_description: Description du jeu de données
            analysis_text: Texte de l'analyse générée
            metadata: Métadonnées supplémentaires (dimensions, types, etc.)
                
        Returns:
            str: ID de l'analyse créée ou None en cas d'échec
        """
        try:
            # Générer un identifiant unique pour cette analyse
            timestamp = datetime.datetime.now().isoformat()
            timestamp_safe = timestamp.replace(':', '-').replace('.', '_')  # Rendre le timestamp sûr pour les noms de fichiers
            analysis_id = f"analysis_{len(self.index['analyses'])+1}_{timestamp_safe}"
            
            # Fonction pour convertir les types NumPy en types Python standards
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_numpy_types(item) for item in obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return convert_numpy_types(obj.tolist())
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                else:
                    return obj
            
            # Convertir les métadonnées pour s'assurer que tous les types NumPy sont convertis
            metadata_converted = convert_numpy_types(metadata or {})
            
            # Créer l'entrée pour cette analyse
            analysis_entry = {
                "id": analysis_id,
                "dataset_name": dataset_name,
                "dataset_description": dataset_description,
                "analysis": analysis_text,
                "timestamp": timestamp,
                "metadata": metadata_converted
            }
            
            # S'assurer que le répertoire existe
            if not os.path.exists(self.storage_dir):
                print(f"Création du répertoire de stockage: {self.storage_dir}")
                os.makedirs(self.storage_dir, exist_ok=True)
            
            # Sauvegarder l'analyse dans un fichier séparé
            analysis_file = os.path.normpath(os.path.join(self.storage_dir, f"{analysis_id}.json"))
            print(f"Tentative de sauvegarde de l'analyse dans: {analysis_file}")
            
            # Définir un encodeur JSON personnalisé pour les types NumPy
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    return super(NumpyEncoder, self).default(obj)
            
            # Écrire le fichier JSON avec l'encodeur personnalisé
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_entry, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            
            # Vérifier que le fichier a bien été créé
            if not os.path.exists(analysis_file):
                print(f"Erreur: Le fichier d'analyse n'a pas été créé: {analysis_file}")
                # Sauvegarde de secours dans le répertoire courant
                backup_file = f"analyse_backup_{timestamp_safe}.json"
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis_entry, f, ensure_ascii=False, indent=2)
                print(f"Sauvegarde de secours créée: {backup_file}")
                
            else:
                print(f"Analyse sauvegardée avec succès dans: {analysis_file}")
            
            # Ajouter à l'index
            self.index["analyses"].append({
                "id": analysis_id,
                "dataset_name": dataset_name,
                "timestamp": timestamp,
                "summary": analysis_text[:100] + "..." if len(analysis_text) > 100 else analysis_text
            })
            
            # Mettre à jour les informations du dataset dans l'index
            if dataset_name not in self.index["datasets"]:
                self.index["datasets"][dataset_name] = {
                    "description": dataset_description,
                    "first_analysis": timestamp,
                    "last_analysis": timestamp,
                    "analyses_count": 1,
                    "analyses": [analysis_id]
                }
            else:
                self.index["datasets"][dataset_name]["last_analysis"] = timestamp
                self.index["datasets"][dataset_name]["analyses_count"] += 1
                self.index["datasets"][dataset_name]["analyses"].append(analysis_id)
            
            # Sauvegarder l'index mis à jour
            try:
                self._save_index()
                print(f"Index mis à jour avec succès")
            except Exception as e:
                print(f"Erreur lors de la sauvegarde de l'index: {e}")
                # Sauvegarde de secours de l'index
                backup_index = f"index_backup_{timestamp_safe}.json"
                with open(backup_index, 'w', encoding='utf-8') as f:
                    json.dump(self.index, f, ensure_ascii=False, indent=2)
                print(f"Sauvegarde de secours de l'index créée: {backup_index}")
            
            return analysis_id
            
        except Exception as e:
            print(f"Erreur lors de l'ajout de l'analyse: {e}")
            
            # Sauvegarde de secours en cas d'erreur
            try:
                backup_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = f"analyse_erreur_{backup_timestamp}.txt"
                with open(backup_file, 'w', encoding='utf-8') as f:
                    f.write(f"ERREUR DE SAUVEGARDE - {backup_timestamp}\n\n")
                    f.write(f"Dataset: {dataset_name}\n")
                    f.write(f"Description: {dataset_description}\n\n")
                    f.write("ANALYSE:\n" + analysis_text)
                print(f"Sauvegarde d'urgence créée: {backup_file}")
            except Exception as backup_err:
                print(f"Échec également de la sauvegarde d'urgence: {backup_err}")
            
            return None
    
    def get_analysis(self, analysis_id: str) -> Dict[str, Any]:
        """
        Récupère une analyse spécifique par son ID
        
        Args:
            analysis_id: Identifiant de l'analyse
                
        Returns:
            Dict: Contenu de l'analyse ou None si non trouvée
        """
        # Normalisation du chemin pour éviter les problèmes de séparateurs
        analysis_file = os.path.normpath(os.path.join(self.storage_dir, f"{analysis_id}.json"))
        
        # Vérification de l'existence du fichier
        if not os.path.exists(analysis_file):
            print(f"Fichier d'analyse non trouvé: {analysis_file}")
            return None
        
        try:
            # Ouverture et lecture du fichier
            with open(analysis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Analyse {analysis_id} chargée avec succès")
                return data
        except json.JSONDecodeError as e:
            # Erreur de décodage JSON
            print(f"Erreur de décodage JSON pour le fichier {analysis_file}: {e}")
            return None
        except IOError as e:
            # Erreur d'entrée/sortie (lecture du fichier)
            print(f"Erreur d'accès au fichier {analysis_file}: {e}")
            return None
        except Exception as e:
            # Autres erreurs potentielles
            print(f"Erreur inattendue lors de la lecture de l'analyse {analysis_id}: {e}")
            return None
    
    def get_dataset_analyses(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Récupère toutes les analyses associées à un jeu de données
        
        Args:
            dataset_name: Nom du jeu de données
            
        Returns:
            List: Liste des analyses complètes
        """
        if dataset_name not in self.index["datasets"]:
            print(f"Aucune analyse trouvée pour le dataset {dataset_name}")
            return []
        
        analyses = []
        for analysis_id in self.index["datasets"][dataset_name]["analyses"]:
            analysis = self.get_analysis(analysis_id)
            if analysis:
                analyses.append(analysis)
        
        # Trier par date (plus récent en premier)
        analyses.sort(key=lambda x: x["timestamp"], reverse=True)
        return analyses
    
    def get_recent_analyses(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Récupère les analyses les plus récentes
        
        Args:
            limit: Nombre maximum d'analyses à récupérer
            
        Returns:
            List: Liste des analyses récentes
        """
        # Vérifier si l'index existe et contient des analyses
        if not self.index or "analyses" not in self.index or not self.index["analyses"]:
            print("Aucune analyse dans l'index")
            return []
        
        # Trier l'index des analyses par date
        try:
            sorted_analyses = sorted(
                self.index["analyses"], 
                key=lambda x: x.get("timestamp", ""), 
                reverse=True
            )
        except Exception as e:
            print(f"Erreur lors du tri des analyses: {e}")
            # Tentative de récupération sans tri
            sorted_analyses = self.index["analyses"][:limit]
        
        # Récupérer les analyses complètes
        recent_analyses = []
        for analysis_info in sorted_analyses[:limit]:
            try:
                if "id" not in analysis_info:
                    print(f"Erreur: ID manquant dans l'info d'analyse: {analysis_info}")
                    continue
                    
                analysis_id = analysis_info["id"]
                analysis = self.get_analysis(analysis_id)
                
                if analysis:
                    recent_analyses.append(analysis)
                else:
                    print(f"Impossible de récupérer l'analyse avec ID: {analysis_id}")
                    
                    # Tentative de récupération alternative à partir de l'index
                    if all(key in analysis_info for key in ["dataset_name", "timestamp", "summary"]):
                        print(f"Utilisation des informations de l'index pour {analysis_id}")
                        # Créer une version allégée de l'analyse à partir des données de l'index
                        minimal_analysis = {
                            "id": analysis_id,
                            "dataset_name": analysis_info["dataset_name"],
                            "timestamp": analysis_info["timestamp"],
                            "analysis": analysis_info.get("summary", "Analyse non disponible"),
                            "dataset_description": "Information non disponible"
                        }
                        recent_analyses.append(minimal_analysis)
            except Exception as e:
                print(f"Erreur lors de la récupération de l'analyse {analysis_info.get('id', 'ID inconnu')}: {e}")
        
        return recent_analyses
    
    def generate_context(self, dataset_name: Optional[str] = None, max_analyses: int = 3) -> str:
        """
        Génère un contexte à partir des analyses précédentes pour alimenter le modèle
        
        Args:
            dataset_name: Nom du jeu de données pour lequel générer le contexte
                         (Si None, utilise les analyses les plus récentes tous datasets confondus)
            max_analyses: Nombre maximum d'analyses à inclure dans le contexte
            
        Returns:
            str: Texte de contexte formaté pour le modèle
        """
        if dataset_name:
            analyses = self.get_dataset_analyses(dataset_name)[:max_analyses]
        else:
            analyses = self.get_recent_analyses(max_analyses)
        
        if not analyses:
            return ""
        
        # Formater le contexte pour le modèle - clairement marqué comme référence secondaire
        context = "CONTEXTE D'ANALYSES PRÉCÉDENTES (RÉFÉRENCE SECONDAIRE):\n"
        context += "======================================================\n\n"
        context += "NOTE: Ce contexte historique est fourni uniquement comme référence et ne doit pas\n"
        context += "remplacer les instructions actuelles de l'utilisateur, qui ont toujours priorité.\n\n"
        
        for i, analysis in enumerate(analyses, 1):
            context += f"ANALYSE HISTORIQUE {i}: {analysis['dataset_name']}\n"
            context += f"Date: {analysis['timestamp']}\n"
            if analysis.get('dataset_description'):
                context += f"Description: {analysis['dataset_description']}\n"
            
            # Ajouter les métadonnées pertinentes
            if analysis.get('metadata'):
                meta = analysis['metadata']
                if meta.get('dimensions'):
                    context += f"Dimensions: {meta['dimensions']}\n"
                if meta.get('columns_types'):
                    context += f"Types de colonnes: {meta['columns_types']}\n"
            
            # Ajouter le texte de l'analyse
            analysis_text = analysis.get('analysis', '')
            if isinstance(analysis_text, str):
                context += f"\nANALYSE HISTORIQUE:\n{analysis_text}\n\n"
            else:
                context += f"\nANALYSE HISTORIQUE:\n(Format non textuel)\n\n"
            
            context += "-" * 40 + "\n\n"
        
        return context
    
    def find_dataset_by_similarity(self, df: pd.DataFrame, threshold: float = 0.7) -> Optional[str]:
        """
        Tente d'identifier un dataset déjà analysé qui pourrait être similaire
        à celui actuellement en cours d'analyse
        
        Args:
            df: DataFrame à comparer
            threshold: Seuil de similarité pour considérer les datasets comme identiques
            
        Returns:
            str: Nom du dataset similaire ou None si aucun trouvé
        """
        # Cette implémentation est simplifiée et pourrait être améliorée
        # avec des méthodes plus avancées de comparaison de datasets
        
        current_columns = set(df.columns)
        current_shape = df.shape
        
        for dataset_name, dataset_info in self.index["datasets"].items():
            # Récupérer la première analyse pour ce dataset
            if not dataset_info["analyses"]:
                continue
                
            first_analysis = self.get_analysis(dataset_info["analyses"][0])
            if not first_analysis or not first_analysis.get("metadata"):
                continue
                
            meta = first_analysis["metadata"]
            
            # Vérifier si les colonnes et dimensions sont similaires
            if meta.get("columns") and meta.get("dimensions"):
                stored_columns = set(meta["columns"])
                stored_shape = meta["dimensions"]
                
                # Calculer un score de similarité basique
                columns_similarity = len(current_columns.intersection(stored_columns)) / max(len(current_columns), len(stored_columns))
                
                # Les dimensions sont considérées comme similaires si le nombre de lignes est proche
                # et si le nombre de colonnes est identique ou très proche
                shape_similarity = 0
                if stored_shape and len(stored_shape) == 2 and len(current_shape) == 2:
                    if abs(stored_shape[1] - current_shape[1]) <= 2:  # Nombre de colonnes similaire
                        # Similarité sur le nombre de lignes
                        rows_ratio = min(stored_shape[0], current_shape[0]) / max(stored_shape[0], current_shape[0])
                        shape_similarity = rows_ratio
                
                # Combiner les scores de similarité
                total_similarity = (columns_similarity * 0.7) + (shape_similarity * 0.3)
                
                if total_similarity >= threshold:
                    return dataset_name
        
        return None
    
    def clear_history(self):
        """
        Efface tout l'historique d'analyse
        """
        # Créer un nouvel index vide
        self.index = {
            "analyses": [],
            "datasets": {},
            "last_updated": datetime.datetime.now().isoformat()
        }
        self._save_index()
        
        # Supprimer tous les fichiers d'analyse
        for filename in os.listdir(self.storage_dir):
            if filename.startswith("analysis_") and filename.endswith(".json"):
                try:
                    os.remove(os.path.join(self.storage_dir, filename))
                except Exception as e:
                    print(f"Erreur lors de la suppression du fichier {filename}: {e}")