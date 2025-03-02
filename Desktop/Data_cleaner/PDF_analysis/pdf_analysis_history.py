import os
import json
import datetime
import hashlib
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFAnalysisHistory:
    """
    Extension pour gérer l'historique des analyses de fichiers PDF
    Cette classe peut être utilisée séparément ou intégrée à AnalysisHistory
    """
    
    def __init__(self, storage_dir: str = "analysis_history/pdf"):
        """
        Initialise le gestionnaire d'historique d'analyse PDF
        
        Args:
            storage_dir: Répertoire où seront stockés les fichiers d'historique PDF
        """
        self.storage_dir = storage_dir
        logger.info(f"Initialisation de l'historique d'analyse PDF, répertoire: {storage_dir}")
        
        # Créer le répertoire de stockage s'il n'existe pas
        if not os.path.exists(storage_dir):
            try:
                os.makedirs(storage_dir)
                logger.info(f"Répertoire PDF créé: {storage_dir}")
            except Exception as e:
                logger.error(f"Erreur lors de la création du répertoire PDF: {e}")
        else:
            logger.info(f"Répertoire PDF existant: {storage_dir}")
        
        # Fichier qui contient l'index de tous les documents PDF analysés
        self.index_file = os.path.join(storage_dir, "pdf_analysis_index.json")
        
        # Charger l'index existant ou en créer un nouveau
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.index = json.load(f)
                logger.info(f"Index PDF chargé: {len(self.index.get('analyses', []))} analyses trouvées")
            except Exception as e:
                logger.error(f"Erreur lors du chargement de l'index PDF: {e}")
                self.index = self._create_empty_index()
        else:
            self.index = self._create_empty_index()
            self._save_index()
    
    def _create_empty_index(self) -> Dict[str, Any]:
        """Crée un index vide avec la structure appropriée"""
        return {
            "analyses": [],  # Liste des analyses effectuées
            "documents": {},  # Dictionnaire des documents analysés (par pdf_id)
            "last_updated": datetime.datetime.now().isoformat()
        }
    
    def _save_index(self) -> bool:
        """Sauvegarde l'index dans le fichier"""
        try:
            self.index["last_updated"] = datetime.datetime.now().isoformat()
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, ensure_ascii=False, indent=2)
            logger.info(f"Index PDF sauvegardé dans: {self.index_file}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'index PDF: {e}")
            return False
    
    def add_pdf_analysis(self, 
                      pdf_id: str, 
                      pdf_name: str, 
                      analysis_result: Dict[str, Any], 
                      metadata: Dict[str, Any] = None) -> str:
        """
        Ajoute une nouvelle analyse PDF à l'historique
        
        Args:
            pdf_id: Identifiant unique du PDF (hash)
            pdf_name: Nom du fichier PDF
            analysis_result: Résultats de l'analyse
            metadata: Métadonnées du document
                
        Returns:
            str: ID de l'analyse créée ou None en cas d'échec
        """
        try:
            # Générer un identifiant unique pour cette analyse
            timestamp = datetime.datetime.now().isoformat()
            timestamp_safe = timestamp.replace(':', '-').replace('.', '_')  # Pour les noms de fichiers
            analysis_id = f"pdf_analysis_{len(self.index['analyses'])+1}_{timestamp_safe}"
            
            # Préparer les métadonnées du document
            if not metadata:
                metadata = {}
            
            # S'assurer que les métadonnées sont sérialisables
            def sanitize_for_json(obj):
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                elif isinstance(obj, (list, tuple)):
                    return [sanitize_for_json(item) for item in obj]
                elif isinstance(obj, dict):
                    return {str(k): sanitize_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return sanitize_for_json(obj.tolist())
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                else:
                    return str(obj)
            
            metadata = sanitize_for_json(metadata)
            
            # Structure de l'analyse
            analysis_entry = {
                "id": analysis_id,
                "pdf_id": pdf_id,
                "pdf_name": pdf_name,
                "timestamp": timestamp,
                "metadata": metadata,
                "analysis": sanitize_for_json(analysis_result)
            }
            
            # Sauvegarder l'analyse dans un fichier séparé
            analysis_file = os.path.join(self.storage_dir, f"{analysis_id}.json")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_entry, f, ensure_ascii=False, indent=2)
            
            # Vérifier que le fichier a bien été créé
            if not os.path.exists(analysis_file):
                logger.error(f"Erreur: Le fichier d'analyse PDF n'a pas été créé: {analysis_file}")
                return None
            
            # Ajouter à l'index
            self.index["analyses"].append({
                "id": analysis_id,
                "pdf_id": pdf_id,
                "pdf_name": pdf_name,
                "timestamp": timestamp,
                "summary": analysis_result.get("summary", "Pas de résumé disponible")[:100] + "..." 
                            if analysis_result.get("summary") and len(analysis_result.get("summary")) > 100 
                            else analysis_result.get("summary", "Pas de résumé disponible")
            })
            
            # Mettre à jour les informations du document dans l'index
            if pdf_id not in self.index["documents"]:
                self.index["documents"][pdf_id] = {
                    "name": pdf_name,
                    "first_analysis": timestamp,
                    "last_analysis": timestamp,
                    "analyses_count": 1,
                    "analyses": [analysis_id],
                    "metadata": {
                        "title": metadata.get("title", ""),
                        "author": metadata.get("author", ""),
                        "page_count": metadata.get("page_count", 0)
                    }
                }
            else:
                self.index["documents"][pdf_id]["last_analysis"] = timestamp
                self.index["documents"][pdf_id]["analyses_count"] += 1
                self.index["documents"][pdf_id]["analyses"].append(analysis_id)
                # Mettre à jour le nom si nécessaire
                if pdf_name != self.index["documents"][pdf_id]["name"]:
                    self.index["documents"][pdf_id]["name"] = pdf_name
            
            # Sauvegarder l'index mis à jour
            self._save_index()
            
            return analysis_id
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout de l'analyse PDF: {e}")
            return None
    
    def get_pdf_analysis(self, analysis_id: str) -> Dict[str, Any]:
        """
        Récupère une analyse PDF spécifique par son ID
        
        Args:
            analysis_id: Identifiant de l'analyse
                
        Returns:
            Dict: Contenu de l'analyse ou None si non trouvée
        """
        analysis_file = os.path.join(self.storage_dir, f"{analysis_id}.json")
        
        if not os.path.exists(analysis_file):
            logger.warning(f"Fichier d'analyse PDF non trouvé: {analysis_file}")
            return None
        
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erreur lors de la lecture de l'analyse PDF {analysis_id}: {e}")
            return None
    
    def get_pdf_analyses(self, pdf_id: str) -> List[Dict[str, Any]]:
        """
        Récupère toutes les analyses associées à un PDF
        
        Args:
            pdf_id: Identifiant du PDF
            
        Returns:
            List: Liste des analyses complètes
        """
        if pdf_id not in self.index["documents"]:
            logger.info(f"Aucune analyse trouvée pour le PDF {pdf_id}")
            return []
        
        analyses = []
        for analysis_id in self.index["documents"][pdf_id]["analyses"]:
            analysis = self.get_pdf_analysis(analysis_id)
            if analysis:
                analyses.append(analysis)
        
        # Trier par date (plus récent en premier)
        analyses.sort(key=lambda x: x["timestamp"], reverse=True)
        return analyses
    
    def get_recent_pdf_analyses(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Récupère les analyses PDF les plus récentes
        
        Args:
            limit: Nombre maximum d'analyses à récupérer
            
        Returns:
            List: Liste des analyses récentes
        """
        # Vérifier si l'index existe et contient des analyses
        if not self.index or "analyses" not in self.index or not self.index["analyses"]:
            logger.info("Aucune analyse PDF dans l'index")
            return []
        
        # Trier l'index des analyses par date
        sorted_analyses = sorted(self.index["analyses"], key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Récupérer les analyses complètes
        recent_analyses = []
        for analysis_info in sorted_analyses[:limit]:
            if "id" not in analysis_info:
                continue
                
            analysis_id = analysis_info["id"]
            analysis = self.get_pdf_analysis(analysis_id)
            
            if analysis:
                recent_analyses.append(analysis)
        
        return recent_analyses
    
    def generate_pdf_context(self, pdf_id=None, max_analyses=3):
        """
        Génère un contexte textuel à partir des analyses PDF précédentes.
        
        Args:
            pdf_id (str, optional): Identifiant du PDF pour lequel générer le contexte.
            max_analyses (int, optional): Nombre maximum d'analyses à inclure.
        
        Returns:
            str: Le contexte formaté contenant les analyses précédentes.
        """
        context_parts = []
        
        # En-tête du contexte
        context_parts.append("CONTEXTE D'ANALYSES PDF PRÉCÉDENTES (RÉFÉRENCE SECONDAIRE):")
        context_parts.append("========================================================")
        context_parts.append("")
        context_parts.append("NOTE: Ce contexte historique est fourni uniquement comme référence et ne doit pas")
        context_parts.append("remplacer les instructions actuelles de l'utilisateur, qui ont toujours priorité.")
        context_parts.append("")
        
        # Analyses à inclure
        analyses_to_include = []
        
        if pdf_id and pdf_id in self.index["documents"]:
            # Si un PDF spécifique est demandé, récupérer ses analyses
            analysis_ids = self.index["documents"][pdf_id]["analyses"]
            for analysis_id in analysis_ids[-max_analyses:]:  # Prendre les plus récentes
                analysis = self.get_pdf_analysis(analysis_id)
                if analysis:
                    analyses_to_include.append(analysis)
        else:
            # Sinon, prendre les analyses les plus récentes
            analyses_to_include = self.get_recent_pdf_analyses(limit=max_analyses)
        
        # Formater chaque analyse
        for i, analysis in enumerate(analyses_to_include, 1):
            context_parts.append(f"ANALYSE PDF HISTORIQUE {i}: {analysis.get('pdf_name', 'Sans nom')}")
            context_parts.append(f"Date: {analysis.get('timestamp', 'Date inconnue')}")
            
            # Ajouter les métadonnées
            if "metadata" in analysis:
                metadata = analysis["metadata"]
                context_parts.append(f"Titre: {metadata.get('title', 'Document sans titre')}")
                context_parts.append(f"Pages: {metadata.get('page_count', 'N/A')}")
                
                if "author" in metadata and metadata["author"]:
                    context_parts.append(f"Auteur: {metadata['author']}")
                
                if "word_count" in metadata:
                    context_parts.append(f"Nombre de mots: {metadata['word_count']}")
            
            # Ajouter le contenu de l'analyse
            if "analysis" in analysis:
                analysis_content = analysis["analysis"]
                
                # Ajouter le résumé
                if "summary" in analysis_content and analysis_content["summary"]:
                    context_parts.append("\nRÉSUMÉ:")
                    context_parts.append(analysis_content["summary"])
                
                # Ajouter les thèmes clés
                if "key_themes" in analysis_content and analysis_content["key_themes"]:
                    context_parts.append("\nTHÈMES CLÉS:")
                    for theme in analysis_content["key_themes"]:
                        context_parts.append(f"- {theme}")
                
                # Ajouter les insights
                if "insights" in analysis_content and analysis_content["insights"]:
                    context_parts.append("\nINSIGHTS:")
                    for insight in analysis_content["insights"]:
                        context_parts.append(f"- {insight}")
                
                # Ajouter une version condensée de l'analyse complète si disponible
                if "full_analysis" in analysis_content and analysis_content["full_analysis"]:
                    # Limiter la taille pour ne pas surcharger le contexte
                    full_text = analysis_content["full_analysis"]
                    max_chars = 1000  # Ajuster selon vos besoins
                    
                    if len(full_text) > max_chars:
                        context_parts.append("\nEXTRAIT DE L'ANALYSE COMPLÈTE:")
                        context_parts.append(full_text[:max_chars] + "... [tronqué]")
                    else:
                        context_parts.append("\nANALYSE COMPLÈTE:")
                        context_parts.append(full_text)
            
            # Séparateur entre les analyses
            context_parts.append("----------------------------------------")
        
        # Si aucune analyse n'a été trouvée
        if not analyses_to_include:
            context_parts.append("Aucune analyse PDF disponible dans l'historique.")
        
        # Joindre toutes les parties avec des sauts de ligne
        return "\n".join(context_parts)
    
    def find_similar_pdf(self, pdf_metadata: Dict[str, Any], threshold: float = 0.7) -> Optional[str]:
        """
        Tente d'identifier un PDF déjà analysé qui pourrait être similaire
        à celui actuellement en cours d'analyse
        
        Args:
            pdf_metadata: Métadonnées du PDF à comparer
            threshold: Seuil de similarité
            
        Returns:
            str: ID du PDF similaire ou None si aucun trouvé
        """
        if not pdf_metadata or not self.index["documents"]:
            return None
        
        # Facteurs de similarité
        title = pdf_metadata.get("title", "").lower()
        author = pdf_metadata.get("author", "").lower()
        page_count = pdf_metadata.get("page_count", 0)
        
        # Si le titre ou l'auteur est vide, la détection est moins fiable
        if not title and not author:
            return None
        
        best_match = None
        best_score = 0.0
        
        for pdf_id, doc_info in self.index["documents"].items():
            doc_meta = doc_info.get("metadata", {})
            doc_title = doc_meta.get("title", "").lower()
            doc_author = doc_meta.get("author", "").lower()
            doc_pages = doc_meta.get("page_count", 0)
            
            # Calcul de similarité
            score = 0.0
            
            # Similarité du titre (poids important)
            if title and doc_title:
                # Similarité simple basée sur les mots communs
                title_words = set(title.split())
                doc_title_words = set(doc_title.split())
                if title_words and doc_title_words:
                    common_words = title_words.intersection(doc_title_words)
                    title_sim = len(common_words) / max(len(title_words), len(doc_title_words))
                    score += title_sim * 0.6  # Le titre compte pour 60%
            
            # Similarité de l'auteur
            if author and doc_author:
                if author == doc_author:
                    score += 0.3  # L'auteur compte pour 30%
            
            # Nombre de pages (faible poids)
            if page_count > 0 and doc_pages > 0:
                page_ratio = min(page_count, doc_pages) / max(page_count, doc_pages)
                score += page_ratio * 0.1  # Le nombre de pages compte pour 10%
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = pdf_id
        
        return best_match
    
    def clear_pdf_history(self):
        """
        Efface tout l'historique d'analyse PDF
        """
        # Créer un nouvel index vide
        self.index = self._create_empty_index()
        self._save_index()
        
        # Supprimer tous les fichiers d'analyse
        for filename in os.listdir(self.storage_dir):
            if filename.startswith("pdf_analysis_") and filename.endswith(".json"):
                try:
                    os.remove(os.path.join(self.storage_dir, filename))
                except Exception as e:
                    logger.error(f"Erreur lors de la suppression du fichier {filename}: {e}")
