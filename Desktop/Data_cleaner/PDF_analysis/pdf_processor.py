import os
import tempfile
import pandas as pd
import numpy as np
import logging
import fitz  # PyMuPDF
import re
import hashlib
from typing import Dict, List, Any, Tuple, Optional
from io import BytesIO

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Classe qui gère l'extraction, l'analyse et la transformation de données à partir de fichiers PDF
    """
    
    def __init__(self, model_name="mistral:latest", context_size=4096):
        """
        Initialise le processeur PDF avec un modèle LLM
        
        Args:
            model_name: Nom du modèle Ollama à utiliser
            context_size: Taille du contexte pour le modèle
        """
        self.model_name = model_name
        self.context_size = context_size
        self.logger = logging.getLogger(__name__)
        
        # Tenter une connexion à Ollama
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                if self.model_name in model_names:
                    self.logger.info(f"Modèle Ollama disponible: {self.model_name}")
                else:
                    self.logger.warning(f"Modèle {self.model_name} non trouvé dans Ollama. Modèles disponibles: {model_names}")
            else:
                self.logger.error("Impossible de se connecter à Ollama API")
        except Exception as e:
            self.logger.error(f"Erreur de connexion à Ollama: {e}")
    
    def extract_text_from_pdf(self, pdf_file) -> Tuple[str, Dict[str, Any]]:
        """
        Extrait le texte et les métadonnées d'un fichier PDF
        
        Args:
            pdf_file: Fichier PDF (BytesIO ou chemin vers un fichier)
            
        Returns:
            Tuple[str, Dict]: Texte extrait et métadonnées du PDF
        """
        try:
            # Gérer les cas où pdf_file est un BytesIO ou un chemin
            if isinstance(pdf_file, BytesIO):
                doc = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")
            else:
                doc = fitz.open(pdf_file)
            
            # Extraire les métadonnées
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "page_count": len(doc),
                "file_size": len(pdf_file.getvalue()) if isinstance(pdf_file, BytesIO) else os.path.getsize(pdf_file),
            }
            
            # Extraire le texte page par page
            full_text = []
            for page_num, page in enumerate(doc):
                text = page.get_text()
                full_text.append(text)
                
                # Extraire les tableaux détectés (approximation simplifiée)
                tables = self._detect_tables(page)
                if tables:
                    metadata[f"tables_page_{page_num+1}"] = len(tables)
            
            # Assembler le texte complet
            complete_text = "\n\n".join(full_text)
            
            # Fermer le document
            doc.close()
            
            # Ajouter des statistiques au texte
            metadata["word_count"] = len(re.findall(r'\b\w+\b', complete_text))
            metadata["character_count"] = len(complete_text)
            
            return complete_text, metadata
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction du texte PDF: {e}")
            return "", {"error": str(e)}
    
    def _detect_tables(self, page) -> List[Dict]:
        """
        Tente de détecter des tableaux dans une page
        Implémentation simplifiée - pourrait être améliorée avec des bibliothèques spécialisées
        
        Args:
            page: Objet page PyMuPDF
            
        Returns:
            List[Dict]: Liste des tableaux détectés
        """
        # Cette implémentation est une approximation basique
        # Pour une détection robuste, on pourrait utiliser des bibliothèques comme camelot-py
        tables = []
        
        # Rechercher les lignes horizontales et verticales qui pourraient être des bordures de tableaux
        rect_areas = []
        for rect in page.get_drawings():
            if rect["type"] == "r":  # Rectangle
                rect_areas.append(rect["rect"])
        
        # Détecter des alignements réguliers de texte (indice d'un tableau)
        blocks = page.get_text("blocks")
        x_positions = {}
        
        for block in blocks:
            x = round(block[0])  # Position X
            if x not in x_positions:
                x_positions[x] = 0
            x_positions[x] += 1
        
        # Si plusieurs blocs sont alignés verticalement, c'est un indice de tableau
        aligned_columns = [x for x, count in x_positions.items() if count >= 3]
        
        if len(aligned_columns) >= 3 or len(rect_areas) > 0:
            # Probable présence d'un tableau
            tables.append({"type": "detected_table", "columns": len(aligned_columns)})
        
        return tables
    
    def extract_tables_from_pdf(self, pdf_file) -> List[pd.DataFrame]:
        """
        Tente d'extraire des tableaux structurés d'un PDF
        
        Args:
            pdf_file: Fichier PDF
            
        Returns:
            List[pd.DataFrame]: Liste des tableaux extraits en tant que DataFrames
        """
        # Remarque: Une extraction robuste de tableaux nécessite généralement des bibliothèques
        # spécialisées comme tabula-py ou camelot-py. Cette implémentation est simplifiée.
        try:
            import tabula
            
            # Gérer les cas où pdf_file est un BytesIO ou un chemin
            if isinstance(pdf_file, BytesIO):
                # Enregistrer temporairement le PDF
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                    tmp.write(pdf_file.getvalue())
                    tmp_path = tmp.name
                
                tables = tabula.read_pdf(tmp_path, pages='all', multiple_tables=True)
                os.unlink(tmp_path)  # Supprimer le fichier temporaire
            else:
                tables = tabula.read_pdf(pdf_file, pages='all', multiple_tables=True)
            
            return tables
            
        except ImportError:
            self.logger.warning("La bibliothèque tabula-py n'est pas installée. Extraction de tableaux limitée.")
            return self._fallback_table_extraction(pdf_file)
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction des tableaux: {e}")
            return []
    
    def _fallback_table_extraction(self, pdf_file) -> List[pd.DataFrame]:
        """
        Méthode alternative d'extraction de tableaux basée sur des heuristiques
        
        Args:
            pdf_file: Fichier PDF
            
        Returns:
            List[pd.DataFrame]: Liste des tableaux extraits (approximation)
        """
        # Cette méthode est une solution de repli très simplifiée
        try:
            # Ouvrir le PDF
            if isinstance(pdf_file, BytesIO):
                doc = fitz.open(stream=pdf_file.getvalue(), filetype="pdf")
            else:
                doc = fitz.open(pdf_file)
            
            tables = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extraire les blocs de texte
                blocks = page.get_text("blocks")
                
                # Identifier les blocs qui pourraient être des parties de tableaux
                # (recherche de blocs alignés avec des caractères de séparation comme | ou ,)
                candidates = []
                for block in blocks:
                    text = block[4]
                    if ('|' in text or '\t' in text or ',' in text) and len(text.split()) > 3:
                        candidates.append(text)
                
                if candidates:
                    # Traiter les candidats comme des tableaux potentiels
                    for candidate in candidates:
                        # Essayer de séparer par différents délimiteurs
                        if '|' in candidate:
                            lines = candidate.strip().split('\n')
                            data = [line.split('|') for line in lines if line.strip()]
                        elif '\t' in candidate:
                            lines = candidate.strip().split('\n')
                            data = [line.split('\t') for line in lines if line.strip()]
                        elif ',' in candidate:
                            lines = candidate.strip().split('\n')
                            data = [line.split(',') for line in lines if line.strip()]
                        else:
                            continue
                        
                        # Créer un DataFrame si les données sont cohérentes
                        if data and all(len(row) == len(data[0]) for row in data):
                            if len(data) > 1 and len(data[0]) > 1:
                                headers = data[0]
                                df = pd.DataFrame(data[1:], columns=headers)
                                tables.append(df)
            
            doc.close()
            return tables
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction de secours des tableaux: {e}")
            return []
    
    def generate_with_ollama(self, prompt, max_tokens=800, temperature=0.3):
        """
        Génère une réponse avec l'API Ollama
        
        Args:
            prompt: Texte du prompt
            max_tokens: Nombre maximum de tokens à générer
            temperature: Température pour la génération
            
        Returns:
            dict: Réponse formatée
        """
        try:
            import requests
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                    "top_k": 40,
                    "frequency_penalty": 1.0,
                    "presence_penalty": 0.6 
                }
            }

            response = requests.post("http://localhost:11434/api/generate", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return {"choices": [{"text": result.get("response", "")}]}
            else:
                self.logger.error(f"Erreur Ollama: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération Ollama: {e}")
            return None
    
    def analyze_pdf_content(self, text, metadata, context=None) -> Dict[str, Any]:
        """
        Analyse le contenu extrait d'un PDF
        
        Args:
            text: Texte extrait du PDF
            metadata: Métadonnées du PDF
            context: Contexte additionnel pour l'analyse
            
        Returns:
            Dict: Résultats de l'analyse
        """
        # Préparation des données pour l'analyse
        # Prendre des échantillons de différentes parties du document
        text_length = len(text)
        sample_size = 3000
        begin_sample = text[:sample_size] if text_length > sample_size else text
        middle_sample = text[text_length//2-sample_size//2:text_length//2+sample_size//2] if text_length > sample_size else ""
        end_sample = text[-sample_size:] if text_length > sample_size else ""

        text_sample = f"DÉBUT: {begin_sample}\n\nMILIEU: {middle_sample}\n\nFIN: {end_sample}"
        word_count = metadata.get("word_count", 0)
        
        # Construction du système de prompt
        system_prompt = """Tu es un expert en analyse des données.
Ton rôle est d'extraire des insights pertinents et de résumer efficacement le contenu d'un document PDF ou d'un dataset.

RÈGLE IMPORTANTE: Pour chaque affirmation importante, CITE EXPLICITEMENT ta source d'information 
(soit les données actuelles, soit une analyse historique spécifique).
Si une information ne provient pas de ces sources, précise-le clairement.

Structure ton analyse en 3 parties:
1. RÉSUMÉ PRINCIPAL (5-7 lignes capturant l'essence du document)
2. THÈMES CLÉS (liste des 3-5 thèmes principaux identifiés)
3. INSIGHTS (2-3 observations importantes ou conclusions)"""

        # Construction du prompt utilisateur
        user_prompt = f"""Analyse ce document PDF avec les caractéristiques suivantes:

Titre: {metadata.get('title', 'Non spécifié')}
Auteur: {metadata.get('author', 'Non spécifié')}
Nombre de pages: {metadata.get('page_count', 0)}
Nombre de mots: {word_count}

EXTRAIT DU TEXTE:
{text_sample}

"""

        # Ajout du contexte utilisateur si disponible
        if context:
            if "CONTEXTE D'ANALYSES PRÉCÉDENTES" in context:
                # Le contexte contient à la fois des instructions utilisateur et des analyses historiques
                parts = context.split("CONTEXTE D'ANALYSES PRÉCÉDENTES", 1)
                user_instructions = parts[0].strip()
                historical_context = "CONTEXTE D'ANALYSES PRÉCÉDENTES" + parts[1]
                
                user_prompt += f"""
INSTRUCTIONS PRINCIPALES (À PRIORISER):
{user_instructions}

RÉFÉRENCE HISTORIQUE (SECONDAIRE):
{historical_context}
"""
            else:
                # Seulement des instructions utilisateur
                user_prompt += f"""
INSTRUCTIONS PRINCIPALES:
{context}
"""

        # Finalisation du prompt
        user_prompt += """
En te basant principalement sur mes instructions actuelles et le contenu du document,
fournit ton analyse structurée comme demandé."""

        # Utiliser le format de système+prompt d'Ollama
        full_prompt = f"<s>[INST] {system_prompt} [/INST]</s>\n\n<s>[INST] {user_prompt} [/INST]"
        
        # Appel à Ollama
        response = self.generate_with_ollama(
            full_prompt,
            max_tokens=1000,
            temperature=0.3
        )
        
        if response and "choices" in response and response["choices"][0]["text"]:
            analysis = response["choices"][0]["text"].strip()
            
            # Extraction structurée des parties de l'analyse
            result = {
                "summary": "",
                "key_themes": [],
                "insights": [],
                "full_analysis": analysis
            }
            
            # Tentative d'extraction de la structure (si présente)
            summary_match = re.search(r"RÉSUMÉ PRINCIPAL[:\s]+(.*?)(?=THÈMES CLÉS|\n\n|$)", analysis, re.DOTALL)
            if summary_match:
                result["summary"] = summary_match.group(1).strip()
            
            themes_match = re.search(r"THÈMES CLÉS[:\s]+(.*?)(?=INSIGHTS|\n\n|$)", analysis, re.DOTALL)
            if themes_match:
                themes_text = themes_match.group(1).strip()
                # Extraire les thèmes individuels (numérotés ou avec puces)
                themes = re.findall(r"(?:^|\n)[•\-*\d.]+\s*(.*?)(?=\n[•\-*\d.]|\n\n|$)", themes_text, re.DOTALL)
                if themes:
                    result["key_themes"] = [theme.strip() for theme in themes]
                else:
                    # Essayer une approche plus simple - lignes individuelles
                    result["key_themes"] = [line.strip() for line in themes_text.split('\n') if line.strip()]
            
            insights_match = re.search(r"INSIGHTS[:\s]+(.*?)(?=\n\n|$)", analysis, re.DOTALL)
            if insights_match:
                insights_text = insights_match.group(1).strip()
                insights = re.findall(r"(?:^|\n)[•\-*\d.]+\s*(.*?)(?=\n[•\-*\d.]|\n\n|$)", insights_text, re.DOTALL)
                if insights:
                    result["insights"] = [insight.strip() for insight in insights]
                else:
                    result["insights"] = [line.strip() for line in insights_text.split('\n') if line.strip()]
            
            return result
        else:
            self.logger.error("Réponse vide ou incorrecte d'Ollama")
            return {
                "error": "Analyse échouée",
                "full_analysis": "Impossible de générer une analyse pour ce document."
            }
    
    def process_pdf(self, pdf_file, context=None) -> Dict[str, Any]:
        """
        Traite un fichier PDF complet (extraction + analyse)
        
        Args:
            pdf_file: Fichier PDF (BytesIO ou chemin)
            context: Contexte additionnel pour l'analyse
            
        Returns:
            Dict: Résultats complets du traitement
        """
        result = {
            "success": False,
            "metadata": {},
            "analysis": {},
            "tables": []
        }
        
        try:
            # Extraire le texte et les métadonnées
            self.logger.info("Extraction du texte du PDF...")
            text, metadata = self.extract_text_from_pdf(pdf_file)
            
            if not text:
                return {
                    "success": False,
                    "error": "Extraction du texte échouée"
                }
            
            # Générer un identifiant unique pour ce PDF
            pdf_hash = self._generate_pdf_hash(text, metadata)
            
            # Extraire les tableaux si présents
            try:
                self.logger.info("Tentative d'extraction des tableaux...")
                tables = self.extract_tables_from_pdf(pdf_file)
                if tables:
                    result["tables"] = [table.to_dict() for table in tables]
                    metadata["table_count"] = len(tables)
            except Exception as e:
                self.logger.warning(f"Erreur lors de l'extraction des tableaux: {e}")
                metadata["table_extraction_error"] = str(e)
            
            # Analyser le contenu
            self.logger.info("Analyse du contenu du PDF...")
            analysis = self.analyze_pdf_content(text, metadata, context)
            
            # Assemblage des résultats
            result = {
                "success": True,
                "pdf_id": pdf_hash,
                "text_length": len(text),
                "metadata": metadata,
                "analysis": analysis,
                "tables": result["tables"]
            }
            
            self.logger.info(f"Traitement du PDF terminé avec succès (id: {pdf_hash})")
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement du PDF: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_pdf_hash(self, text, metadata) -> str:
        """
        Génère un identifiant unique pour un PDF basé sur son contenu
        
        Args:
            text: Texte extrait du PDF
            metadata: Métadonnées du PDF
            
        Returns:
            str: Hash unique du PDF
        """
        # Construire une chaîne représentative du contenu
        content_str = f"{metadata.get('title', '')}_{metadata.get('author', '')}_{metadata.get('page_count', 0)}_{len(text)}"
        content_str += text[:1000] if text else ""  # Ajouter un échantillon du texte
        
        # Générer le hash
        return hashlib.md5(content_str.encode('utf-8')).hexdigest()
