import streamlit as st
import pandas as pd
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import hashlib
import io
import base64
from data_transformer_ollama2 import DataTransformer
from analysis_history2 import AnalysisHistory
from pdf_processor import PDFProcessor  # Notre nouveau module
from pdf_analysis_history import PDFAnalysisHistory  # Extension pour l'historique PDF
import logging
from add_graphs import create_charts


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Data Transformation avec PDF & Mémoire",
    page_icon="C:\\Users\\Francis\\Desktop\\Data_cleaner\\PDF_analysis\\orbit-icon-13.jpg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialiser l'historique des analyses CSV
@st.cache_resource
def get_analysis_history():
    """Create or retrieve the analysis history manager"""
    return AnalysisHistory(storage_dir="./analysis_history")

# Initialiser l'historique des PDF
@st.cache_resource
def get_pdf_history():
    """Create or retrieve the PDF analysis history manager"""
    return PDFAnalysisHistory(storage_dir="./analysis_history/pdf")

# Fonction pour calculer un hash de dataframe pour identifier les datasets
def get_dataframe_hash(df):
    """Generate a hash for a dataframe to identify it"""
    column_str = "_".join(df.columns)
    shape_str = f"{df.shape[0]}_{df.shape[1]}"
    sample_data = df.head(5).to_json()
    
    hash_input = column_str + shape_str + sample_data
    return hashlib.md5(hash_input.encode()).hexdigest()

def create_download_link(df, filename):
    """Create a download link for dataframe"""
    csv = df.to_csv(index=False)
    return f'<a href="data:file/csv;base64,{base64.b64encode(csv.encode()).decode()}" download="{filename}">Télécharger {filename}</a>'

def create_text_download_link(text, filename):
    """Crée un lien de téléchargement pour un texte formaté"""
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Télécharger {filename}</a>'

def plot_correlations(df):
    """Create correlation plot for numeric columns"""
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns_plot = sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        return plt
    return None

def plot_missing_values(df):
    """Create missing values visualization"""
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing) > 0:
        plt.figure(figsize=(10, 6))
        missing.plot(kind='bar')
        plt.title('Missing Values by Column')
        plt.ylabel('Count')
        plt.xlabel('Columns')
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt
    return None

def get_dataset_info(df):
    """Extract basic information about the dataset"""
    dataset_info = {
        "dimensions": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "columns_types": {
            "numeric": list(df.select_dtypes(include=['number']).columns),
            "categorical": list(df.select_dtypes(exclude=['number']).columns)
        },
        "missing_values": df.isna().sum().sum(),
        "duplicate_rows": df.duplicated().sum()
    }
    return dataset_info

def get_combined_context(dataset_context=None, pdf_context=None, max_analyses=3):
    """Combine les contextes d'analyse de datasets et de PDF"""
    history = get_analysis_history()
    pdf_history = get_pdf_history()
    
    logger.info(f"Chemin d'historique CSV: {history.storage_dir}")
    logger.info(f"Chemin d'historique PDF: {pdf_history.storage_dir}")
    
    # Récupérer le contexte des datasets si demandé
    csv_context = ""
    if dataset_context is not None:
        csv_context = history.generate_context(
            dataset_name=dataset_context, 
            max_analyses=max_analyses
        )
        logger.info(f"Contexte CSV généré: {len(csv_context)} caractères")
    
    # Récupérer le contexte des PDF
    pdf_history_context = ""
    # Même si pdf_context est None, récupérons les analyses PDF récentes
    pdf_analyses = pdf_history.get_recent_pdf_analyses(limit=max_analyses)
    logger.info(f"Nombre d'analyses PDF récentes trouvées: {len(pdf_analyses)}")
    
    if pdf_analyses:
        if pdf_context:
            # Si un PDF spécifique est demandé
            pdf_history_context = pdf_history.generate_pdf_context(
                pdf_id=pdf_context,
                max_analyses=max_analyses
            )
        else:
            # Sinon utiliser les analyses récentes
            pdf_history_context = pdf_history.generate_pdf_context(
                max_analyses=max_analyses
            )
        logger.info(f"Contexte PDF généré: {len(pdf_history_context)} caractères")
    else:
        logger.info("Aucune analyse PDF disponible pour générer un contexte")
    
    # Combiner les contextes
    combined_context = ""
    if csv_context and pdf_history_context:
        combined_context = f"{csv_context}\n\n{pdf_history_context}"
        logger.info(f"Contexte combiné (CSV+PDF): {len(combined_context)} caractères")
    else:
        combined_context = csv_context or pdf_history_context
        if csv_context:
            logger.info(f"Contexte CSV uniquement: {len(csv_context)} caractères")
        elif pdf_history_context:
            logger.info(f"Contexte PDF uniquement: {len(pdf_history_context)} caractères")
        else:
            logger.info("Aucun contexte généré")
    
    return combined_context

def process_csv_data():
    """Fonction pour traiter les données CSV"""
    # Initialiser l'historique des analyses
    history = get_analysis_history()
    
    st.header("📈 Analyse de données CSV")
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv", key="csv_uploader")
    
    if uploaded_file is not None:
        # Load the data
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Données chargées avec succès : {df.shape[0]} lignes, {df.shape[1]} colonnes")
            
            # Generate a unique identifier for this dataset
            df_hash = get_dataframe_hash(df)
            dataset_name = uploaded_file.name
            
            # Show data preview
            with st.expander("Aperçu des données", expanded=True):
                st.dataframe(df.head())
            
            # Après l'expander "Aperçu des données"
            with st.expander("Visualisations interactives", expanded=False):
                # Enregistrer le dataframe dans session_state pour create_charts
                st.session_state["df"] = df
                # Appeler la fonction de création de graphiques
                create_charts()
                
                # Show basic statistics
                st.subheader("Résumé des données")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Colonnes numériques :", len(df.select_dtypes(include=['number']).columns))
                    st.write("Colonnes catégorielles :", len(df.select_dtypes(exclude=['number']).columns))
                with col2:
                    st.write("Valeurs manquantes :", df.isna().sum().sum())
                    st.write("Lignes dupliquées :", df.duplicated().sum())
                    
                # Plot missing values
                missing_plot = plot_missing_values(df)
                if missing_plot:
                    st.subheader("Distribution des valeurs manquantes")
                    st.pyplot(missing_plot)
            
            # Try to find similar datasets in history
            similar_dataset = None
            if st.session_state.get("use_history", True):
                similar_dataset = history.find_dataset_by_similarity(df)
                if similar_dataset:
                    st.info(f"📌 Ce jeu de données semble similaire à '{similar_dataset}' que vous avez déjà analysé. Les analyses précédentes seront prises en compte, mais vos instructions actuelles auront priorité.")
            
            # Process the data
            if st.button("Transformer les données", key="transform_csv"):
                st.header("Résultats de la transformation")
                
                # Initialize progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Initialisation du transformateur...")
                
                # Create a temporary file for the transformer
                with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_input:
                    df.to_csv(tmp_input.name, index=False)
                    tmp_input_path = tmp_input.name
                
                try:
                    # Initialize transformer
                    status_text.text("Initialisation du transformateur Mistral...")
                    progress_bar.progress(10)
                    
                    # Prepare context from history if enabled
                    analysis_context = ""
                    
                    if st.session_state.get("use_history", True):
                        status_text.text("Récupération de l'historique d'analyse...")
                        progress_bar.progress(15)
                        
                        # Get historical context
                        analysis_context = get_combined_context(
                            dataset_context=similar_dataset, 
                            pdf_context=None,  # Nous pourrions ici sélectionner des PDFs pertinents si besoin
                            max_analyses=st.session_state.get("max_history", 3)
                        )
                        
                        # Afficher le contexte pour vérification (debug)
                        with st.expander("Contexte d'analyse combiné (debug)", expanded=False):
                            st.code(analysis_context)

                    # Build the full context with clear priority for user instructions
                    user_context = st.session_state.get("user_context", "")
                    if user_context and analysis_context:
                        # Emphasize user instructions based on history_weight
                        if st.session_state.get("history_weight", 5) <= 3:  # Low weight to history
                            full_context = f"""INSTRUCTIONS UTILISATEUR (PRIORITÉ ÉLEVÉE):
{user_context}

{analysis_context}"""
                        elif st.session_state.get("history_weight", 5) >= 7:  # High weight to history
                            full_context = f"""{user_context}

{analysis_context}"""
                        else:  # Balanced approach
                            full_context = f"""INSTRUCTIONS UTILISATEUR:
{user_context}

{analysis_context}"""
                    elif user_context:
                        full_context = user_context
                    else:
                        full_context = analysis_context
                    
                    if analysis_context:
                        st.info("ℹ️ Analyses précédentes intégrées au contexte. Vos instructions actuelles restent prioritaires.")
                        
                    transformer = DataTransformer(model_name=st.session_state.get("model_name", "mistral:latest"), 
                                                context_size=st.session_state.get("context_size", 4096))
                    
                    # Transform data
                    status_text.text("Application des transformations...")
                    progress_bar.progress(30)
                    
                    transformations = None if st.session_state.get("auto_detect", True) else st.session_state.get("transformations", [])
                    df_transformed, metadata = transformer.transform(df, transformations, full_context)
                    
                    # Display results
                    status_text.text("Traitement des résultats...")
                    progress_bar.progress(70)
                    
                    # Display transformed data
                    st.subheader("Jeu de données transformé")
                    st.dataframe(df_transformed.head())
                    
                    # Display transformation details
                    st.subheader("Transformations appliquées")
                    
                    # Create columns for before/after stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Dimensions d'origine", f"{metadata['original_shape'][0]} × {metadata['original_shape'][1]}")
                    with col2:
                        st.metric("Dimensions transformées", f"{metadata['transformed_shape'][0]} × {metadata['transformed_shape'][1]}")
                    with col3:
                        missing_before = metadata['missing_values'].get('before', 0)
                        missing_after = metadata['missing_values'].get('after', 0)
                        reduction = missing_before - missing_after
                        reduction_pct = (1 - missing_after/max(1, missing_before))*100 if missing_before > 0 else 0
                        st.metric("Réduction des valeurs manquantes", f"{reduction} ({reduction_pct:.1f}%)")
                    
                    # List all applied transformations
                    if metadata.get('transformations'):
                        for i, t in enumerate(metadata['transformations'], 1):
                            st.write(f"{i}. {t['details']}")
                    else:
                        st.info("Aucune transformation n'a été appliquée.")
                    
                    # Show new columns
                    if metadata.get('new_columns'):
                        st.subheader("Nouvelles colonnes créées")
                        st.write(", ".join(metadata['new_columns']))
                    
                    # Show removed columns
                    if metadata.get('removed_columns'):
                        st.subheader("Colonnes supprimées")
                        st.write(", ".join(metadata['removed_columns']))
                    
                    # Display Mistral's analysis if available
                    analysis_text = ""
                    if "analysis" in metadata and metadata["analysis"]:
                        st.subheader("Analyse IA")
                        st.info(metadata["analysis"])
                        analysis_text = metadata["analysis"]
                    
                    # Create download link for transformed data
                    st.markdown(create_download_link(df_transformed, "donnees_transformees.csv"), unsafe_allow_html=True)
                    
                    # Plot correlations for transformed data
                    correlation_plot = plot_correlations(df_transformed)
                    if correlation_plot:
                        st.subheader("Corrélations des variables (après transformation)")
                        st.pyplot(correlation_plot)
                    
                    # Save analysis to history
                    if st.session_state.get("use_history", True) and analysis_text:
                        dataset_info = get_dataset_info(df)
                        
                        # Create a descriptive name
                        dataset_description = f"Dataset avec {df.shape[0]} lignes et {df.shape[1]} colonnes"
                        if uploaded_file.name:
                            dataset_description += f" (Fichier: {uploaded_file.name})"
                        
                        # Add analysis to history
                        analysis_id = history.add_analysis(
                            dataset_name=dataset_name,
                            dataset_description=dataset_description,
                            analysis_text=analysis_text,
                            metadata=dataset_info
                        )
                        
                        if analysis_id:
                            st.success(f"✓ Cette analyse a été sauvegardée et sera utilisée pour améliorer les analyses futures (ID: {analysis_id})")
                        else:
                            st.warning("⚠️ L'analyse n'a pas pu être sauvegardée dans l'historique.")
                    
                    # Complete the progress
                    progress_bar.progress(100)
                    status_text.text("Transformation terminée avec succès !")
                    
                except Exception as e:
                    st.error(f"Erreur pendant la transformation: {e}")
                    logger.error(f"Erreur de transformation: {e}")
                
                finally:
                    # Clean up the temporary file
                    if 'tmp_input_path' in locals() and os.path.exists(tmp_input_path):
                        os.unlink(tmp_input_path)
        
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier: {e}")
            logger.error(f"Erreur de chargement du fichier: {e}")

def process_pdf_data():
    """Fonction pour traiter les fichiers PDF"""
    # Initialiser l'historique des analyses PDF
    pdf_history = get_pdf_history()
    
    st.header("📄 Analyse de documents PDF")
    uploaded_pdf = st.file_uploader("Choisir un fichier PDF", type="pdf", key="pdf_uploader")
    
    if uploaded_pdf is not None:
        # Afficher les informations basiques sur le PDF
        pdf_size = len(uploaded_pdf.getvalue()) / 1024  # taille en KB
        st.success(f"PDF chargé avec succès: {uploaded_pdf.name} ({pdf_size:.1f} KB)")
        
        # Créer un processeur PDF
        processor = PDFProcessor(model_name=st.session_state.get("model_name", "mistral:latest"),
                              context_size=st.session_state.get("context_size", 4096))
        
        # Extraire un aperçu des métadonnées
        with st.expander("Aperçu du document", expanded=True):
            # Utiliser PyMuPDF pour récupérer les métadonnées de base
            pdf_bytes = io.BytesIO(uploaded_pdf.getvalue())
            
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=pdf_bytes.getvalue(), filetype="pdf")
                
                # Afficher les métadonnées
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Titre:", doc.metadata.get("title", "Non spécifié"))
                    st.write("Auteur:", doc.metadata.get("author", "Non spécifié"))
                    st.write("Nombre de pages:", len(doc))
                
                with col2:
                    st.write("Sujet:", doc.metadata.get("subject", "Non spécifié"))
                    st.write("Créateur:", doc.metadata.get("creator", "Non spécifié"))
                    st.write("Date de création:", doc.metadata.get("creationDate", "Non spécifiée"))
                
                # Afficher la première page comme aperçu
                if len(doc) > 0:
                    st.subheader("Aperçu du texte (première page)")
                    first_page = doc[0]
                    text_preview = first_page.get_text()[:500] + "..." if len(first_page.get_text()) > 500 else first_page.get_text()
                    st.text_area("Texte extrait", text_preview, height=200)
                
                doc.close()
            except Exception as e:
                st.warning(f"Impossible d'afficher l'aperçu du PDF: {e}")
        
        # Essayer de trouver un PDF similaire dans l'historique
        similar_pdf = None
        if st.session_state.get("use_history", True):
            try:
                import fitz
                doc = fitz.open(stream=pdf_bytes.getvalue(), filetype="pdf")
                metadata = {
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", ""),
                    "page_count": len(doc)
                }
                doc.close()
                
                similar_pdf = pdf_history.find_similar_pdf(metadata)
                if similar_pdf and similar_pdf in pdf_history.index["documents"]:
                    pdf_name = pdf_history.index["documents"][similar_pdf]["name"]
                    st.info(f"📌 Ce document PDF semble similaire à '{pdf_name}' que vous avez déjà analysé. Les analyses précédentes seront prises en compte.")
            except Exception as e:
                logger.error(f"Erreur lors de la recherche de PDF similaires: {e}")
        
        # Contexte utilisateur pour l'analyse
        user_context_pdf = st.text_area(
            "Instructions pour l'analyse du PDF (optionnel)",
            help="Fournissez des instructions spécifiques pour guider l'analyse de ce document",
            key="user_context_pdf"
        )
        
        # Bouton d'analyse du PDF
        if st.button("Analyser le PDF", key="analyze_pdf"):
            st.header("Résultats de l'analyse")
            
            # Initialize progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Initialisation de l'analyse du PDF...")
            
            try:
                # Préparer le contexte à partir de l'historique si activé
                analysis_context = ""
                
                if st.session_state.get("use_history", True):
                    status_text.text("Récupération de l'historique d'analyse PDF...")
                    progress_bar.progress(10)
                    
                    # Récupérer le contexte des analyses précédentes
                    analysis_context = get_combined_context(
                        dataset_context=None,  # Nous pourrions ici sélectionner des datasets pertinents
                        pdf_context=similar_pdf,
                        max_analyses=st.session_state.get("max_history", 3)
                    )
                    # Afficher le contexte pour vérification
                    with st.expander("Contexte d'analyse combiné (debug)", expanded=False):
                        st.code(analysis_context)
                
                # Construire le contexte complet
                if user_context_pdf and analysis_context:
                    # Mettre l'accent sur les instructions utilisateur
                    full_context = f"""INSTRUCTIONS UTILISATEUR (PRIORITÉ ÉLEVÉE):
{user_context_pdf}

{analysis_context}"""
                elif user_context_pdf:
                    full_context = user_context_pdf
                else:
                    full_context = analysis_context
                
                # Informer l'utilisateur sur l'utilisation du contexte
                if analysis_context:
                    st.info("ℹ️ Analyses précédentes intégrées au contexte. Vos instructions actuelles restent prioritaires.")
                
                # Traiter le PDF
                status_text.text("Extraction et analyse du texte du PDF...")
                progress_bar.progress(30)
                
                # Analyser le PDF (réinitialiser BytesIO pour éviter les erreurs EOF)
                pdf_bytes = io.BytesIO(uploaded_pdf.getvalue())
                pdf_result = processor.process_pdf(pdf_bytes, context=full_context)
                
                if pdf_result["success"]:
                    # Extraction réussie
                    progress_bar.progress(70)
                    status_text.text("Analyse du PDF terminée, préparation des résultats...")
                    
                    # Afficher les métadonnées extraites
                    st.subheader("Métadonnées du document")
                    metadata = pdf_result["metadata"]
                    
                    # Créer une présentation en colonnes des métadonnées
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Pages", metadata.get("page_count", "N/A"))
                        st.write("Titre:", metadata.get("title", "Non spécifié"))
                    with col2:
                        st.metric("Mots", metadata.get("word_count", "N/A"))
                        st.write("Auteur:", metadata.get("author", "Non spécifié"))
                    with col3:
                        st.metric("Caractères", metadata.get("character_count", "N/A"))
                        st.write("Sujet:", metadata.get("subject", "Non spécifié"))
                    
                    # Afficher les résultats de l'analyse
                    st.subheader("Analyse du contenu")
                    analysis = pdf_result["analysis"]
                    
                    # Résumé principal
                    if analysis.get("summary"):
                        st.markdown("#### Résumé principal")
                        st.info(analysis["summary"])
                    
                    # Thèmes clés
                    if analysis.get("key_themes"):
                        st.markdown("#### Thèmes clés")
                        for theme in analysis["key_themes"]:
                            st.markdown(f"• {theme}")
                    
                    # Insights
                    if analysis.get("insights"):
                        st.markdown("#### Insights")
                        for insight in analysis["insights"]:
                            st.markdown(f"• {insight}")
                    
                    # Tableau extrait (si disponible)
                    if pdf_result.get("tables") and len(pdf_result["tables"]) > 0:
                        st.subheader(f"Tableaux extraits ({len(pdf_result['tables'])})")
                        for i, table_dict in enumerate(pdf_result["tables"], 1):
                            try:
                                # Convertir le dictionnaire en DataFrame
                                table_df = pd.DataFrame.from_dict(table_dict)
                                st.markdown(f"**Tableau {i}:**")
                                st.dataframe(table_df)
                            except Exception as e:
                                st.warning(f"Impossible d'afficher le tableau {i}: {e}")
                    
                    # Créer un lien de téléchargement pour le résumé d'analyse
                    if analysis.get("full_analysis"):
                        full_analysis = analysis["full_analysis"]
                        formatted_analysis = f"""# Analyse du document: {uploaded_pdf.name}
Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}

## Résumé
{analysis.get("summary", "Résumé non disponible")}

## Thèmes clés
{chr(10).join("- " + theme for theme in analysis.get("key_themes", ["Thèmes non disponibles"]))}

## Insights
{chr(10).join("- " + insight for insight in analysis.get("insights", ["Insights non disponibles"]))}

## Analyse complète
{full_analysis}
"""
                        st.markdown(
                            create_text_download_link(formatted_analysis, f"analyse_{uploaded_pdf.name.replace('.pdf', '.txt')}"),
                            unsafe_allow_html=True
                        )
                    
                    # Sauvegarder l'analyse dans l'historique PDF
                    if st.session_state.get("use_history", True):
                        pdf_id = pdf_result["pdf_id"]
                        analysis_id = pdf_history.add_pdf_analysis(
                            pdf_id=pdf_id,
                            pdf_name=uploaded_pdf.name,
                            analysis_result=analysis,
                            metadata=metadata
                        )
                        
                        if analysis_id:
                            st.success(f"✓ Cette analyse PDF a été sauvegardée et sera utilisée pour améliorer les analyses futures (ID: {analysis_id})")
                        else:
                            st.warning("⚠️ L'analyse PDF n'a pas pu être sauvegardée dans l'historique.")
                
                else:
                    # Échec de l'extraction
                    st.error(f"Échec de l'analyse du PDF: {pdf_result.get('error', 'Erreur inconnue')}")
                
                # Terminer la progression
                progress_bar.progress(100)
                status_text.text("Analyse du PDF terminée avec succès !")
                
            except Exception as e:
                st.error(f"Erreur lors de l'analyse du PDF: {e}")
                logger.error(f"Erreur d'analyse PDF: {e}")
    
    else:
        st.info("Veuillez charger un fichier PDF pour commencer l'analyse.")

def process_visualizations():
    """Fonction pour gérer les visualisations interactives"""
    
    st.header("📊 Visualisations interactives")
    
    # Vérifier si un DataFrame est déjà chargé dans la session
    if "df" in st.session_state:
        # Afficher un aperçu du DataFrame chargé
        st.write(f"**Données chargées**: {st.session_state.get('df_name', 'Données sans nom')} - {st.session_state['df'].shape[0]} lignes, {st.session_state['df'].shape[1]} colonnes")
        
        # Appeler la fonction de visualisation
        create_charts()
    else:
        # Permettre à l'utilisateur de charger un fichier directement depuis cet onglet
        st.info("Aucune donnée chargée. Vous pouvez charger un fichier CSV ci-dessous.")
        uploaded_file = st.file_uploader("Choisir un fichier CSV pour les visualisations", type="csv", key="viz_uploader")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state["df"] = df
                st.session_state["df_name"] = uploaded_file.name
                st.success(f"Données chargées avec succès : {df.shape[0]} lignes, {df.shape[1]} colonnes")
                
                # Afficher un aperçu
                st.dataframe(df.head())
                
                # Appeler la fonction de visualisation
                create_charts()
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier: {e}")

def show_history_tab():
    """Affiche l'historique des analyses CSV et PDF"""
    
    st.header("📚 Historique des analyses")
    
    # Récupérer les historiques
    history = get_analysis_history()
    pdf_history = get_pdf_history()
    
    # Créer des sous-onglets pour différencier CSV et PDF
    tabs = st.tabs(["CSV", "PDF"])
    
    # Onglet CSV
    with tabs[0]:
        recent_analyses = history.get_recent_analyses(limit=10)
        
        if recent_analyses:
            st.write(f"Vous avez {len(history.index['analyses'])} analyses CSV sauvegardées sur {len(history.index['datasets'])} jeux de données différents.")
            
            # Afficher les analyses récentes
            st.subheader("Analyses CSV récentes")
            for analysis in recent_analyses:
                with st.expander(f"{analysis['dataset_name']} - {analysis['timestamp'][:16].replace('T', ' ')}"):
                    st.write(f"**Description**: {analysis.get('dataset_description', 'Non disponible')}")
                    if "metadata" in analysis and analysis["metadata"]:
                        meta = analysis["metadata"]
                        st.write(f"**Dimensions**: {meta.get('dimensions', 'Non disponible')}")
                    st.write("**Analyse**:")
                    st.info(analysis.get("analysis", "Contenu non disponible"))
            
            # Afficher les datasets
            st.subheader("Datasets analysés")
            for dataset_name, dataset_info in history.index["datasets"].items():
                with st.expander(f"{dataset_name} ({dataset_info['analyses_count']} analyses)"):
                    st.write(f"**Première analyse**: {dataset_info['first_analysis'][:16].replace('T', ' ')}")
                    st.write(f"**Dernière analyse**: {dataset_info['last_analysis'][:16].replace('T', ' ')}")
                    
                    # Afficher la dernière analyse pour ce dataset
                    last_analysis_id = dataset_info['analyses'][-1]
                    last_analysis = history.get_analysis(last_analysis_id)
                    if last_analysis:
                        st.write("**Dernière analyse**:")
                        st.info(last_analysis.get("analysis", "Aucune analyse disponible"))

                    else:
                        st.info("Aucune analyse CSV dans l'historique")

    # Onglet PDF
    with tabs[1]:
        pdf_analyses = pdf_history.get_recent_pdf_analyses(limit=10)
        
        if pdf_analyses:
            st.write(f"Vous avez {len(pdf_history.index['analyses'])} analyses PDF sauvegardées sur {len(pdf_history.index['documents'])} documents différents.")
            
            # Afficher les analyses récentes
            st.subheader("Analyses PDF récentes")
            for analysis in pdf_analyses:
                with st.expander(f"{analysis['pdf_name']} - {analysis['timestamp'][:16].replace('T', ' ')}"):
                    st.write(f"**ID du PDF**: {analysis.get('pdf_id', 'Non disponible')}")
                    
                    # Afficher les métadonnées
                    if "metadata" in analysis and analysis["metadata"]:
                        meta = analysis["metadata"]
                        st.write(f"**Pages**: {meta.get('page_count', 'N/A')}, **Mots**: {meta.get('word_count', 'N/A')}")
                    
                    # Afficher le résumé
                    if "analysis" in analysis and "summary" in analysis["analysis"]:
                        st.write("**Résumé**:")
                        st.info(analysis["analysis"]["summary"])
                    
                    # Afficher les thèmes
                    if "analysis" in analysis and "key_themes" in analysis["analysis"]:
                        st.write("**Thèmes clés**:")
                        for theme in analysis["analysis"]["key_themes"]:
                            st.markdown(f"• {theme}")
        else:
            st.info("Aucune analyse PDF dans l'historique")

def show_settings():
    """Affiche les paramètres de l'application"""
    st.header("⚙️ Paramètres")
    
    # Paramètres du modèle
    st.subheader("Paramètres du modèle")
    
    # Initialiser les paramètres par défaut s'ils n'existent pas
    if "model_name" not in st.session_state:
        st.session_state["model_name"] = "mistral:latest"
    if "context_size" not in st.session_state:
        st.session_state["context_size"] = 4096
    
    # Modèle à utiliser
    st.session_state["model_name"] = st.selectbox(
        "Modèle à utiliser",
        ["mistral:latest", "llama3:latest", "llama2:13b", "mixtral:latest"],
        index=["mistral:latest", "llama3:latest", "llama2:13b", "mixtral:latest"].index(st.session_state["model_name"])
    )
    
    # Taille du contexte
    st.session_state["context_size"] = st.slider(
        "Taille du contexte (tokens)",
        min_value=1024,
        max_value=16384,
        value=st.session_state["context_size"],
        step=1024
    )
    
    # Paramètres de l'historique
    st.subheader("Paramètres de l'historique")
    
    # Activation de l'historique
    st.session_state["use_history"] = st.toggle(
        "Utiliser l'historique des analyses",
        value=st.session_state.get("use_history", True),
        help="Si activé, les analyses précédentes sont utilisées pour améliorer les analyses futures"
    )
    
    # Nombre maximal d'analyses à utiliser
    st.session_state["max_history"] = st.slider(
        "Nombre maximum d'analyses historiques à utiliser",
        min_value=1,
        max_value=10,
        value=st.session_state.get("max_history", 3),
        step=1,
        help="Nombre maximum d'analyses précédentes à inclure dans le contexte"
    )
    
    # Poids de l'historique
    st.session_state["history_weight"] = st.slider(
        "Poids de l'historique (1-10)",
        min_value=1,
        max_value=10,
        value=st.session_state.get("history_weight", 5),
        step=1,
        help="Détermine l'importance des analyses passées par rapport aux instructions actuelles (1 = faible, 10 = élevé)"
    )
    
    # Paramètres de transformation
    st.subheader("Paramètres de transformation")
    
    # Auto-détection des transformations
    st.session_state["auto_detect"] = st.toggle(
        "Auto-détection des transformations",
        value=st.session_state.get("auto_detect", True),
        help="Si activé, l'IA détecte automatiquement les transformations à appliquer"
    )
    
    # Transformations manuelles (visible si auto-détection désactivée)
    if not st.session_state["auto_detect"]:
        # Liste des transformations possibles
        all_transformations = [
            "Traitement des valeurs manquantes",
            "Normalisation des données",
            "Détection et traitement des outliers",
            "Encodage des variables catégorielles",
            "Création de nouvelles variables",
            "Réduction de dimensionnalité",
            "Filtrage des données",
            "Agrégation des données"
        ]
        
        # Initialiser les transformations si non définies
        if "transformations" not in st.session_state:
            st.session_state["transformations"] = ["Traitement des valeurs manquantes"]
        
        # Sélection multiple des transformations
        st.session_state["transformations"] = st.multiselect(
            "Transformations à appliquer",
            all_transformations,
            default=st.session_state["transformations"]
        )

# Page principale
def main():
    """Application principale"""
    
    # Titre et description
    st.title("ORBIT - Data Transformation")
    st.image("C:\\Users\\Francis\\Desktop\\Data_cleaner\\PDF_analysis\\orbit-icon-13.jpg")
    st.write("Une application pour transformer vos données CSV et analyser vos PDF avec IA")
    
    # Onglets pour les différentes sections
# Onglets pour les différentes sections
tabs = st.tabs(["CSV", "PDF", "Visualisations", "Historique", "Paramètres"])

# Onglet CSV
with tabs[0]:
    process_csv_data()

# Onglet PDF
with tabs[1]:
    process_pdf_data()

# Onglet Visualisations
with tabs[2]:
    process_visualizations()  # Nouvelle fonction à créer

# Onglet Historique
with tabs[3]:
    show_history_tab()

# Onglet Paramètres
with tabs[4]:
    show_settings()
    

    # Contexte utilisateur pour l'ensemble de l'application
    with st.sidebar:
        st.header("Instructions utilisateur")
        
        st.session_state["user_context"] = st.text_area(
            "Instructions pour l'analyse",
            value=st.session_state.get("user_context", ""),
            height=300,
            help="Décrivez votre objectif d'analyse ou posez des questions sur vos données. Par exemple: 'Identifiez les tendances principales', 'Analysez les relations entre variables', ou 'Que pouvez-vous me dire sur ces données?'",
            key="user_context_area"
        )
        
        # Ajouter un exemple de prompt utile
        with st.expander("📝 Exemples de prompts efficaces"):
            st.markdown("""
            **Prompts génériques utiles:**
            - "Analysez ce jeu de données et identifiez les insights principaux."
            - "Pouvez-vous me dire quelles variables semblent avoir le plus d'impact sur [variable cible]?"
            - "Quelles tendances ou patterns voyez-vous dans ces données?"
            - "Comment pourrais-je améliorer la qualité de ce jeu de données?"
            
            **Pour les CSV:**
            - "Y a-t-il des corrélations significatives entre les variables?"
            - "Comment pourrais-je segmenter ces données pour mieux comprendre [phénomène]?"
            
            **Pour les PDF:**
            - "Résumez les points clés de ce document et identifiez les thèmes principaux."
            - "Quels sont les arguments principaux présentés dans ce document?"
            """)
        
        # Afficher des informations sur l'historique
        if st.session_state.get("use_history", True):
            history = get_analysis_history()
            pdf_history = get_pdf_history()
            
            st.info(f"""
            **État de l'historique**:
            - {len(history.index['analyses'])} analyses CSV
            - {len(pdf_history.index['analyses'])} analyses PDF
            
            Historique activé (poids: {st.session_state.get("history_weight", 5)}/10)
            """)
        else:
            st.warning("L'historique est actuellement désactivé")
        
        # Afficher les crédits
        st.markdown("---")
        st.markdown("© 2024 - Application de transformation de données avec mémoire")

if __name__ == "__main__":
    main()