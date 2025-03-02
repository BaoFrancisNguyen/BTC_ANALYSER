import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from data_transformer_ollama2 import DataTransformer

def generate_graph_description(graph_type, x_var, y_var, df):
    """
    Génère une description explicative du graphique sélectionné.
    
    Args:
        graph_type (str): Type de graphique
        x_var (str): Variable x
        y_var (str): Variable y
        df (pd.DataFrame): Données du graphique
    
    Returns:
        str: Description générée par Mistral
    """
    try:
        # Initialiser le transformateur avec Mistral
        transformer = DataTransformer(model_name="mistral:latest")
        
        # Préparer un contexte pour la description du graphique
        context = f"""
        Décris en détail le graphique suivant :
        - Type de graphique : {graph_type}
        - Variable X : {x_var}
        - Variable Y : {y_var}
        """
        
        # Générer l'analyse descriptive
        analysis = transformer.generate_dataset_analysis(df[[x_var, y_var]], context)
        
        return analysis
    except Exception as e:
        st.error(f"Erreur lors de la génération de la description : {e}")
        return "Impossible de générer la description automatique."

def create_charts():
    """Fonction pour créer et afficher des graphiques dynamiques et interactifs"""
    
    st.header("📊 Visualisations interactives")
    
    # Vérifier si un fichier CSV a été chargé dans la session
    if "df" not in st.session_state:
        uploaded_file = st.file_uploader("Choisir un fichier CSV pour les visualisations", type="csv", key="chart_uploader")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state["df"] = df
                st.session_state["df_name"] = uploaded_file.name
                st.success(f"Données chargées avec succès : {df.shape[0]} lignes, {df.shape[1]} colonnes")
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier: {e}")
                return
        else:
            st.info("Veuillez charger un fichier CSV pour générer des visualisations.")
            return
    
    # À ce stade, nous avons un DataFrame à visualiser
    df = st.session_state["df"]
    
    # Barre latérale pour les options de visualisation
    with st.sidebar:
        st.subheader("Options de visualisation")
        chart_type = st.selectbox(
            "Type de graphique",
            ["Distribution", "Relation", "Catégoriel", "Matrice de corrélation", 
             "Boxplot", "Carte de chaleur", "Timeseries", "Dashboard interactif"]
        )
    
    # Afficher un aperçu des données
    with st.expander("Aperçu des données", expanded=False):
        st.dataframe(df.head())
        
        # Afficher les informations sur les colonnes
        st.subheader("Types de données")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Colonnes numériques:")
            st.write(df.select_dtypes(include=['number']).columns.tolist())
        with col2:
            st.write("Colonnes catégorielles:")
            st.write(df.select_dtypes(exclude=['number']).columns.tolist())
    
    # Logique pour différents types de graphiques
    if chart_type == "Distribution":
        st.subheader("Distribution des variables")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            st.warning("Aucune colonne numérique trouvée pour créer un histogramme.")
            return
            
        selected_col = st.selectbox("Sélectionner une variable numérique", numeric_cols)
        
        # Options de l'histogramme
        col1, col2, col3 = st.columns(3)
        with col1:
            bins = st.slider("Nombre de bins", min_value=5, max_value=100, value=30)
        with col2:
            kde = st.checkbox("Afficher la courbe KDE", value=True)
        with col3:
            color_scale = st.selectbox("Palette de couleurs", 
                                      ["viridis", "plasma", "inferno", "magma", "cividis", "turbo"])
        
        # Créer l'histogramme interactif avec Plotly
        if kde:
            # Histogramme avec densité
            fig = px.histogram(df, x=selected_col, 
                              nbins=bins,
                              marginal="box", # Affiche un boxplot en marge
                              opacity=0.8,
                              histnorm="probability density")
            
            # Ajouter une courbe KDE
            x = df[selected_col].dropna()
            x_range = np.linspace(min(x), max(x), 1000)
            kde_function = stats.gaussian_kde(x)
            y_kde = kde_function(x_range)
            
            fig.add_trace(go.Scatter(x=x_range, y=y_kde, mode='lines', name='KDE',
                                    line=dict(color='red', width=2)))
        else:
            # Histogramme simple
            fig = px.histogram(df, x=selected_col, 
                              nbins=bins,
                              marginal="box")
        
        fig.update_layout(
            title=f"Distribution de {selected_col}",
            xaxis_title=selected_col,
            yaxis_title="Fréquence",
            hovermode="closest",
            bargap=0.1
        )
        
        # Afficher le graphique interactif
        st.plotly_chart(fig, use_container_width=True)
        
        
        # Statistiques descriptives
        st.subheader("Statistiques descriptives")
        stats_df = pd.DataFrame(df[selected_col].describe()).T
        
        # Ajouter des métriques supplémentaires
        skewness = stats.skew(df[selected_col].dropna())
        kurtosis = stats.kurtosis(df[selected_col].dropna())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Moyenne", f"{df[selected_col].mean():.2f}")
        with col2:
            st.metric("Médiane", f"{df[selected_col].median():.2f}")
        with col3:
            st.metric("Écart-type", f"{df[selected_col].std():.2f}")
            
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Asymétrie (Skewness)", f"{skewness:.2f}", 
                     delta="Positive" if skewness > 0 else "Négative" if skewness < 0 else "Symétrique")
        with col2:
            st.metric("Aplatissement (Kurtosis)", f"{kurtosis:.2f}")
        with col3:
            # Test de normalité (Shapiro-Wilk)
            sample = df[selected_col].dropna()
            if len(sample) > 5000:
                sample = sample.sample(5000)  # Limite pour le test de Shapiro
            shapiro_test = stats.shapiro(sample)
            is_normal = shapiro_test.pvalue > 0.05
            st.metric("Test de normalité (p-value)", f"{shapiro_test.pvalue:.4f}", 
                     delta="Distribution normale" if is_normal else "Non-normale")
        
    elif chart_type == "Relation":
        st.subheader("Relation entre variables")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("Au moins deux colonnes numériques sont nécessaires pour créer un scatter plot.")
            return
            
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("Variable X", numeric_cols, index=0)
        with col2:
            y_col = st.selectbox("Variable Y", numeric_cols, index=min(1, len(numeric_cols)-1))
        with col3:
            plot_type = st.selectbox("Type de visualisation", ["Scatter", "Ligne", "Scatter+Ligne"])
        
        # Options supplémentaires
        col1, col2 = st.columns(2)
        with col1:
            hue_option = st.selectbox("Coloration par catégorie (optionnel)", 
                                    ["Aucune"] + df.select_dtypes(exclude=['number']).columns.tolist())
        with col2:
            size_option = st.selectbox("Taille des points (optionnel)", 
                                     ["Uniforme"] + numeric_cols)
        
        # Options avancées
        with st.expander("Options avancées"):
            col1, col2 = st.columns(2)
            with col1:
                trend_line = st.selectbox("Ligne de tendance", ["Aucune", "OLS", "Lowess", "Quadratique", "Exponentielle"])
            with col2:
                animation_col = st.selectbox("Animation par (optionnel)", 
                                           ["Aucune"] + df.columns.tolist())
        
        # Créer le scatter plot interactif
        if hue_option != "Aucune":
            if animation_col != "Aucune":
                # Animation + coloration
                if plot_type == "Scatter":
                    fig = px.scatter(df, x=x_col, y=y_col, color=hue_option,
                                   size=None if size_option == "Uniforme" else size_option,
                                   animation_frame=animation_col, animation_group=hue_option,
                                   trendline=None if trend_line == "Aucune" else trend_line.lower().replace('quadratique', 'ols(order=2)').replace('exponentielle', 'ols(log_y=True)'),
                                   hover_name=df.index if "Name" not in df.columns else df["Name"] if "Name" in df.columns else None)
                elif plot_type == "Ligne":
                    fig = px.line(df, x=x_col, y=y_col, color=hue_option,
                                animation_frame=animation_col, animation_group=hue_option,
                                hover_name=df.index if "Name" not in df.columns else df["Name"] if "Name" in df.columns else None)
                else:  # Scatter+Ligne
                    fig = px.scatter(df, x=x_col, y=y_col, color=hue_option,
                                   size=None if size_option == "Uniforme" else size_option,
                                   animation_frame=animation_col, animation_group=hue_option,
                                   trendline=None if trend_line == "Aucune" else trend_line.lower().replace('quadratique', 'ols(order=2)').replace('exponentielle', 'ols(log_y=True)'),
                                   hover_name=df.index if "Name" not in df.columns else df["Name"] if "Name" in df.columns else None)
                    # Ajouter des lignes reliant les points par groupe
                    for group in df[hue_option].unique():
                        group_data = df[df[hue_option] == group].sort_values(by=x_col)
                        fig.add_trace(go.Scatter(x=group_data[x_col], y=group_data[y_col], 
                                               mode='lines', showlegend=False, 
                                               line=dict(width=1, dash='dot')))
            else:
                # Coloration sans animation
                if plot_type == "Scatter":
                    fig = px.scatter(df, x=x_col, y=y_col, color=hue_option,
                                   size=None if size_option == "Uniforme" else size_option,
                                   trendline=None if trend_line == "Aucune" else trend_line.lower().replace('quadratique', 'ols(order=2)').replace('exponentielle', 'ols(log_y=True)'),
                                   hover_name=df.index if "Name" not in df.columns else df["Name"] if "Name" in df.columns else None)
                elif plot_type == "Ligne":
                    fig = px.line(df, x=x_col, y=y_col, color=hue_option,
                                hover_name=df.index if "Name" not in df.columns else df["Name"] if "Name" in df.columns else None)
                else:  # Scatter+Ligne
                    fig = px.scatter(df, x=x_col, y=y_col, color=hue_option,
                                   size=None if size_option == "Uniforme" else size_option,
                                   trendline=None if trend_line == "Aucune" else trend_line.lower().replace('quadratique', 'ols(order=2)').replace('exponentielle', 'ols(log_y=True)'),
                                   hover_name=df.index if "Name" not in df.columns else df["Name"] if "Name" in df.columns else None)
                    # Ajouter des lignes reliant les points par groupe
                    for group in df[hue_option].unique():
                        group_data = df[df[hue_option] == group].sort_values(by=x_col)
                        fig.add_trace(go.Scatter(x=group_data[x_col], y=group_data[y_col], 
                                               mode='lines', showlegend=False, 
                                               line=dict(width=1, dash='dot')))
        else:
            # Sans coloration
            if animation_col != "Aucune":
                # Animation sans coloration
                if plot_type == "Scatter":
                    fig = px.scatter(df, x=x_col, y=y_col,
                                   size=None if size_option == "Uniforme" else size_option,
                                   animation_frame=animation_col,
                                   trendline=None if trend_line == "Aucune" else trend_line.lower().replace('quadratique', 'ols(order=2)').replace('exponentielle', 'ols(log_y=True)'),
                                   hover_name=df.index if "Name" not in df.columns else df["Name"] if "Name" in df.columns else None)
                elif plot_type == "Ligne":
                    fig = px.line(df, x=x_col, y=y_col,
                                animation_frame=animation_col,
                                hover_name=df.index if "Name" not in df.columns else df["Name"] if "Name" in df.columns else None)
                else:  # Scatter+Ligne
                    fig = px.scatter(df, x=x_col, y=y_col,
                                   size=None if size_option == "Uniforme" else size_option,
                                   animation_frame=animation_col,
                                   trendline=None if trend_line == "Aucune" else trend_line.lower().replace('quadratique', 'ols(order=2)').replace('exponentielle', 'ols(log_y=True)'),
                                   hover_name=df.index if "Name" not in df.columns else df["Name"] if "Name" in df.columns else None)
                    # Ajouter une ligne
                    sorted_data = df.sort_values(by=x_col)
                    fig.add_trace(go.Scatter(x=sorted_data[x_col], y=sorted_data[y_col], 
                                           mode='lines', showlegend=False, 
                                           line=dict(width=1, dash='dot')))
            else:
                # Simple, sans coloration ni animation
                if plot_type == "Scatter":
                    fig = px.scatter(df, x=x_col, y=y_col,
                                   size=None if size_option == "Uniforme" else size_option,
                                   trendline=None if trend_line == "Aucune" else trend_line.lower().replace('quadratique', 'ols(order=2)').replace('exponentielle', 'ols(log_y=True)'),
                                   hover_name=df.index if "Name" not in df.columns else df["Name"] if "Name" in df.columns else None)
                elif plot_type == "Ligne":
                    fig = px.line(df, x=x_col, y=y_col,
                                hover_name=df.index if "Name" not in df.columns else df["Name"] if "Name" in df.columns else None)
                else:  # Scatter+Ligne
                    fig = px.scatter(df, x=x_col, y=y_col,
                                   size=None if size_option == "Uniforme" else size_option,
                                   trendline=None if trend_line == "Aucune" else trend_line.lower().replace('quadratique', 'ols(order=2)').replace('exponentielle', 'ols(log_y=True)'),
                                   hover_name=df.index if "Name" not in df.columns else df["Name"] if "Name" in df.columns else None)
                    # Ajouter une ligne
                    sorted_data = df.sort_values(by=x_col)
                    fig.add_trace(go.Scatter(x=sorted_data[x_col], y=sorted_data[y_col], 
                                           mode='lines', showlegend=False, 
                                           line=dict(width=1, dash='dot')))
        
        fig.update_layout(
            title=f"Relation entre {x_col} et {y_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
            hovermode="closest"
        )
        
        # Afficher le graphique interactif
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistique de corrélation
        st.subheader("Analyse de la relation")
        
        # Calcul des corrélations
        corr_pearson = df[[x_col, y_col]].corr().iloc[0, 1]
        corr_spearman = df[[x_col, y_col]].corr(method='spearman').iloc[0, 1]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Corrélation de Pearson", f"{corr_pearson:.4f}", 
                     delta="Forte" if abs(corr_pearson) > 0.7 else "Modérée" if abs(corr_pearson) > 0.3 else "Faible")
        with col2:
            st.metric("Corrélation de Spearman (rangs)", f"{corr_spearman:.4f}",
                     delta="Forte" if abs(corr_spearman) > 0.7 else "Modérée" if abs(corr_spearman) > 0.3 else "Faible")
        
        # Si une ligne de tendance OLS est demandée, ajouter des statistiques de régression
        if trend_line in ["OLS", "Quadratique", "Exponentielle"]:
            st.subheader("Statistiques de régression")
            
            if trend_line == "OLS":
                formula = f"{y_col} ~ {x_col}"
                model = ols(formula, data=df).fit()
                
                # Afficher les coefficients
                st.write(f"**Équation**: {y_col} = {model.params[1]:.4f} × {x_col} + {model.params[0]:.4f}")
                
                # Afficher R² et p-value
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² (coefficient de détermination)", f"{model.rsquared:.4f}")
                with col2:
                    st.metric("R² ajusté", f"{model.rsquared_adj:.4f}")
                with col3:
                    st.metric("p-value", f"{model.f_pvalue:.6f}",
                             delta="Significatif" if model.f_pvalue < 0.05 else "Non significatif")
                
                with st.expander("Résumé complet de la régression"):
                    st.text(model.summary().as_text())
            
            elif trend_line == "Quadratique":
                # Ajouter un terme quadratique
                df_temp = df.copy()
                df_temp[f"{x_col}_squared"] = df_temp[x_col] ** 2
                
                formula = f"{y_col} ~ {x_col} + {x_col}_squared"
                model = ols(formula, data=df_temp).fit()
                
                # Afficher les coefficients
                st.write(f"**Équation**: {y_col} = {model.params[2]:.4f} × {x_col}² + {model.params[1]:.4f} × {x_col} + {model.params[0]:.4f}")
                
                # Afficher R² et p-value
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² (coefficient de détermination)", f"{model.rsquared:.4f}")
                with col2:
                    st.metric("R² ajusté", f"{model.rsquared_adj:.4f}")
                with col3:
                    st.metric("p-value", f"{model.f_pvalue:.6f}",
                             delta="Significatif" if model.f_pvalue < 0.05 else "Non significatif")
            
            elif trend_line == "Exponentielle":
                # Transformation logarithmique
                df_temp = df.copy()
                df_temp[f"log_{y_col}"] = np.log(df_temp[y_col])
                
                formula = f"log_{y_col} ~ {x_col}"
                model = ols(formula, data=df_temp).fit()
                
                # Afficher les coefficients (forme y = a * e^(bx))
                a = np.exp(model.params[0])
                b = model.params[1]
                st.write(f"**Équation**: {y_col} = {a:.4f} × e^({b:.4f} × {x_col})")
                
                # Afficher R² et p-value
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² (coefficient de détermination)", f"{model.rsquared:.4f}")
                with col2:
                    st.metric("R² ajusté", f"{model.rsquared_adj:.4f}")
                with col3:
                    st.metric("p-value", f"{model.f_pvalue:.6f}",
                             delta="Significatif" if model.f_pvalue < 0.05 else "Non significatif")
        
    elif chart_type == "Catégoriel":
        st.subheader("Analyse catégorielle")
        
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        if not cat_cols:
            st.warning("Aucune colonne catégorielle trouvée. Conversion de variables numériques en catégories...")
            # Proposer de convertir des variables numériques en catégories
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            col_to_cat = st.selectbox("Variable numérique à convertir en catégorie", num_cols)
            
            # Méthode de conversion
            cat_method = st.selectbox("Méthode de catégorisation", 
                                    ["Quantiles", "Intervalles égaux", "Valeurs uniques"])
            
            if cat_method == "Quantiles":
                n_bins = st.slider("Nombre de catégories", min_value=2, max_value=10, value=4)
                df_mod = df.copy()
                df_mod[f"{col_to_cat}_cat"] = pd.qcut(df_mod[col_to_cat], q=n_bins, duplicates='drop')
                cat_col = f"{col_to_cat}_cat"
            elif cat_method == "Intervalles égaux":
                n_bins = st.slider("Nombre de catégories", min_value=2, max_value=10, value=4)
                df_mod = df.copy()
                df_mod[f"{col_to_cat}_cat"] = pd.cut(df_mod[col_to_cat], bins=n_bins)
                cat_col = f"{col_to_cat}_cat"
            else:  # Valeurs uniques
                df_mod = df.copy()
                df_mod[f"{col_to_cat}_cat"] = df_mod[col_to_cat].astype(str)
                cat_col = f"{col_to_cat}_cat"
        else:
            df_mod = df.copy()
            cat_col = st.selectbox("Sélectionner une variable catégorielle", cat_cols)
        
        # Options de visualisation
        viz_type = st.selectbox("Type de visualisation", 
                               ["Barres", "Camembert", "Treemap", "Sunburst"])
        
        # Options pour une deuxième variable
        second_cat = st.selectbox("Seconde variable catégorielle (optionnel)", 
                                ["Aucune"] + [col for col in df_mod.select_dtypes(exclude=['number']).columns if col != cat_col])
        
        # Variable numérique pour les valeurs (au lieu du comptage)
        value_col = st.selectbox("Valeur à agréger (optionnel)", 
                               ["Comptage"] + df.select_dtypes(include=['number']).columns.tolist())
        
        # Méthode d'agrégation si valeur numérique
        if value_col != "Comptage":
            agg_method = st.selectbox("Méthode d'agrégation", ["Somme", "Moyenne", "Médiane", "Max", "Min"])
            agg_func = {"Somme": "sum", "Moyenne": "mean", "Médiane": "median", "Max": "max", "Min": "min"}[agg_method]
        
        # Construire le DataFrame pour la visualisation
        if second_cat != "Aucune":
            if value_col == "Comptage":
                # Tableau croisé pour le comptage
                pivot_df = pd.crosstab(df_mod[cat_col], df_mod[second_cat])
                # Convertir en format long pour Plotly
                plot_df = pivot_df.reset_index().melt(id_vars=cat_col, var_name=second_cat, value_name="Comptage")
            else:
                # Agrégation d'une valeur numérique
                plot_df = df_mod.groupby([cat_col, second_cat])[value_col].agg(agg_func).reset_index()
                plot_df.columns = [cat_col, second_cat, value_col]
        else:
            if value_col == "Comptage":
                # Simple comptage
                plot_df = df_mod[cat_col].value_counts().reset_index()
                plot_df.columns = [cat_col, "Comptage"]
            else:
                # Agrégation d'une valeur numérique
                plot_df = df_mod.groupby(cat_col)[value_col].agg(agg_func).reset_index()
        
        # Créer la visualisation selon le type choisi
        if viz_type == "Barres":
            if second_cat != "Aucune":
                if value_col == "Comptage":
                    fig = px.bar(plot_df, x=cat_col, y="Comptage", color=second_cat, barmode="group")
                else:
                    fig = px.bar(plot_df, x=cat_col, y=value_col, color=second_cat, barmode="group")
            else:
                if value_col == "Comptage":
                    fig = px.bar(plot_df, x=cat_col, y="Comptage")
                else:
                    fig = px.bar(plot_df, x=cat_col, y=value_col)
            
            fig.update_layout(title=f"{cat_col} - {second_cat if second_cat != 'Aucune' else ''}")
        
        elif viz_type == "Camembert":
            if second_cat != "Aucune":
                st.warning("Le camembert ne peut pas afficher deux variables catégorielles simultanément. Utilisation d'une seule variable.")
            
            if value_col == "Comptage":
                fig = px.pie(plot_df, names=cat_col, values="Comptage")
            else:
                fig = px.pie(plot_df, names=cat_col, values=value_col)
            
            fig.update_layout(title=f"Répartition de {cat_col}")
        
        elif viz_type == "Treemap":
            if second_cat != "Aucune":
                if value_col == "Comptage":
                    fig = px.treemap(plot_df, path=[cat_col, second_cat], values="Comptage")
                else:
                    fig = px.treemap(plot_df, path=[cat_col, second_cat], values=value_col)
            else:
                if value_col == "Comptage":
                    fig = px.treemap(plot_df, path=[cat_col], values="Comptage")
                else:
                    fig = px.treemap(plot_df, path=[cat_col], values=value_col)
            
            fig.update_layout(title=f"Treemap de {cat_col} {('× ' + second_cat) if second_cat != 'Aucune' else ''}")
        
        elif viz_type == "Sunburst":
            if second_cat != "Aucune":
                if value_col == "Comptage":
                    fig = px.sunburst(plot_df, path=[cat_col, second_cat], values="Comptage")
                else:
                    fig = px.sunburst(plot_df, path=[cat_col, second_cat], values=value_col)
            else:
                if value_col == "Comptage":
                    fig = px.sunburst(plot_df, path=[cat_col], values="Comptage")
                else:
                    fig = px.sunburst(plot_df, path=[cat_col], values=value_col)
            
            fig.update_layout(title=f"Sunburst de {cat_col} {('× ' + second_cat) if second_cat != 'Aucune' else ''}")
        
        # Afficher le graphique
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau des fréquences
        st.subheader("Tableau de fréquence")
        if second_cat != "Aucune":
            # Tableau croisé
            if value_col == "Comptage":
                cross_tab = pd.crosstab(df_mod[cat_col], df_mod[second_cat], normalize="index") * 100
                cross_tab_count = pd.crosstab(df_mod[cat_col], df_mod[second_cat])
                
                st.write("Tableau croisé (pourcentages par ligne) :")
                st.dataframe(cross_tab.style.format("{:.1f}%"))
                
                st.write("Tableau croisé (comptages) :")
                st.dataframe(cross_tab_count)
            else:
                # Agrégation
                cross_tab = df_mod.pivot_table(index=cat_col, columns=second_cat, values=value_col, aggfunc=agg_func)
                
                st.write(f"Tableau croisé ({agg_method.lower()} de {value_col}) :")
                st.dataframe(cross_tab.style.format("{:.2f}"))
        else:
            # Tableau simple
            if value_col == "Comptage":
                freq_table = df_mod[cat_col].value_counts().reset_index()
                freq_table.columns = [cat_col, "Comptage"]
                freq_table["Pourcentage"] = freq_table["Comptage"] / freq_table["Comptage"].sum() * 100
                
                st.dataframe(freq_table.style.format({"Pourcentage": "{:.1f}%"}))
            else:
                freq_table = df_mod.groupby(cat_col)[value_col].agg([agg_func, 'count']).reset_index()
                freq_table.columns = [cat_col, agg_method, "Comptage"]
                
                st.dataframe(freq_table)

    elif chart_type == "Matrice de corrélation":
        st.subheader("Matrice de corrélation")
        
        # Obtenir uniquement les colonnes numériques
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("Au moins deux colonnes numériques sont nécessaires pour créer une matrice de corrélation.")
            return
        
        # Options de la matrice
        corr_method = st.selectbox("Méthode de corrélation", ["Pearson", "Spearman"])
        
        # Sélection des variables
        selected_cols = st.multiselect(
            "Variables à inclure (laisser vide pour toutes)",
            options=numeric_cols,
            default=numeric_cols[:min(len(numeric_cols), 10)]  # Par défaut, prendre les 10 premières si plus de 10
        )
        
        if not selected_cols:  # Si aucune colonne n'est sélectionnée, prendre toutes les colonnes numériques
            selected_cols = numeric_cols
        
        # Calculer la matrice de corrélation
        corr_matrix = df[selected_cols].corr(method=corr_method.lower())
        
        # Options d'affichage
        col1, col2 = st.columns(2)
        with col1:
            color_scale = st.selectbox("Palette de couleurs", 
                                     ["RdBu_r", "Viridis", "Plasma", "Blues", "Greens", "YlOrRd"])
        with col2:
            show_values = st.checkbox("Afficher les valeurs", value=True)
        
        # Créer la heatmap avec Plotly
        if show_values:
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale=color_scale,
                labels=dict(color="Corrélation"),
                text_auto=".2f",  # Affiche les valeurs avec 2 décimales
                aspect="auto"     # Ajuste l'aspect pour s'adapter à la taille
            )
        else:
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale=color_scale,
                labels=dict(color="Corrélation"),
                aspect="auto"
            )
        
        fig.update_layout(
            title=f"Matrice de corrélation ({corr_method})",
            height=600
        )
        
        # Afficher la heatmap
        st.plotly_chart(fig, use_container_width=True)
        
        # Afficher les paires avec les plus fortes corrélations (positives et négatives)
        st.subheader("Paires avec les plus fortes corrélations")
        
        # Obtenir les paires uniques (triangle supérieur de la matrice)
        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                pairs.append((var1, var2, corr_val))
        
        # Trier par valeur absolue de corrélation
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Créer un DataFrame pour afficher les résultats
        top_pairs = pd.DataFrame(pairs, columns=["Variable 1", "Variable 2", "Corrélation"])
        
        # Afficher les 15 premières paires avec la corrélation la plus forte (en valeur absolue)
        st.dataframe(top_pairs.head(15).style.format({"Corrélation": "{:.4f}"}))
        
        # Option pour explorer une paire spécifique
        st.subheader("Explorer une paire de variables")
        
        col1, col2 = st.columns(2)
        with col1:
            var1 = st.selectbox("Variable 1", selected_cols, index=0)
        with col2:
            var2 = st.selectbox("Variable 2", selected_cols, index=min(1, len(selected_cols)-1))
        
        # Créer un scatter plot pour la paire sélectionnée
        fig = px.scatter(df, x=var1, y=var2, trendline="ols")
        
        fig.update_layout(
            title=f"Relation entre {var1} et {var2} (r = {corr_matrix.loc[var1, var2]:.4f})",
            xaxis_title=var1,
            yaxis_title=var2,
            hovermode="closest"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Boxplot":
        st.subheader("Analyse de distributions (Boxplot)")
        
        # Obtenir les colonnes numériques
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            st.warning("Aucune colonne numérique trouvée pour créer un boxplot.")
            return
        
        # Sélectionner la variable numérique pour le boxplot
        y_var = st.selectbox("Variable numérique", numeric_cols)
        
        # Options pour la variable catégorielle (facultatif)
        cat_option = st.selectbox(
            "Grouper par (optionnel)",
            ["Aucun"] + df.select_dtypes(exclude=['number']).columns.tolist()
        )
        
        # Options de visualisation
        col1, col2 = st.columns(2)
        with col1:
            orientation = st.radio("Orientation", ["Verticale", "Horizontale"])
        with col2:
            points = st.radio("Affichage des points", ["Outliers", "Tous les points", "Aucun"])
        
        # Créer le boxplot
        if cat_option != "Aucun":
            if orientation == "Verticale":
                if points == "Outliers":
                    fig = px.box(df, x=cat_option, y=y_var, points="outliers")
                elif points == "Tous les points":
                    fig = px.box(df, x=cat_option, y=y_var, points="all")
                else:
                    fig = px.box(df, x=cat_option, y=y_var, points=False)
            else:  # Horizontale
                if points == "Outliers":
                    fig = px.box(df, y=cat_option, x=y_var, points="outliers")
                elif points == "Tous les points":
                    fig = px.box(df, y=cat_option, x=y_var, points="all")
                else:
                    fig = px.box(df, y=cat_option, x=y_var, points=False)
        else:
            if orientation == "Verticale":
                if points == "Outliers":
                    fig = px.box(df, y=y_var, points="outliers")
                elif points == "Tous les points":
                    fig = px.box(df, y=y_var, points="all")
                else:
                    fig = px.box(df, y=y_var, points=False)
            else:  # Horizontale
                if points == "Outliers":
                    fig = px.box(df, x=y_var, points="outliers")
                elif points == "Tous les points":
                    fig = px.box(df, x=y_var, points="all")
                else:
                    fig = px.box(df, x=y_var, points=False)
        
        fig.update_layout(
            title=f"Distribution de {y_var}" + (f" par {cat_option}" if cat_option != "Aucun" else ""),
            boxmode="group",
            hovermode="closest"
        )
        
        # Afficher le boxplot
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques descriptives
        st.subheader("Statistiques par groupe")
        
        if cat_option != "Aucun":
            # Calculer les statistiques par groupe
            stats_by_group = df.groupby(cat_option)[y_var].describe().reset_index()
            st.dataframe(stats_by_group.style.format({col: "{:.2f}" for col in stats_by_group.columns if col != cat_option}))
            
            # Test ANOVA si plus de 2 groupes, t-test si 2 groupes
            n_groups = df[cat_option].nunique()
            if n_groups >= 2:
                st.subheader("Test statistique")
                
                if n_groups == 2:
                    # T-test pour 2 groupes
                    groups = df[cat_option].unique()
                    group1 = df[df[cat_option] == groups[0]][y_var].dropna()
                    group2 = df[df[cat_option] == groups[1]][y_var].dropna()
                    
                    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
                    
                    st.write(f"**Test t de Welch** (variables indépendantes, variances inégales)")
                    st.write(f"Comparaison de {groups[0]} et {groups[1]}")
                    st.write(f"t = {t_stat:.4f}, p-value = {p_val:.4f}")
                    
                    if p_val < 0.05:
                        st.success(f"Différence significative entre les groupes (p < 0.05)")
                    else:
                        st.info(f"Pas de différence significative entre les groupes (p > 0.05)")
                
                else:
                    # ANOVA pour plus de 2 groupes
                    formula = f"{y_var} ~ C({cat_option})"
                    model = ols(formula, data=df).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    
                    st.write("**ANOVA à un facteur**")
                    st.dataframe(anova_table.style.format({col: "{:.4f}" for col in anova_table.columns}))
                    
                    p_val = anova_table["PR(>F)"][0]
                    if p_val < 0.05:
                        st.success(f"Différence significative entre au moins deux groupes (p < 0.05)")
                    else:
                        st.info(f"Pas de différence significative entre les groupes (p > 0.05)")
        else:
            # Statistiques globales
            stats = df[y_var].describe()
            stats_df = pd.DataFrame(stats).T
            st.dataframe(stats_df.style.format({col: "{:.2f}" for col in stats_df.columns}))

    elif chart_type == "Carte de chaleur":
        st.subheader("Carte de chaleur")
        
        # Vérifier s'il y a des colonnes numériques
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("Au moins deux colonnes numériques sont nécessaires pour créer une carte de chaleur.")
            return
        
        # Deux options : matrice de corrélation ou pivot table
        heatmap_type = st.radio("Type de carte de chaleur", ["Matrice de corrélation", "Tableau croisé (pivot)"])
        
        if heatmap_type == "Matrice de corrélation":
            # Similaire à la matrice de corrélation précédente, mais avec des options visuelles différentes
            corr_method = st.selectbox("Méthode de corrélation", ["Pearson", "Spearman"])
            
            # Sélection des variables
            selected_cols = st.multiselect(
                "Variables à inclure (laisser vide pour toutes)",
                options=numeric_cols,
                default=numeric_cols[:min(len(numeric_cols), 10)]
            )
            
            if not selected_cols:
                selected_cols = numeric_cols
            
            # Calculer la matrice
            corr_matrix = df[selected_cols].corr(method=corr_method.lower())
            
            # Options de visualisation
            color_scale = st.selectbox("Palette de couleurs", 
                                     ["RdBu_r", "Viridis", "Plasma", "YlGnBu", "YlOrRd"])
            
            # Créer la heatmap
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale=color_scale,
                labels=dict(color="Corrélation"),
                text_auto=".2f"
            )
            
            fig.update_layout(
                title=f"Matrice de corrélation ({corr_method})",
                height=700
            )
            
            # Afficher la heatmap
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # Tableau croisé (pivot)
            # Sélectionner les variables pour le pivot
            st.write("Sélectionner les variables pour le tableau croisé:")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                row_var = st.selectbox("Variable en ligne", df.columns.tolist())
            with col2:
                col_var = st.selectbox("Variable en colonne", 
                                     [col for col in df.columns if col != row_var], 
                                     index=min(1, len(df.columns)-1))
            with col3:
                if len(numeric_cols) > 0:
                    value_var = st.selectbox("Variable de valeur", 
                                           ["Comptage"] + numeric_cols)
                else:
                    value_var = "Comptage"
                    st.info("Aucune colonne numérique trouvée. Utilisation du comptage.")
            
            # Options pour la transformation des variables
            max_unique = 20  # Limite pour éviter trop de catégories
            
            if df[row_var].nunique() > max_unique and df[row_var].dtype.kind in 'ifc':
                row_transform = st.selectbox(f"Transformer {row_var} (trop de valeurs uniques)", 
                                          ["Quantiles", "Intervalles égaux"])
                if row_transform == "Quantiles":
                    n_bins_row = st.slider(f"Nombre de bins pour {row_var}", 3, 10, 5)
                    df_mod = df.copy()
                    df_mod[f"{row_var}_binned"] = pd.qcut(df_mod[row_var], q=n_bins_row, duplicates='drop')
                    row_var_mod = f"{row_var}_binned"
                else:  # Intervalles égaux
                    n_bins_row = st.slider(f"Nombre de bins pour {row_var}", 3, 10, 5)
                    df_mod = df.copy()
                    df_mod[f"{row_var}_binned"] = pd.cut(df_mod[row_var], bins=n_bins_row)
                    row_var_mod = f"{row_var}_binned"
            else:
                df_mod = df.copy()
                row_var_mod = row_var
            
            if df[col_var].nunique() > max_unique and df[col_var].dtype.kind in 'ifc':
                col_transform = st.selectbox(f"Transformer {col_var} (trop de valeurs uniques)", 
                                          ["Quantiles", "Intervalles égaux"])
                if col_transform == "Quantiles":
                    n_bins_col = st.slider(f"Nombre de bins pour {col_var}", 3, 10, 5)
                    df_mod[f"{col_var}_binned"] = pd.qcut(df_mod[col_var], q=n_bins_col, duplicates='drop')
                    col_var_mod = f"{col_var}_binned"
                else:  # Intervalles égaux
                    n_bins_col = st.slider(f"Nombre de bins pour {col_var}", 3, 10, 5)
                    df_mod[f"{col_var}_binned"] = pd.cut(df_mod[col_var], bins=n_bins_col)
                    col_var_mod = f"{col_var}_binned"
            else:
                col_var_mod = col_var
            
            # Créer le tableau pivot
            if value_var == "Comptage":
                pivot_table = pd.crosstab(df_mod[row_var_mod], df_mod[col_var_mod], normalize=st.checkbox("Normaliser (pourcentage)"))
            else:
                agg_func = st.selectbox("Fonction d'agrégation", ["Moyenne", "Somme", "Médiane", "Max", "Min"])
                agg_map = {"Moyenne": "mean", "Somme": "sum", "Médiane": "median", "Max": "max", "Min": "min"}
                
                pivot_table = pd.pivot_table(
                    df_mod, 
                    values=value_var,
                    index=row_var_mod,
                    columns=col_var_mod,
                    aggfunc=agg_map[agg_func],
                    fill_value=0
                )
            
            # Options de visualisation
            color_scale = st.selectbox("Palette de couleurs pour heatmap", 
                                     ["Viridis", "YlGnBu", "YlOrRd", "RdBu", "Plasma"])
            
            # Créer la heatmap
            fig = px.imshow(
                pivot_table,
                color_continuous_scale=color_scale,
                labels=dict(color=value_var if value_var != "Comptage" else "Fréquence"),
                text_auto=True
            )
            
            title_text = f"Carte de chaleur: {row_var} vs {col_var}"
            if value_var != "Comptage":
                title_text += f" ({agg_func.lower()} de {value_var})"
            
            fig.update_layout(
                title=title_text,
                height=700
            )
            
            # Afficher la heatmap
            st.plotly_chart(fig, use_container_width=True)
            
            # Afficher le tableau pivot
            with st.expander("Afficher le tableau de données"):
                st.dataframe(pivot_table)

    elif chart_type == "Timeseries":
        st.subheader("Analyse de séries temporelles")
        
        # Vérifier s'il y a une colonne de date
        date_cols = []
        for col in df.columns:
            # Vérifier si la colonne est déjà une date
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
            # Essayer de convertir en date
            elif df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col], errors='raise')
                    date_cols.append(col)
                except:
                    pass
        
        if not date_cols:
            st.warning("Aucune colonne de date détectée dans le jeu de données.")
            return
        
        # Sélectionner la colonne de date
        date_col = st.selectbox("Sélectionner la colonne de date", date_cols)
        
        # Convertir la colonne en datetime si ce n'est pas déjà fait
        df_time = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_time[date_col]):
            try:
                df_time[date_col] = pd.to_datetime(df_time[date_col])
                st.success(f"Colonne {date_col} convertie en format date")
            except Exception as e:
                st.error(f"Erreur lors de la conversion en date: {e}")
                return
        
        # Sélectionner les variables numériques à analyser
        numeric_cols = df_time.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            st.warning("Aucune colonne numérique trouvée pour l'analyse temporelle.")
            return
        
        value_cols = st.multiselect("Sélectionner les variables à analyser", numeric_cols)
        if not value_cols:
            st.info("Veuillez sélectionner au moins une variable à analyser.")
            return
        
        # Options de regroupement temporel
        time_group = st.selectbox("Regroupement temporel", 
                                ["Aucun", "Jour", "Semaine", "Mois", "Trimestre", "Année"])
        
        # Options d'agrégation
        agg_func = st.selectbox("Fonction d'agrégation", ["Moyenne", "Somme", "Médiane", "Max", "Min"])
        agg_map = {"Moyenne": "mean", "Somme": "sum", "Médiane": "median", "Max": "max", "Min": "min"}
        
        # Préparer les données
        if time_group != "Aucun":
            # Définir la fréquence de regroupement
            freq_map = {"Jour": "D", "Semaine": "W", "Mois": "M", "Trimestre": "Q", "Année": "Y"}
            
            # Regrouper les données
            df_time = df_time.set_index(date_col)
            df_time = df_time[value_cols].resample(freq_map[time_group]).agg(agg_map[agg_func])
            df_time = df_time.reset_index()
        
        # Options de visualisation
        col1, col2 = st.columns(2)
        with col1:
            viz_type = st.selectbox("Type de visualisation", ["Ligne", "Ligne+Marqueurs", "Aire", "Barre"])
        with col2:
            if len(value_cols) > 1:
                multi_mode = st.radio("Mode d'affichage", ["Superposé", "Séparé"])
            else:
                multi_mode = "Superposé"  # Par défaut pour une seule variable
        
        # Créer la visualisation
        if multi_mode == "Séparé" and len(value_cols) > 1:
            # Créer des sous-graphiques
            fig = make_subplots(rows=len(value_cols), cols=1, shared_xaxes=True,
                              subplot_titles=value_cols)
            
            for i, col in enumerate(value_cols):
                if viz_type == "Ligne":
                    fig.add_trace(
                        go.Scatter(x=df_time[date_col], y=df_time[col], mode='lines', name=col),
                        row=i+1, col=1
                    )
                elif viz_type == "Ligne+Marqueurs":
                    fig.add_trace(
                        go.Scatter(x=df_time[date_col], y=df_time[col], mode='lines+markers', name=col),
                        row=i+1, col=1
                    )
                elif viz_type == "Aire":
                    fig.add_trace(
                        go.Scatter(x=df_time[date_col], y=df_time[col], mode='lines', 
                                 fill='tozeroy', name=col),
                        row=i+1, col=1
                    )
                else:  # Barre
                    fig.add_trace(
                        go.Bar(x=df_time[date_col], y=df_time[col], name=col),
                        row=i+1, col=1
                    )
            
            # Mettre à jour la hauteur du graphique
            fig.update_layout(height=300 * len(value_cols))
            
        else:
            # Graphique unique avec toutes les variables
            fig = go.Figure()
            
            for col in value_cols:
                if viz_type == "Ligne":
                    fig.add_trace(go.Scatter(x=df_time[date_col], y=df_time[col], mode='lines', name=col))
                elif viz_type == "Ligne+Marqueurs":
                    fig.add_trace(go.Scatter(x=df_time[date_col], y=df_time[col], mode='lines+markers', name=col))
                elif viz_type == "Aire":
                    fig.add_trace(go.Scatter(x=df_time[date_col], y=df_time[col], mode='lines', 
                                           fill='tozeroy', name=col))
                else:  # Barre
                    fig.add_trace(go.Bar(x=df_time[date_col], y=df_time[col], name=col))
        
        # Mise à jour du layout
        title_text = f"Évolution temporelle"
        if time_group != "Aucun":
            title_text += f" ({time_group.lower()}, {agg_func.lower()})"
            
        fig.update_layout(
            title=title_text,
            xaxis_title="Date",
            yaxis_title="Valeur",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        # Afficher le graphique
        st.plotly_chart(fig, use_container_width=True)
        
        # Analyse des tendances
        if len(value_cols) == 1 and st.checkbox("Afficher l'analyse de tendance"):
            trend_var = value_cols[0]
            
            # Préparation des données pour l'analyse
            df_trend = df_time.copy()
            df_trend['time_idx'] = range(len(df_trend))
            
            # Ajuster une régression linéaire
            X = df_trend['time_idx'].values.reshape(-1, 1)
            y = df_trend[trend_var].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculer les prédictions et les résidus
            df_trend['trend'] = model.predict(X)
            df_trend['residual'] = df_trend[trend_var] - df_trend['trend']
            
            # Coefficient directeur
            slope = model.coef_[0]
            intercept = model.intercept_
            
            # Afficher les résultats
            st.subheader("Analyse de tendance")
            st.write(f"Équation de la tendance: {trend_var} = {slope:.4f} × temps + {intercept:.4f}")
            
            # Interpréter la tendance
            if slope > 0:
                st.success(f"Tendance à la hausse (+{slope:.4f} par période)")
            elif slope < 0:
                st.error(f"Tendance à la baisse ({slope:.4f} par période)")
            else:
                st.info("Pas de tendance apparente")
            
            # Graphique avec la ligne de tendance
            fig = go.Figure()
            
            # Données originales
            fig.add_trace(go.Scatter(x=df_trend[date_col], y=df_trend[trend_var], 
                                   mode='lines+markers', name='Données'))
            
            # Ligne de tendance
            fig.add_trace(go.Scatter(x=df_trend[date_col], y=df_trend['trend'], 
                                   mode='lines', name='Tendance',
                                   line=dict(color='red', dash='dash')))
            
            fig.update_layout(
                title="Données avec ligne de tendance",
                xaxis_title="Date",
                yaxis_title=trend_var,
                hovermode="closest"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Graphique des résidus
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=df_trend[date_col], y=df_trend['residual'], 
                                   mode='lines+markers', name='Résidus'))
            
            # Ligne à zéro
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title="Résidus (écarts à la tendance)",
                xaxis_title="Date",
                yaxis_title="Résidu",
                hovermode="closest"
            )
            
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Dashboard interactif":
            st.subheader("Dashboard interactif")
        
        # Obtenir les colonnes numériques
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) < 1:
            st.warning("Au moins une colonne numérique est nécessaire pour créer un dashboard.")
            return
        
        # Paramètres du dashboard
        st.write("Configurer votre dashboard interactif")
        
        # Sélection des graphiques à afficher
        graph_options = [
            "Histogramme", 
            "Boîte à moustaches", 
            "Scatter plot", 
            "Carte de chaleur", 
            "Graphique en barres"
        ]
        
        # Configuration du nombre de graphiques
        n_graphs = st.slider("Nombre de graphiques", min_value=1, max_value=4, value=2)
        
        # Conteneurs pour les graphiques
        graph_configs = []
        
        # Créer des configurations pour chaque graphique
        for i in range(n_graphs):
            st.subheader(f"Graphique {i+1}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                graph_type = st.selectbox(f"Type de graphique {i+1}", graph_options, key=f"graph_type_{i}")
            
            with col2:
                # Variables pour le graphique
                x_var = st.selectbox(f"Variable X {i+1}", 
                                    df.columns.tolist(), 
                                    key=f"x_var_{i}")
            
            with col3:
                # Sélectionner la variable Y (si nécessaire)
                if graph_type in ["Scatter plot", "Boîte à moustaches", "Histogramme"]:
                    y_var = st.selectbox(f"Variable Y {i+1}", 
                                         numeric_cols, 
                                         key=f"y_var_{i}")
                else:
                    y_var = None
            
            # Options supplémentaires
            with st.expander(f"Options avancées graphique {i+1}"):
                # Options de couleur
                color_option = st.selectbox(
                    "Coloration", 
                    ["Aucune"] + df.select_dtypes(exclude=['number']).columns.tolist(),
                    key=f"color_{i}"
                )
                
                # Filtres supplémentaires
                filter_option = st.multiselect(
                    "Filtrer par", 
                    df.columns.tolist(),
                    key=f"filter_{i}"
                )
            
            # Stocker la configuration
            graph_configs.append({
                "type": graph_type,
                "x_var": x_var,
                "y_var": y_var,
                "color_option": color_option,
                "filter_option": filter_option
            })
        
        # Bouton de génération du dashboard
            
    if st.button("Générer le Dashboard"):
            # Créer une mise en page en colonnes pour les graphiques
            cols = st.columns(n_graphs)
            
            for i, (col, config) in enumerate(zip(cols, graph_configs)):
                with col:
                    # Appliquer les filtres si nécessaire
                    df_filtered = df.copy()
                    for filter_col in config['filter_option']:
                        # Demander les valeurs à filtrer
                        unique_vals = df[filter_col].unique()
                        selected_vals = st.multiselect(
                            f"Filtrer {filter_col}", 
                            unique_vals, 
                            default=unique_vals,
                            key=f"filter_select_{i}_{filter_col}"
                        )
                        df_filtered = df_filtered[df_filtered[filter_col].isin(selected_vals)]
                    
                    # Créer le graphique selon le type sélectionné
                    if config['type'] == "Histogramme":
                        fig = px.histogram(
                            df_filtered, 
                            x=config['x_var'], 
                            y=config['y_var'], 
                            color=config['color_option'] if config['color_option'] != "Aucune" else None,
                            title=f"Histogramme de {config['x_var']}"
                        )
                    
                    elif config['type'] == "Boîte à moustaches":
                        fig = px.box(
                            df_filtered, 
                            x=config['x_var'], 
                            y=config['y_var'], 
                            color=config['color_option'] if config['color_option'] != "Aucune" else None,
                            title=f"Boîte à moustaches de {config['y_var']}"
                        )
                    
                    elif config['type'] == "Scatter plot":
                        fig = px.scatter(
                            df_filtered, 
                            x=config['x_var'], 
                            y=config['y_var'], 
                            color=config['color_option'] if config['color_option'] != "Aucune" else None,
                            title=f"Scatter plot de {config['x_var']} vs {config['y_var']}"
                        )
                    
                    elif config['type'] == "Carte de chaleur":
                        # Pour la carte de chaleur, on va utiliser un tableau pivot
                        pivot_table = pd.pivot_table(
                            df_filtered, 
                            values=numeric_cols[0] if numeric_cols else None,
                            index=config['x_var'],
                            columns=config['color_option'] if config['color_option'] != "Aucune" else None,
                            aggfunc='mean'
                        )
                        
                        fig = px.imshow(
                            pivot_table,
                            labels=dict(color="Valeur moyenne"),
                            title=f"Carte de chaleur: {config['x_var']}"
                        )
                    
                    elif config['type'] == "Graphique en barres":
                        fig = px.bar(
                            df_filtered, 
                            x=config['x_var'], 
                            y=numeric_cols[0] if numeric_cols else None,
                            color=config['color_option'] if config['color_option'] != "Aucune" else None,
                            title=f"Graphique en barres de {config['x_var']}"
                        )
                    
                    # Afficher le graphique
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Bouton pour générer l'analyse Mistral
                    if st.button(f"Analyser le graphique {i+1} avec Mistral", key=f"analyze_graph_{i}"):
                        analysis = generate_graph_description(
                            graph_type=config['type'], 
                            x_var=config['x_var'], 
                            y_var=config['y_var'], 
                            df=df_filtered
                        )
                        
                        st.write(f"### Analyse Mistral du graphique {i+1}")
                        st.write(analysis)
            
            # Section informative
            st.info("""
            🔍 Guide du Dashboard Interactif :
            - Sélectionnez jusqu'à 4 graphiques
            - Choisissez le type de graphique et les variables
            - Utilisez les options avancées pour filtrer et colorier
            - Cliquez sur 'Générer le Dashboard' pour visualiser
            """)

