import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Analyse Interactive des Indicateurs Économiques",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("📊 Analyse Interactive des Indicateurs Économiques")
st.markdown("### Visualisation, Analyse et Modélisation des Données de la Banque Mondiale")

# Cache pour optimiser les requêtes API
@st.cache_data(ttl=3600)
def get_world_bank_data(indicator, countries, start_year, end_year):
    """Récupère les données de la Banque Mondiale via API"""
    
    # Construire l'URL de l'API
    countries_str = ";".join(countries)
    url = f"https://api.worldbank.org/v2/country/{countries_str}/indicator/{indicator}"
    
    params = {
        'format': 'json',
        'date': f"{start_year}:{end_year}",
        'per_page': 1000,
        'page': 1
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if len(data) < 2 or not data[1]:
            return pd.DataFrame()
        
        # Convertir en DataFrame
        df = pd.DataFrame(data[1])
        df['date'] = pd.to_numeric(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['value'])
        
        # Renommer les colonnes
        df = df.rename(columns={
            'date': 'Année',
            'value': 'Valeur',
            'country': 'Pays'
        })
        
        # Extraire le nom du pays
        df['Pays'] = df['Pays'].apply(lambda x: x['value'] if isinstance(x, dict) else x)
        
        return df[['Année', 'Pays', 'Valeur']].sort_values(['Pays', 'Année'])
        
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données : {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_countries_list():
    """Récupère la liste des pays disponibles"""
    try:
        url = "https://api.worldbank.org/v2/country?format=json&per_page=300"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        countries = []
        for country in data[1]:
            if country['region']['value'] != 'Aggregates':
                countries.append({
                    'code': country['id'],
                    'name': country['name']
                })
        
        return sorted(countries, key=lambda x: x['name'])
    except:
        # Liste de pays par défaut si l'API ne fonctionne pas
        return [
            {'code': 'USA', 'name': 'États-Unis'},
            {'code': 'FRA', 'name': 'France'},
            {'code': 'DEU', 'name': 'Allemagne'},
            {'code': 'JPN', 'name': 'Japon'},
            {'code': 'GBR', 'name': 'Royaume-Uni'},
            {'code': 'CHN', 'name': 'Chine'},
            {'code': 'BRA', 'name': 'Brésil'},
            {'code': 'IND', 'name': 'Inde'},
            {'code': 'CAN', 'name': 'Canada'},
            {'code': 'AUS', 'name': 'Australie'}
        ]

def calculate_statistics(df):
    """Calcule les statistiques descriptives"""
    stats_dict = {}
    
    for country in df['Pays'].unique():
        country_data = df[df['Pays'] == country]['Valeur']
        
        stats_dict[country] = {
            'Moyenne': country_data.mean(),
            'Médiane': country_data.median(),
            'Écart-type': country_data.std(),
            'Min': country_data.min(),
            'Max': country_data.max(),
            'Tendance': 'Croissante' if country_data.iloc[-1] > country_data.iloc[0] else 'Décroissante',
            'Variation (%)': ((country_data.iloc[-1] - country_data.iloc[0]) / country_data.iloc[0] * 100) if country_data.iloc[0] != 0 else 0
        }
    
    return stats_dict

def perform_regression_analysis(df, degree=1):
    """Effectue une analyse de régression"""
    results = {}
    
    for country in df['Pays'].unique():
        country_data = df[df['Pays'] == country].copy()
        
        if len(country_data) < 3:
            continue
            
        X = country_data['Année'].values.reshape(-1, 1)
        y = country_data['Valeur'].values
        
        # Régression polynomiale
        if degree > 1:
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(X)
            model = LinearRegression().fit(X_poly, y)
            predictions = model.predict(X_poly)
        else:
            model = LinearRegression().fit(X, y)
            predictions = model.predict(X)
        
        # Calcul des métriques
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        
        # Prédictions futures
        future_years = np.arange(country_data['Année'].max() + 1, country_data['Année'].max() + 6).reshape(-1, 1)
        if degree > 1:
            future_pred = model.predict(poly_features.transform(future_years))
        else:
            future_pred = model.predict(future_years)
        
        results[country] = {
            'model': model,
            'r2': r2,
            'mse': mse,
            'predictions': predictions,
            'future_years': future_years.flatten(),
            'future_predictions': future_pred,
            'poly_features': poly_features if degree > 1 else None
        }
    
    return results

def create_correlation_matrix(df):
    """Crée une matrice de corrélation entre pays"""
    pivot_df = df.pivot(index='Année', columns='Pays', values='Valeur')
    correlation_matrix = pivot_df.corr()
    
    fig = px.imshow(
        correlation_matrix,
        title="Matrice de Corrélation entre Pays",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    
    return fig

# Interface utilisateur
st.sidebar.header("⚙️ Configuration")

# Indicateurs économiques populaires
indicators = {
    'PIB ($ US courants)': 'NY.GDP.MKTP.CD',
    'PIB par habitant ($ US courants)': 'NY.GDP.PCAP.CD',
    'Croissance du PIB (% annuel)': 'NY.GDP.MKTP.KD.ZG',
    'Inflation (%)': 'FP.CPI.TOTL.ZG',
    'Chômage (% de la population active)': 'SL.UEM.TOTL.ZS',
    'Population totale': 'SP.POP.TOTL',
    'Espérance de vie': 'SP.DYN.LE00.IN',
    'Dépenses de santé (% du PIB)': 'SH.XPD.CHEX.GD.ZS',
    'Dépenses d\'éducation (% du PIB)': 'SE.XPD.TOTL.GD.ZS',
    'Émissions de CO2 (tonnes métriques par habitant)': 'EN.ATM.CO2E.PC'
}

# Sélection de l'indicateur
selected_indicator_name = st.sidebar.selectbox(
    "📈 Sélectionnez un indicateur économique",
    list(indicators.keys())
)
selected_indicator = indicators[selected_indicator_name]

# Récupération des pays
countries_list = get_countries_list()
country_names = [country['name'] for country in countries_list]
country_codes = {country['name']: country['code'] for country in countries_list}

# Sélection des pays
selected_countries_names = st.sidebar.multiselect(
    "🌍 Sélectionnez les pays",
    country_names,
    default=['États-Unis', 'France', 'Allemagne'] if 'États-Unis' in country_names else country_names[:3]
)

selected_countries = [country_codes[name] for name in selected_countries_names]

# Sélection de la période
col1, col2 = st.sidebar.columns(2)
with col1:
    start_year = st.number_input("Année de début", min_value=1960, max_value=2023, value=2000)
with col2:
    end_year = st.number_input("Année de fin", min_value=1960, max_value=2023, value=2022)

# Bouton pour charger les données
if st.sidebar.button("🔄 Charger les données", type="primary"):
    with st.spinner("Chargement des données..."):
        df = get_world_bank_data(selected_indicator, selected_countries, start_year, end_year)
        st.session_state['df'] = df

# Affichage des données si disponibles
if 'df' in st.session_state and not st.session_state['df'].empty:
    df = st.session_state['df']
    
    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualisation", "📈 Analyse", "🔮 Modélisation", "📋 Données"])
    
    with tab1:
        st.subheader(f"Visualisation : {selected_indicator_name}")
        
        # Graphique linéaire interactif
        fig = px.line(
            df, 
            x='Année', 
            y='Valeur', 
            color='Pays',
            title=f"Évolution de {selected_indicator_name} ({start_year}-{end_year})",
            markers=True
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Graphique en barres pour comparaison
        col1, col2 = st.columns(2)
        
        with col1:
            # Dernières valeurs
            latest_data = df.groupby('Pays')['Valeur'].last().reset_index()
            fig_bar = px.bar(
                latest_data,
                x='Pays',
                y='Valeur',
                title=f"Valeurs les plus récentes ({df['Année'].max()})"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Variation en pourcentage
            variation_data = []
            for country in df['Pays'].unique():
                country_data = df[df['Pays'] == country].sort_values('Année')
                if len(country_data) >= 2:
                    first_val = country_data.iloc[0]['Valeur']
                    last_val = country_data.iloc[-1]['Valeur']
                    variation = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
                    variation_data.append({'Pays': country, 'Variation (%)': variation})
            
            if variation_data:
                var_df = pd.DataFrame(variation_data)
                fig_var = px.bar(
                    var_df,
                    x='Pays',
                    y='Variation (%)',
                    title=f"Variation totale ({start_year}-{end_year})",
                    color='Variation (%)',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_var, use_container_width=True)
        
        # Matrice de corrélation
        if len(selected_countries) > 1:
            st.subheader("🔗 Analyse de Corrélation")
            correlation_fig = create_correlation_matrix(df)
            st.plotly_chart(correlation_fig, use_container_width=True)
    
    with tab2:
        st.subheader(f"Analyse Statistique : {selected_indicator_name}")
        
        # Statistiques descriptives
        stats = calculate_statistics(df)
        
        # Affichage des statistiques
        stats_df = pd.DataFrame(stats).T
        st.dataframe(stats_df.round(2), use_container_width=True)
        
        # Graphique de distribution
        fig_dist = px.box(
            df,
            x='Pays',
            y='Valeur',
            title="Distribution des valeurs par pays"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Analyse de tendance
        st.subheader("📈 Analyse de Tendance")
        for country in df['Pays'].unique():
            country_data = df[df['Pays'] == country].sort_values('Année')
            if len(country_data) >= 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    country_data['Année'], country_data['Valeur']
                )
                
                trend = "Croissante" if slope > 0 else "Décroissante"
                significance = "Significative" if p_value < 0.05 else "Non significative"
                
                st.write(f"**{country}** : Tendance {trend} (R² = {r_value**2:.3f}, p = {p_value:.3f}) - {significance}")
    
    with tab3:
        st.subheader(f"Modélisation et Prédictions : {selected_indicator_name}")
        
        # Paramètres de modélisation
        col1, col2 = st.columns(2)
        with col1:
            degree = st.selectbox("Degré de polynôme", [1, 2, 3], index=0)
        with col2:
            show_predictions = st.checkbox("Afficher les prédictions futures", value=True)
        
        # Analyse de régression
        regression_results = perform_regression_analysis(df, degree)
        
        if regression_results:
            # Graphique avec prédictions
            fig = go.Figure()
            
            for country in regression_results.keys():
                country_data = df[df['Pays'] == country]
                
                # Données réelles
                fig.add_trace(go.Scatter(
                    x=country_data['Année'],
                    y=country_data['Valeur'],
                    mode='markers',
                    name=f"{country} (Réel)",
                    marker=dict(size=8)
                ))
                
                # Prédictions sur les données existantes
                fig.add_trace(go.Scatter(
                    x=country_data['Année'],
                    y=regression_results[country]['predictions'],
                    mode='lines',
                    name=f"{country} (Modèle)",
                    line=dict(dash='dash')
                ))
                
                # Prédictions futures
                if show_predictions:
                    fig.add_trace(go.Scatter(
                        x=regression_results[country]['future_years'],
                        y=regression_results[country]['future_predictions'],
                        mode='lines+markers',
                        name=f"{country} (Prédiction)",
                        line=dict(dash='dot'),
                        marker=dict(symbol='diamond')
                    ))
            
            fig.update_layout(
                title=f"Modélisation et Prédictions - {selected_indicator_name}",
                xaxis_title="Année",
                yaxis_title="Valeur",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Métriques de performance
            st.subheader("📊 Performance des Modèles")
            
            performance_data = []
            for country, results in regression_results.items():
                performance_data.append({
                    'Pays': country,
                    'R² Score': results['r2'],
                    'MSE': results['mse'],
                    'Qualité': 'Excellente' if results['r2'] > 0.8 else 'Bonne' if results['r2'] > 0.6 else 'Moyenne'
                })
            
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df.round(4), use_container_width=True)
            
            # Prédictions futures en tableau
            if show_predictions:
                st.subheader("🔮 Prédictions Futures")
                future_data = []
                for country, results in regression_results.items():
                    for year, pred in zip(results['future_years'], results['future_predictions']):
                        future_data.append({
                            'Pays': country,
                            'Année': year,
                            'Prédiction': pred
                        })
                
                future_df = pd.DataFrame(future_data)
                pivot_future = future_df.pivot(index='Année', columns='Pays', values='Prédiction')
                st.dataframe(pivot_future.round(2), use_container_width=True)
    
    with tab4:
        st.subheader("📋 Données Brutes")
        
        # Affichage des données
        st.dataframe(df, use_container_width=True)
        
        # Statistiques générales
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre de pays", len(df['Pays'].unique()))
        with col2:
            st.metric("Période", f"{df['Année'].min()} - {df['Année'].max()}")
        with col3:
            st.metric("Observations", len(df))
        
        # Téléchargement des données
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger les données (CSV)",
            data=csv,
            file_name=f"{selected_indicator_name}_{start_year}_{end_year}.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Configurez les paramètres dans la barre latérale et cliquez sur 'Charger les données' pour commencer l'analyse.")
    
    # Affichage des indicateurs disponibles
    st.subheader("📊 Indicateurs Économiques Disponibles")
    
    indicators_df = pd.DataFrame([
        {"Indicateur": name, "Code": code} 
        for name, code in indicators.items()
    ])
    
    st.dataframe(indicators_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>📊 Application d'Analyse des Indicateurs Économiques | Données : Banque Mondiale</p>
        <p>Développé avec Python, Streamlit et Plotly</p>
    </div>
    """,
    unsafe_allow_html=True
)