"""
Complete Working Crime Analytics Dashboard
No external dependencies with syntax errors
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

st.set_page_config(page_title="Crime Dashboard", layout="wide")

if 'df' not in st.session_state:
    st.session_state.df = None

st.markdown("""
<style>
.stApp { background-color: #0e1117; }
.header-container { text-align: center; padding: 2rem; background: linear-gradient(135deg, #1f6feb 0%, #58A6FF 100%); border-radius: 15px; color: white; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(31, 111, 235, 0.3); }
.subtitle { font-size: 1.2rem; opacity: 0.9; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-container"><h1>🚓 Crime Analytics Dashboard</h1><p class="subtitle">Interactive Analysis & Prediction</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Data")
    
    if st.button("📥 Load Default Data"):
        csv_path = "data/20_Victims_of_rape.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.lower().str.strip()
            st.session_state.df = df
            st.success(f"Loaded {len(df)} records")
        else:
            st.error("CSV not found")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        df.columns = df.columns.str.lower().str.strip()
        st.session_state.df = df
        st.success("Data uploaded")

if st.session_state.df is not None:
    df = st.session_state.df.copy()
    
    # Features
    df['hour'] = pd.to_numeric(df.get('hour', 12))
    df['lat'] = pd.to_numeric(df.get('lat', 34.05))
    df['lon'] = pd.to_numeric(df.get('lon', -118.24))
    df['crm_cd_desc'] = df.get('type', 'Unknown')
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Records", len(df))
    with col2:
        top = df['crm_cd_desc'].value_counts().iloc[0]
        st.metric("Top Crime", top)

    tab1, tab2, tab3 = st.tabs(["📊 Charts", "🔥 Hotspots", "🤖 Predict"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.bar(df.groupby('hour').size(), title="Crimes by Hour")
            fig1.update_layout(template="plotly_dark")
            st.plotly_chart(fig1)
        
        with col2:
            fig2 = px.bar(df['crm_cd_desc'].value_counts().head(10), orientation='h', title="Top Crimes")
            fig2.update_layout(template="plotly_dark")
            st.plotly_chart(fig2)

    with tab2:
        if st.button("Detect Hotspots"):
            coords = df[['lat', 'lon']].values
            kmeans = KMeans(n_clusters=4)
            clusters = kmeans.fit_predict(coords)
            df['cluster'] = clusters
            fig = px.scatter(df, x='lon', y='lat', color='cluster', title="Hotspots")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig)

    with tab3:
        if st.button("Train & Predict"):
            X = df[['hour', 'lat', 'lon']]
            y = LabelEncoder().fit_transform(df['crm_cd_desc'])
            model = RandomForestClassifier()
            model.fit(X, y)
            pred = model.predict([[12, 34.05, -118.24]])[0]
            st.success("Prediction ready!")
            st.bar_chart(model.feature_importances_)

else:
    st.info("Load data from sidebar")

