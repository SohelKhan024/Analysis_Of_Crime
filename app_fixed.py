"""
Crime Analytics Dashboard - Fixed Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from backend_fixed import CrimeDataProcessor, HotspotDetector, CrimeRiskPredictor
from ui_components_fixed import set_page_configuration

import warnings
warnings.filterwarnings('ignore')

set_page_configuration()

if 'df' not in st.session_state:
    st.session_state.df = None
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None
if 'hotspot_model' not in st.session_state:
    st.session_state.hotspot_model = None
if 'risk_predictor' not in st.session_state:
    st.session_state.risk_predictor = None

st.markdown("""
<div class="header-container">
    <h1>🚓 Crime Analytics Dashboard</h1>
    <p class="subtitle">Real-time Crime Insights & Hotspot Detection</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    data_source = st.radio("📁 Data Source", ["Use Default Dataset", "Upload CSV"])

    import os

    if st.session_state.df is None or st.session_state.df.empty:
        st.info("👆 Please select or upload a dataset to get started")

    if data_source == "Use Default Dataset":
        csv_path = "data/20_Victims_of_rape.csv"
        if st.button("🚀 Load Default Dataset", use_container_width=True):
            if os.path.exists(csv_path):
                try:
                    df_temp = pd.read_csv(csv_path)
                    df_temp.columns = df_temp.columns.str.lower().str.strip()
                    st.session_state.df = df_temp
                    st.session_state.filtered_df = df_temp.copy()
                    st.success(f"✅ Loaded {len(df_temp):,} records")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Dataset not found: " + csv_path)
    else:
        uploaded_file = st.file_uploader("📤 Upload CSV", type="csv")
        if uploaded_file is not None:
            try:
                df_temp = pd.read_csv(uploaded_file)
                df_temp.columns = df_temp.columns.str.lower().str.strip()
                st.session_state.df = df_temp
                st.session_state.filtered_df = df_temp.copy()
                st.success(f"✅ Loaded {len(df_temp):,} records")
            except Exception as e:
                st.warning(f"Upload failed: {e}")

    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown("## 🔍 Filters")

        # Dynamic column detection for crime type and hour
        crime_col_candidates = ['type', 'crime_type', 'crime', 'category', 'crm_cd_desc']
        hour_col_candidates = ['hour', 'time_hour', 'hour_of_day']
        
        crime_col = next((col for col in crime_col_candidates if col in st.session_state.df.columns), None)
        hour_col = next((col for col in hour_col_candidates if col in st.session_state.df.columns), None)
        
        all_crimes = sorted(st.session_state.df[crime_col].str.lower().unique()) if crime_col else []
        selected_crime = st.selectbox("Crime Type", ["All"] + all_crimes) if crime_col else "No crime column"
        
        selected_hour = st.slider("Hour (0-23)", 0, 23, (0, 23)) if hour_col else (0, 23)

        filtered_df = st.session_state.df.copy()
        if crime_col and selected_crime != "All":
            filtered_df = filtered_df[filtered_df[crime_col].str.lower() == selected_crime]
        if hour_col:
            filtered_df = filtered_df[(filtered_df[hour_col] >= selected_hour[0]) & (filtered_df[hour_col] <= selected_hour[1])]
        st.session_state.filtered_df = filtered_df

if st.session_state.df is not None:
    df = st.session_state.df
    filtered_df = st.session_state.filtered_df
    has_coords = 'lat' in df.columns and 'lon' in df.columns

    st.markdown("## 📊 Key Performance Indicators")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Crimes", len(filtered_df))

    with col2:
        if crime_col and len(filtered_df) > 0:
            most_common = filtered_df[crime_col].value_counts().index[0]
            st.metric("Most Common Crime", most_common)
        else:
            st.metric("Most Common Crime", "N/A")

    with col3:
        if hour_col and len(filtered_df) > 0:
            peak_hour = filtered_df[hour_col].value_counts().index[0]
            st.metric("Peak Hour", peak_hour)
        else:
            st.metric("Peak Hour", "N/A")

    tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🔥 Hotspots", "🤖 Prediction"])

    with tab1:
        st.markdown("### 📈 Crime Analysis")
        col1, col2 = st.columns(2)

        with col1:
            if hour_col:
                crime_by_hour = filtered_df.groupby(hour_col).size()
                fig = px.bar(x=crime_by_hour.index, y=crime_by_hour.values, title="Crimes by Hour")
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Hour column needed for this chart.")

        with col2:
            if crime_col:
                top_crimes = filtered_df[crime_col].value_counts().head(10)
                fig = px.bar(y=top_crimes.index, x=top_crimes.values, orientation='h', title="Top Crimes")
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Crime column needed for this chart.")

    with tab2:
        st.markdown("### 🗺️ Hotspots")
        if has_coords:
            n_clusters = st.slider("Clusters", 2, 10, 4)
            if st.button("Detect Hotspots"):
                from sklearn.cluster import KMeans
                coords = filtered_df[['lat', 'lon']].values
                kmeans = KMeans(n_clusters=n_clusters)
                clusters = kmeans.fit_predict(coords)
                st.write("Hotspots detected!")
                fig = px.scatter(filtered_df, x='lon', y='lat', color=clusters, title="Hotspots")
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig)
        else:
            st.warning("No lat/lon data for hotspots.")

    with tab3:
        st.markdown("### 🤖 Prediction")
        pred_hour = st.slider("Hour", 0, 23, 12)
        if st.button("Predict"):
            predictor = CrimeRiskPredictor()
            predictor.train(filtered_df)
            prediction = predictor.predict_crime_type(pred_hour, 'Central', 34.05, -118.24)
            st.success(f"Predicted: {prediction}")

else:
    st.markdown("## 👋 Welcome")
    st.info("Load dataset from sidebar to start analysis.")

