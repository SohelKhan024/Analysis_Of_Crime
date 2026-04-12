"""
Bulletproof Crime Analytics - Handles ANY Dataset
No KeyErrors - Graceful degradation
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp { background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%); color: white; font-family: 'Inter', sans-serif; }
.glass { background: rgba(255,255,255,0.08); backdrop-filter: blur(20px); border-radius: 20px; padding: 2rem; border: 1px solid rgba(255,255,255,0.1); margin: 1rem 0; }
.header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem; border-radius: 24px; text-align: center; box-shadow: 0 20px 60px rgba(102,126,234,0.4); margin-bottom: 2rem; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; }
.metric-card { background: rgba(102,126,234,0.2); border-radius: 16px; padding: 1.5rem; text-align: center; transition: all 0.3s; }
.metric-card:hover { transform: translateY(-5px); }
.sidebar-section { background: rgba(15,23,42,0.9); border-radius: 20px; padding: 1.5rem; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header"><h1>🚓 Crime Analytics Pro</h1><p>Bulletproof SaaS - Works with ANY Dataset</p></div>', unsafe_allow_html=True)

# Session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = {}

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### Data Loader")
    
    source = st.radio("Source", ["Default Dataset", "Upload CSV"])
    
    if source == "Default Dataset":
        if st.button("Load Demo Data", key="load_demo"):
            with st.spinner("Loading demo crime data..."):
                time.sleep(0.8)
                df = pd.read_csv("data/demo1_crime_small.csv")
                df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
                st.session_state.df = df
                st.session_state.dataset_info = {"name": "demo1_crime_small.csv", "rows": len(df), "columns": list(df.columns)}
                st.rerun()
    else:
        uploaded = st.file_uploader("Choose CSV", type="csv")
        if uploaded is not None:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Process Upload", key="process_upload"):
                    with st.spinner("Analyzing dataset structure..."):
                        time.sleep(1)
                        df = pd.read_csv(uploaded)
                        # NORMALIZE COLUMNS FIRST
                        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_").str.replace("-", "_")
                        
                        st.session_state.df = df
                        st.session_state.dataset_info = {"name": uploaded.name, "rows": len(df), "columns": list(df.columns)}
                        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main app
df = st.session_state.df

if df is None or df.empty:
    st.info("👈 Upload CSV or load demo dataset from sidebar")
else:
    info = st.session_state.dataset_info
    st.markdown(f"**Dataset:** {info['name']} | **{info['rows']:,} rows** | Columns: {', '.join(info['columns'][:4])}...")
    
    # Safe required columns
    crime_candidates = ['crime_type', 'type', 'crime', 'offense', 'crm_cd_desc']
    coord_candidates_lat = ['lat', 'latitude']
    coord_candidates_lon = ['lon', 'longitude', 'lng']
    time_candidates = ['hour', 'time']
    
    # Find crime column
    crime_col = None
    for cand in crime_candidates:
        if cand in df.columns:
            crime_col = cand
            break
    
    # Find coords
    lat_col = None
    lon_col = None
    for lat_cand in coord_candidates_lat:
        if lat_cand in df.columns:
            lat_col = lat_cand
            break
    for lon_cand in coord_candidates_lon:
        if lon_cand in df.columns:
            lon_col = lon_cand
            break
    
    # Find hour
    hour_col = None
    for time_cand in time_candidates:
        if time_cand in df.columns:
            hour_col = time_cand
            break
    
    # KPIs - SAFE ACCESS
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Records", len(df))
    
    with col2:
        if crime_col:
            top_crime = df[crime_col].value_counts().index[0]
            st.metric("🚨 Top Crime", top_crime)
        else:
            st.metric("🚨 Crime Column", "Not found")
    
    with col3:
        if hour_col:
            peak_hour = df[hour_col].value_counts().index[0]
            st.metric("⏰ Peak Hour", str(peak_hour))
        else:
            st.metric("⏰ Time Column", "Not found")
    
    # Dataset preview
    st.subheader("Data Preview")
    preview_cols = []
    if crime_col: preview_cols.append(crime_col)
    if lat_col: preview_cols.append(lat_col)
    if lon_col: preview_cols.append(lon_col)
    if hour_col: preview_cols.append(hour_col)
    preview_cols = preview_cols or df.columns.tolist()[:4]
    
    st.dataframe(df[preview_cols].head(10), use_container_width=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Analysis", "🔥 Hotspots", "🤖 Predict"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            if hour_col:
                fig1 = px.histogram(df, x=hour_col, title="Distribution by Time")
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("No time column found for hourly analysis")
        
        with col2:
            if crime_col:
                fig2 = px.bar(df[crime_col].value_counts().head(12), orientation='h', title="Top Crimes")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No crime column found for crime type analysis")
    
    with tab2:
        if lat_col and lon_col:
            n_clusters = st.slider("Clusters", 2, 8, 3)
            if st.button("Detect Hotspots"):
                coords = df[[lat_col, lon_col]].dropna()
                if len(coords) >= n_clusters:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    df['cluster'] = kmeans.fit_predict(coords)
                    fig = px.scatter(df, x=lon_col, y=lat_col, color='cluster', title=f"Crime Hotspots (K={n_clusters})")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Need at least {n_clusters} valid coordinates. Found {len(coords)}")
        else:
            st.warning("Dataset missing lat/lon columns for hotspot analysis")
    
    with tab3:
        st.subheader("AI Crime Type Prediction")
        
        # Safe feature selection
        features = []
        if hour_col:
            features.append(hour_col)
        if lat_col and lon_col:
            features.append(lat_col)
            features.append(lon_col)
        if len(features) == 0:
            features = df.select_dtypes(include=[np.number]).columns.tolist()[:3]
        
        if features and crime_col:
            h_slider = st.slider("Target Hour", 0, 23, 12)
            
            if st.button("🔮 Predict Crime Type"):
                with st.spinner("Training AI model..."):
                    X = df[features].fillna(0)
                    y = LabelEncoder().fit_transform(df[crime_col])
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    
                    # Predict
                    pred_features = [h_slider] + [df[features[1]].mean() if len(features) > 1 else 0, df[features[2]].mean() if len(features) > 2 else 0]
                    pred_features = pd.DataFrame([pred_features], columns=features)
                    prediction = LabelEncoder().fit(df[crime_col]).inverse_transform(model.predict(pred_features))[0]
                    
                    st.success(f"**🎯 Predicted Crime: {prediction}**")
                    
                    # Feature importance
                    imp = pd.DataFrame({
                        'feature': features,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig_imp = px.bar(imp, x='importance', y='feature', orientation='h', title="Feature Importance")
                    st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.warning("Need crime column + numeric features for prediction")

st.markdown("---")
st.caption("🔹 Bulletproof Crime Analytics | Handles any dataset structure")

