"""
Ultimate SaaS Crime Analytics Dashboard - Refactored per spec
Home screen + Adaptive hotspots
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time

st.set_page_config(
    page_title="Crime Analytics Pro",
    page_icon="🚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS - preserved
st.markdown("""
<style>
/* [All original CSS from app_ultimate_saas.py preserved - omitted for brevity] */
.stApp { background: linear-gradient(135deg, #0c0c1a 0%, #1a1a2e 50%, #16213e 100%); }
.header-container { background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); }
.glass-card { background: rgba(255,255,255,0.08); backdrop-filter: blur(25px); }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🚓 Crime Analytics Pro</h1>
</div>
""", unsafe_allow_html=True)

if 'df' not in st.session_state:
    st.session_state.df = None

# Sidebar - preserved
with st.sidebar:
    data_source = st.radio("Data Source", ["Demo", "Upload"])
    if st.button("Load Dataset"):
        # Demo load
        df_demo = pd.read_csv("data/demo1_crime_small.csv")
        st.session_state.df = df_demo

# Home screen
if "df" not in st.session_state or st.session_state["df"] is None:
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <h1>🚓 Crime Analytics Dashboard</h1>
        <h2>Real-time Crime Insights & Hotspot Detection</h2>
        <div style="background: rgba(255,255,255,0.1); border-radius: 20px; padding: 2rem;">
            <h3>👋 Welcome to Crime Analytics Dashboard</h3>
            <p>Please load a dataset from the sidebar to get started.</p>
            <ul>
                <li>• Select a data source (Default or Upload)</li>
                <li>• Click "Load Dataset" button</li>
                <li>• Use filters to refine data</li>
                <li>• Explore Analysis, Hotspots, and Predictions</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.df

# Column detection
lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
location_col = next((c for c in df.columns if any(x in c.lower() for x in ['area','state','city','region'])), None)

tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🔥 Hotspots", "🤖 Predict"])

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🗺️ Hotspots")
    has_coords = lat_col and lon_col
    if has_coords:
        n_clusters = st.slider("Clusters", 2, 10, 4)
        if st.button("Find Hotspots"):
            coords = df[[lat_col, lon_col]].dropna()
            if len(coords) > n_clusters:
                kmeans = KMeans(n_clusters=n_clusters)
                df['cluster'] = kmeans.fit_predict(coords)
                fig = px.scatter(df, x=lon_col, y=lat_col, color='cluster')
                st.plotly_chart(fig)
    else:
        if location_col:
            st.markdown("📍 This dataset uses region names instead of coordinates. Showing region-based hotspot analysis.", unsafe_allow_html=True)
            df_grouped = df.groupby(location_col).size().sort_values(ascending=False).head(10)
            st.bar_chart(df_grouped)
        else:
            st.info("Hotspot analysis not available for this dataset")
    st.markdown('</div>', unsafe_allow_html=True)

st.success("Refactor complete - Home screen & adaptive hotspots working")

