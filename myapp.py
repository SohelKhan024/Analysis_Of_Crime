"""
Ultimate SaaS Crime Analytics Dashboard - FIXED UI + Logic
Premium design preserved + Home screen + Adaptive hotspots
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time
from io import StringIO
import csv
import os


def load_csv_robust(uploaded_file):
    """Safe CSV loader"""
    try:
        raw_data = uploaded_file.read()
        try:
            text = raw_data.decode("utf-8")
        except:
            text = raw_data.decode("latin-1")
        uploaded_file.seek(0)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(text[:5000])
        df = pd.read_csv(StringIO(text), delimiter=dialect.delimiter)
        return df
    except:
        uploaded_file.seek(0)
        try:
            df = pd.read_excel(uploaded_file)
            return df
        except:
            return None

st.set_page_config(
    page_title="Crime Analytics Pro",
    page_icon="🚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PREMIUM CSS - FULL ORIGINAL
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
.stApp { background: linear-gradient(135deg, #0c0c1a 0%, #1a1a2e 50%, #16213e 100%); font-family: 'Inter', sans-serif; }
.header-container { background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); padding: 4rem 2rem; border-radius: 32px; text-align: center; margin: 1.5rem 0 3rem 0; box-shadow: 0 35px 80px rgba(102,126,234,0.4); color: white; }
.header-title { font-size: 4.2rem; font-weight: 800; margin: 0; }
.header-subtitle { font-size: 1.5rem; opacity: 0.95; margin: 1rem 0 0 0; }
.glass-card { background: rgba(255,255,255,0.08); backdrop-filter: blur(25px); border: 1px solid rgba(255,255,255,0.15); border-radius: 24px; padding: 2.5rem; margin: 1.5rem 0; box-shadow: 0 25px 60px rgba(0,0,0,0.4); }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 2rem; }
.kpi-card { background: linear-gradient(145deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2)); border-radius: 24px; padding: 2.5rem 2rem; text-align: center; }
.sidebar-panel { background: rgba(15,23,42,0.95); backdrop-filter: blur(20px); border-radius: 24px; padding: 2rem; }
</style>
""", unsafe_allow_html=True)

# HEADER - UNCHANGED
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🚓 Crime Analytics Pro</h1>
    <p class="header-subtitle" id="status-subtitle">Ready for Analysis</p>
</div>
""", unsafe_allow_html=True)

# Session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'dataset_status' not in st.session_state:
    st.session_state.dataset_status = "No dataset loaded"

# Sidebar - UNCHANGED
with st.sidebar:
    st.markdown('<div class="sidebar-panel">', unsafe_allow_html=True)
    st.markdown("### Data Loader")
    data_source = st.radio("Source", ["Demo Dataset", "Upload CSV"])
    
    if data_source == "Demo Dataset":
        if st.button("Load Demo"):
            with st.spinner("Loading demo dataset..."):
                try:
                    demo_path = "data/demo1_crime_small.csv"
                    if not os.path.exists(demo_path):
                        st.error(f"Demo file missing: {demo_path}")
                        st.stop()
                    
                    df_demo = pd.read_csv(demo_path, encoding='utf-8', low_memory=False)
                    if df_demo.empty or len(df_demo) == 0:
                        raise ValueError("Demo dataset is empty")
                    
                    st.session_state.df = df_demo.copy()
                    st.session_state.dataset_status = f"✅ Demo loaded | {len(df_demo):,} rows, {len(df_demo.columns)} cols"
                    st.success(f"✅ Demo dataset loaded successfully! {len(df_demo):,} records")
                    st.rerun()
                except Exception as e:
                    st.error(f"Demo load failed: {str(e)}")
                    st.info("File exists but check encoding/permissions. Try upload CSV instead.")
    else:
        uploaded = st.file_uploader("CSV", type="csv")
        if uploaded and st.button("Process"):
            df_upload = load_csv_robust(uploaded)
            if df_upload is not None:
                st.session_state.df = df_upload
                st.session_state.dataset_status = f"Uploaded | {len(df_upload)} records"
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# HOME SCREEN - FIXED in glass-card, preserves header
if "df" not in st.session_state or st.session_state["df"] is None:
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <h2>👋 Welcome to Crime Analytics Dashboard</h2>
        <p style="font-size: 1.3rem;">Please load a dataset from the sidebar to get started.</p>
        <ul style="text-align: left; font-size: 1.1rem;">
            <li>Select a data source (Default or Upload)</li>
            <li>Click "Load Dataset" button</li>
            <li>Use filters to refine data</li>
            <li>Explore Analysis, Hotspots, and Predictions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.df

# Column detection
lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
location_col = next((c for c in df.columns if any(x in c.lower() for x in ['area','state','city','region'])), None)

# Enhanced dynamic detection
category_col = next(
    (c for c in df.columns if any(x in c.lower() for x in ['crime','type','category','group','sub_group'])), 
    df.columns[0] if len(df.columns) > 0 else None
)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# KPIs - UNCHANGED structure
st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
col1, col2 = st.columns(2)
col1.metric("Records", len(df))
col2.metric("Columns", len(df.columns))
st.markdown('</div>', unsafe_allow_html=True)

# Dataset Overview Table
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📋 Dataset Preview")
# Dynamic clean preview - adapt to any schema
st.markdown("**Schema Overview**")
col1, col2, col3 = st.columns(3)
col1.metric("Columns", len(df.columns))
col2.metric("Rows", len(df))
col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")


col_info = st.expander("📋 Column Details (first 6)", expanded=False)
with col_info:
    col_df = pd.DataFrame({
        'Column': df.columns[:6],
        'Type': [str(dtype) for dtype in df.dtypes[:6]],
        'Unique': [len(df[col].unique()) for col in df.columns[:6]],
        'Nulls': [df[col].isnull().sum() for col in df.columns[:6]]
    })
    st.dataframe(col_df, use_container_width=True, hide_index=True)

st.markdown("**Data Preview**")
st.dataframe(
    df.head(10), 
    use_container_width=True, 
    height=300,
    hide_index=True
)
st.caption(f"✅ Fully adaptive to any schema: {len(df.columns)} cols × {len(df)} rows")
st.markdown('</div>', unsafe_allow_html=True)

# Tabs - preserve glass-card wrappers
tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🔥 Hotspots", "🔮 Predictions"])

with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Analysis")
    # Simple chart
    try:
        if category_col and category_col in df.columns:
            top_categories = df[category_col].value_counts().head(10)
        else:
            # Fallback to first column or numeric summary
            top_categories = df.iloc[:, 0].value_counts().head(10) if len(df.columns) > 0 else pd.Series([1,2,3], index=['A','B','C'])
        fig = px.bar(x=top_categories.index, y=top_categories.values, title=f"Top {category_col or 'Values'}")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {str(e)}")
        st.info("Try different dataset.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🗺️ Hotspots - Adaptive")
    
    if lat_col and lon_col:
        # Original map logic
        if st.button("Generate Map"):
            coords = df[[lat_col, lon_col]].dropna()
            if len(coords) > 5:
                kmeans = KMeans(n_clusters=4)
                df['cluster'] = kmeans.fit_predict(coords)
                fig = px.scatter(df, x=lon_col, y=lat_col, color='cluster')
                st.plotly_chart(fig)
            else:
                st.info("Insufficient geo data")
    else:
        if location_col:
            st.markdown('<div style="background: rgba(88,166,255,0.2); padding: 1rem; border-radius: 12px; border-left: 4px solid #58A6FF;">📍 This dataset uses region names instead of coordinates. Showing region-based hotspot analysis.</div>', unsafe_allow_html=True)
            df_grouped = df.groupby(location_col).size().sort_values(ascending=False).head(10)
            st.bar_chart(df_grouped)
        else:
            st.info("Hotspot analysis not available for this dataset")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔮 Predictions")
    
    if st.button("Train & Predict"):
        try:
            # Dynamic RF model
            le = LabelEncoder()
            
            if category_col and len(numeric_cols) >= 1:
                features = numeric_cols[:2]  # Top 2 numeric
                df_ml = df[[category_col] + features].dropna()
                
                if len(df_ml) > 10:  # Min data
                    X = df_ml[features]
                    y = le.fit_transform(df_ml[category_col])
                    
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                    model.fit(X, y)
                    
                    # Predict with same features
                    sample_idx = 0
                    sample = [[df_ml[features[0]].iloc[sample_idx], df_ml[features[1]].iloc[sample_idx]]]
                    pred_idx = model.predict(sample)[0]
                    pred_value = le.inverse_transform([pred_idx])[0]
                    
                    st.success(f"✅ Predicted **{pred_value}**")
                    st.metric("Accuracy", f"{model.score(X, y):.1%}")
                    st.caption(f"Using: {category_col} → predict | Features: {', '.join(features)}")
                else:
                    st.warning("Insufficient data rows (>10 needed)")
            else:
                st.info("🔄 Needs category column + 1+ numeric columns for predictions")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Ensure dataset has category + numeric columns")
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; opacity: 0.7;">
    Premium SaaS Dashboard - UI preserved + Logic enhanced
</div>
""", unsafe_allow_html=True)

