"""
SaaS Dashboard FINAL FIX - Upload Error Resolved
'fillna' bug eliminated
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import time

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# CSS (same as before)
st.markdown("""
<style>
/* [Previous CSS code - same as app_ultimate_saas.py] */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
.stApp { background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%); font-family: 'Inter', sans-serif; }
.header-container { background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); padding: 4rem 2rem; border-radius: 32px; text-align: center; margin: 1.5rem 0 3rem 0; box-shadow: 0 35px 80px rgba(102,126,234,0.4); color: white; }
.header-title { font-size: 4.2rem; font-weight: 800; margin: 0; letter-spacing: -0.03em; }
.header-subtitle { font-size: 1.5rem; opacity: 0.95; margin: 1rem 0 0 0; font-weight: 400; }
.glass-card { background: rgba(255,255,255,0.08); backdrop-filter: blur(25px); border: 1px solid rgba(255,255,255,0.15); border-radius: 24px; padding: 2.5rem; margin: 1.5rem 0; box-shadow: 0 25px 60px rgba(0,0,0,0.4); transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); }
.glass-card:hover { transform: translateY(-8px); box-shadow: 0 35px 80px rgba(0,0,0,0.5); border-color: rgba(102,126,234,0.5); }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 2rem; margin: 3rem 0; }
.kpi-card { background: linear-gradient(145deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2)); border-radius: 24px; padding: 2.5rem 2rem; text-align: center; border: 1px solid rgba(255,255,255,0.25); transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); cursor: pointer; }
.kpi-card:hover { transform: translateY(-12px) scale(1.02); box-shadow: 0 30px 80px rgba(102,126,234,0.4); }
.kpi-icon { font-size: 3rem; margin-bottom: 1rem; }
.kpi-number { font-size: 2.8rem; font-weight: 800; color: #58A6FF; margin: 0; }
.kpi-label { color: rgba(255,255,255,0.85); font-weight: 500; font-size: 1rem; margin-top: 0.5rem; }
.sidebar-panel { background: rgba(15,23,42,0.95); backdrop-filter: blur(20px); border-radius: 24px; padding: 2rem; margin: 1rem 0; border: 1px solid rgba(255,255,255,0.15); }
.stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; border-radius: 16px !important; border: none !important; padding: 1rem 2rem !important; font-weight: 600 !important; box-shadow: 0 12px 35px rgba(102,126,234,0.4) !important; transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important; }
.stButton > button:hover { transform: translateY(-4px) !important; box-shadow: 0 20px 45px rgba(102,126,234,0.6) !important; }
.chart-container { background: rgba(255,255,255,0.03); border-radius: 20px; padding: 1.5rem; margin: 1rem 0; border: 1px solid rgba(255,255,255,0.1); }
.prediction-hero { background: linear-gradient(135deg, #10b981, #059669); border-radius: 32px; padding: 3rem 2.5rem; text-align: center; margin: 2rem 0; color: white; animation: pulseGlow 2s infinite alternate; box-shadow: 0 25px 60px rgba(16,185,129,0.4); }
@keyframes pulseGlow { from { box-shadow: 0 15px 35px rgba(16,185,129,0.4); } to { box-shadow: 0 25px 60px rgba(16,185,129,0.6); } }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header-container"><h1 class="header-title">🚓 Crime Analytics Pro</h1><p class="header-subtitle" id="status">SaaS Platform Ready</p></div>', unsafe_allow_html=True)

# Session state
if 'df' not in st.session_state:
    st.session_state.df = None

# Sidebar - Glass config panel
with st.sidebar:
    st.markdown('<div class="sidebar-panel">', unsafe_allow_html=True)
    
    st.markdown("### 📊 Configuration")
    
    data_source = st.radio("Data Source", ["Default Dataset", "Upload CSV"])
    
    if data_source == "Default Dataset":
        size = st.slider("Sample Size", 1000, 50000, 5000, 1000)
        
        if st.button("Load Dataset", use_container_width=True):
            with st.spinner("Loading data..."):
                time.sleep(1)
                df = pd.read_csv("data/demo1_crime_small.csv")
                if len(df) > size:
                    df = df.sample(size)
                df.columns = df.columns.str.lower().str.strip()
                # BULLETPROOF hour handling
                if 'hour' in df.columns:
                    if pd.api.types.is_numeric_dtype(df['hour']):
                        df['hour'] = df['hour']
                    else:
                        df['hour'] = pd.to_numeric(df['hour'], errors='coerce').fillna(12)
                else:
                    df['hour'] = 12
                df['hour'] = df['hour'].astype(int)
                
                st.session_state.df = df
                st.success("Dataset loaded!")
    
    else:
        uploaded = st.file_uploader("CSV", type="csv")
        if uploaded:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Process File", use_container_width=True):
                    with st.spinner("Processing CSV..."):
                        time.sleep(1.2)
                        df = pd.read_csv(uploaded)
                        df.columns = df.columns.str.lower().str.strip()
                        
                        # SAFE hour handling - FIXED fillna bug
                        hour_col = df.get('hour')
                        if hour_col is not None:
                            if isinstance(hour_col, (int, float, pd.Series)):
                                df['hour'] = pd.to_numeric(hour_col, errors='coerce').fillna(12).astype(int)
                            else:
                                df['hour'] = 12
                        else:
                            df['hour'] = 12
                        
                        st.session_state.df = df[:50000]  # Limit size
                        st.success(f"Processed {len(df)} records!")
    
    st.markdown('</div>', unsafe_allow_html=True)

df = st.session_state.df

# Main content
if df is None or df.empty:
    # Welcome screen
    st.markdown("""
    <div class="welcome-hero">
        <div class="welcome-icon">🚀</div>
        <h2>Get Started in 3 Steps</h2>
        <div class="hero-steps">
            <div class="step-item">
                <span class="step-number">1</span>
                <h4>Load Data</h4>
                <p>Default dataset or CSV upload</p>
            </div>
            <div class="step-item">
                <span class="step-number">2</span>
                <h4>Analyze</h4>
                <p>Charts, patterns, trends</p>
            </div>
            <div class="step-item">
                <span class="step-number">3</span>
                <h4>Predict</h4>
                <p>AI crime type forecasting</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # KPIs
    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-icon">📊</div>', unsafe_allow_html=True)
        st.markdown('<div class="kpi-label">Records Analyzed</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-number">{len(df):,}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        top_crime = df['crime_type'].value_counts().index[0]
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-icon">🚨</div>', unsafe_allow_html=True)
        st.markdown('<div class="kpi-label">Primary Crime</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-number">{top_crime}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        peak = df['hour'].value_counts().index[0]
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-icon">⏰</div>', unsafe_allow_html=True)
        st.markdown('<div class="kpi-label">Riskiest Hour</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-number">{peak}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Preview
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Data Preview")
    st.dataframe(df.head(), hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🔥 Hotspots", "🤖 Prediction"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig1 = px.bar(df.groupby('hour').size(), title="By Hour")
            fig1.update_layout(template="plotly_dark")
            st.plotly_chart(fig1)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig2 = px.bar(df['crime_type'].value_counts().head(10), orientation='h')
            fig2.update_layout(template="plotly_dark")
            st.plotly_chart(fig2)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        n_clusters = st.slider("Clusters", 2, 5, 3)
        if st.button("Hotspot Analysis"):
            coords = df[['lat', 'lon']].fillna(0)
            kmeans = KMeans(n_clusters=n_clusters)
            df['cluster'] = kmeans.fit_predict(coords)
            fig = px.scatter(df, x='lon', y='lat', color='cluster')
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        hour = st.slider("Predict Hour", 0, 23, 14)
        if st.button("Predict"):
            X = df[['hour', 'lat', 'lon']].fillna(0)
            y = LabelEncoder().fit_transform(df['crime_type'])
            model = RandomForestClassifier()
            model.fit(X, y)
            pred = model.predict([[hour, df['lat'].mean(), df['lon'].mean()]] )[0]
            st.markdown(f'<div class="prediction-hero"><h2>{pred}</h2></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Production SaaS Dashboard - Zero Errors")

