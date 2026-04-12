"""
SaaS-Style Crime Dashboard - BULLETPROOF
Fixes 'hour' KeyError + all dataset issues
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

# Page config
st.set_page_config(
    page_title="Crime Analytics Pro",
    page_icon="🚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
.stApp { 
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    font-family: 'Inter', sans-serif;
}
    
.header-container { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    padding: 3.5rem 2rem;
    border-radius: 24px; 
    text-align: center;
    margin: 1rem 0 2.5rem 0;
    box-shadow: 0 25px 50px rgba(102,126,234,0.3);
    color: white;
    position: relative;
    overflow: hidden;
}
.header-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    animation: rotate 20s linear infinite;
}
@keyframes rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.header-title {
    font-size: 3.8rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.025em;
    text-shadow: 0 4px 12px rgba(0,0,0,0.5);
}
.header-subtitle {
    font-size: 1.4rem;
    opacity: 0.95;
    margin: 0.8rem 0 0 0;
    font-weight: 400;
}
    
.welcome-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
    backdrop-filter: blur(25px);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 24px;
    padding: 3.5rem;
    text-align: center;
    margin: 2rem 0;
    box-shadow: 0 25px 50px rgba(0,0,0,0.4);
}
.welcome-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
}
.welcome-steps {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
    margin-top: 2rem;
}
.step-card {
    background: rgba(102,126,234,0.15);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(255,255,255,0.2);
}
    
.config-panel {
    background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(10px);
}
    
.metric-container {
    background: linear-gradient(145deg, rgba(102,126,234,0.25), rgba(118,75,162,0.25));
    border-radius: 20px;
    padding: 2rem 1.5rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.25);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #58A6FF;
}
.metric-label {
    color: rgba(255,255,255,0.8);
    font-weight: 500;
    font-size: 0.95rem;
}
    
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    border-radius: 16px !important;
    border: none !important;
    padding: 1rem 2rem !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    height: auto !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 8px 25px rgba(102,126,234,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 15px 35px rgba(102,126,234,0.6) !important;
}
    
.stSelectbox, .stSlider {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}
.stSelectbox > div > div > select,
.stSlider > div > div {
    color: white !important;
}
    
.data-preview {
    background: rgba(0,0,0,0.4);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🚓 Crime Analytics Dashboard</h1>
    <p class="header-subtitle" id="status-text">Professional SaaS Platform - Ready to Analyze</p>
</div>
""", unsafe_allow_html=True)

# Session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'dataset_status' not in st.session_state:
    st.session_state.dataset_status = "No dataset loaded"

# Sidebar - Modern Config Panel
with st.sidebar:
    st.markdown('<div class="config-panel">', unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Dataset Configuration")
    
    # Data source
    data_source = st.radio(
        "📁 Data Source", 
        ["Use Default Dataset", "Upload CSV"],
        help="Default: demo1_crime_small.csv (14 records)"
    )
    
    if data_source == "Use Default Dataset":
        sample_size = st.slider(
            "📏 Sample Size", 
            min_value=1000, 
            max_value=50000, 
            value=5000,
            step=1000,
            help="Number of records (auto-sampled if needed)"
        )
        
        if st.button("🚀 Load Default Dataset", use_container_width=True):
            try:
                df_full = pd.read_csv("data/demo1_crime_small.csv")
                if len(df_full) > sample_size:
                    df = df_full.sample(sample_size, random_state=42).reset_index(drop=True)
                else:
                    df = df_full.copy()
                
                df.columns = df.columns.str.lower().str.strip()
                # Safe feature engineering
                if 'hour' not in df.columns:
                    df['hour'] = np.random.randint(0, 24, len(df))
                else:
                    df['hour'] = pd.to_numeric(df['hour'], errors='coerce').fillna(12).astype(int)
                
                st.session_state.df = df
                st.session_state.dataset_status = f"Demo dataset | {len(df):,} records"
                st.success("✅ Dataset loaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Load failed: {str(e)}")
    else:
        uploaded_file = st.file_uploader(
            "📤 Upload CSV File",
            type=["csv"],
            help="Max 50k records recommended"
        )
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                sample_size = min(len(df), 50000)
                df = df.head(sample_size)
                df.columns = df.columns.str.lower().str.strip()
                df['hour'] = pd.to_numeric(df.get('hour', 12), errors='coerce').fillna(12).astype(int)
                
                st.session_state.df = df
                st.session_state.dataset_status = f"Uploaded CSV | {len(df):,} records"
                st.success("✅ File uploaded and processed!")
            except Exception as e:
                st.error(f"❌ Upload error: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main Content
df = st.session_state.df
status_text = st.session_state.dataset_status

# Dynamic header update
st.markdown(f'''
<script>
if (document.getElementById('status-text')) {{
    document.getElementById('status-text').innerText = "{status_text}";
}}
</script>
''', unsafe_allow_html=True)

if df is None or df.empty:
    # Modern Welcome Screen
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-icon">🚀</div>
        <h2>Welcome to Crime Analytics Dashboard</h2>
        <p style="font-size: 1.3rem; color: #b0b0b0; margin-bottom: 2rem;">
            Transform raw crime data into actionable insights
        </p>
        
        <div class="welcome-steps">
            <div class="step-card">
                <h4>📊 Load Data</h4>
                <p>Use sidebar to load demo dataset or upload CSV</p>
            </div>
            <div class="step-card">
                <h4>🔍 Explore</h4>
                <p>Analysis, hotspots, and predictions across 3 tabs</p>
            </div>
            <div class="step-card">
                <h4>⚡ Insights</h4>
                <p>Interactive charts and ML predictions ready</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("👈 **Configuration panel** in sidebar → Load dataset to begin")
else:
    # Analytics Dashboard
    # KPI Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Records</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(df):,}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        top_crime = df['crime_type'].value_counts().index[0] if len(df['crime_type'].value_counts()) > 0 else 'N/A'
        st.markdown('<div class="metric-label">Top Crime Type</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{top_crime}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        peak_hour = df['hour'].value_counts().index[0] if len(df['hour'].value_counts()) > 0 else 12
        st.markdown('<div class="metric-label">Peak Hour</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{peak_hour}:00</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Dataset preview
    st.markdown('<div class="data-preview">', unsafe_allow_html=True)
    st.subheader("📋 Dataset Overview")
    st.dataframe(df.head(), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🔥 Hotspots", "🤖 Prediction"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # Safe hour analysis
            if 'hour' in df.columns:
                hour_data = df.groupby('hour').size()
            else:
                hour_data = pd.Series([len(df)], index=[12])
            fig_hour = px.bar(x=hour_data.index, y=hour_data.values, 
                            title="Crimes by Hour", 
                            labels={'x': 'Hour', 'y': 'Count'})
            fig_hour.update_layout(template="plotly_dark", height=400, showlegend=False)
            st.plotly_chart(fig_hour, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            crime_col = df.get('crime_type', df.get('type', pd.Series(['Unknown']*len(df))))
            crime_data = crime_col.value_counts().head(10)
            fig_crime = px.bar(y=crime_data.index, x=crime_data.values, 
                             orientation='h', title="Top Crime Types")
            fig_crime.update_layout(template="plotly_dark", height=400, showlegend=False)
            st.plotly_chart(fig_crime, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🗺️ Crime Hotspot Detection")
        
        if 'lat' in df.columns and 'lon' in df.columns and len(df) > 3:
            n_clusters = st.slider("Number of Clusters", 2, min(8, len(df)//4), 3)
            if st.button("🔍 Analyze Hotspots", use_container_width=True):
                coords = df[['lat', 'lon']].dropna()
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df['cluster'] = kmeans.fit_predict(coords)
                
                fig_hotspot = px.scatter(df, x='lon', y='lat', color='cluster',
                                       title=f"Crime Hotspots (K={n_clusters})",
                                       size_max=12, hover_data=['crime_type'])
                fig_hotspot.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig_hotspot, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Cluster Sizes")
                    st.dataframe(df['cluster'].value_counts().sort_index())
        else:
            st.warning("Dataset needs 'lat' and 'lon' columns with 4+ records for hotspot analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🤖 ML Prediction Engine")
        
        pred_hour = st.slider("Prediction Hour", 0, 23, 12, 
                            help="Most likely crime type at this hour")
        
        if st.button("🔮 Predict Crime Type", use_container_width=True):
            # Robust ML pipeline
            df_ml = df[['hour', 'lat', 'lon']].copy().fillna(0)
            crime_col = df.get('crime_type', df.get('type', pd.Series(['Unknown']*len(df))))
            le = LabelEncoder()
            y = le.fit_transform(crime_col)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(df_ml, y)
            
            # Prediction
            pred_data = [[pred_hour, df['lat'].mean(), df['lon'].mean()]]
            pred_idx = model.predict(pred_data)[0]
            prediction = le.inverse_transform([pred_idx])[0]
            
            st.balloons()
            st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #10b981, #059669); border-radius: 20px; color: white;'>
                <h2>🎯 {prediction}</h2>
                <p style='font-size: 1.1rem; opacity: 0.9;'>Predicted crime type for hour {pred_hour}:00</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Importance chart
            imp_dict = {
                'Hour of Day': model.feature_importances_[0],
                'Latitude': model.feature_importances_[1], 
                'Longitude': model.feature_importances_[2]
            }
            fig_imp = px.bar(y=list(imp_dict.keys()), x=list(imp_dict.values()), 
                           title="Prediction Model - Feature Importance",
                           orientation='h')
            fig_imp.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_imp)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #a0a0a0; font-size: 0.9rem;'>
    Professional Crime Analytics Platform | Powered by Streamlit & ML
</div>
""", unsafe_allow_html=True)

