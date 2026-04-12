"""
Modern SaaS-Style Crime Analytics Dashboard
Professional UI with all requested features
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Page config
st.set_page_config(
    page_title="Crime Analytics Pro",
    page_icon="🚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        color: white;
    }
    
    .header-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .header-subtitle {
        font-size: 1.3rem;
        opacity: 0.95;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    .welcome-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }
    
    .config-panel {
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .metric-container {
        background: linear-gradient(145deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 12px;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(102,126,234,0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(102,126,234,0.6);
    }
    
    .stSelectbox > div > div > select,
    .stSlider > div > div {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .data-preview {
        background: rgba(0,0,0,0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🚓 Crime Analytics Dashboard</h1>
    <p class="header-subtitle" id="status-subtitle">Professional Crime Data Analysis Platform</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'sample_size' not in st.session_state:
    st.session_state.sample_size = 10000

# Sidebar Configuration
with st.sidebar:
    st.markdown('<div class="config-panel">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Configuration")
    
    data_source = st.radio("📁 Data Source", ["Use Default Dataset", "Upload CSV"], 
                          help="Choose your data source")
    
    if data_source == "Use Default Dataset":
        st.session_state.sample_size = st.slider("📏 Sample Size", 1000, 50000, 5000, 
                                                 step=1000, help="Number of records to analyze")
        if st.button("🚀 Load Dataset", use_container_width=True):
            try:
                # Load demo dataset
                df = pd.read_csv("data/demo1_crime_small.csv")
                if len(df) > st.session_state.sample_size:
                    df = df.sample(st.session_state.sample_size).reset_index(drop=True)
                df.columns = df.columns.str.lower().str.strip()
                st.session_state.df = df
                st.success(f"✅ Dataset loaded! {len(df):,} records ready")
                st.markdown(f'<div class="data-preview"><b>Dataset:</b> demo1_crime_small.csv | <b>Records:</b> {len(df)}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Load error: {e}")
    else:
        uploaded_file = st.file_uploader("📤 Upload CSV", type="csv", 
                                        help="Upload your crime data CSV")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.sample_size = min(len(df), 50000)
                df.columns = df.columns.str.lower().str.strip()
                st.session_state.df = df
                st.success(f"✅ Uploaded {len(df):,} records")
            except Exception as e:
                st.error(f"Upload failed: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main content
if st.session_state.df is None or st.session_state.df.empty:
    # Welcome screen
    st.markdown("""
    <div class="welcome-card">
        <h2>👋 Welcome to Crime Analytics Dashboard</h2>
        <p style="font-size: 1.2rem; color: #a0a0a0;">Real-time Crime Insights & Hotspot Detection</p>
        
        <div style="display: flex; justify-content: space-around; margin: 2rem 0;">
            <div style="text-align: center;">
                <h3>📊 Analysis</h3>
                <p>Interactive charts & trends</p>
            </div>
            <div style="text-align: center;">
                <h3>🔥 Hotspots</h3>
                <p>K-Means clustering</p>
            </div>
            <div style="text-align: center;">
                <h3>🤖 Prediction</h3>
                <p>ML crime type prediction</p>
            </div>
        </div>
        
        <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px; margin-top: 2rem;">
            <h4 style="margin: 0 0 1rem 0;">🚀 Get Started</h4>
            <ol style="color: #b0b0b0;">
                <li>Load dataset from sidebar</li>
                <li>Adjust sample size & filters</li>
                <li>Explore 3 analysis tabs</li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Dataset loaded - analytics
    df = st.session_state.df.copy()
    
    # Update header subtitle
    st.markdown(f'''
    <script>
    document.getElementById("status-subtitle").innerText = "Dataset loaded: {len(df):,} records • Ready for analysis";
    </script>
    ''', unsafe_allow_html=True)
    
    # Dataset preview
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Total Records", len(df))
        st.markdown('</div>', unsafe_allow_html=True)

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🔥 Hotspots", "🤖 Prediction"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            hour_data = df.groupby('hour').size()
            fig_hour = px.bar(x=hour_data.index, y=hour_data.values, 
                            title="Crimes by Hour", 
                            labels={'x':'Hour', 'y':'Count'})
            fig_hour.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_hour)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            crime_data = df['crime_type'].value_counts().head(10) if 'crime_type' in df else pd.Series()
            fig_crime = px.bar(y=crime_data.index, x=crime_data.values, 
                             orientation='h', title="Top Crimes")
            fig_crime.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_crime)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Hotspot Detection")
        
        if 'lat' in df.columns and 'lon' in df.columns:
            n_clusters = st.slider("Clusters", 2, min(8, len(df)//4 + 1), 3)
            if st.button("🔍 Detect Hotspots", use_container_width=True):
                coords = df[['lat', 'lon']].dropna()
                if len(coords) > n_clusters:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    df['cluster'] = kmeans.fit_predict(coords)
                    fig = px.scatter(df, x='lon', y='lat', color='cluster',
                                   title=f"Crime Hotspots (K={n_clusters})",
                                   hover_data=['crime_type'])
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Cluster Summary")
                    st.dataframe(df['cluster'].value_counts().sort_index())
                else:
                    st.warning(f"Need at least {n_clusters} data points for clustering")
        else:
            st.warning("Dataset missing lat/lon columns for hotspot analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Crime Type Prediction")
        
        pred_hour = st.slider("🔮 Prediction Hour", 0, 23, 12, 
                            help="Select hour for crime prediction")
        
        if st.button("🤖 Generate Prediction", use_container_width=True):
            # Safe ML pipeline
            df_ml = df[['hour', 'lat', 'lon']].copy()
            le = LabelEncoder()
            y = le.fit_transform(df['crime_type'].fillna('Unknown'))
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(df_ml, y)
            
            # Predict
            pred_data = [[pred_hour, df['lat'].mean(), df['lon'].mean()]]
            pred_idx = model.predict(pred_data)[0]
            prediction = le.inverse_transform([pred_idx])[0]
            
            st.success(f"**Predicted Crime Type:** {prediction}")
            st.info(f"**Input:** Hour {pred_hour} | Location: ({df['lat'].mean():.4f}, {df['lon'].mean():.4f})")
            
            # Feature importance
            imp_names = ['Hour', 'Latitude', 'Longitude']
            imp_values = model.feature_importances_
            fig_imp = px.bar(x=imp_values, y=imp_names, orientation='h',
                           title="Model Feature Importance")
            fig_imp.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_imp)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("*Professional Crime Analytics Dashboard | Ready for production*")

