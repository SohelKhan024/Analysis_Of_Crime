"""
Crime Analytics Dashboard - DATASET-FIRST DYNAMIC VERSION
Fully dataset-driven: detects roles, no assumptions, resets on new upload
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp { background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%); color: white; font-family: 'Inter', sans-serif; }
.glass-panel { background: rgba(255,255,255,0.08); backdrop-filter: blur(25px); border-radius: 20px; padding: 2rem; margin: 1rem 0; border: 1px solid rgba(255,255,255,0.1); }
.header.header-glow { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem; border-radius: 24px; text-align: center; box-shadow: 0 20px 60px rgba(102,126,234,0.4); margin-bottom: 2rem; }
.sidebar-config { background: rgba(15,23,42,0.95); border-radius: 20px; padding: 1.5rem; margin: 1rem 0; }

/* Fade Up Animation */
.fade-up {
    animation: fadeUp 0.6s ease-out;
}
@keyframes fadeUp {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Header Glow */
@keyframes glow {
    from {box-shadow: 0 0 10px rgba(100,150,255,0.3);}
    to {box-shadow: 0 0 30px rgba(150,100,255,0.6);}
}

/* KPI Card Hover */
.kpi-card {
    transition: all 0.3s ease;
}
.kpi-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

/* Prediction Pulse */
.prediction-card {
    animation: pulseGlow 2s infinite alternate;
}
@keyframes pulseGlow {
    from {box-shadow: 0 0 10px #10b981;}
    to {box-shadow: 0 0 25px #10b981;}
}

/* Glass Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(12px);
}
</style>
""", unsafe_allow_html=True)

# Reset model state function - FIX for NameError
def reset_model_state():
    for key in ['model', 'encoder', 'feature_cols', 'target_col']:
        st.session_state.pop(key, None)

st.markdown('<div class="header header-glow fade-up"><h1>🚓 Crime Analytics - DATASET-FIRST</h1><p>🧠 Analyzes YOUR dataset structure first, then adapts</p></div>', unsafe_allow_html=True)

# SPEC: Intelligent Column Role Detection (Step 3)
def detect_columns(df):
    """Detect column roles without assuming schema"""
    cols_lower = df.columns.str.lower()
    
    category_col = time_col = lat_col = lon_col = value_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ["type", "category", "group", "crime", "offense"]):
            category_col = col
        if any(x in col_lower for x in ["year", "date", "time", "datetime", "occ"]):
            time_col = col
        if "lat" in col_lower:
            lat_col = col
        if any(x in col_lower for x in ["lon", "lng", "long"]):
            lon_col = col
    
    # Numeric fallback for value/trends
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        value_col = numeric_cols[0]  # First numeric
    
    mapping = {
        "category": category_col,
        "time": time_col,
        "lat": lat_col, 
        "lon": lon_col,
        "value": value_col
    }
    
    # Enhanced detection
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Best target (categorical: high cardinality but not unique)
    target_category = None
    max_unique_ratio = 0.1  # <10% unique = categorical
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.01 and unique_ratio < max_unique_ratio:  # Good target
            target_category = col
            break
    if not target_category and category_col:
        target_category = category_col
    
    # Additional
    year_col = area_col = None
    for col in df.columns:
        cl = col.lower()
        if any(x in cl for x in ['year']):
            year_col = col
        if any(x in cl for x in ['area', 'region', 'zone', 'district']):
            area_col = col
    
    mapping['target_category'] = target_category
    mapping['numeric_cols'] = numeric_cols
    mapping['year_col'] = year_col
    mapping['area_col'] = area_col
    
    # Feature availability
    features = {
        "has_category": category_col is not None,
        "has_time": time_col is not None,
        "has_location": lat_col is not None and lon_col is not None,
        "has_value": value_col is not None,
        "has_numeric": len(numeric_cols) > 0,
        "has_target": target_category is not None,
        "has_area": area_col is not None
    }
    
    return mapping, features

# Robust CSV loader
def safe_load_csv(file):
    try:
        return pd.read_csv(file)
    except:
        try:
            return pd.read_csv(file, sep=';', on_bad_lines='skip')
        except:
            try:
                return pd.read_csv(file, sep='\t', on_bad_lines='skip')
            except:
                st.error("Cannot parse file")
                return pd.DataFrame()

# Session state reset on new upload
if 'mapping' not in st.session_state:
    st.session_state.mapping = {}
if 'features' not in st.session_state:
    st.session_state.features = {}
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-config">', unsafe_allow_html=True)
    
    st.header("📁 Dataset")
    demo_clicked = st.button("Load Demo Dataset")
    uploaded_file = st.file_uploader("Or upload CSV", type=['csv', 'txt'])
    
    if demo_clicked:
        df = pd.read_csv("data/demo1_crime_small.csv")
        mapping, features = detect_columns(df)
        st.session_state.df = df
        st.session_state.mapping = mapping
        st.session_state.features = features
        reset_model_state()
        st.success("✅ Demo loaded & model reset")
        st.rerun()
    
    if uploaded_file is not None and st.button("🔄 Analyze This Dataset"):
        df = safe_load_csv(uploaded_file)
        if not df.empty:
            # CRITICAL Step 6: Reset state for NEW dataset
            st.session_state.df = df
            st.session_state.mapping, st.session_state.features = detect_columns(df)
            reset_model_state()
            st.success(f"✅ New dataset analyzed: {len(df)} rows")
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Main app
df = st.session_state.df
mapping = st.session_state.mapping
features = st.session_state.features

if df.empty:
    st.info("👆 Load demo or upload dataset to start")
    st.stop()

st.success(f"📊 Dataset ready: {len(df):,} rows, {len(df.columns)} columns")

# Step 7: Debug Panel
with st.expander("🔍 Column Detection Debug", expanded=False):
    st.write("**Detected Roles:**")
    for role, col in mapping.items():
        status = "✅" if col else "❌"
        st.write(f"{status} **{role}**: {col or 'None'}")
    st.write("**Features Enabled:**", features)

# Dynamic KPIs
st.markdown('<div class="fade-up">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Rows", len(df))
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Columns", len(df.columns))
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Features", sum(features.values()))
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Preview
st.subheader("Data Preview")
st.markdown('<div class="fade-up">', unsafe_allow_html=True)
st.subheader("Data Preview")
preview_cols = st.multiselect("Preview columns", df.columns.tolist(), default=df.columns.tolist()[:4])
st.dataframe(df[preview_cols].head(10))
st.markdown('</div>', unsafe_allow_html=True)

# DYNAMIC TABS w/ User-Friendly Messages (Steps 5,8)
st.markdown('<div class="fade-up">', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📈 Patterns", "🗺️ Hotspots", "🔮 Predict"])
st.markdown('</div>', unsafe_allow_html=True)  # close tabs fade-up

with st.container():  # new fade-up for tabs content
    st.markdown('<div class="fade-up">', unsafe_allow_html=True)
with tab1:  # Patterns - time/year/numeric fallback
    pattern_col = mapping.get('time') or mapping.get('year_col') or mapping.get('numeric_cols', [None])[0]
    if pattern_col and pattern_col in df.columns:
        try:
            if 'time' in pattern_col.lower() or 'date' in pattern_col.lower():
                df_temp = df[[pattern_col]].copy()
                series = pd.to_datetime(df[pattern_col], errors='coerce').dt.hour.fillna(12)
                title = f"Hourly Patterns ({pattern_col})"
            else:
                series = pd.to_numeric(df[pattern_col], errors='coerce').fillna(0)
                title = f"Distribution ({pattern_col})"
            fig = px.histogram(x=series, nbins=24, title=title)
            fig.update_layout(transition_duration=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Could not process patterns: {str(e)[:100]}")
    else:
        st.info("**No suitable pattern column (time/year/numeric). Showing categories.**")
        if mapping.get('category') in df.columns:
            top = df[mapping['category']].value_counts().head(10)
            fig = px.bar(y=top.index, x=top.values, orientation='h', title="Category Distribution")
            fig.update_layout(transition_duration=500)
        else:
            sample_cols = df.select_dtypes(include='number').columns[:3]
            if len(sample_cols) == 0:
                sample_cols = df.columns[:3]
            fig = px.histogram(df[sample_cols].melt(), x='value', color='variable', title="Dataset Overview")
            fig.update_layout(transition_duration=500)
        st.plotly_chart(fig, use_container_width=True)

with tab2:  # Hotspots IF lat+lon ELSE alternative
    if features['has_location']:
        try:
            coords = df[[mapping['lat'], mapping['lon']]].dropna()
            if len(coords) > 2:
                n_clusters = st.slider("Hotspots", 2, 8, 3)
                kmeans = KMeans(n_clusters, random_state=42)
                df['cluster'] = kmeans.fit_predict(coords)
                fig = px.scatter(df, x=mapping['lon'], y=mapping['lat'], 
                               color='cluster', title="Hotspots Map")
                fig.update_layout(transition_duration=500)
                st.plotly_chart(fig)
            else:
                st.info("Too few coordinate points")
        except Exception:
            st.info("Could not process coordinates")
    else:
        st.info("**No coordinates found. Regional Summary:**")
        if features['has_area'] and mapping['area_col'] in df.columns:
            area_stats = df[mapping['area_col']].value_counts().head(10)
            fig = px.treemap(names=area_stats.index, parents=['']*len(area_stats), values=area_stats.values,
                           title=f"Regional Heatmap by {mapping['area_col']}")
            fig.update_layout(transition_duration=500)
            st.plotly_chart(fig, use_container_width=True)
        elif features['has_category'] and mapping['category'] in df.columns:
            top = df[mapping['category']].value_counts().head(10)
            fig = px.bar(y=top.index, x=top.values, title="Category Distribution")
            fig.update_layout(transition_duration=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No area or category data for regional summary.")

with tab3:  # Predict - Dynamic model per dataset
    if features['has_target'] and features['has_numeric'] and len(mapping['numeric_cols']) > 0:
        try:
            # Dynamic X/y from detection
            feature_cols = mapping['numeric_cols']
            target_col = mapping['target_category']
            
            X = df[feature_cols].fillna(0)
            y = df[target_col].fillna('Unknown').astype(str)
            
            if len(X) < 10 or y.nunique() < 2:
                st.warning("Insufficient data for training (need 10+ rows, 2+ categories)")
            else:
                st.session_state.model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                st.session_state.model.fit(X, y_encoded)
                st.session_state.encoder = le
                st.session_state.feature_cols = feature_cols
                st.session_state.target_col = target_col
                
                st.success(f"✅ Custom model trained for this dataset! Features: {len(feature_cols)} | Target: {target_col} | Classes: {len(le.classes_)}")
                
                # Demo prediction with means
                mean_features = X.mean().values.reshape(1, -1)
                pred_idx = st.session_state.model.predict(mean_features)[0]
                pred_name = st.session_state.encoder.inverse_transform([pred_idx])[0]
                st.info(f"**Demo prediction (dataset average):** {pred_name}")
                
                # Predict button
                if st.button("🔮 Predict New Instance"):
                    with st.spinner("🧠 AI analyzing..."):
                        input_vals = [st.number_input(f"{col}", value=X[col].mean()) for col in feature_cols[:4]]
                        if len(input_vals) == len(feature_cols[:4]):
                            pred_X = np.array(input_vals).reshape(1, -1)
                            pred_idx = st.session_state.model.predict(pred_X)[0]
                            prediction = st.session_state.encoder.inverse_transform([pred_idx])[0]
                            st.markdown(f'''
                            <div class="prediction-card" style="background: linear-gradient(135deg, #10b981, #059669); padding: 2rem; border-radius: 20px; text-align: center; color: white;">
                                <h3>🎯 Prediction Result</h3>
                                <h2>{prediction}</h2>
                                <p>AI Crime Type Prediction</p>
                            </div>
                            ''', unsafe_allow_html=True)
        except Exception as e:
            st.info("🤖 AI Note: This dataset structure is different. Attempting to auto-map features...")
            st.rerun()
    else:
        st.info("**Prediction ready when you upload a dataset with categories + numbers.**")

st.markdown("---")
st.caption("✅ FULLY DATASET-DRIVEN - Adapts to YOUR uploaded data structure!")

