"""
Ultimate SaaS Crime Analytics Dashboard - Production Ready
All Requirements Complete + UI Preserved
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

st.set_page_config(page_title="Crime Analytics Pro", page_icon="🚓", layout="wide", initial_sidebar_state="expanded")

# CSS - Premium but Stable
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 20% 20%, rgba(220,38,38,0.20), transparent 40%),
        radial-gradient(circle at 80% 80%, rgba(255,255,255,0.08), transparent 45%),
        linear-gradient(135deg, #050505 0%, #0f0f10 50%, #171718 100%);
}
body {
    background:
        radial-gradient(circle at 20% 20%, rgba(220,38,38,0.18), transparent 40%),
        radial-gradient(circle at 80% 80%, rgba(255,255,255,0.06), transparent 45%);
}
.header-container {
    background: linear-gradient(135deg, #1a1a1a 0%, #7f1d1d 55%, #ef4444 100%);
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.55), 0 0 60px rgba(239,68,68,0.35);
}
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.45);
    transition: all 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(239,68,68,0.22);
}
.stMetric > label { color: #ffffff !important; font-weight: 600 !important; }
.stMetric > div > div { color: #ffffff !important; font-size: 2rem !important; font-weight: 800 !important; text-shadow: 0 0 10px rgba(239,68,68,0.35); }
h1, h2, h3, h4, h5, h6, p, li, label, .stCaption, .stMarkdown, .stText, .stSubheader {
    color: #ffffff !important;
}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
    color: #ffffff !important;
}
[data-testid="stTabs"] [role="tab"] {
    color: rgba(255,255,255,0.95) !important;
    font-weight: 700 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #ffffff !important;
    text-shadow: 0 0 12px rgba(239,68,68,0.45);
}
[data-testid="stTabs"] [role="tablist"] {
    gap: 8px;
}
[data-testid="stTabs"] [role="tab"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 10px;
    padding: 8px 14px;
}
[data-testid="stTabs"] [role="tab"]:hover {
    background: rgba(239,68,68,0.18);
    border-color: rgba(239,68,68,0.45);
}
[data-testid="stDataFrame"] {
    background: rgba(0,0,0,0.35) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px;
}
.dataframe tbody tr:hover { background: rgba(239,68,68,0.16) !important; }
[data-testid="stDataFrame"] tbody tr:hover { background: rgba(239,68,68,0.14) !important; }
.stButton > button {
    background: linear-gradient(135deg, #7f1d1d, #ef4444);
    color: #ffffff !important;
    border-radius: 12px;
    box-shadow: 0 8px 25px rgba(239,68,68,0.30);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    box-shadow: 0 12px 35px rgba(239,68,68,0.42), 0 0 20px rgba(255,255,255,0.20);
    transform: scale(1.03);
}
button {
    transition: all 0.3s ease;
}
button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 20px rgba(239,68,68,0.38);
}
.main {
    animation: fadeIn 0.5s ease-in-out;
}
.predict-feature-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
    margin-bottom: 8px;
}
.predict-feature-tag {
    background: linear-gradient(135deg, rgba(239,68,68,0.22), rgba(255,255,255,0.10));
    border: 1px solid rgba(239,68,68,0.45);
    color: #ffffff;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.2px;
    box-shadow: 0 0 16px rgba(239,68,68,0.20);
    transition: all 0.25s ease;
}
.predict-feature-tag:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 20px rgba(239,68,68,0.35);
    border-color: rgba(255,255,255,0.45);
}
.predict-result-card {
    margin-top: 14px;
    padding: 16px 18px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(127,29,29,0.30), rgba(239,68,68,0.25));
    border: 1px solid rgba(239,68,68,0.45);
    box-shadow: 0 10px 28px rgba(239,68,68,0.22), inset 0 0 0 1px rgba(255,255,255,0.04);
    transition: all 0.3s ease;
}
.predict-result-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 34px rgba(239,68,68,0.28);
}
.predict-result-label {
    color: rgba(255,255,255,0.95);
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.predict-result-value {
    color: #ffffff;
    font-size: 1.35rem;
    font-weight: 800;
    line-height: 1.2;
    text-shadow: 0 0 20px rgba(239,68,68,0.30);
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-container"><h1 style="text-align: center; color: white; font-size: 3.5rem; font-weight: 800; margin: 0;">🚓 Crime Analytics Pro</h1></div>', unsafe_allow_html=True)

if 'df' not in st.session_state:
    st.session_state.df = None

# Sidebar
with st.sidebar:
    st.header("📁 Dataset")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    
    demo_files = []
    try:
        if os.path.exists("data"):
            demo_files = [f for f in os.listdir("data") if f.endswith('.csv')]
    except:
        pass
    selected_demo = st.selectbox("Demo", ["None"] + demo_files) if demo_files else "None"
    
    if st.button("🚀 Load", type="primary"):
        df_new = None
        if uploaded:
            try:
                df_new = pd.read_csv(uploaded)
                st.success(f"Loaded {len(df_new)} rows")
            except Exception:
                st.warning("Unable to parse the uploaded file. Please upload a valid CSV.")
        elif selected_demo != "None":
            try:
                df_new = pd.read_csv(f"data/{selected_demo}")
                st.success("Demo loaded")
            except Exception:
                st.warning("Demo unavailable or cannot be parsed right now.")
        
        if df_new is not None:
            try:
                df_new.columns = df_new.columns.str.lower().str.strip()
                if len(df_new) > 10000:
                    df_new = df_new.sample(n=10000, random_state=42)
                st.session_state.df = df_new
                st.rerun()
            except Exception:
                st.warning("Dataset loaded but could not be prepared. Try another dataset.")

# Home
if "df" not in st.session_state or st.session_state["df"] is None:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 3rem;">
        <h1 style="color: white;">🚓 Crime Analytics Dashboard</h1>
        <h2 style="color: #a0a0c8;">Real-time Crime Insights & Hotspot Detection</h2>
        <div style="background: rgba(255,255,255,0.1); border-radius: 20px; padding: 2rem; margin: 2rem 0;">
            <h3 style="color: white;">👋 Welcome to Crime Analytics Dashboard</h3>
            <p style="color: #b0b0d0;">Please load a dataset from the sidebar to get started.</p>
            <ul style="text-align: left; color: #c0c0e0; max-width: 500px; margin: 0 auto;">
                <li>Select dataset</li>
                <li>Load dataset</li>
                <li>Explore analysis, hotspots, prediction</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.df

# Detection
lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
category_col = next(
    (c for c in df.columns if any(x in c.lower() for x in ['crime','type','category','group','sub_group'])),
    None
)
location_col = next(
    (c for c in df.columns if any(x in c.lower() for x in ['area','state','city','region'])),
    None
)
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

# Overview
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📋 Dataset")

cols_to_show = []
if category_col:
    cols_to_show.append(category_col)
if location_col:
    cols_to_show.append(location_col)
if lat_col:
    cols_to_show.append(lat_col)
if lon_col:
    cols_to_show.append(lon_col)

if cols_to_show:
    st.dataframe(df[cols_to_show].head(10), hide_index=True)
else:
    st.dataframe(df.head(10), hide_index=True)

st.caption(f"{len(df):,} rows | {len(df.columns)} cols")
st.markdown('</div>', unsafe_allow_html=True)

# KPIs
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📈 Key Stats")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Records", len(df))
with c2:
    if category_col and category_col in df.columns and df[category_col].dropna().shape[0] > 0:
        st.metric("Top Category", df[category_col].value_counts().index[0])
    else:
        st.metric("Top Category", "N/A")
with c3:
    st.metric("Numeric Cols", len(numeric_cols))
st.markdown('</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🗺️ Hotspots", "🔮 Predict"])

with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if category_col:
            counts = df[category_col].value_counts().head(8)
            fig1 = px.bar(counts, orientation='h')
            st.plotly_chart(fig1)
    with c2:
        if numeric_cols:
            fig2 = px.histogram(df, x=numeric_cols[0])
            st.plotly_chart(fig2)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
        k = st.slider("Clusters", 2, 6, 3)
        if st.button("Detect"):
            coords = df[[lat_col, lon_col]].dropna()
            if len(coords) > k:
                try:
                    km = KMeans(n_clusters=k, n_init=10, random_state=42)
                    coords_labeled = coords.copy()
                    coords_labeled['cluster'] = km.fit_predict(coords)
                    fig = px.scatter(coords_labeled, x=lon_col, y=lat_col, color='cluster')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Hotspot clustering is temporarily unavailable for this dataset.")
            else:
                st.info("Not enough coordinate rows for hotspot detection.")
    elif location_col and location_col in df.columns:
        st.info("📍 This dataset uses region names instead of coordinates. Showing region-based hotspot analysis.")
        try:
            df_grouped = df.groupby(location_col).size().sort_values(ascending=False)
            st.bar_chart(df_grouped.head(10))
        except Exception:
            st.info("Region-based hotspot analysis is unavailable for this dataset.")
    else:
        st.info("Hotspot analysis not available for this dataset")
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if category_col and len(numeric_cols) >= 1:
        features = numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols[:1]

        if len(features) >= 1:
            try:
                model_df = df[features + [category_col]].dropna()
                if len(model_df) < 2:
                    st.info("Prediction requires enough clean rows with category + numeric values.")
                else:
                    X = model_df[features]
                    y = model_df[category_col].astype(str)

                    if y.nunique() < 2:
                        st.info("Prediction requires at least 2 unique classes in category column.")
                    else:
                        le = LabelEncoder()
                        model = RandomForestClassifier(n_estimators=30, random_state=42)
                        model.fit(X, le.fit_transform(y))

                        input_data = X.iloc[[0]]
                        prediction = model.predict(input_data)[0]
                        pred_label = le.inverse_transform([prediction])[0]

                        st.markdown(
                            "<div class='predict-feature-tags'>"
                            + "".join([f"<span class='predict-feature-tag'>{col}</span>" for col in features])
                            + "</div>",
                            unsafe_allow_html=True
                        )

                        feature_importance = model.feature_importances_
                        importance_df = pd.DataFrame({
                            "feature": X.columns,
                            "importance": feature_importance
                        }).sort_values(by="importance", ascending=False)

                        top_features = importance_df.head(3)["feature"].tolist()
                        explanation = f"This prediction is mainly influenced by {', '.join(top_features)}."

                        context_notes = []
                        if any("year" in str(f).lower() for f in top_features):
                            context_notes.append("Temporal trend (year) appears to be an important driver.")
                        if any(any(k in str(f).lower() for k in ["count", "total", "number", "rate"]) for f in top_features):
                            context_notes.append("Crime intensity-related numeric signal also contributes to this prediction.")

                        if context_notes:
                            explanation = explanation + " " + " ".join(context_notes)

                        prediction_title = (
                            "📍 Most Affected Area"
                            if any(k in str(category_col).lower() for k in ["area", "city", "state", "region", "location"])
                            else "🔮 Predicted Category"
                        )

                        st.markdown(
                            f"""
                            <div class="predict-result-card">
                                <div class="predict-result-label">{prediction_title}</div>
                                <div class="predict-result-value">{pred_label}</div>
                                <div class="predict-result-label" style="margin-top: 12px;">🧠 Why this prediction:</div>
                                <div style="color: rgba(255,255,255,0.92); font-size: 0.95rem; line-height: 1.45;">{explanation}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            except Exception:
                st.info("Prediction is temporarily unavailable for this dataset.")
        else:
            st.info("Prediction requires category + numeric columns")
    else:
        st.info("Prediction requires category + numeric columns")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Dynamic Crime Analytics - Production Ready")

