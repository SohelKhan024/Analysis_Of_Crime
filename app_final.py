"""
Crime Analytics Dashboard - Complete Working Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from backend_fixed import CrimeDataProcessor, HotspotDetector, CrimeRiskPredictor
from ui_components import set_page_configuration

set_page_configuration()

if 'df' not in st.session_state:
    st.session_state.df = None
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None

st.markdown("""
<div class="header-container">
    <h1>🚓 Crime Analytics Dashboard</h1>
    <p class="subtitle">Data Analysis • Hotspot Detection • Risk Prediction</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Controls")

    data_source = st.radio("Data", ["Default Dataset", "Upload CSV"])

    if data_source == "Default Dataset":
        csv_path = "data/20_Victims_of_rape.csv"
        if st.button("📥 Load Data"):
            if os.path.exists(csv_path):
                df_temp = pd.read_csv(csv_path)
                df_temp.columns = df_temp.columns.str.lower().str.strip()
                st.session_state.df = df_temp
                st.session_state.filtered_df = df_temp.copy()
                st.success(f"Loaded {len(df_temp)} records")
            else:
                st.error("File not found")
    else:
        uploaded = st.file_uploader("CSV", type="csv")
        if uploaded:
            df_temp = pd.read_csv(uploaded)
            df_temp.columns = df_temp.columns.str.lower().str.strip()
            st.session_state.df = df_temp
            st.session_state.filtered_df = df_temp.copy()
            st.success("Data uploaded")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("### Filters")
        crime_filter = st.selectbox("Crime", ["All"] + sorted(df['type'].unique()))
        hour_range = st.slider("Hour", 0, 23, (0, 23))

        filtered = df.copy()
        if crime_filter != "All":
            filtered = filtered[filtered['type'] == crime_filter]
        filtered = filtered[(filtered['hour'] >= hour_range[0]) & (filtered['hour'] <= hour_range[1])]
        st.session_state.filtered_df = filtered

if st.session_state.df is not None:
    filtered_df = st.session_state.filtered_df
    has_coords = 'lat' in filtered_df.columns and 'lon' in filtered_df.columns

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Records", len(filtered_df))
    with col2:
        top_crime = filtered_df['type'].value_counts().index[0]
        st.metric("Top Crime", top_crime)
    with col3:
        peak_h = filtered_df['hour'].mode().iloc[0]
        st.metric("Peak Hour", peak_h)

    tab1, tab2, tab3 = st.tabs(["📈 Analysis", "🔥 Hotspots", "🤖 Predict"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            hour_data = filtered_df.groupby('hour').size()
            fig1 = px.bar(x=hour_data.index, y=hour_data.values, title="Crimes by Hour")
            fig1.update_layout(template="plotly_dark")
            st.plotly_chart(fig1)

        with col2:
            type_data = filtered_df['type'].value_counts().head(10)
            fig2 = px.bar(y=type_data.index, x=type_data.values, title="Top Crimes", orientation='h')
            fig2.update_layout(template="plotly_dark")
            st.plotly_chart(fig2)

    with tab2:
        if has_coords:
            n_clusters = st.slider("Clusters", 2, 8, 4)
            if st.button("Find Hotspots"):
                coords = filtered_df[['lat', 'lon']].values
                kmeans = KMeans(n_clusters=n_clusters, n_init=10)
                clusters = kmeans.fit_predict(coords)
                fig = px.scatter(filtered_df, x='lon', y='lat', color=clusters, title="Crime Hotspots")
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig)
                st.write("Hotspots detected!")
        else:
            st.warning("No lat/lon for hotspots")

    with tab3:
        hour_pred = st.slider("Predict Hour", 0, 23, 12)
        if st.button("Predict Crime Type"):
            predictor = CrimeRiskPredictor()
            predictor.train(filtered_df)
            pred = predictor.predict_crime_type(hour_pred, 'test', 34.05, -118.24)
            st.success(f"Predicted: {pred}")
            st.bar_chart(predictor.get_feature_importance())

else:
    st.info("👆 Load data from sidebar")

