"""
Crime Dashboard with 2 Default Datasets
demo1_crime_small.csv + 20_Victims_of_rape.csv
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

st.set_page_config(layout="wide")

st.markdown("""
<style>
.stApp { background: #0e1117; color: white; }
.header { text-align: center; padding: 2rem; background: linear-gradient(135deg, #1f6feb 0%, #58A6FF 100%); border-radius: 15px; margin-bottom: 2rem; }
.card { background: rgba(255,255,255,0.05); border-radius: 10px; padding: 1rem; margin: 0.5rem 0; }
.dataset-card { background: rgba(35, 134, 54, 0.2); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>🚓 Crime Analytics - Dual Dataset</h1></div>', unsafe_allow_html=True)

if 'df1' not in st.session_state:
    st.session_state.df1 = pd.DataFrame()
if 'df2' not in st.session_state:
    st.session_state.df2 = pd.DataFrame()

with st.sidebar:
    st.header("📁 Datasets")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load demo1_crime_small.csv"):
            df = pd.read_csv("data/demo1_crime_small.csv")
            df.columns = df.columns.str.lower()
            df['hour'] = pd.to_numeric(df.get('hour', 12), errors='coerce').fillna(12).astype(int)
            st.session_state.df1 = df
            st.success(f"✅ Demo1: {len(df)} records")
    
    with col2:
        if st.button("Load 20_Victims_of_rape.csv"):
            df = pd.read_csv("data/20_Victims_of_rape.csv")
            df.columns = df.columns.str.lower()
            df['hour'] = pd.to_numeric(df.get('hour', 12), errors='coerce').fillna(12).astype(int)
            st.session_state.df2 = df
            st.success(f"✅ Victims: {len(df)} records")

df1 = st.session_state.df1
df2 = st.session_state.df2

if not df1.empty or not df2.empty:
    # Display loaded datasets
    if not df1.empty:
        st.markdown('<div class="dataset-card card">', unsafe_allow_html=True)
        st.metric("Demo1 Records", len(df1))
        st.dataframe(df1.head(3))
        st.markdown('</div>', unsafe_allow_html=True)
    
    if not df2.empty:
        st.markdown('<div class="dataset-card card">', unsafe_allow_html=True)
        st.metric("Victims Records", len(df2))
        st.dataframe(df2.head(3))
        st.markdown('</div>', unsafe_allow_html=True)

    # Combined analysis
    if not df1.empty and not df2.empty:
        combined = pd.concat([df1, df2], ignore_index=True)
        combined['crime_type'] = combined.get('crime_type', combined.get('type', 'Unknown'))
        combined['lat'] = pd.to_numeric(combined.get('lat', 34.05), errors='coerce')
        combined['lon'] = pd.to_numeric(combined.get('lon', -118.24), errors='coerce')

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", len(combined))
        with col2:
            top_crime = combined['crime_type'].value_counts().index[0]
            st.metric("Most Common", top_crime)

        tab1, tab2, tab3 = st.tabs(["📊 Charts", "🗺️ Hotspots", "🔮 Predict"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                hour_chart = px.bar(combined.groupby('hour').size(), title="Hour Distribution")
                hour_chart.update_layout(template="plotly_dark")
                st.plotly_chart(hour_chart)
            with col2:
                crime_chart = px.bar(combined['crime_type'].value_counts().head(10), orientation='h')
                crime_chart.update_layout(template="plotly_dark")
                st.plotly_chart(crime_chart)

        with tab2:
            if len(combined) > 3:
                n_clusters = st.slider("Clusters", 2, min(6, len(combined)//2), 2)
                if st.button("Detect Hotspots"):
                    coords = combined[['lat', 'lon']].dropna()
                    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                    combined['cluster'] = kmeans.fit_predict(coords)
                    fig = px.scatter(combined, x='lon', y='lat', color='cluster', title="Crime Hotspots")
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig)

        with tab3:
            pred_hour = st.slider("Predict Hour", 0, 23, 12)
            if st.button("Make Prediction"):
                X = combined[['hour', 'lat', 'lon']].fillna(0)
                le = LabelEncoder()
                y = le.fit_transform(combined['crime_type'])
                model = RandomForestClassifier()
                model.fit(X, y)
                pred = model.predict([[pred_hour, combined['lat'].mean(), combined['lon'].mean()]])

                st.success(f"Predicted: {le.inverse_transform(pred)[0]}")
                st.bar_chart({'hour': model.feature_importances_[0], 'lat': model.feature_importances_[1], 'lon': model.feature_importances_[2]})
    else:
        st.info("Load both datasets for analysis")

else:
    st.info("👈 Load the 2 default datasets from sidebar")

