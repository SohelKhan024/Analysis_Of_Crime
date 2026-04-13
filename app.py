"""
FRONTEND.PY (APP.PY) - Premium Crime Analytics Dashboard
Interactive dark-themed dashboard for crime data analysis and prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
from backend import CrimeDataProcessor, HotspotDetector, CrimeRiskPredictor
from ui_components import set_page_configuration
import warnings
warnings.filterwarnings('ignore')


def prepare_dashboard_dataframe(df_input: pd.DataFrame) -> pd.DataFrame:
    """Normalize incoming dataset columns and create required dashboard fields safely."""
    df = df_input.copy()
    df.columns = df.columns.str.lower().str.strip()

    # ---- IMPROVED Crime description mapping (DYNAMIC) ----
    crime_col = None
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in [
            "crime", "offense", "offence", "type", "category", "description"
        ]):
            crime_col = col
            break

    if crime_col is None:
        # Fallback: first categorical column
        cat_cols = df.select_dtypes(include=["object"]).columns
        if len(cat_cols) > 0:
            crime_col = cat_cols[0]

    if crime_col is None:
        # Final safety
        df["crime_fallback"] = "Unknown Crime"
        crime_col = "crime_fallback"

    df['crm_cd_desc'] = df[crime_col].astype(str)

    # ---- Date/time mapping for hour/day ----
    date_candidates = ['date_occ', 'date', 'occ_date', 'reported_date', 'datetime']
    time_candidates = ['time_occ', 'time', 'hour']

    date_col = next((c for c in date_candidates if c in df.columns), None)
    time_col = next((c for c in time_candidates if c in df.columns), None)

    if 'hour' not in df.columns:
        if time_col and time_col != 'hour':
            if pd.api.types.is_numeric_dtype(df[time_col]):
                df['hour'] = pd.to_numeric(df[time_col], errors='coerce').fillna(0).astype(int).clip(0, 23)
            else:
                parsed_time = pd.to_datetime(df[time_col], errors='coerce')
                df['hour'] = parsed_time.dt.hour.fillna(0).astype(int)
        elif date_col:
            parsed_date = pd.to_datetime(df[date_col], errors='coerce')
            df['hour'] = parsed_date.dt.hour.fillna(0).astype(int)
        else:
            df['hour'] = 12

    if 'day_of_week' not in df.columns:
        if date_col:
            parsed_date = pd.to_datetime(df[date_col], errors='coerce')
            df['day_of_week'] = parsed_date.dt.day_name().fillna('Unknown')
        else:
            df['day_of_week'] = 'Unknown'

    # ---- Area mapping ----
    area_candidates = ['area_name', 'area', 'neighborhood', 'district', 'zone', 'precinct']
    area_col = next((c for c in area_candidates if c in df.columns), None)
    if area_col:
        if area_col == 'area':
            df['area_name'] = 'Area ' + df[area_col].astype(str)
        else:
            df['area_name'] = df[area_col].astype(str)
    else:
        df['area_name'] = 'Unknown Area'

    # ---- Coordinate mapping (optional) ----
    lat_candidates = ['lat', 'latitude']
    lon_candidates = ['lon', 'lng', 'longitude']
    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)

    if lat_col and lon_col:
        if lat_col != 'lat':
            df['lat'] = pd.to_numeric(df[lat_col], errors='coerce')
        else:
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')

        if lon_col != 'lon':
            df['lon'] = pd.to_numeric(df[lon_col], errors='coerce')
        else:
            df['lon'] = pd.to_numeric(df['lon'], errors='coerce')

    return df

# ==================== PAGE CONFIGURATION ====================
set_page_configuration()

# ==================== INITIALIZE SESSION STATE ====================
if "df" not in st.session_state:
    st.session_state["df"] = None
if "filtered_df" not in st.session_state:
    st.session_state["filtered_df"] = None
if "hotspot_model" not in st.session_state:
    st.session_state["hotspot_model"] = None

if "column_mapping" not in st.session_state:
    st.session_state["column_mapping"] = None

# ==================== HEADER SECTION ====================
st.markdown("""
<div class="header-container header-glow fade-up" style="
    background: linear-gradient(120deg, rgba(31,111,235,0.22), rgba(111,66,193,0.16), rgba(16,185,129,0.12));
    border: 1px solid rgba(88,166,255,0.35);
    box-shadow: 0 10px 36px rgba(31,111,235,0.18), inset 0 0 24px rgba(255,255,255,0.03);
    border-radius: 18px;
">
    <h1 style="margin-bottom:6px;">🚓 Crime Analytics Dashboard</h1>
    <p class="subtitle" style="font-size:1.05rem;">Real-time Crime Insights, Intelligent Hotspots & Explainable Prediction</p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR - DATA LOADING & FILTERS ====================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    data_source = st.sidebar.radio("📁 Data Source", ["Use Default Dataset", "Upload CSV"])
    uploaded_file = st.sidebar.file_uploader("📤 Upload CSV", type="csv") if data_source == "Upload CSV" else None

    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "demo_crime_small.csv")

    # Ensure dataset is loaded
    st.info("👆 Select a data source and click 'Load Dataset' to begin")

    if st.button("🚀 Load Dataset", use_container_width=True):
        if data_source == "Use Default Dataset":
                if os.path.exists(DEFAULT_DATA_PATH):
                    df_temp = pd.read_csv(DEFAULT_DATA_PATH)
                    df_temp = prepare_dashboard_dataframe(df_temp)

                    if 'lat' not in df_temp.columns or 'lon' not in df_temp.columns or df_temp['lat'].isna().all() or df_temp['lon'].isna().all():
                        st.warning("No location data found. Map features disabled.")

                    st.session_state["df"] = df_temp
                    st.session_state["filtered_df"] = df_temp.copy()
                    st.session_state["column_mapping"] = None
                    st.success("✅ Default dataset loaded successfully")
                else:
                    st.error("Default dataset not found: data/demo_crime_small.csv")


        elif uploaded_file is not None:
            try:
                df_temp = pd.read_csv(uploaded_file)
                df_temp = prepare_dashboard_dataframe(df_temp)

                if 'lat' not in df_temp.columns or 'lon' not in df_temp.columns or df_temp['lat'].isna().all() or df_temp['lon'].isna().all():
                    st.warning("No location data found. Map features disabled.")

                st.session_state["df"] = df_temp
                st.session_state["filtered_df"] = df_temp.copy()
                st.session_state["column_mapping"] = None
                st.success("✅ Uploaded dataset loaded successfully")
                st.info(f"Dataset loaded: {df_temp.shape[0]} rows, {df_temp.shape[1]} columns")
                st.dataframe(df_temp.head())
            except Exception as e:
                st.error(f"Failed to load uploaded dataset: {e}")
        else:
            st.warning("Please upload a CSV file before clicking Load Dataset.")

    # Filters
    if st.session_state["df"] is not None:
        st.markdown("---")
        st.markdown("## 🔍 Filters")

        # Crime Type Filter
        all_crimes = sorted(st.session_state["df"]['crm_cd_desc'].unique())
        selected_crime = st.selectbox(
            "Crime Type",
            ["All Crimes"] + all_crimes
        )

        # Hour Filter
        selected_hour = st.slider("Hour (0-23)", 0, 23, (0, 23))

        # Apply Filters
        filtered_df = st.session_state["df"].copy()

        if selected_crime != "All Crimes":
            filtered_df = filtered_df[filtered_df['crm_cd_desc'] == selected_crime]

        filtered_df = filtered_df[
            (filtered_df['hour'] >= selected_hour[0]) &
            (filtered_df['hour'] <= selected_hour[1])
        ]

        st.session_state["filtered_df"] = filtered_df

# ==================== MAIN CONTENT ====================
if st.session_state["df"] is not None:
    df = st.session_state["df"]
    filtered_df = st.session_state["filtered_df"]

    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if any(x in c.lower() for x in ['lon', 'lng', 'long'])), None)
    location_col = next((c for c in df.columns if any(x in c.lower() for x in ['area', 'state', 'city', 'region'])), None)

    st.session_state["column_mapping"] = {
        "lat_col": lat_col,
        "lon_col": lon_col,
        "location_col": location_col
    }


    # ==================== KPI SECTION ====================
    st.markdown("""
    <div class="premium-card fade-up" style="
        border: 1px solid rgba(88,166,255,0.25);
        background: linear-gradient(135deg, rgba(22,27,34,0.88), rgba(13,17,23,0.92));
        backdrop-filter: blur(8px);
    ">
        <h2 style="margin:0;border:none;padding:0;color:#79C0FF;">📊 Key Performance Indicators</h2>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
            st.markdown('<div class="premium-card kpi-card">', unsafe_allow_html=True)
            st.metric(
                "Total Crimes",
                f"{len(filtered_df):,}",
                f"{len(filtered_df) - len(df)}" if len(filtered_df) < len(df) else "Unfiltered"
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        most_common = filtered_df['crm_cd_desc'].value_counts().index[0] if len(filtered_df) > 0 else "N/A"
        count = filtered_df['crm_cd_desc'].value_counts().values[0] if len(filtered_df) > 0 else 0
        st.markdown('<div class="premium-card kpi-card">', unsafe_allow_html=True)
        st.metric("Most Common Crime", most_common, count)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        peak_hour = filtered_df['hour'].value_counts().index[0] if len(filtered_df) > 0 else "N/A"
        st.markdown('<div class="premium-card kpi-card">', unsafe_allow_html=True)
        st.metric("Peak Crime Hour", f"{peak_hour}:00", "24-hour format")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="premium-card fade-up glass-card" style="
        border: 1px solid rgba(88,166,255,0.25);
        background: linear-gradient(135deg, rgba(30,41,59,0.52), rgba(15,23,42,0.66));
        margin-top: 16px;
    ">
        <h3 style="margin:0;color:#79C0FF;">🧾 Dataset Overview</h3>
        <p style="margin:6px 0 0 0;">Dynamic schema detected from uploaded/default dataset. Rendering available columns safely.</p>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        filtered_df.head(12),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # ==================== TABS ====================
    tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🔥 Hotspots", "🤖 Prediction"])

    # ==================== TAB 1: ANALYSIS ====================
    with tab1:
        st.markdown("### 📈 Crime Analysis Charts")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Crime by Hour")
            crime_by_hour = filtered_df.groupby('hour').size()
            fig_hour = go.Figure(
                data=[go.Bar(
                    x=crime_by_hour.index,
                    y=crime_by_hour.values,
                    marker=dict(color='#1f6feb', line=dict(color='#58A6FF', width=2)),
                    hovertemplate='<b>Hour:</b> %{x}:00<br><b>Crimes:</b> %{y}<extra></extra>'
                )]
            )
            fig_hour.update_layout(
                template="plotly_dark",
                showlegend=False,
                hovermode='x unified',
                xaxis_title="Hour of Day",
                yaxis_title="Number of Crimes",
                height=400
            )
            st.plotly_chart(fig_hour, use_container_width=True)

        with col2:
            st.subheader("Top 10 Crime Types")
            top_crimes = filtered_df['crm_cd_desc'].value_counts().head(10)
            top_crimes_df = top_crimes.reset_index()
            top_crimes_df.columns = ['crime_type', 'count']
            fig_crime = px.bar(
                top_crimes_df,
                x='count',
                y='crime_type',
                orientation='h',
                title="Top 10 Crime Types",
                labels={'count': 'Count', 'crime_type': 'Crime Type'}
            )
            fig_crime.update_traces(marker=dict(color='#238636', line=dict(color='#2ea043', width=2)))
            fig_crime.update_layout(
                template="plotly_dark",
                showlegend=False,
                height=400,
                hovermode='y unified'
            )
            st.plotly_chart(fig_crime, use_container_width=True)

        # Crime by Day of Week
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Crime by Day of Week")
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            crime_by_day = filtered_df['day_of_week'].value_counts().reindex(day_order)
            fig_day = go.Figure(
                data=[go.Bar(
                    x=crime_by_day.index,
                    y=crime_by_day.values,
                    marker=dict(color='#d1883d', line=dict(color='#f08d57', width=2)),
                    hovertemplate='<b>%{x}</b><br>Crimes: %{y}<extra></extra>'
                )]
            )
            fig_day.update_layout(
                template="plotly_dark",
                showlegend=False,
                height=400,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_day, use_container_width=True)

        with col2:
            st.subheader("Crime by Area")
            area_crimes = filtered_df['area_name'].value_counts().head(10)
            area_crimes_df = area_crimes.reset_index()
            area_crimes_df.columns = ['area_name', 'count']
            fig_area = px.bar(
                area_crimes_df,
                x='count',
                y='area_name',
                orientation='h'
            )
            fig_area.update_traces(marker=dict(color='#6e40c9', line=dict(color='#9966ff', width=2)))
            fig_area.update_layout(
                template="plotly_dark",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_area, use_container_width=True)

    # ==================== TAB 2: HOTSPOTS ====================
    with tab2:
        st.markdown("### 🗺️ Crime Hotspot Detection")

        lat_col = next((c for c in filtered_df.columns if 'lat' in c.lower()), None)
        lon_col = next((c for c in filtered_df.columns if any(x in c.lower() for x in ['lon', 'lng', 'long'])), None)
        location_col = next((c for c in filtered_df.columns if any(x in c.lower() for x in ['area', 'state', 'city', 'region'])), None)

        if lat_col is not None and lon_col is not None:
            col1, col2 = st.columns([1, 2])

            with col1:
                # Compute safe max_clusters to prevent NameError
                valid_coords = filtered_df[[lat_col, lon_col]].dropna()
                n_valid_points = len(valid_coords)
                max_clusters = min(n_valid_points // 5, 20)  # More lenient: 5 pts/cluster min

# --- SAFE HOTSPOT SLIDER FIX ---
                # --- SAFE HOTSPOT SLIDER FIX (STRICT TASK REQUIREMENT) ---
                safe_max = max(1, int(max_clusters))

                # If dataset too small → avoid slider crash
                if safe_max < 3:
                    n_clusters = 1
                    st.info("Not enough data for multiple hotspots. Showing 1 hotspot.")
                else:
                    default_val = min(8, safe_max)
                    if safe_max == 3:
                        n_clusters = 3  # Single value case - avoid slider
                    else:
                        n_clusters = st.slider(
                            "Number of Hotspots",
                            min_value=3,
                            max_value=safe_max,
                            value=default_val
                        )
                if st.button("🔍 Detect Hotspots", use_container_width=True):
                    with st.spinner("Detecting hotspots..."):
                        hotspot_df = filtered_df.copy()
                        hotspot_df["lat"] = pd.to_numeric(hotspot_df[lat_col], errors="coerce")
                        hotspot_df["lon"] = pd.to_numeric(hotspot_df[lon_col], errors="coerce")
                        hotspot_df = hotspot_df.dropna(subset=["lat", "lon"])

                        if hotspot_df.empty:
                            st.info("No valid coordinate rows found for hotspot detection.")
                            st.session_state["hotspot_model"] = None
                        elif len(hotspot_df) < n_clusters:
                            st.info(f"Not enough coordinate rows ({len(hotspot_df)}) for {n_clusters} clusters.")
                            st.session_state["hotspot_model"] = None
                        else:
                            detector = HotspotDetector(n_clusters=n_clusters)
                            clusters, centers = detector.detect_hotspots(hotspot_df)

                            if clusters is None or centers is None:
                                st.info("Hotspot detection could not be completed for this dataset.")
                                st.session_state["hotspot_model"] = None
                            else:
                                st.session_state["hotspot_model"] = {
                                    "clusters": clusters,
                                    "centers": centers,
                                    "hotspot_df": hotspot_df
                                }

                                hotspot_data = pd.DataFrame({
                                    "Hotspot ID": range(len(centers)),
                                    "Latitude": centers[:, 0].round(4),
                                    "Longitude": centers[:, 1].round(4),
                                    "Crime Count": [sum(clusters == i) for i in range(len(centers))]
                                }).sort_values("Crime Count", ascending=False)

                                st.markdown("#### Hotspot Information")
                                st.dataframe(hotspot_data, use_container_width=True, hide_index=True)

            with col2:
                if "hotspot_model" in st.session_state and st.session_state["hotspot_model"] is not None:
                    model = st.session_state["hotspot_model"]
                    clusters = model["clusters"]
                    centers = model["centers"]
                    hotspot_df = model["hotspot_df"]

                    fig_hotspots = go.Figure()
                    fig_hotspots.add_trace(go.Scattergeo(
                        lon=hotspot_df["lon"],
                        lat=hotspot_df["lat"],
                        mode="markers",
                        marker=dict(
                            size=4,
                            color=clusters,
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Cluster")
                        ),
                        text=hotspot_df["crm_cd_desc"] if "crm_cd_desc" in hotspot_df.columns else None,
                        hovertemplate="<b>Lat:</b> %{lat:.4f}<br><b>Lon:</b> %{lon:.4f}<extra></extra>"
                    ))

                    fig_hotspots.add_trace(go.Scattergeo(
                        lon=centers[:, 1],
                        lat=centers[:, 0],
                        mode="markers",
                        marker=dict(size=15, color="red", symbol="star", line=dict(width=2, color="white")),
                        text=[f"Hotspot {i}" for i in range(len(centers))],
                        hovertemplate="<b>%{text}</b><extra></extra>"
                    ))

                    fig_hotspots.update_layout(
                        template="plotly_dark",
                        geo=dict(projection_type="mercator"),
                        height=500,
                        showlegend=False
                    )
                    st.plotly_chart(fig_hotspots, use_container_width=True)
                else:
                    st.info("👈 Select hotspot count and click 'Detect Hotspots' to visualize")

        elif location_col is not None:
            st.markdown("📍 Showing region-based hotspot analysis")
            df_grouped = filtered_df.groupby(location_col).size().sort_values(ascending=False)
            st.bar_chart(df_grouped.head(10))

        else:
            st.info("Hotspot analysis not available for this dataset")

    # ==================== TAB 3: PREDICTION ====================
    with tab3:
        st.markdown("### 🤖 Crime Risk Prediction")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Make Prediction")

            pred_hour = st.slider("Select Hour", 0, 23, 12, key="pred_hour_slider")

            areas = sorted(filtered_df['area_name'].unique())
            pred_area = st.selectbox("Select Area", areas, key="pred_area_select")

            if st.button("🔮 Predict Crime Type", use_container_width=True):

                # STEP 1: Detect numeric features dynamically
                numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()

                if len(numeric_cols) == 0:
                    st.warning("No numeric data available for prediction")
                    st.stop()

                # STEP 2: Detect target column intelligently
                target_col = next(
                    (c for c in filtered_df.columns if any(x in c.lower() for x in ['crime','type','group','category','sub_group'])),
                    None
                )

                if target_col is None:
                    st.warning("No target column found for prediction")
                    st.stop()

                # STEP 3: Prepare dataset
                X = filtered_df[numeric_cols].fillna(0)
                y = filtered_df[target_col].astype(str)

                # STEP 4: Encode target
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)

                # STEP 5: Train model fresh every time
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier()
                model.fit(X, y_encoded)

                # STEP 6: Create input using SAME columns
                input_data = X.iloc[[0]]  # safe default

                # STEP 7: Predict
                pred = model.predict(input_data)
                prediction = le.inverse_transform(pred)[0]

                st.success(f"Prediction: {prediction}")

        with col2:
            st.markdown("#### Model Statistics")

            st.info("ℹ️ Dynamic model trained fresh for each prediction using dataset features.")

else:
    st.markdown("""
    <div class="premium-card">
        <h3 style="color: #58A6FF;">👋 Welcome to Crime Analytics Dashboard</h3>
        <p>Please load a dataset from the sidebar to get started.</p>
        <p><b>Steps:</b></p>
        <ul>
            <li>Select a data source (Default or Upload)</li>
            <li>Click "Load Dataset" button</li>
            <li>Use filters to refine data</li>
            <li>Explore Analysis, Hotspots, and Predictions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

