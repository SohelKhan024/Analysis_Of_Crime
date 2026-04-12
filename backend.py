"""
BACKEND.PY - Data Processing & Machine Learning Models
Handles all data operations and ML algorithms
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


class CrimeDataProcessor:
    def __init__(self, csv_path=''):
        self.csv_path = csv_path
        self.df = pd.DataFrame()
        self.has_coords = False

    def load_data(self, sample=False, n_rows=None):
        if self.csv_path:
            self.df = pd.read_csv(self.csv_path)
        if sample and n_rows is not None and len(self.df) > n_rows:
            self.df = self.df.sample(n_rows).reset_index(drop=True)

    def clean_data(self):
        """Robust data cleaning with lat/lon detection, normalization, safe conversion, no crashes. Compatible with Streamlit."""
        if self.df.empty:
            self.has_coords = False
            return

        try:
            self.df.columns = self.df.columns.str.strip().str.lower()

            lat_candidates = ['lat', 'latitude']
            lon_candidates = ['lon', 'lng', 'longitude']

            lat_col = next((c for c in lat_candidates if c in self.df.columns), None)
            lon_col = next((c for c in lon_candidates if c in self.df.columns), None)

            if lat_col and lon_col:
                self.df = self.df.rename(columns={lat_col: 'lat', lon_col: 'lon'})
                self.df['lat'] = pd.to_numeric(self.df['lat'], errors='coerce')
                self.df['lon'] = pd.to_numeric(self.df['lon'], errors='coerce')

                initial_rows = len(self.df)
                self.df = self.df.dropna(subset=['lat', 'lon'])
                self.has_coords = True
                print(f"Kept {len(self.df)}/{initial_rows} rows with valid coordinates")
            else:
                import streamlit as st
                st.warning("Dataset does not contain location columns (lat/lon). Map features disabled.")
                self.has_coords = False

            self.df = self.df.dropna()

        except Exception as e:
            print(f"Cleaning error (non-fatal): {e}")
            self.has_coords = False

    def extract_features(self):
        if self.df.empty:
            return

        self.df['date'] = pd.to_datetime(self.df.get('date', pd.Timestamp.now()))
        self.df['hour'] = self.df['date'].dt.hour
        self.df['day_of_week'] = self.df['date'].dt.day_name()

        if 'area' in self.df.columns:
            self.df['area_name'] = 'Area ' + self.df['area'].astype(str)


class HotspotDetector:
    """Detect crime hotspots using KMeans clustering"""

    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters

    def detect_hotspots(self, df):
        """Detect hotspots from lat/lon coordinates"""
        if 'lat' not in df.columns or 'lon' not in df.columns:
            import streamlit as st
            st.warning("No valid coordinates for hotspot detection.")
            return None, None

        coords = df[['lat', 'lon']].dropna().values
        if len(coords) < self.n_clusters:
            import streamlit as st
            st.warning("Insufficient data points for clustering.")
            return None, None

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(coords)
        return clusters, kmeans.cluster_centers_


class CrimeRiskPredictor:
    """Predict crime risk based on hour, location, and type"""

    def __init__(self):
        self.model = None
        self.feature_encoders = {}
        self.feature_columns = None

    def prepare_features(self, df):
        features_df = df[['hour', 'area_name', 'crm_cd_desc', 'lat', 'lon']].copy()

        for col in ['area_name', 'crm_cd_desc']:
            le = LabelEncoder()
            features_df[col] = le.fit_transform(features_df[col].astype(str))
            self.feature_encoders[col] = le

        return features_df

    def train(self, df, target_col='crm_cd_desc'):
        required_cols = ['hour', 'area_name', 'lat', 'lon', target_col]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for training: {missing_cols}")

        train_df = df[required_cols].copy().dropna()
        if train_df.empty:
            raise ValueError("No valid rows available for training after preprocessing/filtering.")

        X = train_df[['hour', 'area_name', 'lat', 'lon']].copy()

        le_area = LabelEncoder()
        X['area_name'] = le_area.fit_transform(X['area_name'].astype(str))
        self.feature_encoders['area_name'] = le_area

        le_crime = LabelEncoder()
        y = le_crime.fit_transform(train_df[target_col].astype(str))
        self.feature_encoders['crm_cd_desc'] = le_crime

        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=15
        )
        self.model.fit(X, y)
        self.feature_columns = X.columns

        return self.model

    def predict_crime_type(self, hour, area_name, lat, lon):
        if self.model is None:
            return None

        le_area = self.feature_encoders.get('area_name')
        try:
            area_encoded = le_area.transform([area_name])[0]
        except Exception:
            area_encoded = 0

        X_input = np.array([[hour, area_encoded, lat, lon]])
        prediction = self.model.predict(X_input)[0]

        le_crime = self.feature_encoders.get('crm_cd_desc')
        crime_type = le_crime.inverse_transform([prediction])[0]

        return crime_type

    def get_feature_importance(self):
        if self.model is None:
            return None

        importances = self.model.feature_importances_
        features = self.feature_columns
        return dict(zip(features, importances))
