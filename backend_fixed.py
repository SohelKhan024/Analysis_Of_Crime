"""
BACKEND.PY - Data Processing & Machine Learning Models
Handles all data operations and ML algorithms
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

class CrimeDataProcessor:
    def __init__(self, csv_path=''):
        self.csv_path = csv_path
        self.df = pd.DataFrame()
        self.has_coords = False

    def load_data(self, sample=False, n_rows=None):
        if self.csv_path:
            self.df = pd.read_csv(self.csv_path)
        if sample and len(self.df) > n_rows:
            self.df = self.df.sample(n_rows).reset_index(drop=True)

    def clean_data(self):
        """Robust data cleaning with lat/lon detection, normalization, safe conversion, no crashes. Compatible with Streamlit."""
        if self.df.empty:
            self.has_coords = False
            return

        try:
            # Step 1: Normalize column names (lowercase, strip spaces)
            self.df.columns = self.df.columns.str.strip().str.lower()

            # Step 2: Detect lat/lon variations
            lat_candidates = ['lat', 'latitude']
            lon_candidates = ['lon', 'lng', 'longitude']

            lat_col = None
            for candidate in lat_candidates:
                if candidate in self.df.columns:
                    lat_col = candidate
                    break

            lon_col = None
            for candidate in lon_candidates:
                if candidate in self.df.columns:
                    lon_col = candidate
                    break

            # Step 3: Rename if found
            if lat_col and lon_col:
                self.df = self.df.rename(columns={lat_col: 'lat', lon_col: 'lon'})
                self.df['lat'] = pd.to_numeric(self.df['lat'], errors='coerce')
                self.df['lon'] = pd.to_numeric(self.df['lon'], errors='coerce')
                initial_rows = len(self.df)
                self.df = self.df.dropna(subset=['lat', 'lon'])
                self.has_coords = True
                print(f"Kept {len(self.df)}/{initial_rows} rows with valid coordinates")
            else:
                st.warning("Dataset does not contain location columns (lat/lon). Map features disabled.")
                self.has_coords = False
            
            # Safe cleaning
            self.df = self.df.fillna('Unknown')
            
        except Exception as e:
            print(f"Cleaning error (non-fatal): {e}")
            self.has_coords = False

    def extract_features(self):
        if self.df.empty:
            return

        try:
            self.df['date'] = pd.to_datetime(self.df.get('date', pd.Timestamp.now()), errors='coerce')
            self.df['hour'] = self.df['date'].dt.hour.fillna(12).astype(int)
            self.df['day_of_week'] = self.df['date'].dt.day_name().fillna('Unknown')
        except:
            self.df['hour'] = 12
            self.df['day_of_week'] = 'Unknown'

        if 'area' in self.df.columns:
            self.df['area_name'] = 'Area ' + self.df['area'].astype(str)


class HotspotDetector:
    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters

    def detect_hotspots(self, df):
        """Detect hotspots from lat/lon coordinates"""
        if not df.has_coords:
            st.warning("No valid coordinates for hotspot detection.")
            return None, None
        coords = df[['lat', 'lon']].dropna().values
        if len(coords) < self.n_clusters:
            st.warning("Insufficient data points for clustering.")
            return None, None
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(coords)
        return clusters, kmeans.cluster_centers_


class CrimeRiskPredictor:
    def __init__(self):
        self.model = None

    def train(self, df):
        """Train model to predict crime type"""
        try:
            X = pd.DataFrame({'hour': df['hour'].fillna(12)})
            if df.has_coords:
                X['lat_mean'] = df['lat'].mean()
                X['lon_mean'] = df['lon'].mean()
            target_col = next((col for col in ['type', 'crime_type', 'crm_cd_desc'] if col in df.columns), list(df.columns)[0] if df.columns.size > 0 else 'type')
            y = LabelEncoder().fit_transform(df[target_col].astype(str))
            self.model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
            self.model.fit(X, y)
        except Exception as e:
            st.warning(f"Prediction model training skipped: {e}")

    def predict_crime_type(self, hour, area_name, lat, lon):
        """Predict crime type for given inputs"""
        if self.model is None:
            return "Unknown"
        try:
            X_input = pd.DataFrame({'hour': [hour], 'lat_mean': [lat], 'lon_mean': [lon]})
            pred = self.model.predict(X_input)[0]
            return f"Crime Type {pred}"
        except:
            return "Prediction unavailable"

