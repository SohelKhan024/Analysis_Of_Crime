"""
BACKEND.PY - Data Processing & Machine Learning Models
Handles all data operations and ML algorithms
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st  # for optional warnings


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
        """Dynamic feature preparation using available numeric + categorical cols."""
        if df.empty:
            return pd.DataFrame()

        # Dynamic numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Categorical candidates for encoding
        cat_candidates = ['area_name', 'crm_cd_desc', 'day_of_week']
        cat_cols = [c for c in cat_candidates if c in df.columns]
        
        features_df = df[numeric_cols + cat_cols].copy().fillna(0)
        
        # Encode categoricals
        for col in cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                features_df[col] = le.fit_transform(df[col].astype(str))
                self.feature_encoders[col] = le
        
        return features_df

    def train(self, df, target_col=None):
        """Dynamic training with auto target selection and available features."""
        if df.empty:
            self.model = None
            return None

        # Auto-select target
        target_candidates = ['crm_cd_desc', 'crime', 'crime_type', 'offense', 'area_name']
        target_col = target_col or next((c for c in target_candidates if c in df.columns), None)
        
        if not target_col:
            print("No suitable target column found")
            self.model = None
            return None

        features_df = self.prepare_features(df)
        if features_df.empty or len(features_df.columns) < 2:
            print("Insufficient features for training")
            self.model = None
            return None

        train_df = features_df.dropna()
        if len(train_df) < 10:
            print("Insufficient training rows")
            self.model = None
            return None

        le_target = LabelEncoder()
        y = le_target.fit_transform(df[target_col].astype(str).iloc[train_df.index])
        self.feature_encoders['target'] = le_target

        X = train_df
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=15
        )
        self.model.fit(X, y)
        self.feature_columns = X.columns.tolist()

        print(f"✅ Trained on {len(train_df)} rows, {len(X.columns)} features")
        return self.model

    def predict_crime_type(self, hour, area_name, lat=0.0, lon=0.0):
        if self.model is None:
            return None

        # Create input matching training features
        input_data = {}
        
        # Always include hour if available
        if 'hour' in self.feature_columns:
            input_data['hour'] = hour
        
        # Encode area_name if feature exists
        if 'area_name' in self.feature_columns and 'area_name' in self.feature_encoders:
            le = self.feature_encoders['area_name']
            try:
                input_data['area_name'] = le.transform([area_name])[0]
            except:
                input_data['area_name'] = 0
        
        # Add lat/lon only if they were training features
        if 'lat' in self.feature_columns:
            input_data['lat'] = lat
        if 'lon' in self.feature_columns:
            input_data['lon'] = lon
        
        # Fill remaining features with 0
        for col in self.feature_columns:
            if col not in input_data:
                input_data[col] = 0.0
        
        X_input = np.array([[input_data[col] for col in self.feature_columns]])
        prediction = self.model.predict(X_input)[0]
        
        le_target = self.feature_encoders.get('target')
        if le_target:
            return le_target.inverse_transform([prediction])[0]
        return "Unknown"

    def get_feature_importance(self):
        if self.model is None:
            return None

        importances = self.model.feature_importances_
        features = self.feature_columns
        return dict(zip(features, importances))
