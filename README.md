# 🚨 Crime Data Analysis & Prediction Dashboard

A comprehensive data science project that analyzes crime data, detects hotspots, and predicts crime risk using machine learning.

## 📋 Features

✅ **Data Processing**
- Load and clean crime dataset (1M+ records)
- Remove duplicates and handle missing values
- Standardize column names and formats
- Extract temporal features (hour, day, month, year)

✅ **Visualizations**
- Crime distribution by hour, day, type, and location
- Top crime areas and crime types
- Temporal trends and patterns
- Victim demographics analysis

✅ **Crime Hotspot Detection**
- KMeans clustering on latitude/longitude coordinates
- Identify 3-15 crime concentration zones
- Interactive visualization of hotspots
- Crime count per hotspot

✅ **Crime Risk Prediction**
- RandomForest model to predict crime type
- Predictions based on: hour, location, area
- Feature importance analysis
- Interactive prediction interface

✅ **Web Application**
- Streamlit-based interactive dashboard
- Multiple tabs for different analyses
- CSV upload support
- Real-time predictions

## 🚀 Installation & Setup

### 1. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### 2. Project Structure (3 Main Files)

```
c:/Users/saraj/Desktop/DS Project/
├── app.py                   # FRONTEND - Streamlit web application
├── backend.py               # BACKEND - Data processing & ML models
├── ui_components.py         # UI/UX - Styling & reusable components
└── README.md                # This file
```

### 3. Data Location

The crime dataset should be located at:
```
c:/Users/saraj/Desktop/Csv file DS/Crime_Data_from_2020_to_Present.csv
```

## 📊 Running the Application

### Start the Streamlit App

```bash
cd c:/Users/saraj/Desktop/DS\ Project
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 🎯 Usage Guide

### Tab 1: Overview 📊
- View total crimes, unique locations, crime types
- Dataset preview and statistics
- Data info and memory usage

### Tab 2: Visualizations 📈
- Crime count by hour (identify peak hours)
- Top 10 crime types
- Top 10 crime areas
- Crime distribution by day of week

### Tab 3: Hotspots 🎯
- Adjust number of hotspots (3-15)
- Detect crime concentration zones
- View hotspot locations and statistics
- Interactive scatter plot with cluster centers

### Tab 4: Prediction 🔮
- **Option 1:** Predict crime type by hour and area
- **Option 2:** Interactive map-based prediction
- View model feature importance
- Enter custom coordinates and time for predictions

### Tab 5: Analytics ⚙️
- Crime trends over time
- Year-over-year comparisons
- Victim demographics:
  - Gender distribution
  - Age distribution
  - Descent/ethnicity analysis

## 📈 Model Details

### Hotspot Detection
- **Algorithm:** K-Means Clustering
- **Features:** Latitude, Longitude
- **Scaling:** StandardScaler normalization
- **Output:** Cluster centers and assignments

### Crime Risk Prediction
- **Algorithm:** Random Forest Classifier
- **Features:** Hour, Area (encoded), Latitude, Longitude
- **Target:** Crime Type (Crm Cd Desc)
- **Parameters:** 100 trees, max_depth=15
- **Accuracy:** Varies based on data sample size

## 🔧 Configuration

### Data Source Options
1. **Use Default Dataset** - Loads from CSV with configurable sample size (10K-100K records)
2. **Upload CSV** - Upload your own crime dataset

### Hotspot Parameters
- Minimum hotspots: 3
- Maximum hotspots: 15
- Default: 8

### Sample Size
- Minimum: 10,000 records
- Maximum: 100,000 records
- Recommended: 50,000 for balance

## 📊 Dataset Info

**Default Dataset:** Crime_Data_from_2020_to_Present.csv
- **Records:** 1,005,198
- **Columns:** 28
- **Date Range:** 2020-Present
- **Key Columns:**
  - DATE OCC: Occurrence date/time
  - TIME OCC: Crime time (HHMM format)
  - Crm Cd Desc: Crime type description
  - LAT, LON: Geographic coordinates
  - AREA NAME: Police reporting area
  - Vict Sex, Vict Age, Vict Descent: Victim info

## 🐛 Troubleshooting

### Libraries not found
```bash
pip install --upgrade pandas numpy matplotlib seaborn scikit-learn streamlit
```

### CSV file not found
Ensure the data file exists at:
`c:/Users/saraj/Desktop/Csv file DS/Crime_Data_from_2020_to_Present.csv`

### Streamlit port already in use
```bash
streamlit run app.py --server.port 8502
```

### Out of memory with large samples
Reduce sample size in sidebar to 10K-30K records

## 📝 Code Overview

### 1. **backend.py** - Data Processing & Machine Learning
Core business logic and ML models:
- **CrimeDataProcessor**: Data loading and cleaning
  - `load_data()`: Load CSV with optional sampling
  - `clean_data()`: Remove duplicates, standardize column names, handle missing values
  - `extract_features()`: Convert dates, extract temporal features

- **HotspotDetector**: KMeans clustering for geographic hotspots
  - `detect_hotspots()`: Find crime concentration zones

- **CrimeRiskPredictor**: RandomForest model for crime type prediction
  - `train()`: Train prediction model
  - `predict_crime_type()`: Make predictions on new data
  - `get_feature_importance()`: Analyze model features

### 2. **ui_components.py** - Styling & Reusable UI Components
Visual elements and UI styling:
- **Styling Functions:**
  - `set_page_configuration()`: Configure page settings
  - `configure_chart_style()`: Setup matplotlib/seaborn styles

- **Chart Components:**
  - `create_bar_chart()`: Bar chart visualization
  - `create_pie_chart()`: Pie chart visualization
  - `create_line_chart()`: Time series visualization
  - `create_scatter_plot()`: Geographic scatter plots
  - `create_histogram()`: Histogram visualization

- **Form Components:**
  - `create_data_source_selector()`: Data loading UI
  - `create_prediction_inputs()`: Prediction form inputs
  - `display_metrics_4col()`: KPI cards

- **Status Components:**
  - Alerts and messages (success, error, info, warning)

### 3. **app.py** - Frontend Web Application
Interactive Streamlit dashboard with 5 main tabs:

- **Tab 1 - Overview 📊:** Dataset stats, preview, data info
- **Tab 2 - Visualizations 📈:** Crime patterns (hour, type, area, day)
- **Tab 3 - Hotspots 🎯:** KMeans clustering visualization
- **Tab 4 - Prediction 🔮:** Crime type prediction interface
- **Tab 5 - Analytics ⚙️:** Trends, demographics, analysis

## 💡 Tips & Best Practices

1. **Start with smaller samples** (10K) to test, then increase for accuracy
2. **Monitor memory usage** - large samples may slow down visualizations
3. **Predictions work better with** more training data
4. **Hotspots are most effective** with 6-10 clusters for this dataset
5. **Hour 0** = midnight, **Hour 23** = 11 PM

## 🎓 Learning Goals

This project demonstrates:
- ✅ Data cleaning and preprocessing
- ✅ Feature engineering and extraction
- ✅ Unsupervised learning (KMeans clustering)
- ✅ Supervised learning (RandomForest classification)
- ✅ Interactive web applications (Streamlit)
- ✅ Data visualization (Matplotlib, Seaborn)
- ✅ Geospatial analysis

## 📚 References

- [Streamlit Documentation](https://docs.streamlit.io)
- [Scikit-learn Documentation](https://scikit-learn.org)
- [Pandas Documentation](https://pandas.pydata.org)
- [Matplotlib Documentation](https://matplotlib.org)

---

**Created:** 2026-04-08
**Last Updated:** 2026-04-08
