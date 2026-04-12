# 🚓 Analysis Of Crime

A smart, dataset-adaptive crime analytics dashboard built with Streamlit that provides real-time insights, hotspot detection, and AI-powered predictions.

---

## 🔥 Features

- 📊 **Dynamic Data Analysis**  
  Works with any uploaded dataset (no fixed schema required)

- 🗺️ **Smart Hotspot Detection**  
  - Uses map if lat/lon available  
  - Falls back to region-based analysis automatically  

- 🔮 **AI Prediction System**  
  - Predicts most affected area or category  
  - Adapts based on dataset structure  

- 🧠 **Explainable AI**  
  - Shows why a prediction was made  
  - Uses feature importance  

- ⚡ **Robust Data Handling**  
  - No crashes on missing columns  
  - Handles different formats intelligently  

---

## 🛠️ Tech Stack

- Python 🐍
- Streamlit
- Pandas
- Scikit-learn
- Plotly

---

## 🚀 How to Run

```bash
git clone https://github.com/your-username/Analysis_Of_Crime.git
cd Analysis_Of_Crime
pip install -r requirements.txt
streamlit run app_ultimate_saas.py
