import streamlit as st

def set_page_configuration():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Crime Analytics Dashboard",
        page_icon="🚓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Dark theme CSS
    st.markdown("""
    <style>
    .stApp { 
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%); 
        color: white; 
        font-family: 'Inter', sans-serif; 
    }
    .header-container {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #1f6feb 0%, #58A6FF 100%);
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(31, 111, 235, 0.3);
    }
    .subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    .premium-card {
        background: linear-gradient(145deg, #1a1d2e, #16213e);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #1f6feb;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin: 1rem 0;
    }
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(12px);
    }
    </style>
    """, unsafe_allow_html=True)
