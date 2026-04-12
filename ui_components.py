"""
UI_COMPONENTS.PY - UI/UX Styling & Components
Handles all visual elements, styling, and reusable UI components
"""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ==================== STYLING ====================
def apply_custom_styling():
    """Apply premium SaaS styling with gradient, glow, and glass cards."""
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

            :root {
                --bg-1: #0b1020;
                --bg-2: #0f172a;
                --bg-3: #111827;
                --text-main: #e6edf3;
                --text-soft: #b8c1d1;
                --accent: #58a6ff;
                --accent-2: #8b5cf6;
                --accent-3: #10b981;
                --line: rgba(148, 163, 184, 0.22);
                --glass: rgba(17, 25, 40, 0.56);
            }

            html, body, [class*="css"]  {
                font-family: 'Inter', sans-serif;
            }

            .stApp {
                background:
                    radial-gradient(circle at 12% 10%, rgba(88,166,255,0.18), transparent 36%),
                    radial-gradient(circle at 86% 16%, rgba(139,92,246,0.16), transparent 34%),
                    radial-gradient(circle at 72% 86%, rgba(16,185,129,0.12), transparent 38%),
                    linear-gradient(135deg, var(--bg-1) 0%, var(--bg-2) 45%, var(--bg-3) 100%);
                color: var(--text-main);
            }

            [data-testid="stMain"] {
                background: transparent;
            }

            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(2,6,23,0.96) 0%, rgba(15,23,42,0.96) 100%);
                border-right: 1px solid var(--line);
                backdrop-filter: blur(16px);
            }

            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] p,
            section[data-testid="stSidebar"] span {
                color: var(--text-main) !important;
            }

            .header-container {
                text-align: center;
                padding: 2.3rem 1.6rem;
                margin: 1rem 0 2rem 0;
                border-radius: 20px;
                background: linear-gradient(120deg, rgba(31,111,235,0.26), rgba(139,92,246,0.2), rgba(16,185,129,0.18));
                border: 1px solid rgba(148,163,184,0.35);
                box-shadow:
                    0 14px 42px rgba(2,6,23,0.58),
                    0 0 52px rgba(88,166,255,0.22),
                    inset 0 1px 0 rgba(255,255,255,0.08);
                position: relative;
                overflow: hidden;
            }

            .header-container::before {
                content: "";
                position: absolute;
                inset: -120% auto auto -30%;
                width: 42%;
                height: 280%;
                transform: rotate(24deg);
                background: linear-gradient(90deg, rgba(255,255,255,0.00), rgba(255,255,255,0.16), rgba(255,255,255,0.00));
                pointer-events: none;
                animation: sheen 7s linear infinite;
            }

            @keyframes sheen {
                0% { left: -45%; }
                100% { left: 130%; }
            }

            .header-glow {
                animation: glowPulse 2.8s ease-in-out infinite alternate;
            }

            @keyframes glowPulse {
                from { box-shadow: 0 10px 28px rgba(31,111,235,0.20), 0 0 22px rgba(139,92,246,0.14); }
                to { box-shadow: 0 16px 46px rgba(31,111,235,0.26), 0 0 34px rgba(16,185,129,0.18); }
            }

            h1, h2, h3, h4 {
                color: #dbeafe;
                letter-spacing: 0.2px;
            }

            h2 {
                border-left: 4px solid rgba(88,166,255,0.85);
                padding-left: 12px;
            }

            .subtitle, p {
                color: var(--text-soft);
            }

            .premium-card, .glass-card {
                background: linear-gradient(135deg, rgba(30,41,59,0.55), rgba(15,23,42,0.62));
                border: 1px solid rgba(148,163,184,0.26);
                border-radius: 18px;
                padding: 1.1rem 1.2rem;
                margin: 0.7rem 0 1rem 0;
                backdrop-filter: blur(14px);
                box-shadow: 0 10px 30px rgba(2,6,23,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
                transition: all 0.28s ease;
            }

            .premium-card:hover, .glass-card:hover {
                transform: translateY(-3px);
                border-color: rgba(88,166,255,0.45);
                box-shadow: 0 16px 38px rgba(2,6,23,0.5), 0 0 20px rgba(88,166,255,0.16);
            }

            .kpi-card {
                min-height: 132px;
            }

            .prediction-card {
                border-color: rgba(16,185,129,0.4);
                box-shadow: 0 12px 32px rgba(16,185,129,0.13);
            }

            [data-testid="metric-container"] {
                background: linear-gradient(145deg, rgba(30,41,59,0.62), rgba(15,23,42,0.75));
                border: 1px solid rgba(148,163,184,0.24);
                border-radius: 16px;
                padding: 18px;
                box-shadow: 0 10px 28px rgba(2,6,23,0.38);
                transition: all 0.25s ease;
            }

            [data-testid="metric-container"]:hover {
                transform: translateY(-3px);
                border-color: rgba(88,166,255,0.5);
                box-shadow: 0 14px 34px rgba(2,6,23,0.42), 0 0 16px rgba(88,166,255,0.12);
            }

            .stButton > button {
                background: linear-gradient(135deg, #1f6feb 0%, #7c3aed 55%, #10b981 100%);
                color: #ffffff !important;
                border: 1px solid rgba(255,255,255,0.18);
                border-radius: 12px;
                padding: 0.55rem 1rem;
                font-weight: 700;
                transition: all 0.25s ease;
                box-shadow: 0 8px 18px rgba(31,111,235,0.24);
            }

            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 26px rgba(31,111,235,0.30), 0 0 16px rgba(139,92,246,0.20);
                border-color: rgba(147,197,253,0.5);
            }

            [data-testid="stTabs"] [role="tablist"] {
                gap: 0.5rem;
                border-bottom: 1px solid var(--line);
            }

            [data-testid="stTabs"] [role="tab"] {
                color: #cbd5e1;
                background: rgba(30,41,59,0.38);
                border: 1px solid rgba(148,163,184,0.20);
                border-radius: 10px 10px 0 0;
                padding: 0.55rem 0.9rem;
                transition: all 0.2s ease;
            }

            [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
                color: #dbeafe;
                border-color: rgba(88,166,255,0.42);
                box-shadow: inset 0 -2px 0 rgba(88,166,255,0.65);
            }

            [data-testid="stDataFrame"] {
                border-radius: 14px;
                overflow: hidden;
                border: 1px solid rgba(148,163,184,0.24);
                background: rgba(15,23,42,0.50);
            }

            [data-testid="stFileUploaderDropzone"] {
                border: 2px dashed rgba(148,163,184,0.34);
                border-radius: 12px;
                background: rgba(30,41,59,0.28);
                transition: all 0.25s ease;
            }

            [data-testid="stFileUploaderDropzone"]:hover {
                border-color: rgba(88,166,255,0.7);
                box-shadow: 0 0 0 2px rgba(88,166,255,0.12) inset;
            }

            hr {
                border: none;
                height: 1px;
                background: linear-gradient(90deg, transparent, rgba(148,163,184,0.35), transparent);
                margin: 1.2rem 0 1.4rem 0;
            }

            .fade-up {
                animation: fadeUp 0.55s ease-out;
            }

            @keyframes fadeUp {
                from { opacity: 0; transform: translateY(12px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
    """, unsafe_allow_html=True)


def set_page_configuration():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Crime Data Analysis",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    apply_custom_styling()


# ==================== VISUALIZATION COMPONENTS ====================
def create_bar_chart(data, title, xlabel, ylabel, color='steelblue', kind='bar'):
    """Reusable bar chart component"""
    fig, ax = plt.subplots(figsize=(10, 5))
    if kind == 'bar':
        data.plot(kind='bar', ax=ax, color=color, edgecolor='black', linewidth=0.5)
    else:
        data.plot(kind='barh', ax=ax, color=color, edgecolor='black', linewidth=0.5)

    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    return fig


def create_pie_chart(data, title):
    """Reusable pie chart component"""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.Set3(range(len(data)))
    data.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_ylabel("")
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    return fig


def create_line_chart(data, title, xlabel, ylabel):
    """Reusable line chart component"""
    fig, ax = plt.subplots(figsize=(10, 5))
    data.plot(ax=ax, color='#dc3545', linewidth=2, marker='o', markersize=5)
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def create_scatter_plot(df, lon_col, lat_col, color_col, title, centers=None):
    """Reusable scatter plot for geographic data"""
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        df[lon_col],
        df[lat_col],
        c=df[color_col],
        cmap='tab20',
        alpha=0.6,
        s=20,
        edgecolors='none'
    )

    if centers is not None:
        ax.scatter(
            centers[:, 1],
            centers[:, 0],
            c='red',
            marker='X',
            s=300,
            edgecolors='black',
            linewidth=2,
            label='Centers',
            zorder=5
        )

    ax.set_xlabel('Longitude', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.tight_layout()

    return fig


def create_histogram(data, bins=30, title='', xlabel='', ylabel='Count'):
    """Reusable histogram component"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data, bins=bins, color='green', edgecolor='black', alpha=0.7)
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    return fig


# ==================== METRIC COMPONENTS ====================
def display_metrics(metrics_dict):
    """Display multiple metrics in a row"""
    cols = st.columns(len(metrics_dict))
    for i, (label, value) in enumerate(metrics_dict.items()):
        with cols[i]:
            st.metric(label, value)


def display_metrics_4col(metric1, value1, metric2, value2, metric3, value3, metric4, value4):
    """Display 4 metrics in a row with animations"""
    col1, col2, col3, col4 = st.columns(4)

    metrics_data = [
        (col1, metric1, value1),
        (col2, metric2, value2),
        (col3, metric3, value3),
        (col4, metric4, value4)
    ]

    for col, label, value in metrics_data:
        with col:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
                border: 2px solid rgba(255,255,255,0.1);
                transition: all 0.3s ease;
                animation: slideInUp 0.6s ease;
            " onmouseover="this.style.transform='translateY(-10px)'; this.style.boxShadow='0 12px 40px rgba(102, 126, 234, 0.4)';"
              onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 8px 32px rgba(102, 126, 234, 0.2)';">
                <h3 style="margin: 0 0 10px 0; font-size: 1.2em; font-weight: 600;">{label}</h3>
                <p style="margin: 0; font-size: 2em; font-weight: 900;">
                    {f'{value:,}' if isinstance(value, int) else value}
                </p>
            </div>
            """, unsafe_allow_html=True)


# ==================== FORM COMPONENTS ====================
def create_data_source_selector():
    """Create data source selection in sidebar"""
    st.markdown("## Configuration")
    data_source = st.radio("Data Source", ["Use Default Dataset", "Upload CSV"])

    if data_source == "Use Default Dataset":
        sample_size = st.slider("Sample size (records)", 10000, 100000, 50000, step=10000)
        return data_source, sample_size, None

    st.markdown("---")
    st.markdown("### 📤 Upload Your Crime Dataset")
    st.markdown("""
    **Supported Format:** CSV (.csv)
    **Maximum File Size:** 300 MB
    """)

    uploaded_file = st.file_uploader(
        "Drag & Drop or Click to Select CSV File",
        type="csv",
        accept_multiple_files=False,
        help="Upload a crime dataset CSV file (max 300MB). The app will automatically process and analyze your data."
    )

    if uploaded_file is not None:
        st.success(f"✅ File selected: {uploaded_file.name} ({uploaded_file.size / (1024**2):.2f} MB)")

    st.markdown("---")
    return data_source, None, uploaded_file


def create_prediction_inputs():
    """Create input fields for crime prediction"""
    col1, col2 = st.columns(2)

    with col1:
        hour = st.slider("Hour of Day", 0, 23, 12)
        lat = st.number_input("Latitude", value=34.0522, format="%.4f")

    with col2:
        lon = st.number_input("Longitude", value=-118.2437, format="%.4f")

    return hour, lat, lon


# ==================== INFO COMPONENTS ====================
def display_dataset_info(df):
    """Display dataset information"""
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Records:** {len(df):,}")
        st.write(f"**Columns:** {len(df.columns)}")
        st.write(f"**Missing Values:** {df.isnull().sum().sum():,}")

    with col2:
        if 'date_occ' in df.columns:
            st.write(f"**Start Date:** {df['date_occ'].min()}")
            st.write(f"**End Date:** {df['date_occ'].max()}")
        st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def display_hotspot_table(clusters, centers):
    """Display hotspot information in table"""
    hotspot_data = pd.DataFrame({
        'Hotspot ID': range(len(centers)),
        'Latitude': centers[:, 0].round(4),
        'Longitude': centers[:, 1].round(4),
        'Crime Count': [sum(clusters == i) for i in range(len(centers))]
    })

    st.dataframe(hotspot_data, use_container_width=True)
    return hotspot_data


# ==================== STATUS COMPONENTS ====================
def show_loading_spinner(message):
    """Show loading spinner"""
    with st.spinner(message):
        return True


def show_success_message(message):
    """Show success message with animation"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
        margin: 10px 0;
        animation: slideInUp 0.5s ease;
        font-weight: 500;
    ">
        ✅ {message}
    </div>
    """, unsafe_allow_html=True)


def show_error_message(message):
    """Show error message with animation"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);
        margin: 10px 0;
        animation: slideInUp 0.5s ease;
        font-weight: 500;
    ">
        ❌ {message}
    </div>
    """, unsafe_allow_html=True)


def show_info_message(message):
    """Show info message with animation"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        margin: 10px 0;
        animation: slideInUp 0.5s ease;
        font-weight: 500;
    ">
        ℹ️ {message}
    </div>
    """, unsafe_allow_html=True)


def show_warning_message(message):
    """Show warning message with animation"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(250, 112, 154, 0.3);
        margin: 10px 0;
        animation: slideInUp 0.5s ease;
        font-weight: 500;
    ">
        ⚠️ {message}
    </div>
    """, unsafe_allow_html=True)


# ==================== CHART STYLING ====================
def configure_chart_style():
    """Configure matplotlib and seaborn styling"""
    sns.set_style("darkgrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
