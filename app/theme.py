from __future__ import annotations

import streamlit as st


LIGHT_POWDER_BLUE = "#6B9FC0"
SOFT_SKY_BLUE = "#4A7FA8"
LIGHT_AZURE = "#2D6B94"
MEDIUM_STEEL_BLUE = "#1C5C8F"
DEEP_CERULEAN = "#0D4F7A"
RICH_TEAL_BLUE = "#004566"
MIDNIGHT_AZURE = "#003A5F"
WHITE = "#FFFFFF"


def apply_theme() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(135deg, {LIGHT_POWDER_BLUE} 0%, {SOFT_SKY_BLUE} 35%, {LIGHT_AZURE} 70%, {MIDNIGHT_AZURE} 100%);
        }}
        .block-container {{
            background-color: rgba(255, 255, 255, 0.96);
            padding: 2rem 2rem 4rem 2rem;
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        }}
        [data-testid="stSidebar"] {{
            background: {MIDNIGHT_AZURE};
            color: {WHITE};
        }}
        [data-testid="stSidebar"] * {{
            color: {WHITE} !important;
        }}
        h1, h2, h3, h4, h5 {{
            color: {RICH_TEAL_BLUE};
        }}
        span, p, label, li {{
            color: {MIDNIGHT_AZURE};
        }}
        .stButton > button, .stDownloadButton > button {{
            background: {MEDIUM_STEEL_BLUE} !important;
            color: {WHITE} !important;
            border: none;
            border-radius: 999px;
            padding: 0.45rem 1.15rem;
            font-weight: 700;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            background: {DEEP_CERULEAN} !important;
            color: {WHITE} !important;
            transform: translateY(-1px);
        }}
        [data-testid="stSidebar"] .stButton > button {{
            background: {LIGHT_AZURE} !important;
            border: 2px solid {SOFT_SKY_BLUE} !important;
            border-radius: 8px !important;
            margin-bottom: 0.45rem !important;
        }}
        [data-testid="stSidebar"] .stButton > button:hover {{
            background: {SOFT_SKY_BLUE} !important;
            transform: translateX(4px) !important;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            background-color: rgba(153, 192, 222, 0.9);
            border-radius: 999px;
            padding: 0.25rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {MEDIUM_STEEL_BLUE};
            border-radius: 999px;
        }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            background-color: {WHITE};
            color: {RICH_TEAL_BLUE};
            font-weight: 700;
        }}
        [data-testid="stFileUploader"] *,
        [data-testid="stTextInput"] *,
        [data-testid="stTextArea"] *,
        [data-testid="stNumberInput"] *,
        [data-testid="stSelectbox"] *,
        [data-testid="stMultiSelect"] *,
        [data-testid="stRadio"] *,
        [data-testid="stCheckbox"] * {{
            color: #101820 !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {RICH_TEAL_BLUE};
            font-weight: 800;
        }}
        code {{
            color: #101820 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
