import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

# Core modules (unchanged)
from scripts.margin_model import compute_margin
from scripts.scenario_runner import run_sensitivity_scenarios
from scripts.risk_label import risk_label
from scripts.risk_adjuster import adjust_risk
from scripts.confidence_band import confidence_multiplier

# New analytical enhancement modules
from scripts.trend_analysis import (
    analyse_fx_trends,
    analyse_import_trends,
    simulate_historical_margins,
    calculate_margin_trends,
    calculate_fx_percentile_position
)
from scripts.forecast_engine import (
    forecast_fx_rate,
    forecast_margins,
    detect_margin_anomalies,
    detect_cost_anomalies
)
from scripts.advanced_risk_metrics import (
    calculate_var_historical,
    calculate_margin_var,
    calculate_historical_volatility,
    volatility_regime_detection,
    analyse_commodity_fx_correlation,
    decompose_margin_risk,
    run_stress_test,
    get_predefined_stress_scenarios
)


# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="UK SME Import Margin Simulator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# SESSION STATE INIT
# ======================
if "analytics_enabled" not in st.session_state:
    st.session_state.analytics_enabled = True
if "selected_currency" not in st.session_state:
    st.session_state.selected_currency = "EUR"

# ======================
# HS2 CHAPTER DESCRIPTIONS
# ======================
HS2_DESCRIPTIONS = {
    1: "Live animals", 2: "Meat and edible meat offal", 3: "Fish and crustaceans",
    4: "Dairy produce, eggs, honey", 5: "Products of animal origin", 6: "Live trees and plants",
    7: "Edible vegetables", 8: "Edible fruit and nuts", 9: "Coffee, tea, spices",
    10: "Cereals", 11: "Milling products, malt, starches", 12: "Oil seeds, miscellaneous grains",
    13: "Lac, gums, resins", 14: "Vegetable plaiting materials", 15: "Animal or vegetable fats",
    16: "Preparations of meat or fish", 17: "Sugars and sugar confectionery", 18: "Cocoa and cocoa preparations",
    19: "Preparations of cereals", 20: "Preparations of vegetables, fruit", 21: "Miscellaneous edible preparations",
    22: "Beverages, spirits and vinegar", 23: "Food industry residues", 24: "Tobacco and substitutes",
    25: "Salt, sulphur, earth and stone", 26: "Ores, slag and ash", 27: "Mineral fuels, oils",
    28: "Inorganic chemicals", 29: "Organic chemicals", 30: "Pharmaceutical products",
    31: "Fertilisers", 32: "Tanning or dyeing extracts", 33: "Essential oils and perfumery",
    34: "Soap, washing preparations", 35: "Albuminoidal substances, glues", 36: "Explosives, pyrotechnics",
    37: "Photographic goods", 38: "Miscellaneous chemical products", 39: "Plastics and articles",
    40: "Rubber and articles", 41: "Raw hides, skins and leather", 42: "Articles of leather",
    43: "Furskins and artificial fur", 44: "Wood and articles of wood", 45: "Cork and articles",
    46: "Manufactures of straw", 47: "Pulp of wood", 48: "Paper and paperboard",
    49: "Printed books, newspapers", 50: "Silk", 51: "Wool and fine animal hair",
    52: "Cotton", 53: "Other vegetable textile fibres", 54: "Man-made filaments",
    55: "Man-made staple fibres", 56: "Wadding, felt and nonwovens", 57: "Carpets and textile floor coverings",
    58: "Special woven fabrics", 59: "Impregnated textile fabrics", 60: "Knitted or crocheted fabrics",
    61: "Knitted apparel and accessories", 62: "Woven apparel and accessories", 63: "Other made up textile articles",
    64: "Footwear", 65: "Headgear", 66: "Umbrellas, walking sticks",
    67: "Prepared feathers", 68: "Articles of stone, plaster, cement", 69: "Ceramic products",
    70: "Glass and glassware", 71: "Precious stones and metals", 72: "Iron and steel",
    73: "Articles of iron or steel", 74: "Copper and articles", 75: "Nickel and articles",
    76: "Aluminium and articles", 78: "Lead and articles", 79: "Zinc and articles",
    80: "Tin and articles", 81: "Other base metals", 82: "Tools of base metal",
    83: "Miscellaneous articles of base metal", 84: "Nuclear reactors, boilers, machinery", 85: "Electrical machinery",
    86: "Railway locomotives", 87: "Vehicles other than railway", 88: "Aircraft and spacecraft",
    89: "Ships and boats", 90: "Optical and medical instruments", 91: "Clocks and watches",
    92: "Musical instruments", 93: "Arms and ammunition", 94: "Furniture and bedding",
    95: "Toys, games and sports equipment", 96: "Miscellaneous manufactured articles", 97: "Works of art and antiques",
    99: "Special transactions"
}

# ======================
# CUSTOM CSS STYLING
# ======================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

    :root {
        --bg-base: #06060f;
        --bg-surface: #0c0c1a;
        --bg-card: #101020;
        --bg-elevated: #14142a;
        --accent: #6366f1;
        --accent-light: #a5b4fc;
        --accent-violet: #8b5cf6;
        --accent-cyan: #06b6d4;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #f43f5e;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --text-muted: #64748b;
        --border: rgba(255,255,255,0.07);
        --border-accent: rgba(99,102,241,0.3);
        --shadow: 0 4px 24px rgba(0,0,0,0.5);
        --shadow-lg: 0 8px 48px rgba(0,0,0,0.65);
        --glow: 0 0 40px rgba(99,102,241,0.12);
        --glow-lg: 0 0 80px rgba(99,102,241,0.18);
    }

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Inter', 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
        color: var(--text-primary);
    }

    /* ── App Background ── */
    .stApp {
        background:
            radial-gradient(ellipse 140% 60% at 50% -5%, rgba(99,102,241,0.18) 0%, transparent 55%),
            radial-gradient(ellipse 70% 50% at 0% 60%, rgba(139,92,246,0.09) 0%, transparent 50%),
            radial-gradient(ellipse 60% 40% at 100% 90%, rgba(6,182,212,0.07) 0%, transparent 50%),
            linear-gradient(180deg, #06060f 0%, #080814 60%, #060610 100%);
        background-attachment: fixed;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    div[data-testid="stVerticalBlock"] > div {
        gap: 0.75rem;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #07070f 0%, #0a0a1a 100%);
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] p {
        color: #cbd5e1 !important;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
    }

    /* ── Sidebar: Select dropdowns ── */
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #1a1a2e !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] span,
    section[data-testid="stSidebar"] [data-baseweb="select"] [class*="placeholder"],
    section[data-testid="stSidebar"] [data-baseweb="select"] [class*="singleValue"],
    section[data-testid="stSidebar"] [data-baseweb="select"] [class*="Input"] {
        color: #f1f5f9 !important;
    }

    /* ── Sidebar: Number inputs – hard-coded hex so CSS vars can't fail ── */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] input[type="number"],
    section[data-testid="stSidebar"] input[type="text"],
    section[data-testid="stSidebar"] .stNumberInput input,
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
        background-color: #1a1a2e !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        border-radius: 10px !important;
        caret-color: #6366f1 !important;
        -webkit-text-fill-color: #f1f5f9 !important;
    }
    section[data-testid="stSidebar"] input:focus,
    section[data-testid="stSidebar"] .stNumberInput input:focus {
        border-color: rgba(99,102,241,0.55) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
        outline: none !important;
    }
    /* Number input +/– stepper buttons */
    section[data-testid="stSidebar"] .stNumberInput button,
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
        background: rgba(255,255,255,0.06) !important;
        color: #cbd5e1 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }
    section[data-testid="stSidebar"] .stNumberInput button:hover {
        background: rgba(99,102,241,0.15) !important;
        color: #a5b4fc !important;
    }

    /* ── Sidebar collapse / open arrow button — broad net for all Streamlit versions ── */

    /* The button element itself (open sidebar state) */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapseButton"] > button,
    section[data-testid="stSidebar"] button[data-testid="baseButton-headerNoPadding"],
    section[data-testid="stSidebar"] [data-testid="baseButton-headerNoPadding"],
    button[data-testid="baseButton-headerNoPadding"],
    button[data-testid="stBaseButton-headerNoPadding"],
    [data-testid="stBaseButton-headerNoPadding"] {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebarCollapseButton"] button:hover,
    [data-testid="stSidebarCollapseButton"]:hover,
    button[data-testid="baseButton-headerNoPadding"]:hover,
    button[data-testid="stBaseButton-headerNoPadding"]:hover {
        background: rgba(99,102,241,0.25) !important;
        border-color: rgba(99,102,241,0.55) !important;
        color: #ffffff !important;
    }

    /* SVG icon — both fill and currentColor path */
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stSidebarCollapseButton"] svg path,
    [data-testid="stSidebarCollapseButton"] svg polyline,
    [data-testid="stSidebarCollapseButton"] svg line,
    section[data-testid="stSidebar"] button[data-testid="baseButton-headerNoPadding"] svg,
    section[data-testid="stSidebar"] button[data-testid="baseButton-headerNoPadding"] svg path,
    button[data-testid="baseButton-headerNoPadding"] svg,
    button[data-testid="baseButton-headerNoPadding"] svg path,
    button[data-testid="stBaseButton-headerNoPadding"] svg,
    button[data-testid="stBaseButton-headerNoPadding"] svg path {
        fill: #ffffff !important;
        stroke: #ffffff !important;
        color: #ffffff !important;
    }

    /* Collapsed state — arrow shown in main content when sidebar is hidden */
    [data-testid="collapsedControl"],
    [data-testid="collapsedControl"] button,
    [data-testid="collapsedControl"] > button {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }
    [data-testid="collapsedControl"] button:hover,
    [data-testid="collapsedControl"]:hover {
        background: rgba(99,102,241,0.25) !important;
        border-color: rgba(99,102,241,0.55) !important;
    }
    [data-testid="collapsedControl"] svg,
    [data-testid="collapsedControl"] svg path,
    [data-testid="collapsedControl"] svg polyline,
    [data-testid="collapsedControl"] svg line {
        fill: #ffffff !important;
        stroke: #ffffff !important;
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* ── Stress Test Table ── */
    .stress-table-wrap {
        background: rgba(10,10,20,0.8);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 24px rgba(0,0,0,0.5);
        margin-bottom: 1rem;
    }
    .stress-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.83rem;
        font-family: 'Inter', sans-serif;
    }
    .stress-table thead tr {
        background: rgba(99,102,241,0.12);
        border-bottom: 1px solid rgba(99,102,241,0.25);
    }
    .stress-table thead th {
        padding: 0.7rem 1rem;
        text-align: left;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #a5b4fc;
        white-space: nowrap;
    }
    .stress-table tbody tr {
        border-bottom: 1px solid rgba(255,255,255,0.04);
        transition: background 0.15s ease;
    }
    .stress-table tbody tr:last-child { border-bottom: none; }
    .stress-table tbody tr:hover { background: rgba(99,102,241,0.06) !important; }
    .stress-table tbody tr:nth-child(even) { background: rgba(255,255,255,0.02); }
    .stress-table tbody td {
        padding: 0.6rem 1rem;
        color: #cbd5e1;
        vertical-align: middle;
    }
    .stress-table tbody td:first-child { color: #f1f5f9; font-weight: 600; }
    .stress-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.76rem;
        letter-spacing: 0.03em;
    }
    .stress-badge-loss     { background: rgba(244,63,94,0.14);  color: #fca5a5; border: 1px solid rgba(244,63,94,0.3);  }
    .stress-badge-caution  { background: rgba(245,158,11,0.14); color: #fcd34d; border: 1px solid rgba(245,158,11,0.3); }
    .stress-badge-healthy  { background: rgba(16,185,129,0.14); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); }
    .stress-row-loss   { background: rgba(244,63,94,0.05) !important; }
    .stress-row-caution{ background: rgba(245,158,11,0.04) !important; }
    .stress-row-healthy{ background: rgba(16,185,129,0.04) !important; }

    /* ── Dashboard Hero Header ── */
    .dashboard-header {
        background:
            linear-gradient(135deg, rgba(99,102,241,0.10) 0%, rgba(139,92,246,0.05) 50%, rgba(6,182,212,0.04) 100%),
            rgba(12,12,26,0.85);
        border: 1px solid var(--border-accent);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-radius: 22px;
        padding: 2rem 2.4rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-lg), var(--glow-lg);
        position: relative;
        overflow: hidden;
    }

    .dashboard-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(99,102,241,0.7) 30%, rgba(168,85,247,0.7) 60%, transparent 100%);
    }

    .dashboard-header::after {
        content: '';
        position: absolute;
        bottom: -60px; right: -60px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .dashboard-header .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent-light);
        background: rgba(99,102,241,0.14);
        border: 1px solid rgba(99,102,241,0.28);
        padding: 0.28rem 0.75rem;
        border-radius: 999px;
        margin-bottom: 0.85rem;
    }

    .dashboard-header h1 {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
        background: linear-gradient(135deg, #f8fafc 0%, #a5b4fc 45%, #67e8f9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0 0 0.55rem 0;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }

    .dashboard-header p {
        color: var(--text-muted);
        font-size: 0.96rem;
        margin: 0 0 1.1rem 0;
        max-width: 800px;
        line-height: 1.65;
    }

    .hero-tags { display: flex; flex-wrap: wrap; gap: 0.5rem; }

    .hero-tag {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        color: var(--text-secondary);
        font-size: 0.74rem;
        font-weight: 500;
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
        letter-spacing: 0.01em;
        transition: all 0.2s ease;
    }

    /* ── Section Headers ── */
    .section-header {
        display: flex;
        align-items: center;
        background: rgba(99,102,241,0.07);
        border: 1px solid rgba(99,102,241,0.18);
        border-left: 3px solid var(--accent);
        color: var(--text-primary);
        padding: 0.78rem 1.15rem;
        border-radius: 12px;
        margin: 1.25rem 0 0.45rem 0;
        font-size: 0.94rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        box-shadow: 0 4px 20px rgba(99,102,241,0.07);
    }

    .section-description {
        color: var(--text-muted);
        font-size: 0.84rem;
        margin: 0 0 0.8rem 0;
        line-height: 1.65;
    }

    /* ── Generic Card ── */
    .card {
        background: rgba(14,14,28,0.85);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.3rem;
        margin-bottom: 0.75rem;
        box-shadow: var(--shadow);
        backdrop-filter: blur(12px);
    }

    /* ── Metric Cards ── */
    .metric-card {
        padding: 1.4rem 1.2rem;
        border-radius: 18px;
        text-align: left;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        min-height: 148px;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(12px);
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent);
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
        border-color: rgba(255,255,255,0.13);
    }

    .metric-card-profit {
        background: linear-gradient(140deg, rgba(16,185,129,0.16) 0%, rgba(5,150,105,0.07) 100%), rgba(8,8,18,0.95);
        border-color: rgba(16,185,129,0.22);
        box-shadow: var(--shadow), 0 0 40px rgba(16,185,129,0.07);
        color: #ffffff;
    }

    .metric-card-loss {
        background: linear-gradient(140deg, rgba(244,63,94,0.17) 0%, rgba(190,18,60,0.07) 100%), rgba(8,8,18,0.95);
        border-color: rgba(244,63,94,0.22);
        box-shadow: var(--shadow), 0 0 40px rgba(244,63,94,0.07);
        color: #ffffff;
    }

    .metric-card-margin {
        background: linear-gradient(140deg, rgba(99,102,241,0.2) 0%, rgba(79,70,229,0.08) 100%), rgba(8,8,18,0.95);
        border-color: rgba(99,102,241,0.28);
        box-shadow: var(--shadow), 0 0 40px rgba(99,102,241,0.09);
        color: #ffffff;
    }

    .metric-card-cost {
        background: linear-gradient(140deg, rgba(6,182,212,0.14) 0%, rgba(8,145,178,0.06) 100%), rgba(8,8,18,0.95);
        border-color: rgba(6,182,212,0.2);
        box-shadow: var(--shadow), 0 0 40px rgba(6,182,212,0.07);
        color: #ffffff;
    }

    .metric-card-revenue {
        background: linear-gradient(140deg, rgba(245,158,11,0.14) 0%, rgba(180,83,9,0.06) 100%), rgba(8,8,18,0.95);
        border-color: rgba(245,158,11,0.2);
        box-shadow: var(--shadow), 0 0 40px rgba(245,158,11,0.07);
        color: #ffffff;
    }

    .metric-value {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
        font-size: 1.75rem;
        font-weight: 800;
        margin: 0.55rem 0 0.28rem 0;
        letter-spacing: -0.025em;
        color: #ffffff;
    }

    .metric-label {
        font-size: 0.71rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-weight: 600;
    }

    /* ── Badges ── */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.32rem 0.85rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.73rem;
        letter-spacing: 0.05em;
        backdrop-filter: blur(8px);
        margin: 0.3rem 0.2rem 0.3rem 0;
    }

    .risk-high  { background: rgba(244,63,94,0.13);  color: #fca5a5; border: 1px solid rgba(244,63,94,0.3);  }
    .risk-moderate { background: rgba(245,158,11,0.13); color: #fcd34d; border: 1px solid rgba(245,158,11,0.3); }
    .risk-low   { background: rgba(16,185,129,0.13); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); }

    .coverage-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.28rem 0.72rem;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    .coverage-high    { background: rgba(16,185,129,0.13); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3);  }
    .coverage-partial { background: rgba(245,158,11,0.13); color: #fcd34d; border: 1px solid rgba(245,158,11,0.3); }
    .coverage-low     { background: rgba(244,63,94,0.13);  color: #fca5a5; border: 1px solid rgba(244,63,94,0.3);  }
    .coverage-none    { background: rgba(100,116,139,0.13);color: #94a3b8; border: 1px solid rgba(100,116,139,0.3);}

    /* ── Info Box ── */
    .info-box {
        background: rgba(99,102,241,0.06);
        border: 1px solid rgba(99,102,241,0.18);
        border-left: 3px solid var(--accent);
        padding: 1.1rem 1.25rem;
        border-radius: 14px;
        margin: 0.65rem 0;
        font-size: 0.84rem;
        line-height: 1.65;
        color: var(--text-secondary);
        box-shadow: 0 4px 20px rgba(99,102,241,0.05);
    }

    .info-box strong { color: var(--text-primary); }

    /* ── Commodity Card ── */
    .commodity-card {
        background: rgba(12,12,24,0.85);
        padding: 1.1rem;
        border-radius: 14px;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        margin: 0.75rem 0;
        backdrop-filter: blur(12px);
        transition: border-color 0.2s ease;
    }

    .commodity-card:hover { border-color: var(--border-accent); }

    .commodity-category {
        font-size: 0.68rem;
        color: var(--accent-light);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .commodity-name {
        font-size: 0.92rem;
        color: var(--text-primary);
        font-weight: 600;
        margin-top: 0.38rem;
    }

    .coverage-meter {
        background: rgba(255,255,255,0.06);
        border-radius: 999px;
        height: 5px;
        margin-top: 0.65rem;
        overflow: hidden;
    }

    .coverage-fill { height: 100%; border-radius: 999px; }
    .coverage-fill-high    { background: linear-gradient(90deg, #10b981, #34d399); }
    .coverage-fill-partial { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .coverage-fill-low     { background: linear-gradient(90deg, #f43f5e, #fb7185); }
    .coverage-fill-none    { background: rgba(100,116,139,0.45); }

    /* ── Metric Mini ── */
    .metric-mini {
        background: rgba(12,12,24,0.85);
        padding: 1.1rem;
        border-radius: 14px;
        text-align: center;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        backdrop-filter: blur(12px);
        transition: all 0.2s ease;
    }

    .metric-mini:hover { border-color: var(--border-accent); box-shadow: var(--shadow), var(--glow); }

    .metric-mini h3 { color: var(--text-muted); font-size: 0.74rem; font-weight: 600; margin: 0 0 0.35rem 0; text-transform: uppercase; letter-spacing: 0.07em; }
    .metric-mini h2 { color: var(--text-primary); font-size: 1.45rem; font-weight: 800; margin: 0 0 0.2rem 0; letter-spacing: -0.025em; }
    .metric-mini p  { color: var(--text-muted); font-size: 0.74rem; margin: 0; }

    /* ── Native Streamlit st.metric ── */
    div[data-testid="stMetric"] {
        background: rgba(12,12,24,0.85);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        box-shadow: var(--shadow);
        backdrop-filter: blur(12px);
        transition: all 0.2s ease;
    }

    div[data-testid="stMetric"]:hover { border-color: var(--border-accent); }

    div[data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 800 !important;
        font-size: 1.4rem !important;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] [role="tablist"] {
        gap: 0.3rem;
        background: rgba(8,8,18,0.8);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.4rem;
        backdrop-filter: blur(12px);
    }

    [data-testid="stTabs"] [role="tab"] {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.84rem;
        color: var(--text-muted);
        border: 1px solid transparent;
        padding: 0.45rem 1rem;
        transition: all 0.2s ease;
        letter-spacing: 0.01em;
    }

    [data-testid="stTabs"] [role="tab"]:hover {
        color: var(--text-secondary);
        background: rgba(255,255,255,0.04);
    }

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.22), rgba(139,92,246,0.14));
        color: var(--accent-light);
        border-color: rgba(99,102,241,0.28);
        box-shadow: 0 0 20px rgba(99,102,241,0.1);
    }

    /* ── Chart / DataFrame containers ── */
    .stPlotlyChart {
        background: rgba(8,8,18,0.6);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 0.5rem;
        box-shadow: var(--shadow);
        backdrop-filter: blur(12px);
    }

    .stDataFrame {
        background: rgba(8,8,18,0.6);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.5rem;
        box-shadow: var(--shadow);
    }

    .stExpander {
        background: rgba(8,8,18,0.6) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        box-shadow: var(--shadow);
    }

    /* ── Cost Table ── */
    .cost-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.83rem;
        font-family: 'Inter', sans-serif;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border);
    }

    .cost-table th {
        background: rgba(99,102,241,0.10);
        color: var(--text-secondary);
        text-align: left;
        padding: 0.58rem 0.9rem;
        font-weight: 700;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        border-bottom: 1px solid var(--border);
    }

    .cost-table td {
        padding: 0.46rem 0.9rem;
        border-bottom: 1px solid var(--border);
        color: var(--text-secondary);
        background: rgba(8,8,18,0.4);
    }

    .cost-table tr:hover td {
        background: rgba(99,102,241,0.05);
        color: var(--text-primary);
    }

    .cost-table .num {
        text-align: right;
        font-variant-numeric: tabular-nums;
        font-weight: 500;
    }

    .cost-table tr:last-child td {
        background: rgba(99,102,241,0.09);
        font-weight: 700;
        color: var(--accent-light);
        border-bottom: none;
        padding-top: 0.58rem;
        padding-bottom: 0.58rem;
    }

    /* ── Footer Cards ── */
    .footer-card {
        background: rgba(10,10,20,0.85);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        box-shadow: var(--shadow);
        min-height: 150px;
        color: var(--text-muted);
        font-size: 0.84rem;
        line-height: 1.75;
        backdrop-filter: blur(12px);
    }

    .footer-card strong {
        color: var(--text-secondary);
        display: block;
        margin-bottom: 0.45rem;
        font-size: 0.88rem;
        letter-spacing: 0.02em;
    }

    /* ── Buttons ── */
    .stDownloadButton button, .stButton button {
        background: rgba(99,102,241,0.12) !important;
        border: 1px solid rgba(99,102,241,0.3) !important;
        color: var(--accent-light) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        padding: 0.5rem 1.1rem !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.01em !important;
    }

    .stDownloadButton button:hover, .stButton button:hover {
        background: rgba(99,102,241,0.22) !important;
        box-shadow: 0 0 24px rgba(99,102,241,0.2) !important;
        border-color: rgba(99,102,241,0.45) !important;
    }

    /* ── Alerts ── */
    div[data-testid="stAlert"] {
        background: rgba(12,12,24,0.85) !important;
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
        backdrop-filter: blur(12px) !important;
        color: var(--text-secondary) !important;
    }

    /* ── Global: Select dropdowns ── */
    [data-baseweb="select"] > div {
        background: #12121f !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
    }
    [data-baseweb="select"] [class*="placeholder"],
    [data-baseweb="select"] [class*="singleValue"],
    [data-baseweb="select"] span {
        color: #f1f5f9 !important;
    }

    /* ── Global: Number / Text inputs ── */
    input,
    input[type="number"],
    input[type="text"],
    .stNumberInput input,
    .stTextInput input,
    [data-testid="stNumberInput"] input {
        background-color: #12121f !important;
        color: #f1f5f9 !important;
        -webkit-text-fill-color: #f1f5f9 !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        caret-color: #6366f1 !important;
    }
    input:focus,
    .stNumberInput input:focus {
        border-color: rgba(99,102,241,0.55) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
        outline: none !important;
    }

    /* ── Caption / Captions ── */
    .stCaption, caption { color: var(--text-muted) !important; }

    /* ── Markdown headings in main area ── */
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        color: var(--text-primary) !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: var(--text-secondary);
    }

    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 1rem 0;
    }

    /* ── Responsive ── */
    @media (max-width: 992px) {
        .dashboard-header { padding: 1.5rem 1.5rem; }
        .dashboard-header h1 { font-size: 1.8rem; }
        .metric-card { min-height: 126px; }
    }

    @media (max-width: 640px) {
        .main .block-container { padding-top: 0.5rem; padding-bottom: 1.5rem; }
        .dashboard-header h1 { font-size: 1.5rem; }
        .section-header { font-size: 0.88rem; padding: 0.65rem 0.85rem; }
        .metric-value { font-size: 1.45rem; }
    }
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD ONS COVERAGE DATA
# ======================
@st.cache_data
def load_ons_coverage():
    coverage_file = "data/output/ons_coverage_by_commodity_classified.csv"
    if os.path.exists(coverage_file):
        df = pd.read_csv(coverage_file)
        return df
    else:
        st.warning("Coverage data not found. Please run: python scripts/data_merge.py")
        return pd.DataFrame(columns=["commodity", "ons_coverage_pct", "coverage_class", "sitc_category"])

ons_coverage_df = load_ons_coverage()

# ======================
# HELPER FUNCTIONS
# ======================
def get_coverage_badge(coverage_class):
    badges = {
        "High coverage": ("coverage-high", "HIGH COVERAGE"),
        "Partial coverage": ("coverage-partial", "PARTIAL COVERAGE"),
        "Low coverage": ("coverage-low", "LOW COVERAGE"),
        "No coverage": ("coverage-none", "NO COVERAGE")
    }
    css_class, label = badges.get(coverage_class, ("coverage-none", str(coverage_class)))
    return f'<span class="coverage-badge {css_class}">{label}</span>'

def get_risk_badge(risk_level):
    badges = {
        "HIGH": ("risk-high", "HIGH RISK"),
        "MODERATE": ("risk-moderate", "MODERATE RISK"),
        "LOW": ("risk-low", "LOW RISK")
    }
    css_class, label = badges.get(risk_level, ("risk-moderate", str(risk_level)))
    return f'<span class="risk-badge {css_class}">{label}</span>'

def get_coverage_color(coverage_class):
    colors = {
        "High coverage": "coverage-fill-high",
        "Partial coverage": "coverage-fill-partial",
        "Low coverage": "coverage-fill-low",
        "No coverage": "coverage-fill-none"
    }
    return colors.get(coverage_class, "coverage-fill-none")

def get_commodity_info(hs_code, coverage_df):
    """Get commodity information from coverage data."""
    row = coverage_df[coverage_df["commodity"] == hs_code]
    
    if len(row) > 0:
        row = row.iloc[0]
        return {
            "description": HS2_DESCRIPTIONS.get(hs_code, "Unknown"),
            "sitc_category": row.get("sitc_category", "Unknown"),
            "coverage_class": row.get("coverage_class", "No coverage"),
            "coverage_pct": row.get("ons_coverage_pct", 0),
            "total_years": row.get("total_years", 0),
            "covered_years": row.get("ons_covered_years", 0)
        }
    else:
        return {
            "description": HS2_DESCRIPTIONS.get(hs_code, "Unknown"),
            "sitc_category": "Unknown",
            "coverage_class": "No coverage",
            "coverage_pct": 0,
            "total_years": 0,
            "covered_years": 0
        }

# ======================
# HEADER
# ======================
st.markdown("""
<div class="dashboard-header">
    <span class="eyebrow">Intelligence Dashboard</span>
    <h1>UK SME Import Margin Simulator</h1>
    <p>Analyse import profitability, forecast currency movements, and assess risk exposure across dynamic economic scenarios with an enterprise-grade analytics experience.</p>
    <div class="hero-tags">
        <span class="hero-tag">Real-Time Simulation</span>
        <span class="hero-tag">Forecast & Risk Models</span>
        <span class="hero-tag">Confidence-Aware Decisions</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ======================
# SIDEBAR INPUTS
# ======================
with st.sidebar:
    st.markdown("## ⚙️ Control Center")
    st.markdown("<p class='section-description'>Configure scenario assumptions and analytics scope for your decision cockpit.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### Financial Inputs")
    
    import_value = st.number_input(
        "Import Value (GBP)",
        value=1_000_000,
        step=50_000,
        format="%d",
        help="Total value of goods being imported (HMRC baseline)"
    )
    
    revenue = st.number_input(
        "Expected Revenue (GBP)",
        value=1_350_000,
        step=50_000,
        format="%d",
        help="Projected sales revenue from imported goods"
    )
    
    st.markdown("---")
    
    # ======================
    # COMMODITY SELECTION
    # ======================
    st.markdown("### Commodity Selection")
    
    # Get unique SITC categories from coverage data
    if len(ons_coverage_df) > 0 and "sitc_category" in ons_coverage_df.columns:
        sitc_categories = sorted(ons_coverage_df["sitc_category"].dropna().unique())
    else:
        sitc_categories = [
            "0 Food & live animals", "1 Beverages & tobacco", "2 Crude materials",
            "3 Fuels", "5 Chemicals", "6 Manufactured goods",
            "7 Machinery & transport equipment", "8 Miscellaneous manufactures", "9 Other commodities"
        ]
    
    selected_sitc = st.selectbox(
        "Product Category (SITC)",
        options=sitc_categories,
        index=0,
        help="Select the broad SITC category of your import goods"
    )
    
    # Filter HS codes by selected SITC category
    if len(ons_coverage_df) > 0:
        filtered_hs = ons_coverage_df[ons_coverage_df["sitc_category"] == selected_sitc]["commodity"].tolist()
    else:
        filtered_hs = list(range(1, 99))
    
    # Create dropdown options with descriptions
    hs_options = {
        f"HS {hs:02d} - {HS2_DESCRIPTIONS.get(hs, 'Unknown')}": hs 
        for hs in sorted(filtered_hs) if hs in HS2_DESCRIPTIONS
    }
    
    if hs_options:
        selected_hs_label = st.selectbox(
            "Specific Commodity (HS Code)",
            options=list(hs_options.keys()),
            help="Select the specific HS2 chapter for your goods"
        )
        commodity_code = hs_options[selected_hs_label]
    else:
        commodity_code = st.number_input("HS Code", value=1, min_value=1, max_value=99)
    
    # Get commodity info
    commodity_info = get_commodity_info(commodity_code, ons_coverage_df)
    coverage_class = commodity_info["coverage_class"]
    coverage_pct = commodity_info["coverage_pct"]
    
    # Display commodity card
    st.markdown(f"""
    <div class="commodity-card">
        <div class="commodity-category">{commodity_info['sitc_category']}</div>
        <div class="commodity-name">HS {commodity_code:02d}: {commodity_info['description']}</div>
        <div style="margin-top: 0.75rem;">
            {get_coverage_badge(coverage_class)}
        </div>
        <div class="coverage-meter">
            <div class="coverage-fill {get_coverage_color(coverage_class)}" style="width: {coverage_pct}%;"></div>
        </div>
        <div style="font-size: 0.8rem; color: #666; margin-top: 0.25rem;">
            ONS Data Coverage: {coverage_pct:.1f}% ({commodity_info['covered_years']}/{commodity_info['total_years']} years)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Cost Assumptions")
    st.markdown("<p class='section-description'>Adjust cost factors that affect landed cost.</p>", unsafe_allow_html=True)
    
    fx_shock = st.slider(
        "FX Shock (%)",
        min_value=-20.0,
        max_value=20.0,
        value=0.0,
        step=0.5,
        help="Currency fluctuation impact (+ve = weaker GBP)"
    )
    
    shipping_pct = st.slider(
        "Shipping Cost (%)",
        min_value=0.0,
        max_value=30.0,
        value=5.0,
        step=0.5,
        help="Freight cost as percentage of goods value"
    )
    
    insurance_pct = st.slider(
        "Insurance Cost (%)",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Insurance premium as percentage of goods value"
    )
    
    tariff_pct = st.slider(
        "Tariff Rate (%)",
        min_value=0.0,
        max_value=25.0,
        value=2.0,
        step=0.5,
        help="Import duty as percentage of goods value"
    )
    
    st.markdown("---")
    st.caption("Sources: HMRC import valuations, ONS statistical coverage")
    
    # ======================
    # ANALYTICS SETTINGS
    # ======================
    st.markdown("---")
    st.markdown("### Analytics Options")
    
    st.session_state.analytics_enabled = st.checkbox(
        "Enable Advanced Analytics",
        value=st.session_state.analytics_enabled,
        help="Enable trend analysis, forecasting, and risk metrics"
    )
    
    if st.session_state.analytics_enabled:
        st.session_state.selected_currency = st.selectbox(
            "FX Currency for Analysis",
            options=["EUR", "USD", "CNY", "JPY"],
            index=["EUR", "USD", "CNY", "JPY"].index(st.session_state.selected_currency),
            help="Currency for FX trend and volatility analysis"
        )

# ======================
# BASE SCENARIO CALCULATION
# ======================
base_result = compute_margin(
    import_value_gbp=import_value,
    revenue_gbp=revenue,
    fx_shock_pct=fx_shock / 100,
    shipping_pct=shipping_pct / 100,
    insurance_pct=insurance_pct / 100,
    tariff_pct=tariff_pct / 100,
)

profit = base_result["profit"]
margin_pct = base_result["margin_pct"] if base_result["margin_pct"] is not None else 0.0
landed_cost = base_result["landed_cost"]
goods_cost = base_result["goods_cost"]
shipping_cost = base_result["shipping_cost"]
insurance_cost = base_result["insurance_cost"]
tariff_cost = base_result["tariff_cost"]

# Risk Assessment
base_risk = risk_label(margin_pct)
final_risk = adjust_risk(base_risk, coverage_class)
uncertainty = confidence_multiplier(coverage_class)

# ======================
# MAIN DASHBOARD
# ======================

st.markdown('<div class="section-header">Key Performance Metrics</div>', unsafe_allow_html=True)
st.markdown('<p class="section-description">Summary of the current import scenario based on your inputs. Green indicates profitability, red indicates a loss.</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    profit_color = "metric-card-profit" if profit >= 0 else "metric-card-loss"
    st.markdown(f"""
    <div class="metric-card {profit_color}">
        <div class="metric-label">Net Profit</div>
        <div class="metric-value">GBP {profit:,.0f}</div>
        <div class="metric-label">{'Profitable' if profit >= 0 else 'Loss-making'}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card metric-card-margin">
        <div class="metric-label">Profit Margin</div>
        <div class="metric-value">{margin_pct:.1f}%</div>
        <div class="metric-label">of total revenue</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card metric-card-cost">
        <div class="metric-label">Landed Cost</div>
        <div class="metric-value">GBP {landed_cost:,.0f}</div>
        <div class="metric-label">total import cost incl. duties</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card metric-card-revenue">
        <div class="metric-label">Expected Revenue</div>
        <div class="metric-value">GBP {revenue:,.0f}</div>
        <div class="metric-label">projected sales income</div>
    </div>
    """, unsafe_allow_html=True)

# Two column layout for details
left_col, right_col = st.columns([1, 1])

# ======================
# COST BREAKDOWN
# ======================
with left_col:
    st.markdown('<div class="section-header">Cost Breakdown</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-description">How the base import value builds up to the total landed cost. Blue bars are base and total values; red bars are added costs.</p>', unsafe_allow_html=True)
    
    cost_items = ['Import Value', 'FX Impact', 'Shipping', 'Insurance', 'Tariffs', 'Landed Cost']
    cost_values = [
        import_value,
        goods_cost - import_value,
        shipping_cost,
        insurance_cost,
        tariff_cost,
        0
    ]
    
    # Waterfall colour mapping:
    #   increasing (cost additions) = muted red
    #   decreasing (savings)        = green
    #   totals (landed cost)         = primary blue
    fig_waterfall = go.Figure(go.Waterfall(
        name="Cost Breakdown",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "relative", "total"],
        x=cost_items,
        y=cost_values,
        connector={"line": {"color": "rgba(255,255,255,0.12)", "width": 1, "dash": "dot"}},
        increasing={"marker": {"color": "#f43f5e"}},
        decreasing={"marker": {"color": "#10b981"}},
        totals={"marker": {"color": "#6366f1"}},
        text=[f"£{v:,.0f}" if i < 5 else f"£{landed_cost:,.0f}" for i, v in enumerate(cost_values)],
        textposition="outside",
        textfont=dict(size=11, color="#cbd5e1"),
    ))
    
    fig_waterfall.update_layout(
        showlegend=False,
        height=360,
        margin=dict(t=20, b=40, l=45, r=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, Plus Jakarta Sans, sans-serif', size=12, color='#cbd5e1'),
        yaxis=dict(
            title="Amount (GBP)",
            gridcolor="rgba(255,255,255,0.06)",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.1)",
            tickformat=",.0f",
            title_font=dict(size=11, color="#64748b"),
            tickfont=dict(color="#94a3b8"),
        ),
        xaxis=dict(
            tickfont=dict(size=11, color='#94a3b8'),
            gridcolor="rgba(255,255,255,0.04)",
        ),
        bargap=0.35,
    )
    
    st.plotly_chart(fig_waterfall, width="stretch")
    
    # Compact HTML cost table inside the same visual block
    fx_impact = goods_cost - import_value
    st.markdown(f"""
    <table class="cost-table">
        <tr><th>Component</th><th class="num">Amount (GBP)</th><th class="num">% of Import</th></tr>
        <tr><td>Goods (after FX)</td><td class="num">{goods_cost:,.0f}</td><td class="num">{(goods_cost/import_value)*100:.1f}%</td></tr>
        <tr><td>Shipping</td><td class="num">{shipping_cost:,.0f}</td><td class="num">{(shipping_cost/import_value)*100:.1f}%</td></tr>
        <tr><td>Insurance</td><td class="num">{insurance_cost:,.0f}</td><td class="num">{(insurance_cost/import_value)*100:.1f}%</td></tr>
        <tr><td>Tariffs</td><td class="num">{tariff_cost:,.0f}</td><td class="num">{(tariff_cost/import_value)*100:.1f}%</td></tr>
        <tr><td>Total Landed Cost</td><td class="num">{landed_cost:,.0f}</td><td class="num">{(landed_cost/import_value)*100:.1f}%</td></tr>
    </table>
    """, unsafe_allow_html=True)

# ======================
# RISK ASSESSMENT
# ======================
with right_col:
    st.markdown('<div class="section-header">Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-description">Risk classification based on margin level and data quality. Higher risk means tighter monitoring is advised.</p>', unsafe_allow_html=True)
    
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        st.markdown("**Financial Risk**")
        st.markdown("<small>Based on current profit margin</small>", unsafe_allow_html=True)
        st.markdown(get_risk_badge(base_risk), unsafe_allow_html=True)
        
    with risk_col2:
        st.markdown("**Adjusted Risk**")
        st.markdown("<small>Adjusted for ONS data quality</small>", unsafe_allow_html=True)
        st.markdown(get_risk_badge(final_risk), unsafe_allow_html=True)
    
    # Quality descriptions
    quality_description = {
        "High coverage": ("excellent", "reliable", "narrow"),
        "Partial coverage": ("moderate", "reasonably reliable", "moderately widened"),
        "Low coverage": ("limited", "less reliable", "significantly widened"),
        "No coverage": ("no", "unreliable", "maximally widened")
    }
    
    qual_level, qual_reliability, qual_bands = quality_description.get(
        coverage_class, ("unknown", "uncertain", "widened")
    )
    
    st.markdown(f"""
    <div class="info-box">
        <strong>Data Quality Assessment</strong><br><br>
        <strong>Commodity:</strong> HS {commodity_code:02d} - {commodity_info['description']}<br>
        <strong>ONS Coverage:</strong> {get_coverage_badge(coverage_class)}<br><br>
        <strong>What this means:</strong><br>
        ONS provides <strong>{qual_level}</strong> statistical coverage for this commodity, 
        meaning historical data is <strong>{qual_reliability}</strong>.<br><br>
        <strong>Confidence Adjustment:</strong> +/- {uncertainty*100:.0f}%<br>
        <small>Profit and margin confidence bands are <strong>{qual_bands}</strong> to reflect data quality.</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Profit gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=margin_pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Profit Margin (%)"},
        delta={'reference': 10, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#f43f5e"}},
        gauge={
            'axis': {'range': [-20, 40], 'tickwidth': 1, 'tickcolor': '#64748b'},
            'bar': {'color': "#6366f1"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 1,
            'bordercolor': "rgba(255,255,255,0.08)",
            'steps': [
                {'range': [-20, 0],  'color': 'rgba(244,63,94,0.18)'},
                {'range': [0, 5],    'color': 'rgba(245,158,11,0.15)'},
                {'range': [5, 10],   'color': 'rgba(16,185,129,0.15)'},
                {'range': [10, 40],  'color': 'rgba(16,185,129,0.28)'}
            ],
            'threshold': {
                'line': {'color': "rgba(255,255,255,0.5)", 'width': 3},
                'thickness': 0.75,
                'value': margin_pct
            }
        }
    ))
    
    fig_gauge.update_layout(
        height=260,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, Plus Jakarta Sans, sans-serif', color='#cbd5e1'),
    )
    
    st.plotly_chart(fig_gauge, width="stretch")

# ======================
# SENSITIVITY ANALYSIS
# ======================
st.markdown('<div class="section-header">Sensitivity Analysis</div>', unsafe_allow_html=True)
st.markdown('<p class="section-description">Explore how changes in FX rates and shipping costs affect your profit margin. Red zones indicate loss-making scenarios.</p>', unsafe_allow_html=True)

# Run sensitivity scenarios
df = run_sensitivity_scenarios(
    import_value_gbp=import_value,
    revenue_gbp=revenue,
)

# Calculate confidence bands
df["profit_lower"] = df["profit"] - abs(df["profit"]) * uncertainty
df["profit_upper"] = df["profit"] + abs(df["profit"]) * uncertainty
df["margin_lower"] = df["margin_pct"] - abs(df["margin_pct"]) * uncertainty
df["margin_upper"] = df["margin_pct"] + abs(df["margin_pct"]) * uncertainty

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["Margin Heatmap", "Trend Charts", "Scenario Data"])

with tab1:
    pivot_margin = df.pivot_table(
        values='margin_pct', 
        index='fx_shock_pct', 
        columns='shipping_pct',
        aggfunc='mean'
    )
    
    fig_heatmap = px.imshow(
        pivot_margin,
        labels=dict(x="Shipping Cost (%)", y="FX Shock (%)", color="Margin (%)"),
        x=[f"{x:.0f}%" for x in pivot_margin.columns],
        y=[f"{y:.0f}%" for y in pivot_margin.index],
        color_continuous_scale="RdYlGn",
        aspect="auto"
    )
    
    fig_heatmap.update_layout(
        title="Profit Margin Sensitivity: FX Shock vs Shipping Cost",
        height=480,
        margin=dict(t=50, b=40, l=70, r=30),
        font=dict(family='Inter, Plus Jakarta Sans, sans-serif', color='#cbd5e1'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(color="#94a3b8")),
        yaxis=dict(tickfont=dict(color="#94a3b8")),
    )
    
    st.plotly_chart(fig_heatmap, width="stretch")
    
    st.caption(f"Confidence bands widened by +/- {uncertainty*100:.0f}% to reflect ONS data quality ({coverage_class}).")

with tab2:
    trend_col1, trend_col2 = st.columns(2)
    
    with trend_col1:
        df_fx0 = df[df['fx_shock_pct'] == 0].sort_values('shipping_pct')
        
        fig_shipping = go.Figure()
        
        fig_shipping.add_trace(go.Scatter(
            x=df_fx0['shipping_pct'],
            y=df_fx0['margin_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_shipping.add_trace(go.Scatter(
            x=df_fx0['shipping_pct'],
            y=df_fx0['margin_lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(99, 102, 241, 0.12)',
            name=f'Confidence Band (+/- {uncertainty*100:.0f}%)'
        ))
        
        fig_shipping.add_trace(go.Scatter(
            x=df_fx0['shipping_pct'],
            y=df_fx0['margin_pct'],
            mode='lines+markers',
            name='Margin',
            line=dict(color='#6366f1', width=2),
            marker=dict(size=6)
        ))
        
        fig_shipping.add_hline(y=0, line_dash="dash", line_color="rgba(244,63,94,0.5)",
                               annotation_text="Break-even", annotation_position="right",
                               annotation_font_color="#f43f5e")
        
        fig_shipping.update_layout(
            title="Margin vs Shipping Cost (No FX Shock)",
            xaxis_title="Shipping Cost (%)",
            yaxis_title="Profit Margin (%)",
            height=350,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#94a3b8")),
            font=dict(family='Inter, Plus Jakarta Sans, sans-serif', color='#cbd5e1'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8"), zerolinecolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8"), zerolinecolor="rgba(255,255,255,0.08)"),
        )
        
        st.plotly_chart(fig_shipping, width="stretch")
    
    with trend_col2:
        df_ship5 = df[abs(df['shipping_pct'] - 5) < 1].sort_values('fx_shock_pct')
        
        if df_ship5.empty:
            df_ship5 = df[df['shipping_pct'] == df['shipping_pct'].min()].sort_values('fx_shock_pct')
        
        fig_fx = go.Figure()
        
        fig_fx.add_trace(go.Scatter(
            x=df_ship5['fx_shock_pct'],
            y=df_ship5['margin_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_fx.add_trace(go.Scatter(
            x=df_ship5['fx_shock_pct'],
            y=df_ship5['margin_lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(244, 63, 94, 0.10)',
            name=f'Confidence Band (+/- {uncertainty*100:.0f}%)'
        ))
        
        fig_fx.add_trace(go.Scatter(
            x=df_ship5['fx_shock_pct'],
            y=df_ship5['margin_pct'],
            mode='lines+markers',
            name='Margin',
            line=dict(color='#f43f5e', width=2),
            marker=dict(size=6)
        ))
        
        fig_fx.add_hline(y=0, line_dash="dash", line_color="rgba(244,63,94,0.5)",
                         annotation_text="Break-even", annotation_position="right",
                         annotation_font_color="#f43f5e")
        
        fig_fx.update_layout(
            title="Margin vs FX Shock (5% Shipping)",
            xaxis_title="FX Shock (%)",
            yaxis_title="Profit Margin (%)",
            height=350,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#94a3b8")),
            font=dict(family='Inter, Plus Jakarta Sans, sans-serif', color='#cbd5e1'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8"), zerolinecolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8"), zerolinecolor="rgba(255,255,255,0.08)"),
        )
        
        st.plotly_chart(fig_fx, width="stretch")

with tab3:
    st.markdown("#### Scenario Data")
    st.markdown("<p class='section-description'>Full dataset of all simulated scenarios. Download for offline analysis.</p>", unsafe_allow_html=True)
    
    display_df = df.copy()
    display_df = display_df.round(2)
    display_df.columns = ['FX Shock (%)', 'Shipping (%)', 'Profit (GBP)', 'Margin (%)', 
                          'Profit Lower (GBP)', 'Profit Upper (GBP)', 'Margin Lower (%)', 'Margin Upper (%)']
    
    st.dataframe(
        display_df,
        width="stretch",
        height=400
    )
    
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Download Scenario Data (CSV)",
        data=csv,
        file_name=f"import_scenarios_hs{commodity_code}.csv",
        mime="text/csv"
    )

# ======================
# ADVANCED ANALYTICS SECTION
# ======================

# ======================
# CACHED DATA LOADERS
# ======================

@st.cache_data(ttl=3600, show_spinner="Loading FX data...")
def cached_fx_analysis(currency, lookback_days, data_path):
    """Cache FX trend analysis (expensive I/O + computation)."""
    return analyse_fx_trends(currency=currency, lookback_days=lookback_days, data_path=data_path)

@st.cache_data(ttl=3600, show_spinner="Loading import data...")
def cached_import_analysis(commodity_code, data_path):
    """Cache import trend analysis (large CSV reads)."""
    return analyse_import_trends(commodity_code, data_path)

@st.cache_data(ttl=3600, show_spinner="Running forecast...")
def cached_fx_forecast(currency, forecast_days, method, data_path):
    """Cache FX forecast (ARIMA fitting is slow)."""
    return forecast_fx_rate(currency=currency, forecast_days=forecast_days,
                            method=method, data_path=data_path)



if st.session_state.analytics_enabled:
    st.markdown('<div class="section-header">Advanced Analytics</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-description">Historical trends, predictive forecasts, and risk metrics powered by HMRC and BoE data.</p>', unsafe_allow_html=True)
    
    # Create tabs for different analytics features
    analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs([
        "Trends & History",
        "Forecasting",
        "Risk Metrics"
    ])
    
    # ======================
    # TAB 1: TRENDS & HISTORY
    # ======================
    with analytics_tab1:
     try:
        st.markdown("### Historical Trend Analysis")
        st.markdown("<p class='section-description'>Review historical exchange rate movements and import data patterns to identify trends.</p>", unsafe_allow_html=True)
        
        trend_col1, trend_col2 = st.columns(2)
        
        with trend_col1:
            # FX Trend Analysis
            st.markdown("#### Exchange Rate Trends")
            
            fx_analysis = cached_fx_analysis(
                currency=st.session_state.selected_currency,
                lookback_days=730,
                data_path="data/raw/exchange"
            )
            
            if not fx_analysis.get("error"):
                fx_data = fx_analysis.get("data", pd.DataFrame())
                
                if not fx_data.empty:
                    # FX Rate Chart with volatility bands
                    fig_fx_trend = go.Figure()
                    
                    # Main rate line
                    fig_fx_trend.add_trace(go.Scatter(
                        x=fx_data["Date"],
                        y=fx_data["Rate"],
                        mode='lines',
                        name=f'{st.session_state.selected_currency}/GBP Rate',
                        line=dict(color='#a5b4fc', width=2)
                    ))
                    
                    # Rolling mean
                    rolling_stats = fx_analysis.get("rolling_stats", {})
                    if "rolling_mean" in rolling_stats:
                        fig_fx_trend.add_trace(go.Scatter(
                            x=fx_data["Date"],
                            y=rolling_stats["rolling_mean"],
                            mode='lines',
                            name='30-Day Moving Average',
                            line=dict(color='#f43f5e', width=1, dash='dash')
                        ))
                    
                    fig_fx_trend.update_layout(
                        title=f"{st.session_state.selected_currency}/GBP Exchange Rate History",
                        xaxis_title="Date",
                        yaxis_title="Exchange Rate",
                        height=350,
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color="#94a3b8")),
                        font=dict(family='Inter, Plus Jakarta Sans, sans-serif', color='#cbd5e1'),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8")),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8")),
                    )
                    
                    st.plotly_chart(fig_fx_trend, width="stretch")
                    
                    # FX Statistics
                    fx_stats_col1, fx_stats_col2, fx_stats_col3 = st.columns(3)
                    
                    with fx_stats_col1:
                        st.metric(
                            "Current Rate",
                            f"{fx_analysis['current_rate']:.4f}",
                            delta=f"{fx_analysis.get('growth_rates', {}).get('7_period_growth', 0) or 0:.2f}% (7d)"
                        )
                    
                    with fx_stats_col2:
                        st.metric(
                            "Volatility (Ann.)",
                            f"{fx_analysis['current_volatility']:.1f}%",
                            delta=None
                        )
                    
                    with fx_stats_col3:
                        st.metric(
                            "Trend",
                            fx_analysis['trend_direction'].title(),
                            delta=None
                        )
                    
                    # FX Percentile Position
                    fx_percentile = calculate_fx_percentile_position(
                        st.session_state.selected_currency,
                        data_path="data/raw/exchange"
                    )
                    
                    if not fx_percentile.get("error"):
                        st.info(f"**Current Position:** {fx_percentile['interpretation']} "
                               f"(Percentile: {fx_percentile['percentile']:.0f}%). "
                               f"A higher percentile means the currency is relatively expensive.")
            else:
                st.warning(f"FX data not available: {fx_analysis.get('error')}")
        
        with trend_col2:
            # Volatility Analysis
            st.markdown("#### Volatility Analysis")
            st.markdown("<p class='section-description'>Higher volatility indicates greater uncertainty in FX rates.</p>", unsafe_allow_html=True)
            
            if not fx_analysis.get("error") and "data" in fx_analysis:
                vol_data = calculate_historical_volatility(fx_analysis["data"]["Rate"], window=30)
                
                if not vol_data.empty:
                    fig_vol = go.Figure()
                    
                    fig_vol.add_trace(go.Scatter(
                        x=fx_analysis["data"]["Date"],
                        y=vol_data["rolling_vol"] * 100,
                        mode='lines',
                        name='Rolling Volatility',
                        line=dict(color='#6366f1', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(99, 102, 241, 0.10)'
                    ))
                    
                    # Add threshold lines
                    fig_vol.add_hline(y=15, line_dash="dash", line_color="rgba(245,158,11,0.55)",
                                     annotation_text="Elevated", annotation_font_color="#fcd34d")
                    fig_vol.add_hline(y=25, line_dash="dash", line_color="rgba(244,63,94,0.5)",
                                     annotation_text="Critical", annotation_font_color="#f43f5e")
                    
                    fig_vol.update_layout(
                        title="Annualised FX Volatility (30-Day Rolling)",
                        xaxis_title="Date",
                        yaxis_title="Volatility (%)",
                        height=350,
                        showlegend=True,
                        legend=dict(font=dict(color="#94a3b8")),
                        font=dict(family='Inter, Plus Jakarta Sans, sans-serif', color='#cbd5e1'),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8")),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8")),
                    )
                    
                    st.plotly_chart(fig_vol, width="stretch")
                    
                    # Volatility Regime
                    vol_regime = volatility_regime_detection(vol_data["rolling_vol"])
                    
                    if not vol_regime.get("error"):
                        regime_color = {
                            "LOW": "success",
                            "NORMAL": "warning",
                            "HIGH": "error"
                        }.get(vol_regime["current_regime"], "info")
                        
                        st.markdown(f"**Current Regime:** {vol_regime['current_regime']}")
                        st.markdown(f"*{vol_regime['regime_description']}*")
                        st.markdown(f"**Trading Implication:** {vol_regime['trading_implication']}")
            else:
                st.warning("Volatility data not available")
        
        # Historical Margin Simulation
        st.markdown("---")
        st.markdown("#### Historical Margin Simulation")
        st.markdown("<p class='section-description'>Simulated profit margins using historical HMRC import values and BoE exchange rates. Helps identify periods of margin pressure.</p>", unsafe_allow_html=True)
        
        import_analysis = cached_import_analysis(commodity_code, "data/output/merged_hmrc_ons_commodity.csv")
        
        if not import_analysis.get("error") and not fx_analysis.get("error"):
            if "data" in import_analysis and "data" in fx_analysis:
                margin_sim = simulate_historical_margins(
                    import_analysis["data"],
                    fx_analysis["data"],
                    base_revenue_multiplier=revenue / import_value if import_value > 0 else 1.35,
                    shipping_pct=shipping_pct / 100,
                    insurance_pct=insurance_pct / 100,
                    tariff_pct=tariff_pct / 100
                )
                
                if not margin_sim.empty:
                    margin_trends = calculate_margin_trends(margin_sim)
                    
                    # Margin history chart
                    fig_margin_hist = go.Figure()
                    
                    fig_margin_hist.add_trace(go.Scatter(
                        x=margin_sim["date"],
                        y=margin_sim["margin_pct"],
                        mode='lines+markers',
                        name='Simulated Margin',
                        line=dict(color='#10b981', width=2),
                        marker=dict(size=3)
                    ))
                    
                    # Add mean line
                    fig_margin_hist.add_hline(
                        y=margin_trends["mean_margin"],
                        line_dash="dash",
                        line_color="#6366f1",
                        annotation_text=f"Mean: {margin_trends['mean_margin']:.1f}%"
                    )
                    
                    # Break-even line
                    fig_margin_hist.add_hline(
                        y=0,
                        line_dash="solid",
                        line_color="rgba(244,63,94,0.5)",
                        annotation_text="Break-even"
                    )
                    
                    fig_margin_hist.update_layout(
                        title=f"Historical Margin Simulation: HS{commodity_code:02d}",
                        xaxis_title="Date",
                        yaxis_title="Margin (%)",
                        height=400,
                        showlegend=True,
                        legend=dict(font=dict(color="#94a3b8")),
                        font=dict(family='Inter, Plus Jakarta Sans, sans-serif', color='#cbd5e1'),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8")),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8"), zerolinecolor="rgba(255,255,255,0.08)"),
                    )
                    
                    st.plotly_chart(fig_margin_hist, width="stretch")
                    
                    # Margin trend metrics
                    margin_metric_cols = st.columns(5)
                    
                    with margin_metric_cols[0]:
                        st.metric("Mean Margin", f"{margin_trends['mean_margin']:.1f}%")
                    
                    with margin_metric_cols[1]:
                        st.metric("Margin Volatility", f"±{margin_trends['margin_volatility']:.1f}%")
                    
                    with margin_metric_cols[2]:
                        st.metric("Min Margin", f"{margin_trends['min_margin']:.1f}%")
                    
                    with margin_metric_cols[3]:
                        st.metric("Max Margin", f"{margin_trends['max_margin']:.1f}%")
                    
                    with margin_metric_cols[4]:
                        st.metric("Trend", margin_trends['trend_direction'].title())
                else:
                    st.warning("Could not simulate historical margins")
        else:
            st.info("Historical margin simulation requires import and FX data")
     except Exception as e:
        st.error(f"Trends analysis encountered an error: {e}")
    
    # ======================
    # TAB 2: FORECASTING
    # ======================
    with analytics_tab2:
     try:
        st.markdown("### Predictive Analytics")
        st.markdown("<p class='section-description'>Time-series forecasts of exchange rates and margins using ARIMA models. Confidence intervals indicate forecast uncertainty.</p>", unsafe_allow_html=True)
        
        forecast_col1, forecast_col2 = st.columns(2)
        
        with forecast_col1:
            st.markdown("#### Exchange Rate Forecast")
            
            forecast_horizon = st.slider("Forecast Horizon (days)", 7, 90, 30, key="fx_forecast_horizon")
            
            fx_forecast = cached_fx_forecast(
                currency=st.session_state.selected_currency,
                forecast_days=forecast_horizon,
                method="arima",
                data_path="data/raw/exchange"
            )
            
            if not fx_forecast.get("error"):
                forecast_df = fx_forecast.get("forecast", pd.DataFrame())
                historical_df = fx_forecast.get("historical_data", pd.DataFrame())
                
                if not forecast_df.empty and not historical_df.empty:
                    fig_forecast = go.Figure()
                    
                    # Historical data (last 180 days)
                    recent_hist = historical_df.tail(180)
                    fig_forecast.add_trace(go.Scatter(
                        x=recent_hist["Date"],
                        y=recent_hist["Rate"],
                        mode='lines',
                        name='Historical',
                        line=dict(color='#a5b4fc', width=2)
                    ))
                    
                    # Forecast
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_df["date"],
                        y=forecast_df["forecast"],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='#6366f1', width=2, dash='dash')
                    ))
                    
                    # Confidence interval
                    fig_forecast.add_trace(go.Scatter(
                        x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
                        y=pd.concat([forecast_df["upper_bound"], forecast_df["lower_bound"][::-1]]),
                        fill='toself',
                        fillcolor='rgba(99, 102, 241, 0.12)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='95% Confidence'
                    ))
                    
                    fig_forecast.update_layout(
                        title=f"{st.session_state.selected_currency}/GBP Forecast ({forecast_horizon} Days)",
                        xaxis_title="Date",
                        yaxis_title="Exchange Rate",
                        height=400,
                        showlegend=True,
                        legend=dict(font=dict(color="#94a3b8")),
                        font=dict(family='Inter, Plus Jakarta Sans, sans-serif', color='#cbd5e1'),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8")),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8")),
                    )
                    
                    st.plotly_chart(fig_forecast, width="stretch")
                    
                    # Forecast summary
                    forecast_end = forecast_df["forecast"].iloc[-1]
                    forecast_low = forecast_df["lower_bound"].iloc[-1]
                    forecast_high = forecast_df["upper_bound"].iloc[-1]
                    current_rate = historical_df["Rate"].iloc[-1]
                    
                    expected_change = ((forecast_end - current_rate) / current_rate) * 100
                    
                    st.markdown(f"""
                    **Forecast Summary:**
                    - Current Rate: {current_rate:.4f}
                    - Forecast ({forecast_horizon}d): {forecast_end:.4f} ({expected_change:+.2f}%)
                    - 95% Range: [{forecast_low:.4f} - {forecast_high:.4f}]
                    """)
            else:
                st.warning(f"Forecast not available: {fx_forecast.get('error')}")
        
        with forecast_col2:
            st.markdown("#### Margin Forecast")
            
            if "margin_sim" in dir() and not margin_sim.empty:
                margin_forecast = forecast_margins(margin_sim, forecast_periods=6, method="arima")
                
                if not margin_forecast.get("error"):
                    mf_df = margin_forecast.get("forecast", pd.DataFrame())
                    
                    if not mf_df.empty:
                        fig_margin_forecast = go.Figure()
                        
                        # Historical margins
                        fig_margin_forecast.add_trace(go.Scatter(
                            x=margin_sim["date"],
                            y=margin_sim["margin_pct"],
                            mode='lines',
                            name='Historical Margin',
                            line=dict(color='#a5b4fc', width=2)
                        ))
                        
                        # Forecast
                        fig_margin_forecast.add_trace(go.Scatter(
                            x=mf_df["date"],
                            y=mf_df["forecast"],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='#f43f5e', width=2, dash='dash')
                        ))
                        
                        # Confidence band
                        fig_margin_forecast.add_trace(go.Scatter(
                            x=pd.concat([mf_df["date"], mf_df["date"][::-1]]),
                            y=pd.concat([mf_df["upper_bound"], mf_df["lower_bound"][::-1]]),
                            fill='toself',
                            fillcolor='rgba(244, 63, 94, 0.10)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='Confidence Band'
                        ))
                        
                        fig_margin_forecast.add_hline(y=0, line_dash="solid", line_color="rgba(244,63,94,0.4)")
                        
                        fig_margin_forecast.update_layout(
                            title="Margin Forecast (6 Months)",
                            xaxis_title="Date",
                            yaxis_title="Margin (%)",
                            height=400,
                            showlegend=True,
                            legend=dict(font=dict(color="#94a3b8")),
                            font=dict(family='Inter, Plus Jakarta Sans, sans-serif', color='#cbd5e1'),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8")),
                            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8"), zerolinecolor="rgba(244,63,94,0.4)"),
                        )
                        
                        st.plotly_chart(fig_margin_forecast, width="stretch")
                else:
                    st.info("Margin forecast requires more historical data")
            else:
                st.info("Run historical margin simulation first (Trends tab)")
        
        # Anomaly Detection
        st.markdown("---")
        st.markdown("#### Anomaly Detection")
        st.markdown("<p class='section-description'>Flags periods where margins or costs deviated significantly from normal patterns. Anomalies may indicate supply chain disruptions or market shifts.</p>", unsafe_allow_html=True)
        
        if "margin_sim" in dir() and not margin_sim.empty:
            anomaly_col1, anomaly_col2 = st.columns(2)
            
            with anomaly_col1:
                margin_anomalies = detect_margin_anomalies(margin_sim, method="zscore", threshold=2.0)
                
                if not margin_anomalies.get("error"):
                    anomaly_count = margin_anomalies.get("anomaly_count", 0)
                    anomaly_rate = margin_anomalies.get("anomaly_rate", 0)
                    
                    st.metric("Margin Anomalies Detected", anomaly_count)
                    st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
                    
                    if anomaly_count > 0:
                        st.markdown("**Recent Anomalies:**")
                        for anomaly in margin_anomalies.get("anomalies", [])[-3:]:
                            st.markdown(f"- {anomaly['type']}: {anomaly['value']:.1f}% - *{anomaly['explanation']}*")
            
            with anomaly_col2:
                cost_anomalies = detect_cost_anomalies(margin_sim, cost_column="landed_cost", threshold=2.0)
                
                if not cost_anomalies.get("error"):
                    cost_anomaly_count = cost_anomalies.get("anomaly_count", 0)
                    
                    st.metric("Cost Anomalies Detected", cost_anomaly_count)
                    
                    if cost_anomaly_count > 0:
                        st.markdown("**Cost Anomalies:**")
                        for anomaly in cost_anomalies.get("anomalies", [])[-3:]:
                            st.markdown(f"- {anomaly['type']}: GBP {anomaly['value']:,.0f}")
        else:
            st.info("Anomaly detection requires historical margin simulation")
     except Exception as e:
        st.error(f"Forecasting analysis encountered an error: {e}")
    
    # ======================
    # TAB 3: RISK METRICS
    # ======================
    with analytics_tab3:
     try:
        st.markdown("### Risk Metrics")
        st.markdown("<p class='section-description'>Quantitative risk measures including Value-at-Risk, correlation analysis, and stress testing. Higher values generally indicate greater exposure.</p>", unsafe_allow_html=True)
        
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            st.markdown("#### Value-at-Risk (VaR)")
            st.markdown("<p class='section-description'>Maximum expected loss within a 95% confidence interval.</p>", unsafe_allow_html=True)
            
            if "fx_analysis" in dir() and not fx_analysis.get("error"):
                fx_returns = np.log(fx_analysis["data"]["Rate"] / fx_analysis["data"]["Rate"].shift(1))
                
                var_result = calculate_var_historical(fx_returns.dropna(), confidence_level=0.95)
                
                if not var_result.get("error"):
                    st.markdown(f"""
                    <div class="metric-mini">
                        <h3>FX Value-at-Risk (95%)</h3>
                        <h2>{abs(var_result['var_value']):.2f}%</h2>
                        <p>{var_result['risk_level']} Risk Level</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"**Interpretation:** {var_result['var_interpretation']}")
                    
                    if var_result.get("expected_shortfall"):
                        st.markdown(f"**Expected Shortfall:** {var_result['es_interpretation']}")
                    
                    # Distribution stats
                    with st.expander("Distribution Statistics"):
                        stats = var_result.get("statistics", {})
                        st.markdown(f"""
                        - Mean Return: {stats.get('mean_return', 0):.4f}%
                        - Std Deviation: {stats.get('std_return', 0):.4f}%
                        - Skewness: {stats.get('skewness', 0):.4f}
                        - Kurtosis: {stats.get('kurtosis', 0):.4f}
                        """)
            
            # Margin VaR
            st.markdown("#### Margin Value-at-Risk")
            st.markdown("<p class='section-description'>Worst-case margin estimate based on historical variability.</p>", unsafe_allow_html=True)
            
            if "margin_sim" in dir() and not margin_sim.empty:
                margin_var = calculate_margin_var(margin_sim, confidence_level=0.95)
                
                if not margin_var.get("error"):
                    st.metric(
                        "Worst-Case Margin (95%)",
                        f"{margin_var.get('worst_case_margin', 0):.1f}%",
                        delta=f"{margin_var.get('worst_case_margin', 0) - margin_var.get('current_margin', 0):.1f}% from current"
                    )
                    st.markdown(f"*{margin_var.get('interpretation', '')}*")
        
        with risk_col2:
            st.markdown("#### Correlation Analysis")
            st.markdown("<p class='section-description'>Measures how closely import volumes track exchange rate movements. Values near +1 or -1 indicate strong relationships.</p>", unsafe_allow_html=True)
            
            if "import_analysis" in dir() and "fx_analysis" in dir():
                if not import_analysis.get("error") and not fx_analysis.get("error"):
                    if "data" in import_analysis and "data" in fx_analysis:
                        correlation = analyse_commodity_fx_correlation(
                            import_analysis["data"],
                            fx_analysis["data"]
                        )
                        
                        if not correlation.get("error"):
                            # Correlation gauge
                            fig_corr = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=correlation["static_correlation"],
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Import-FX Correlation"},
                                gauge={
                                    'axis': {'range': [-1, 1], 'tickcolor': '#64748b'},
                                    'bar': {'color': "#6366f1"},
                                    'bgcolor': 'rgba(0,0,0,0)',
                                    'borderwidth': 1,
                                    'bordercolor': 'rgba(255,255,255,0.08)',
                                    'steps': [
                                        {'range': [-1, -0.3], 'color': 'rgba(244,63,94,0.18)'},
                                        {'range': [-0.3, 0.3], 'color': 'rgba(245,158,11,0.15)'},
                                        {'range': [0.3, 1],  'color': 'rgba(16,185,129,0.18)'}
                                    ]
                                }
                            ))

                            fig_corr.update_layout(
                                height=250,
                                margin=dict(t=80, b=20),
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(family='Inter, Plus Jakarta Sans, sans-serif', color='#cbd5e1'),
                            )
                            st.plotly_chart(fig_corr, width="stretch")
                            
                            st.markdown(f"**{correlation['interpretation']}**")
                            st.markdown(f"*{correlation['implication']}*")
            
            # Risk Decomposition
            st.markdown("#### Risk Decomposition")
            st.markdown("<p class='section-description'>Attribution of margin risk across cost components.</p>", unsafe_allow_html=True)
            
            if "margin_sim" in dir() and not margin_sim.empty:
                risk_decomp = decompose_margin_risk(margin_sim)
                
                if not risk_decomp.get("error"):
                    contributions = risk_decomp.get("risk_contributions", {})
                    
                    if contributions:
                        fig_decomp = go.Figure(go.Pie(
                            labels=list(contributions.keys()),
                            values=list(contributions.values()),
                            hole=0.45,
                            marker_colors=['#6366f1', '#06b6d4', '#f59e0b', '#f43f5e'],
                            marker=dict(line=dict(color='rgba(0,0,0,0)', width=2)),
                            textfont=dict(color='#cbd5e1'),
                        ))

                        fig_decomp.update_layout(
                            title="Margin Risk Attribution",
                            height=300,
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family='Inter, Plus Jakarta Sans, sans-serif', color='#cbd5e1'),
                            legend=dict(font=dict(color="#94a3b8")),
                        )
                        
                        st.plotly_chart(fig_decomp, width="stretch")
                        
                        st.info(f"{risk_decomp.get('interpretation', '')}")
        
        # Stress Testing
        st.markdown("---")
        st.markdown("#### Stress Testing")
        st.markdown("<p class='section-description'>Simulates margins under adverse economic scenarios. Red bars indicate potential losses; green bars indicate resilience.</p>", unsafe_allow_html=True)
        
        scenarios = get_predefined_stress_scenarios()
        stress_results = run_stress_test(margin_pct, scenarios)
        
        if not stress_results.empty:
            # Build premium dark HTML table
            def _stress_badge(val):
                if val < 0:
                    return f'<span class="stress-badge stress-badge-loss">{val:.1f}%</span>'
                elif val < 5:
                    return f'<span class="stress-badge stress-badge-caution">{val:.1f}%</span>'
                else:
                    return f'<span class="stress-badge stress-badge-healthy">{val:.1f}%</span>'

            def _row_class(val):
                if val < 0:   return 'stress-row-loss'
                elif val < 5: return 'stress-row-caution'
                else:         return 'stress-row-healthy'

            header_html = "".join(
                f'<th>{col.replace("_", " ").title()}</th>'
                for col in stress_results.columns
            )

            rows_html = ""
            for _, row in stress_results.iterrows():
                margin = row.get('stressed_margin', 0)
                row_cls = _row_class(margin)
                cells = ""
                for col in stress_results.columns:
                    val = row[col]
                    if col == 'stressed_margin':
                        cells += f'<td>{_stress_badge(val)}</td>'
                    elif isinstance(val, float):
                        cells += f'<td style="font-variant-numeric:tabular-nums;">{val:+.2f}%</td>'
                    else:
                        cells += f'<td>{val}</td>'
                rows_html += f'<tr class="{row_cls}">{cells}</tr>'

            st.markdown(f"""
            <div class="stress-table-wrap">
              <table class="stress-table">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{rows_html}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)
            
            # Stress test chart
            fig_stress = go.Figure()
            
            colors = ['#10b981' if m > 5 else '#f59e0b' if m > 0 else '#f43f5e'
                     for m in stress_results['stressed_margin']]

            fig_stress.add_trace(go.Bar(
                x=stress_results['scenario'],
                y=stress_results['stressed_margin'],
                marker_color=colors,
                marker=dict(line=dict(color='rgba(0,0,0,0)', width=0)),
                text=[f"{m:.1f}%" for m in stress_results['stressed_margin']],
                textposition='outside',
                textfont=dict(color='#cbd5e1'),
            ))

            fig_stress.add_hline(y=0, line_dash="dash", line_color="rgba(244,63,94,0.5)")
            fig_stress.add_hline(y=margin_pct, line_dash="dash", line_color="rgba(165,180,252,0.6)",
                               annotation_text=f"Current: {margin_pct:.1f}%",
                               annotation_font_color="#a5b4fc")

            fig_stress.update_layout(
                title="Stress Test Results by Scenario",
                xaxis_title="Scenario",
                yaxis_title="Stressed Margin (%)",
                height=400,
                font=dict(family='Inter, Plus Jakarta Sans, sans-serif', color='#cbd5e1'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8")),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#94a3b8"), zerolinecolor="rgba(255,255,255,0.08)"),
                bargap=0.35,
            )
            
            st.plotly_chart(fig_stress, width="stretch")
     except Exception as e:
        st.error(f"Risk metrics encountered an error: {e}")


# ======================
# FOOTER
# ======================
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    <div class="footer-card">
    <strong>Data Sources</strong><br>
    - HMRC: Import valuations<br>
    - ONS: Statistical coverage metrics<br>
    - Bank of England: Exchange rate data
    </div>
    """, unsafe_allow_html=True)

with footer_col2:
    st.markdown("""
    <div class="footer-card">
    <strong>Core Model</strong><br>
    - Landed cost calculation<br>
    - Risk-adjusted margins<br>
    - Confidence band adjustments
    </div>
    """, unsafe_allow_html=True)

with footer_col3:
    st.markdown("""
    <div class="footer-card">
    <strong>Analytics</strong><br>
    - Historical trend analysis<br>
    - ARIMA forecasting<br>
    - VaR and stress testing
    </div>
    """, unsafe_allow_html=True)

st.caption("Disclaimer: This tool provides simulations for educational and analytical purposes only. It does not constitute financial advice. Past data may not predict future outcomes.")
st.caption(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | UK SME Import Margin Simulator v2.0")
