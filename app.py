"""
MedCheck — creative single-page medication interaction checker.
No sidebar. Pretrained embeddings for name matching only.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force-reload local modules so matcher fixes apply without a full Python restart
import importlib
import src.preprocessing as _preprocessing
import src.medication_matcher as _medication_matcher
import src.data_loader as _data_loader
import src.interaction_checker as _interaction_checker
import src.medicine_details as _medicine_details
import src.risk_scoring as _risk_scoring

importlib.reload(_preprocessing)
importlib.reload(_medication_matcher)
importlib.reload(_data_loader)
importlib.reload(_interaction_checker)
importlib.reload(_medicine_details)
importlib.reload(_risk_scoring)

from src.data_loader import build_unique_drug_list, load_datasets
from src.interaction_checker import check_all_pairs, generate_medication_pairs
from src.medication_matcher import (
    create_drug_embeddings,
    load_embedding_model,
    match_all_medications,
)
from src.medicine_details import get_medicine_details
from src.preprocessing import normalize_drug_name, parse_medication_input
from src.risk_scoring import calculate_overall_risk

DATA_DIR = ROOT / "data"
DEFAULT_INTERACTIONS = DATA_DIR / "drug_interactions.csv"
DEFAULT_MEDICINES = DATA_DIR / "medicine_details.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
THRESHOLD = 0.75

st.set_page_config(
    page_title="MedCheck",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,500;9..144,700&display=swap');

:root {
    --ink: #102a2e;
    --muted: #4d6a70;
    --line: rgba(16, 42, 46, 0.10);
    --accent: #0d9488;
    --accent-2: #f97316;
    --glow: #5eead4;
    --card: rgba(255,255,255,0.72);
}

html, body, [class*="css"] {
    font-family: "Outfit", sans-serif;
    color: var(--ink);
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
section.main,
.main,
[data-testid="stMain"],
[data-testid="stAppViewBlockContainer"] {
    background:
        radial-gradient(1000px 560px at 8% -5%, rgba(45, 212, 191, 0.28), transparent 55%),
        radial-gradient(900px 520px at 100% 0%, rgba(251, 146, 60, 0.16), transparent 50%),
        linear-gradient(165deg, #dff7f2 0%, #eaf6f3 40%, #f3efe9 100%) !important;
    background-attachment: fixed !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stToolbar"],
#MainMenu, footer, header {
    display: none !important;
}

.block-container {
    padding-top: 1.6rem !important;
    padding-bottom: 3rem !important;
    max-width: 1080px;
}

/* Hero */
.hero {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    padding: 2rem 2rem 1.7rem 2rem;
    margin-bottom: 1.35rem;
    background:
        linear-gradient(135deg, rgba(13,148,136,0.95) 0%, rgba(15,118,110,0.92) 48%, rgba(249,115,22,0.78) 130%);
    color: #fff;
    box-shadow: 0 20px 50px rgba(13, 148, 136, 0.25);
    animation: floatIn 0.7s ease-out;
}
.hero::after {
    content: "";
    position: absolute;
    width: 280px; height: 280px;
    right: -60px; top: -80px;
    border-radius: 50%;
    background: rgba(255,255,255,0.12);
    animation: pulse 5s ease-in-out infinite;
}
.hero::before {
    content: "";
    position: absolute;
    width: 160px; height: 160px;
    right: 90px; bottom: -70px;
    border-radius: 50%;
    background: rgba(249,115,22,0.25);
}
.hero-kicker {
    position: relative;
    z-index: 1;
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 999px;
    padding: 0.28rem 0.75rem;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.hero h1 {
    position: relative;
    z-index: 1;
    font-family: "Fraunces", Georgia, serif !important;
    font-size: 3rem !important;
    font-weight: 700 !important;
    margin: 0.7rem 0 0.35rem 0 !important;
    color: #fff !important;
    letter-spacing: -0.03em;
    line-height: 1.05 !important;
}
.hero p {
    position: relative;
    z-index: 1;
    margin: 0;
    max-width: 34rem;
    font-size: 1.08rem;
    line-height: 1.5;
    color: rgba(255,255,255,0.92);
}

.glass {
    background: var(--card);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.7);
    border-radius: 22px;
    padding: 1.25rem 1.3rem;
    box-shadow: 0 12px 40px rgba(16, 42, 46, 0.06);
    margin-bottom: 1.1rem;
    animation: floatIn 0.8s ease-out;
}

.section-eyebrow {
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0 0 0.25rem 0;
}
.section-title {
    font-family: "Fraunces", Georgia, serif;
    font-size: 1.55rem;
    font-weight: 700;
    margin: 0 0 0.9rem 0;
    color: var(--ink);
}

.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin: 0.35rem 0 0.8rem 0;
}
.pill {
    background: linear-gradient(180deg, #ffffff, #f0fdfa);
    border: 1px solid rgba(13,148,136,0.2);
    color: #0f766e;
    border-radius: 999px;
    padding: 0.35rem 0.8rem;
    font-size: 0.88rem;
    font-weight: 700;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem;
    margin: 0.6rem 0 0.2rem 0;
    position: relative;
    z-index: 1;
}
@media (max-width: 900px) {
    .metric-grid { grid-template-columns: repeat(2, 1fr); }
    .hero h1 { font-size: 2.3rem !important; }
}
.metric {
    border-radius: 20px;
    padding: 1.15rem 1.15rem 1.05rem 1.15rem;
    position: relative;
    overflow: hidden;
    min-height: 120px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    cursor: pointer;
    box-shadow: 0 10px 26px rgba(16, 42, 46, 0.08);
    border: 2px solid transparent;
    text-align: left;
}
.metric::after {
    content: "";
    position: absolute;
    right: -18px;
    top: -18px;
    width: 88px;
    height: 88px;
    border-radius: 50%;
    background: rgba(255,255,255,0.38);
    pointer-events: none;
}
.metric:hover { transform: translateY(-3px); box-shadow: 0 14px 30px rgba(16, 42, 46, 0.14); }
.metric.is-active {
    border-color: rgba(16, 42, 46, 0.32);
    box-shadow: 0 14px 30px rgba(16, 42, 46, 0.16);
}
.metric .label {
    font-family: "Outfit", sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    opacity: 0.85;
}
.metric .value {
    font-family: "Fraunces", Georgia, serif;
    font-size: 2.15rem;
    font-weight: 700;
    margin-top: 0.28rem;
    line-height: 1;
}
.metric .hint { font-size: 0.78rem; margin-top: 0.45rem; opacity: 0.8; }
.m1 { background: linear-gradient(145deg, #ccfbf1, #99f6e4); color: #115e59; }
.m2 { background: linear-gradient(145deg, #e0f2fe, #bae6fd); color: #075985; }
.m3 { background: linear-gradient(145deg, #fecaca, #fda4af); color: #9f1239; }
.m4 { background: linear-gradient(145deg, #fef3c7, #fde68a); color: #92400e; }
a.metric {
    text-decoration: none !important;
    color: inherit !important;
    display: block;
}

.risk {
    border-radius: 22px;
    padding: 1.35rem 1.4rem;
    margin: 0.4rem 0 1rem 0;
    position: relative;
    overflow: hidden;
    animation: floatIn 0.6s ease-out;
}
.risk::before {
    content: "";
    position: absolute;
    width: 180px; height: 180px;
    right: -40px; top: -50px;
    border-radius: 50%;
    background: rgba(255,255,255,0.28);
}
.risk h3 {
    position: relative;
    font-family: "Fraunces", Georgia, serif !important;
    font-size: 1.7rem !important;
    margin: 0 0 0.4rem 0 !important;
}
.risk p { position: relative; margin: 0; line-height: 1.5; max-width: 46rem; }
.risk-green { background: linear-gradient(135deg, #ccfbf1, #a7f3d0); color: #065f46; }
.risk-grey { background: linear-gradient(135deg, #e8eaed, #d1d5db); color: #374151; }
.risk-yellow { background: linear-gradient(135deg, #fef3c7, #fde68a); color: #92400e; }
.risk-red { background: linear-gradient(135deg, #fecaca, #fda4af); color: #9f1239; }

.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 800;
}
.b-major { background: #fecdd3; color: #9f1239; }
.b-moderate { background: #fde68a; color: #92400e; }
.b-minor { background: #a7f3d0; color: #065f46; }
.b-unknown { background: #e5e7eb; color: #4b5563; }

.match {
    background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(240,253,250,0.9));
    border: 1px solid rgba(13,148,136,0.15);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    margin: 0.55rem 0;
    box-shadow: 0 8px 24px rgba(13,148,136,0.06);
}
.conf {
    display: inline-block;
    background: linear-gradient(90deg, #0d9488, #f97316);
    color: #fff;
    border-radius: 999px;
    padding: 0.18rem 0.7rem;
    font-weight: 800;
    font-size: 0.86rem;
}

.ix-card {
    background: rgba(255,255,255,0.8);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    margin: 0.55rem 0;
}
.ix-card.ix-none {
    background: rgba(240,253,250,0.7);
    border-style: dashed;
}
.ix-pair {
    font-family: "Fraunces", Georgia, serif;
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}
.b-none {
    background: #e2e8f0;
    color: #334155;
}

.disclaimer {
    margin-top: 1.6rem;
    padding: 1rem 1.15rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.55);
    border: 1px solid var(--line);
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.5;
}

/* Force readable dark text — fixes white-on-dark Streamlit theme bleed */
.stApp, .stMarkdown, .stMarkdown p, .stMarkdown li,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stExpander"] p,
[data-testid="stExpander"] div,
.stCaption, label {
    color: var(--ink) !important;
}
[data-testid="stAlert"] {
    color: #102a2e !important;
}
[data-testid="stAlert"] * {
    color: #102a2e !important;
}

.info-soft {
    background: linear-gradient(135deg, #ecfeff, #f0fdfa);
    border: 1px solid rgba(13,148,136,0.18);
    border-radius: 16px;
    padding: 0.95rem 1.1rem;
    color: #134e4a !important;
    font-weight: 600;
}

.med-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-top: 0.75rem;
}
@media (max-width: 800px) {
    .med-grid { grid-template-columns: 1fr; }
}
.med-field {
    background: #ffffff;
    border: 1px solid rgba(16,42,46,0.08);
    border-radius: 16px;
    padding: 0.9rem 1rem;
    box-shadow: 0 6px 18px rgba(16,42,46,0.04);
}
.med-field.wide { grid-column: 1 / -1; }
.med-field .k {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #0d9488 !important;
    margin-bottom: 0.35rem;
}
.med-field .v {
    color: #102a2e !important;
    font-size: 0.98rem;
    line-height: 1.45;
    font-weight: 500;
}
.med-title {
    background: linear-gradient(135deg, #f0fdfa, #fff7ed);
    border: 1px solid rgba(13,148,136,0.15);
    border-radius: 18px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.35rem;
}
.med-title h3 {
    font-family: "Fraunces", Georgia, serif !important;
    margin: 0 !important;
    color: #102a2e !important;
    font-size: 1.45rem !important;
}
.med-note {
    margin-top: 0.75rem;
    background: #fff7ed;
    border-radius: 12px;
    padding: 0.7rem 0.9rem;
    color: #9a3412 !important;
    font-size: 0.88rem;
    font-weight: 600;
}

.welcome-wrap {
    max-width: 560px;
    margin: 2.5rem auto 1rem auto;
    animation: floatIn 0.65s ease-out;
}
.welcome-card {
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.75);
    border-radius: 28px;
    padding: 2rem 1.85rem 1.7rem 1.85rem;
    box-shadow: 0 18px 48px rgba(16, 42, 46, 0.08);
}
.welcome-card h1 {
    font-family: "Fraunces", Georgia, serif !important;
    font-size: 2.35rem !important;
    margin: 0.4rem 0 0.45rem 0 !important;
    color: #102a2e !important;
}
.welcome-card p.lead {
    color: #4d6a70 !important;
    margin: 0 0 1.25rem 0;
    line-height: 1.5;
    font-size: 1.02rem;
}

/* Full-page welcome — large type for easier reading */
.welcome-shell {
    min-height: calc(100vh - 2rem);
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 1.5rem 0 2rem 0;
    animation: floatIn 0.65s ease-out;
}
.welcome-panel {
    width: 100%;
    border-radius: 32px;
    overflow: hidden;
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(255,255,255,0.8);
    box-shadow: 0 24px 60px rgba(16, 42, 46, 0.1);
}
.welcome-banner {
    position: relative;
    overflow: hidden;
    padding: 2.75rem 2.5rem 2.4rem 2.5rem;
    background:
        linear-gradient(135deg, rgba(13,148,136,0.97) 0%, rgba(15,118,110,0.94) 55%, rgba(249,115,22,0.82) 140%);
    color: #fff;
}
.welcome-banner::after {
    content: "";
    position: absolute;
    width: 320px; height: 320px;
    right: -80px; top: -100px;
    border-radius: 50%;
    background: rgba(255,255,255,0.12);
}
.welcome-banner::before {
    content: "";
    position: absolute;
    width: 180px; height: 180px;
    left: -40px; bottom: -70px;
    border-radius: 50%;
    background: rgba(249,115,22,0.28);
}
.welcome-banner .brand {
    position: relative;
    z-index: 1;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.28);
    border-radius: 999px;
    padding: 0.45rem 1rem;
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: 0.04em;
}
.welcome-banner h1 {
    position: relative;
    z-index: 1;
    font-family: "Fraunces", Georgia, serif !important;
    font-size: 3.6rem !important;
    font-weight: 700 !important;
    margin: 1.1rem 0 0.65rem 0 !important;
    color: #fff !important;
    line-height: 1.08 !important;
    letter-spacing: -0.03em;
}
.welcome-banner p {
    position: relative;
    z-index: 1;
    margin: 0;
    font-size: 1.45rem;
    line-height: 1.45;
    color: rgba(255,255,255,0.95);
    max-width: 38rem;
    font-weight: 500;
}
.welcome-body {
    padding: 2.25rem 2.5rem 2.5rem 2.5rem;
}
.welcome-body .field-label {
    font-size: 1.35rem;
    font-weight: 800;
    color: #102a2e;
    margin: 0 0 0.65rem 0;
}
.welcome-body .field-hint {
    font-size: 1.15rem;
    color: #4d6a70;
    margin: 0 0 1.1rem 0;
    line-height: 1.4;
}
.welcome-steps {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1.75rem;
}
.welcome-step {
    background: linear-gradient(180deg, #f0fdfa, #ecfeff);
    border: 1px solid rgba(13,148,136,0.15);
    border-radius: 20px;
    padding: 1.15rem 1.2rem;
}
.welcome-step .n {
    font-size: 1.1rem;
    font-weight: 800;
    color: #0d9488;
    margin-bottom: 0.35rem;
}
.welcome-step .t {
    font-size: 1.15rem;
    font-weight: 700;
    color: #102a2e;
    line-height: 1.3;
}
@media (max-width: 800px) {
    .welcome-banner { padding: 2rem 1.4rem; }
    .welcome-banner h1 { font-size: 2.6rem !important; }
    .welcome-banner p { font-size: 1.2rem; }
    .welcome-body { padding: 1.6rem 1.4rem 1.8rem 1.4rem; }
    .welcome-steps { grid-template-columns: 1fr; }
}
.hi-card {
    text-align: center;
    padding: 2.4rem 1.5rem;
}
.hi-card .wave {
    font-size: 2.6rem;
    margin-bottom: 0.4rem;
}
.hi-card h1 {
    font-family: "Fraunces", Georgia, serif !important;
    font-size: 2.8rem !important;
    margin: 0.2rem 0 0.55rem 0 !important;
    color: #102a2e !important;
}
.hi-card p {
    color: #4d6a70 !important;
    font-size: 1.1rem;
    margin: 0 auto 1.4rem auto;
    max-width: 26rem;
    line-height: 1.5;
}
.patient-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 999px;
    padding: 0.28rem 0.8rem;
    font-size: 0.88rem;
    font-weight: 700;
}

@keyframes floatIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.9; }
    50% { transform: scale(1.08); opacity: 1; }
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #0d9488, #f97316) !important;
    border: none !important;
    color: #fff !important;
    border-radius: 999px !important;
    font-weight: 800 !important;
    height: 3rem !important;
    box-shadow: 0 10px 24px rgba(13,148,136,0.28) !important;
}
div.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.85) !important;
    border: 1px solid rgba(13,148,136,0.2) !important;
    color: var(--ink) !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    height: 3rem !important;
}
/* Invisible click overlays on metric cards — must come AFTER secondary pill styles */
.st-key-filter_found,
.st-key-filter_all,
.st-key-filter_major,
.st-key-filter_moderate {
    margin-top: -124px !important;
    margin-bottom: 0 !important;
    height: 124px !important;
    position: relative !important;
    z-index: 6 !important;
}
.st-key-filter_found div.stButton,
.st-key-filter_all div.stButton,
.st-key-filter_major div.stButton,
.st-key-filter_moderate div.stButton {
    height: 124px !important;
}
.st-key-filter_found div.stButton > button,
.st-key-filter_found div.stButton > button[kind="secondary"],
.st-key-filter_all div.stButton > button,
.st-key-filter_all div.stButton > button[kind="secondary"],
.st-key-filter_major div.stButton > button,
.st-key-filter_major div.stButton > button[kind="secondary"],
.st-key-filter_moderate div.stButton > button,
.st-key-filter_moderate div.stButton > button[kind="secondary"] {
    opacity: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: transparent !important;
    border-radius: 20px !important;
    height: 124px !important;
    min-height: 124px !important;
    width: 100% !important;
    cursor: pointer !important;
}
textarea {
    border-radius: 16px !important;
    border: 1px solid rgba(13,148,136,0.18) !important;
    background: rgba(255,255,255,0.85) !important;
}
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.7);
    border-radius: 16px;
    border: 1px solid var(--line);
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Flow: welcome (patient name) → checker
# ---------------------------------------------------------------------------
if "app_step" not in st.session_state:
    st.session_state.app_step = "welcome"
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""
# Migrate older sessions that were stuck on the removed greeting page
if st.session_state.app_step == "greeting":
    st.session_state.app_step = "checker" if st.session_state.patient_name else "welcome"


def render_welcome_page() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 920px !important;
            padding-top: 1.2rem !important;
            padding-bottom: 1.5rem !important;
        }
        div[data-testid="stTextInput"] input {
            font-size: 1.55rem !important;
            min-height: 4.2rem !important;
            padding: 1.05rem 1.25rem !important;
            border-radius: 18px !important;
            border: 3px solid #0f766e !important;
            background: #ecfdf5 !important;
            color: #102a2e !important;
            font-weight: 700 !important;
        }
        div[data-testid="stTextInput"] input::placeholder {
            color: #5b7c82 !important;
            font-weight: 500 !important;
            opacity: 1 !important;
        }
        div.stButton > button[kind="primary"] {
            height: 4.2rem !important;
            font-size: 1.5rem !important;
            border-radius: 18px !important;
            font-weight: 800 !important;
            background: linear-gradient(90deg, #0f766e, #ea580c) !important;
            box-shadow: 0 14px 28px rgba(15, 118, 110, 0.35) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            border-radius:28px;
            padding:2.4rem 2.2rem 2rem 2.2rem;
            margin-bottom:1.25rem;
            background:linear-gradient(135deg,#0f766e 0%,#0d9488 48%,#ea580c 130%);
            color:#fff;
            box-shadow:0 22px 50px rgba(15,118,110,0.35);
        ">
          <div style="
            display:inline-flex;align-items:center;gap:0.45rem;
            background:rgba(255,255,255,0.2);border:1px solid rgba(255,255,255,0.3);
            border-radius:999px;padding:0.45rem 1rem;font-size:1.1rem;font-weight:800;
          ">💊 MedCheck</div>
          <h1 style="
            font-family:Fraunces,Georgia,serif !important;
            font-size:3.4rem !important;font-weight:700 !important;
            margin:1rem 0 0.55rem 0 !important;color:#ffffff !important;line-height:1.1 !important;
          ">Welcome</h1>
          <p style="margin:0;font-size:1.4rem;line-height:1.45;color:rgba(255,255,255,0.96);font-weight:500;max-width:38rem;">
            Enter the patient&rsquo;s name to begin checking medication interactions.
          </p>
        </div>
        """
        ,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p style="font-size:1.4rem;font-weight:800;color:#102a2e;margin:0 0 0.35rem 0;">
          Patient name
        </p>
        <p style="font-size:1.15rem;color:#134e4a;margin:0 0 0.75rem 0;font-weight:500;">
          Type the full name clearly, then press Continue.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if "welcome_patient_name" not in st.session_state:
        st.session_state.welcome_patient_name = st.session_state.patient_name

    name = st.text_input(
        "Patient name",
        placeholder="Example: Sara Ahmed",
        label_visibility="collapsed",
        key="welcome_patient_name",
    )

    st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)

    if st.button("Continue", type="primary", use_container_width=True, key="welcome_continue"):
        cleaned = (name or "").strip()
        if not cleaned:
            st.error("Please enter the patient's name to continue.")
        else:
            st.session_state.patient_name = cleaned
            st.session_state.app_step = "checker"
            st.rerun()

    st.markdown(
        """
        <div style="
            display:grid;grid-template-columns:repeat(3,1fr);gap:0.9rem;margin-top:1.5rem;
        ">
          <div style="background:rgba(236,253,245,0.92);border:2px solid rgba(15,118,110,0.2);
                      border-radius:18px;padding:1.1rem 1.15rem;">
            <div style="font-size:1.05rem;font-weight:800;color:#0f766e;margin-bottom:0.3rem;">Step 1</div>
            <div style="font-size:1.15rem;font-weight:700;color:#102a2e;line-height:1.3;">Enter the patient name</div>
          </div>
          <div style="background:rgba(255,247,237,0.95);border:2px solid rgba(234,88,12,0.2);
                      border-radius:18px;padding:1.1rem 1.15rem;">
            <div style="font-size:1.05rem;font-weight:800;color:#c2410c;margin-bottom:0.3rem;">Step 2</div>
            <div style="font-size:1.15rem;font-weight:700;color:#102a2e;line-height:1.3;">Add the medication list</div>
          </div>
          <div style="background:rgba(236,253,245,0.92);border:2px solid rgba(15,118,110,0.2);
                      border-radius:18px;padding:1.1rem 1.15rem;">
            <div style="font-size:1.05rem;font-weight:800;color:#0f766e;margin-bottom:0.3rem;">Step 3</div>
            <div style="font-size:1.15rem;font-weight:700;color:#102a2e;line-height:1.3;">Review interaction results</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



if st.session_state.app_step == "welcome":
    render_welcome_page()
    st.stop()


@st.cache_resource(show_spinner=False)
def get_model():
    return load_embedding_model(MODEL_NAME)


@st.cache_data(show_spinner=False)
def get_embeddings(_model_id: str, drug_names: tuple):
    model, error = get_model()
    if model is None or error:
        return None
    return create_drug_embeddings(model, list(drug_names))


@st.cache_data(show_spinner="Loading medication atlas…")
def cached_load_from_disk(interaction_mtime: float, medicine_mtime: float):
    return load_datasets(
        interaction_path=DEFAULT_INTERACTIONS if DEFAULT_INTERACTIONS.exists() else None,
        medicine_path=DEFAULT_MEDICINES if DEFAULT_MEDICINES.exists() else None,
    )


def load_app_data():
    inter_mtime = DEFAULT_INTERACTIONS.stat().st_mtime if DEFAULT_INTERACTIONS.exists() else 0.0
    med_mtime = DEFAULT_MEDICINES.stat().st_mtime if DEFAULT_MEDICINES.exists() else 0.0
    return cached_load_from_disk(inter_mtime, med_mtime)


def badge(severity: str) -> str:
    t = (severity or "").lower()
    if any(k in t for k in ["major", "severe", "high", "contraindic"]):
        return '<span class="badge b-major">Major</span>'
    if "moderate" in t:
        return '<span class="badge b-moderate">Moderate</span>'
    if "minor" in t or "mild" in t:
        return '<span class="badge b-minor">Minor</span>'
    return f'<span class="badge b-unknown">{severity or "Unknown"}</span>'


def display_label(name: str, mapping: dict) -> str:
    return mapping.get(normalize_drug_name(name), name)


def render_risk(risk: dict):
    titles = {
        "grey": "Not enough information",
        "green": "Not enough information",
        "yellow": "Moderate attention needed",
        "red": "High-risk combination",
    }
    css = {
        "grey": "risk-grey",
        "green": "risk-grey",
        "yellow": "risk-yellow",
        "red": "risk-red",
    }.get(risk["level"], "risk-yellow")
    st.markdown(
        f"""
        <div class="risk {css}">
            <h3>{titles.get(risk['level'], risk['title'])}</h3>
            <p>{risk['message']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_bucket(raw) -> str:
    """Map severity text → major | moderate | minor | none | unknown."""
    if raw is None:
        return "none"
    t = str(raw).lower().strip()
    if not t:
        return "none"
    if any(k in t for k in ["major", "severe", "high", "contraindic"]):
        return "major"
    if "moderate" in t:
        return "moderate"
    if "minor" in t or "mild" in t:
        return "minor"
    return "unknown"


def sync_pair_filter_from_url() -> str:
    """Read ?pf= from the URL into session state. Returns active filter."""
    if "pair_filter" not in st.session_state:
        st.session_state.pair_filter = "all"
    try:
        pf = st.query_params.get("pf", None)
        if pf is not None:
            if isinstance(pf, (list, tuple)):
                pf = pf[0] if pf else "all"
            if pf in ("all", "found", "major", "moderate"):
                st.session_state.pair_filter = pf
    except Exception:
        pass
    return st.session_state.get("pair_filter", "all")


def _apply_pair_filter(target: str) -> None:
    current = st.session_state.get("pair_filter", "all")
    st.session_state.pair_filter = "all" if current == target else target


def render_pair_results(results: dict):
    """Show pair cards, optionally filtered by Major / Moderate / etc."""
    pair_filter = st.session_state.get("pair_filter", "all")
    title_map = {
        "all": "All pairs checked",
        "found": "Detected interactions",
        "major": "Major pairs",
        "moderate": "Moderate pairs",
    }
    st.markdown(
        f'<p class="section-eyebrow" style="margin-top:1rem;">Results</p>'
        f'<p class="section-title">{title_map.get(pair_filter, "All pairs checked")}</p>',
        unsafe_allow_html=True,
    )
    all_pairs = results.get("all_pairs") or []
    found = results["found"]

    def _pair_matches_filter(item: dict) -> bool:
        if pair_filter == "all":
            return True
        has_hit = bool(item.get("found", True) and item.get("severity") is not None)
        if pair_filter == "found":
            return has_hit
        if not has_hit:
            return False
        return severity_bucket(item.get("severity")) == pair_filter

    visible = [item for item in all_pairs if _pair_matches_filter(item)]

    if not all_pairs:
        st.markdown(
            '<div class="info-soft">Nice — no interaction showed up for this combination in our dataset. Still check with a doctor or pharmacist if you are unsure.</div>',
            unsafe_allow_html=True,
        )
    elif not visible:
        st.markdown(
            '<div class="info-soft">No pairs in this filter. Tap <strong>Pairs checked</strong> to see everything again.</div>',
            unsafe_allow_html=True,
        )
    else:
        for item in visible:
            a = html.escape(str(item.get("drug_a_display", item["drug_a"])))
            b = html.escape(str(item.get("drug_b_display", item["drug_b"])))
            if item.get("found", True) and item.get("severity") is not None:
                src = "Inferred" if item.get("severity_source") == "keyword-inferred" else "Dataset"
                sev = badge(item["severity"])
                desc = html.escape(str(item.get("description") or ""))
                card_cls = "ix-card"
            else:
                src = ""
                sev = '<span class="badge b-none">No interaction found</span>'
                desc = html.escape(
                    str(
                        item.get("description")
                        or "We don't have enough information to check this combination. To stay safe, please ask your doctor or pharmacist."
                    )
                )
                card_cls = "ix-card ix-none"
            src_html = (
                f'<span style="margin-left:0.5rem;color:#4d6a70;font-size:0.85rem;">{src}</span>'
                if src
                else ""
            )
            st.markdown(
                (
                    f'<div class="{card_cls}">'
                    f'<div class="ix-pair">{a}  +  {b}</div>'
                    f"{sev}"
                    f"{src_html}"
                    f'<p style="margin:0.55rem 0 0 0;color:#102a2e !important;line-height:1.45;">{desc}</p>'
                    f"</div>"
                ),
                unsafe_allow_html=True,
            )
        if any(i.get("severity_source") == "keyword-inferred" for i in found):
            st.caption(
                "Severity is inferred from interaction text when the dataset has no severity column."
            )


@st.fragment
def render_outcome_interactive(results: dict):
    """
    Original polished HTML metric cards + fully invisible Streamlit button overlays.
    Looks like the old cards; clicks still filter results.
    """
    sync_pair_filter_from_url()
    risk = results["risk"]
    current = st.session_state.get("pair_filter", "all")

    # Reinforce invisible overlays inside the fragment (beats theme/button CSS)
    st.html(
        """
        <style>
        .st-key-filter_found,
        .st-key-filter_all,
        .st-key-filter_major,
        .st-key-filter_moderate {
            margin-top: -124px !important;
            margin-bottom: 0.35rem !important;
            height: 124px !important;
            position: relative !important;
            z-index: 6 !important;
        }
        .st-key-filter_found button,
        .st-key-filter_all button,
        .st-key-filter_major button,
        .st-key-filter_moderate button,
        .st-key-filter_found button[kind="secondary"],
        .st-key-filter_all button[kind="secondary"],
        .st-key-filter_major button[kind="secondary"],
        .st-key-filter_moderate button[kind="secondary"] {
            opacity: 0 !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: transparent !important;
            height: 124px !important;
            min-height: 124px !important;
            width: 100% !important;
            cursor: pointer !important;
        }
        </style>
        """
    )

    def active(key: str) -> str:
        return " is-active" if current == key else ""

    cards = [
        ("found", "m1", "Interactions", risk["num_interactions"], "Detected pairs"),
        ("all", "m2", "Pairs checked", risk["num_pairs"], "Unique combinations"),
        ("major", "m3", "Major", risk["num_major"], "High severity"),
        ("moderate", "m4", "Moderate", risk["num_moderate"], "Monitor closely"),
    ]
    key_for = {
        "found": "filter_found",
        "all": "filter_all",
        "major": "filter_major",
        "moderate": "filter_moderate",
    }

    cols = st.columns(4)
    for col, (filt, mcls, title, value, hint) in zip(cols, cards):
        with col:
            st.markdown(
                (
                    f'<div class="metric {mcls}{active(filt)}">'
                    f'<div class="label">{title}</div>'
                    f'<div class="value">{value}</div>'
                    f'<div class="hint">{hint}</div>'
                    f"</div>"
                ),
                unsafe_allow_html=True,
            )
            if st.button(
                title,
                key=key_for[filt],
                use_container_width=True,
                help=f"Show {hint.lower()}",
            ):
                _apply_pair_filter(filt)

    render_pair_results(results)


def render_medicine_details(results: dict) -> None:
    """Medicine info cards — used inside the details dialog."""
    blocks = results.get("details") or []
    if not blocks:
        st.markdown(
            '<div class="info-soft">No medicine details for this check.</div>',
            unsafe_allow_html=True,
        )
        return

    tabs = st.tabs([b["display"] for b in blocks])
    for tab, block in zip(tabs, blocks):
        with tab:
            d = block["details"]
            if not d.get("found"):
                st.markdown(
                    f'<div class="info-soft">No detailed record found for {html.escape(str(block["display"]))}.</div>',
                    unsafe_allow_html=True,
                )
                continue

            uses = html.escape(d.get("uses") or "Not available")
            sides = html.escape(d.get("side_effects") or "Not available")
            subs = html.escape(d.get("substitutes") or "Not available")
            name_safe = html.escape(d.get("name") or block["display"])

            if block["generic_hint"]:
                generic_val = html.escape(block["generic_hint"])
                chem = ""
                if d.get("generic") and normalize_drug_name(d["generic"]) != normalize_drug_name(
                    block["generic_hint"]
                ):
                    chem = (
                        '<div class="med-field">'
                        '<div class="k">Chemical class</div>'
                        f'<div class="v">{html.escape(d["generic"])}</div>'
                        "</div>"
                    )
            else:
                generic_val = html.escape(d.get("generic") or "Not available")
                chem = ""

            note = ""
            if d.get("substitutes"):
                note = '<div class="med-note">Substitutes are informational only. A clinician must approve any change.</div>'

            st.markdown(
                (
                    f'<div class="med-title"><h3>{name_safe}</h3></div>'
                    f'<div class="med-grid">'
                    f'<div class="med-field"><div class="k">Generic / composition</div>'
                    f'<div class="v">{generic_val}</div></div>'
                    f"{chem}"
                    f'<div class="med-field"><div class="k">Uses</div>'
                    f'<div class="v">{uses}</div></div>'
                    f'<div class="med-field"><div class="k">Side effects</div>'
                    f'<div class="v">{sides}</div></div>'
                    f'<div class="med-field wide"><div class="k">Substitutes</div>'
                    f'<div class="v">{subs}</div></div>'
                    f"</div>{note}"
                ),
                unsafe_allow_html=True,
            )


@st.dialog("Medicine information", width="large")
def medicine_details_dialog() -> None:
    results = st.session_state.get("results")
    if not results:
        st.info("Run a check first to see medicine details.")
        return
    render_medicine_details(results)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data = load_app_data()
interactions_df = data.get("interactions")
medicines_df = data.get("medicines")
interaction_map = data.get("interaction_map") or {}
medicine_map = data.get("medicine_map") or {}
drug_names = build_unique_drug_list(
    interactions_df, medicines_df, interaction_map, medicine_map
)
model, model_error = get_model()
embeddings = None
if model is not None and drug_names:
    embeddings = get_embeddings(MODEL_NAME, tuple(drug_names))

if "med_textarea" not in st.session_state:
    st.session_state.med_textarea = ""

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
patient = html.escape(st.session_state.patient_name)
st.markdown(
    f"""
<div class="hero">
  <div class="hero-kicker">💊 AI-assisted · Educational prototype</div>
  <div style="position:relative;z-index:1;margin-top:0.75rem;">
    <span class="patient-chip">👤 {patient}</span>
  </div>
  <h1>Hi, {patient}</h1>
  <p>Enter the medications below to get a clear, visual read on possible interactions — with smart name matching when spelling isn’t perfect.</p>
</div>
""",
    unsafe_allow_html=True,
)

if data.get("errors"):
    for err in data["errors"]:
        st.error(err)
if interactions_df is None and medicines_df is None:
    st.error("Medication data could not be loaded.")
    st.stop()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
st.markdown(
    '<p class="section-eyebrow">Start here</p><p class="section-title">Your medication list</p>',
    unsafe_allow_html=True,
)
st.text_area(
    "Medications",
    height=150,
    placeholder="Warfarin\nAspirin\nIbuprofen",
    label_visibility="collapsed",
    key="med_textarea",
)
b1, b2, _ = st.columns([1.4, 0.9, 2])
with b1:
    check = st.button("Check interactions", type="primary", use_container_width=True)
with b2:
    clear = st.button("Clear", use_container_width=True)

if clear:
    st.session_state.med_textarea = ""
    st.session_state.pop("results", None)
    st.session_state.pair_filter = "all"
    try:
        st.query_params.clear()
    except Exception:
        pass
    st.rerun()

if check:
    st.session_state.pop("results", None)
    st.session_state.pair_filter = "all"
    try:
        st.query_params.clear()
    except Exception:
        pass
    names, _ = parse_medication_input(st.session_state.med_textarea)
    if len(names) < 2:
        st.error("Please enter at least two medication names.")
        st.stop()
    if not drug_names:
        st.error("No medication names found in the data.")
        st.stop()

    with st.spinner("Matching names & scanning interactions…"):
        matches = match_all_medications(
            names,
            drug_names,
            drug_embeddings=embeddings,
            model=model,
            threshold=THRESHOLD,
            medicines_df=medicines_df,
        )
        reviewed = []
        for m in matches:
            display_name = getattr(m, "display_name", None) or m.matched
            generic_name = getattr(m, "generic_name", None) or ""
            accepted = m.status == "accepted"
            reviewed.append(
                {
                    "entered": m.entered,
                    "matched": m.matched,
                    "display_name": display_name,
                    "generic_name": generic_name,
                    "method": m.method,
                    "confidence": m.confidence,
                    "accepted": accepted,
                    "status": m.status,
                    "lookup_name": m.matched if accepted else "",
                    "final_display": display_name if accepted else "",
                }
            )

        unique = []
        seen = set()
        duplicates = []  # entered names that map to an already-included generic
        for r in reviewed:
            if not (r["accepted"] and r["lookup_name"]):
                continue
            key = normalize_drug_name(r["lookup_name"])
            if key not in seen:
                seen.add(key)
                unique.append(r)
            else:
                # Same active ingredient as an earlier entry (e.g. Glucophage + Metformin)
                prior = next(
                    (
                        u
                        for u in unique
                        if normalize_drug_name(u["lookup_name"]) == key
                    ),
                    None,
                )
                duplicates.append(
                    {
                        "entered": r["entered"],
                        "same_as": (prior or {}).get("final_display")
                        or (prior or {}).get("entered")
                        or r["lookup_name"],
                        "generic": r["lookup_name"],
                    }
                )

        if len(unique) < 2:
            failed = [
                r
                for r in reviewed
                if r.get("status") not in ("accepted",) or not r.get("lookup_name")
            ]
            # Still show duplicates-only case clearly
            if len(unique) == 1 and duplicates and not any(
                r.get("status") != "accepted" for r in reviewed if r not in unique
            ):
                pass
            lines = []
            for r in reviewed:
                if r.get("accepted") and r.get("lookup_name"):
                    continue
                if r.get("status") == "not_in_dataset":
                    lines.append(
                        f"- **{r['entered']}** → known as **{r.get('generic_name') or r['matched']}**, "
                        f"but that generic is **not in the interaction dataset** (cannot check pairs)."
                    )
                else:
                    hint = r["matched"] or "no close match"
                    lines.append(
                        f"- **{r['entered']}** → not confident "
                        f"(best guess: {hint}, {r['confidence']*100:.0f}%, {r['method']})"
                    )
            detail = "\n".join(lines) if lines else "- Could not match enough medicines."
            st.error(
                "At least two different medications must be confidently matched "
                "(same brand + generic counts as one).\n\n"
                + detail
            )
            st.stop()

        lookup_names = [r["lookup_name"] for r in unique]
        lookup_to_display = {
            normalize_drug_name(r["lookup_name"]): r.get("final_display") or r["lookup_name"]
            for r in unique
        }
        pairs = generate_medication_pairs(lookup_names)
        found, all_pairs = check_all_pairs(lookup_names, interactions_df, interaction_map)
        for item in found + all_pairs:
            item["drug_a_display"] = display_label(item["drug_a"], lookup_to_display)
            item["drug_b_display"] = display_label(item["drug_b"], lookup_to_display)
        risk = calculate_overall_risk(found, len(unique), len(pairs))

        details_blocks = []
        for r in unique:
            display = r.get("final_display") or r["lookup_name"]
            details = get_medicine_details(
                display,
                medicines_df,
                medicine_map,
                lookup_names=[r["lookup_name"], r.get("generic_name") or "", display],
            )
            details_blocks.append(
                {"display": display, "generic_hint": r.get("generic_name") or "", "details": details}
            )

        st.session_state["results"] = {
            "reviewed": reviewed,
            "unique": unique,
            "duplicates": duplicates,
            "found": found,
            "all_pairs": all_pairs,
            "risk": risk,
            "details": details_blocks,
        }

results = st.session_state.get("results")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if results:
    names = [r.get("final_display") or r["lookup_name"] for r in results["unique"]]
    pills = "".join(f'<span class="pill">{html.escape(str(n))}</span>' for n in names)
    notes = []
    for d in results.get("duplicates") or []:
        notes.append(
            f"<strong>{html.escape(str(d['entered']))}</strong> is the same medicine as "
            f"<strong>{html.escape(str(d['same_as']))}</strong> "
            f"(both are {html.escape(str(d['generic']))}) — counted once."
        )
    for r in results["reviewed"]:
        if r.get("status") == "accepted":
            continue
        if r.get("status") == "not_in_dataset":
            notes.append(
                f"<strong>{html.escape(str(r['entered']))}</strong> → "
                f"{html.escape(str(r.get('generic_name') or r['matched']))} "
                f"is <em>not in the interaction dataset</em>, so pairs cannot be checked."
            )
        else:
            guess = r.get("matched") or "no close match"
            notes.append(
                f"<strong>{html.escape(str(r['entered']))}</strong> skipped "
                f"(low confidence → {html.escape(str(guess))}, {r['confidence']*100:.0f}%). "
                f"Try the generic name."
            )
    note_html = ""
    if notes:
        note_html = (
            '<div class="info-soft" style="margin-top:0.85rem;">'
            + "<br/>".join(notes)
            + "</div>"
        )

    top_l, top_r = st.columns([5.2, 1.15], vertical_alignment="bottom")
    with top_l:
        st.markdown(
            '<p class="section-eyebrow" style="margin-bottom:0.15rem;">Matched list</p>'
            '<p class="section-title" style="margin-bottom:0.35rem;">Medications in this check</p>',
            unsafe_allow_html=True,
        )
    with top_r:
        if st.button("Medicine details", key="open_med_details", use_container_width=True):
            medicine_details_dialog()

    st.markdown(
        f"""
        <div class="glass" style="margin-top:0.35rem;">
          <div class="pill-row">{pills}</div>
          {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="section-eyebrow" style="margin-top:1.35rem;">Outcome</p>'
        '<p class="section-title">Overall risk</p>',
        unsafe_allow_html=True,
    )
    render_risk(results["risk"])
    render_outcome_interactive(results)

    st.markdown(
        '<p class="section-eyebrow">AI matching</p><p class="section-title">How names were recognized</p>',
        unsafe_allow_html=True,
    )
    for r in results["reviewed"]:
        if r["method"] != "Exact match" and r.get("matched"):
            generic = r.get("generic_name") or ""
            generic_line = f"Generic: {generic}" if generic else ""
            conf = f"{r['confidence'] * 100:.0f}%"
            try:
                st.html(
                    f"""
                    <div class="match">
                      <div><strong>Entered</strong> · {html.escape(r['entered'])}</div>
                      <div style="color:#0d9488;font-weight:800;margin:0.2rem 0;">↓</div>
                      <div><strong>Matched</strong> · {html.escape(str(r.get('display_name') or r['matched']))}</div>
                      {"<div style='color:#4d6a70;margin-top:0.25rem;'>" + html.escape(generic_line) + "</div>" if generic_line else ""}
                      <div style="margin-top:0.5rem;">
                        Confidence <span class="conf">{conf}</span>
                        <span style="color:#4d6a70;margin-left:0.4rem;">{html.escape(r['method'])}</span>
                      </div>
                    </div>
                    """
                )
            except Exception:
                st.markdown(
                    f"**Entered:** {r['entered']}  \n"
                    f"**Matched:** {r.get('display_name') or r['matched']}  \n"
                    + (f"**Generic:** {generic}  \n" if generic else "")
                    + f"**Confidence:** {conf} · {r['method']}"
                )
    table = pd.DataFrame(
        [
            {
                "Entered": r["entered"],
                "Matched": (
                    f"{r.get('final_display') or r.get('display_name') or r.get('matched')}"
                    + (
                        f" ({r['generic_name']})"
                        if r.get("generic_name")
                        and normalize_drug_name(r["generic_name"])
                        != normalize_drug_name(
                            str(r.get("final_display") or r.get("display_name") or "")
                        )
                        else ""
                    )
                ),
                "Method": r["method"],
                "Confidence": f"{r['confidence']*100:.0f}%",
            }
            for r in results["reviewed"]
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

st.markdown(
    """
<div class="disclaimer">
<strong>Disclaimer.</strong>
Educational prototype only. Results are based on available datasets and do not replace
advice from a doctor, pharmacist, or other qualified healthcare professional.
</div>
""",
    unsafe_allow_html=True,
)
