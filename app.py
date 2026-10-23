import json
from pathlib import Path

import pandas as pd
import streamlit as st


DATA_PATH = Path("data/processed_patients.csv")
FOLLOWUP_PATH = Path("data/followup.csv")
MODEL_PATH = Path("models/predictmed_model.joblib")
METRICS_PATH = Path("models/metrics.json")

FEATURES = [
    "early_egfr_first",
    "early_egfr_last",
    "early_egfr_mean",
    "early_egfr_min",
    "early_egfr_slope_year",
    "early_visit_count",
    "days_observed",
    "blgfr",
    "trt01pn",
    "blglp1",
    "strata",
]


st.set_page_config(
    page_title="NeuroMedix",
    page_icon="N",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');
    :root {
        --ink: #173b35;
        --muted: #60736d;
        --line: #d9e4de;
        --paper: #f3f6f2;
        --white: #ffffff;
        --green: #176c5b;
        --coral: #c66f50;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: var(--paper); color: var(--ink); }
    .block-container { max-width: 1320px; padding: 1.6rem 2.2rem 3rem; }
    [data-testid="stHeader"] { background: transparent; }
    .nm-masthead {
        display: flex; justify-content: space-between; align-items: flex-end;
        gap: 24px; padding: 26px 30px 28px; margin-bottom: 18px;
        background: #173b35; color: #f5f6ee; border-radius: 8px;
        border-bottom: 4px solid #c66f50;
    }
    .nm-brand { display:flex; align-items:center; gap:10px; color:#b8d6c9;
        font: 500 12px 'DM Mono', monospace; text-transform:uppercase; }
    .nm-brand-mark { display:grid; place-items:center; width:28px; height:28px;
        border:1px solid #7eaa98; border-radius:7px; color:#f5f6ee; font-weight:500; }
    .nm-title { margin:17px 0 5px; font:600 clamp(30px, 4vw, 45px)/1.04 'Newsreader', serif;
        color:#f5f6ee; }
    .nm-subtitle { color:#c2d2ca; font-size:14px; }
    .nm-snapshot { min-width:180px; padding-left:20px; border-left:1px solid #55746a; }
    .nm-snapshot-label { color:#a8c5b7; font:11px 'DM Mono', monospace; text-transform:uppercase; }
    .nm-snapshot-value { margin-top:8px; color:#fff; font-size:15px; font-weight:600; }
    .nm-notice { padding:11px 14px; margin:0 0 20px; border:1px solid #e7c8b8;
        border-left:4px solid var(--coral); border-radius:5px; background:#fffaf6;
        color:#765747; font-size:13px; }
    .section-kicker { margin:2px 0 12px; color:#60736d; font:11px 'DM Mono', monospace;
        text-transform:uppercase; }
    div[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:22px; border-bottom:1px solid var(--line); }
    div[data-testid="stTabs"] [role="tab"] { padding:12px 2px; color:var(--muted) !important; }
    div[data-testid="stTabs"] [role="tab"] p { color:var(--muted) !important; }
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"],
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] p { color:var(--green) !important; }
    div[data-testid="stMetric"] { padding:15px 17px; border:1px solid var(--line);
        border-top:3px solid var(--green); border-radius:6px; background:var(--white); }
    label[data-testid="stMetricLabel"], label[data-testid="stMetricLabel"] *,
    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p { color:var(--muted) !important; font-size:12px; }
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] div { color:var(--ink) !important; font:600 27px 'DM Sans', sans-serif; }
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
        color:var(--muted) !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        min-height:44px; border-color:var(--line) !important; border-radius:5px;
        background:#fff !important; color:var(--ink) !important;
    }
    [data-testid="stSelectbox"] input,
    [data-testid="stSelectbox"] [data-testid="stMarkdownContainer"] p {
        color:var(--ink) !important;
    }
    [data-testid="stSelectbox"] svg { fill:var(--muted) !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--line); border-radius:6px; }
    div[data-testid="stDataFrame"] { width:100%; border:1px solid var(--line); border-radius:6px; background:#fff; }
    div[data-testid="stDataFrame"] iframe { background:#fff; }
    .stButton button[kind="primary"] { border-radius:5px; background:var(--green); border-color:var(--green); }
    .nm-footnote { margin-top:14px; color:#71817c; font-size:12px; }
    @media (max-width: 720px) {
        .block-container { padding:1rem 1rem 2rem; }
        .nm-masthead { align-items:flex-start; flex-direction:column; padding:22px; }
        .nm-snapshot { padding:10px 0 0; border-left:0; border-top:1px solid #55746a; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_patient_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_followup_data():
    data = pd.read_csv(FOLLOWUP_PATH)
    data.columns = data.columns.str.lower().str.strip()
    data["usubjid"] = data["usubjid"].astype(str)
    data["ady"] = pd.to_numeric(data["ady"], errors="coerce")
    data["aval"] = pd.to_numeric(data["aval"], errors="coerce")
    return data


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r") as file:
        return json.load(file)


required_files = [
    DATA_PATH,
    FOLLOWUP_PATH,
    MODEL_PATH,
    METRICS_PATH,
]

if not all(path.exists() for path in required_files):
    st.error(
        "Required files are missing. First run data_pipeline.py "
        "and then train_model.py."
    )
    st.stop()

data = load_patient_data()
followup = load_followup_data()
metrics = load_metrics()

st.markdown(
    """
    <header class="nm-masthead">
      <div>
        <div class="nm-brand"><span class="nm-brand-mark">N</span> NeuroMedix <span>/</span> Research intelligence</div>
        <div class="nm-title">Renal function, in context.</div>
        <div class="nm-subtitle">Chronic kidney disease deterioration monitoring</div>
      </div>
      <div class="nm-snapshot">
        <div class="nm-snapshot-label">Dataset snapshot</div>
        <div class="nm-snapshot-value">Research cohort · {}</div>
      </div>
    </header>
    <div class="nm-notice"><strong>Research prototype</strong> &nbsp; Not clinically validated. Do not use for diagnosis, treatment, or care decisions.</div>
    """.format(len(data)),
    unsafe_allow_html=True,
)

total_patients = len(data)
positive_cases = int(data["future_egfr_below_30"].sum())

patient_ids = sorted(data["usubjid"].astype(str).unique())
overview_tab, patient_tab, population_tab = st.tabs(
    ["Overview", "Patient review", "Population"]
)

with overview_tab:
    st.markdown('<div class="section-kicker">Cohort at a glance</div>', unsafe_allow_html=True)
    summary_cols = st.columns(3)
    summary_cols[0].metric("Usable patient timelines", f"{total_patients:,}")
    summary_cols[1].metric("Observed target-positive", f"{positive_cases:,}")
    summary_cols[2].metric("Held-out test patients", f"{metrics['test_patients']:,}")
    st.markdown(
        '<div class="nm-footnote">Target: any observed follow-up eGFR below 30 after day 365. This is a research label, not a diagnosis.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker" style="margin-top:30px">Model evaluation · Held-out cohort</div>', unsafe_allow_html=True)
    metric_cols = st.columns(4)
    for column, key, label in zip(
        metric_cols,
        ["accuracy", "precision", "recall", "roc_auc"],
        ["Accuracy", "Precision", "Recall", "ROC-AUC"],
    ):
        value = metrics.get(key)
        column.metric(label, "N/A" if value is None else f"{value:.3f}")

    matrix_col, note_col = st.columns([1, 1.3], gap="large")
    with matrix_col:
        st.markdown('<div class="section-kicker" style="margin-top:24px">Confusion matrix · Actual × predicted</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(
                metrics["confusion_matrix"],
                index=["Actual 0", "Actual 1"],
                columns=["Predicted 0", "Predicted 1"],
            ),
            use_container_width=True,
        )
    with note_col:
        st.markdown('<div class="section-kicker" style="margin-top:24px">Interpretation</div>', unsafe_allow_html=True)
        st.info("Evaluation metrics describe this held-out research subset only; they do not establish clinical safety or real-world performance.")

with patient_tab:
    top_left, top_right = st.columns([1, 2], vertical_alignment="bottom")
    with top_left:
        selected_id = st.selectbox("Patient record", patient_ids)
    with top_right:
        st.caption("Historical measurements only · No forecast or clinical recommendation")

    patient = data[data["usubjid"].astype(str) == selected_id].iloc[0]
    patient_history = followup[followup["usubjid"] == selected_id].sort_values("ady")
    chart_data = patient_history[["ady", "aval"]].dropna().set_index("ady")

    patient_metrics = st.columns(4)
    patient_metrics[0].metric("First early eGFR", f"{patient['early_egfr_first']:.1f}")
    patient_metrics[1].metric("Latest early eGFR", f"{patient['early_egfr_last']:.1f}")
    patient_metrics[2].metric("Early mean", f"{patient['early_egfr_mean']:.1f}")
    patient_metrics[3].metric("Early visits", f"{int(patient['early_visit_count'])}")

    chart_col, analysis_col = st.columns([1.35, 0.85], gap="large")
    with chart_col:
        st.markdown('<div class="section-kicker" style="margin-top:20px">eGFR timeline · Days from baseline</div>', unsafe_allow_html=True)
        if not chart_data.empty:
            st.line_chart(chart_data, color="#176c5b", height=300)
        else:
            st.info("No usable timeline is available for this patient.")
    with analysis_col:
        st.markdown('<div class="section-kicker" style="margin-top:20px">Research workflow</div>', unsafe_allow_html=True)
        st.write("Generate a model estimate and trend description for this selected record.")
        if st.button("Run patient analysis", type="primary", use_container_width=True):
            try:
                from agent_workflow import PredictMedWorkflow

                result = PredictMedWorkflow().run(patient)
                st.write("**Status:**", result["status"])
                st.write("**Data quality:**", result["data_quality"])
                if result["status"] == "Cannot predict":
                    st.error(result["data_quality"])
                else:
                    st.metric("Estimated model probability", f"{result['estimated_probability']:.1%}")
                    st.caption("Experimental estimate for the defined research outcome, not a patient's clinical risk.")
                    st.markdown("**Trend-based explanation**")
                    for reason in result["explanation"]:
                        st.write(f"- {reason}")
                    st.markdown("**Care coordination task**")
                    st.write(result["care_task"]["status"])
                    st.write(result["care_task"]["next_step"])
            except Exception as error:
                st.error(f"Analysis failed: {error}")

with population_tab:
    st.markdown('<div class="section-kicker">Patient-level research dataset</div>', unsafe_allow_html=True)
    display_data = data[
        [
            "usubjid",
            "early_egfr_first",
            "early_egfr_last",
            "early_egfr_slope_year",
            "early_visit_count",
            "future_egfr_below_30",
        ]
    ].copy()
    st.dataframe(display_data, use_container_width=True, height=520)
    st.markdown(
        '<div class="nm-footnote">Use only appropriately authorized, de-identified data. This prototype does not connect to claims systems or send clinical alerts.</div>',
        unsafe_allow_html=True,
    )