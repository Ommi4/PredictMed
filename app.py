import json
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib

from agent_workflow import PredictMedWorkflow


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


st.title("NeuroMedix")
st.subheader(
    "Agentic AI for Chronic Kidney Disease Deterioration Monitoring"
)

st.warning(
    "Research prototype only. This system is not clinically validated "
    "and must not be used to diagnose, treat, or make care decisions."
)

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
model = joblib.load(MODEL_PATH)

st.header("Population Overview")

total_patients = len(data)
positive_cases = int(data["future_egfr_below_30"].sum())

col1, col2, col3 = st.columns(3)

col1.metric("Patients with usable timelines", total_patients)
col2.metric("Observed target-positive patients", positive_cases)
col3.metric(
    "Held-out test patients",
    metrics["test_patients"],
)

st.caption(
    "The target counts patients with an observed follow-up eGFR below "
    "30 after day 365. It is a research label, not a diagnosis."
)

st.header("Model Evaluation")

metric_cols = st.columns(4)

for column, key, label in zip(
    metric_cols,
    ["accuracy", "precision", "recall", "roc_auc"],
    ["Accuracy", "Precision", "Recall", "ROC-AUC"],
):
    value = metrics.get(key)

    column.metric(
        label,
        "N/A" if value is None else f"{value:.3f}",
    )

st.write("Confusion matrix (actual rows, predicted columns):")
st.write(pd.DataFrame(
    metrics["confusion_matrix"],
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"],
))

st.caption(
    "Metrics are from a held-out subset of this research dataset. "
    "They do not establish clinical safety or real-world performance."
)

st.header("Patient Monitoring")

patient_ids = sorted(data["usubjid"].astype(str).unique())

selected_id = st.selectbox(
    "Select a patient record",
    patient_ids,
)

patient = data[
    data["usubjid"].astype(str) == selected_id
].iloc[0]

patient_history = followup[
    followup["usubjid"] == selected_id
].sort_values("ady")

left, right = st.columns(2)

with left:
    st.subheader("Available early measurements")

    st.write({
        "First early eGFR": patient["early_egfr_first"],
        "Last early eGFR": patient["early_egfr_last"],
        "Mean early eGFR": patient["early_egfr_mean"],
        "Early visit count": int(patient["early_visit_count"]),
    })

with right:
    st.subheader("eGFR timeline")

    if not patient_history.empty:
        chart_data = patient_history[
            ["ady", "aval"]
        ].dropna().set_index("ady")

        st.line_chart(chart_data)
        st.caption("Historical observations only; not a forecast.")
    else:
        st.info("No usable timeline is available.")

st.header("Agentic AI Workflow")

if st.button("Run PredictMed analysis", type="primary"):
    try:
        workflow = PredictMedWorkflow()
        result = workflow.run(patient)

        st.write("**Workflow status:**", result["status"])
        st.write("**Data quality:**", result["data_quality"])

        if result["status"] == "Cannot predict":
            st.error(result["data_quality"])
        else:
            probability = result["estimated_probability"]

            st.metric(
                "Estimated model probability",
                f"{probability:.1%}",
            )

            st.caption(
                "This is an experimental model estimate for the "
                "defined research outcome, not a patient's clinical "
                "risk and not a validated risk category."
            )

            st.subheader("Trend-based explanation")

            for reason in result["explanation"]:
                st.write(f"- {reason}")

            st.subheader("Care coordination task")
            st.write(result["care_task"]["status"])
            st.write(result["care_task"]["next_step"])

    except Exception as error:
        st.error(f"Analysis failed: {error}")

st.header("Patient Population Table")

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

st.dataframe(display_data, use_container_width=True)

st.caption(
    "Use only appropriately authorized, de-identified data. "
    "This prototype does not connect to insurance claims systems "
    "or send real clinical alerts."
)