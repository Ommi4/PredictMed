from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/predictmed_model.joblib")


class DataQualityAgent:
    """Checks whether required prediction inputs are available."""

    def check(self, patient):
        if patient.isna().all():
            return False, "Patient data is empty."

        required = [
            "early_egfr_last",
            "early_egfr_mean",
            "blgfr",
        ]

        missing = [
            column for column in required
            if column not in patient.index
            or pd.isna(patient[column])
        ]

        if missing:
            return False, (
                "Prediction paused. Missing inputs: "
                + ", ".join(missing)
            )

        return True, "Basic data-quality checks passed."


class PredictionAgent:
    """Uses the saved ML model to estimate the target probability."""

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Train the model first by running train_model.py."
            )

        self.model = joblib.load(MODEL_PATH)

    def predict(self, patient):
        inputs = pd.DataFrame([patient])

        probability = self.model.predict_proba(inputs)[0, 1]

        return float(probability)


class ExplanationAgent:
    """Creates a simple explanation from observed input trends."""

    def explain(self, patient):
        reasons = []

        first = patient.get("early_egfr_first")
        last = patient.get("early_egfr_last")
        slope = patient.get("early_egfr_slope_year")

        if pd.notna(first) and pd.notna(last):
            change = float(last) - float(first)

            if change < 0:
                reasons.append(
                    f"Observed eGFR decreased by {abs(change):.1f} "
                    "between the first and last early measurements."
                )
            elif change > 0:
                reasons.append(
                    f"Observed eGFR increased by {change:.1f} "
                    "between the first and last early measurements."
                )
            else:
                reasons.append(
                    "The first and last early eGFR values were equal."
                )

        if pd.notna(slope):
            reasons.append(
                f"Estimated early eGFR trend: {float(slope):.1f} "
                "mL/min/1.73 m² per year."
            )

        if not reasons:
            reasons.append(
                "Not enough information for a trend explanation."
            )

        reasons.append(
            "These are descriptive observations, not proof of "
            "disease progression or causal explanations for the model."
        )

        return reasons


class CareCoordinationAgent:
    """Prepares a review task; does not make clinical decisions."""

    def create_task(self, probability):
        return {
            "status": "Requires human review",
            "estimated_probability": round(probability, 4),
            "next_step": (
                "A qualified care-management professional should "
                "review the record and determine appropriate follow-up."
            ),
        }


class PredictMedWorkflow:
    def __init__(self):
        self.quality_agent = DataQualityAgent()
        self.prediction_agent = PredictionAgent()
        self.explanation_agent = ExplanationAgent()
        self.care_agent = CareCoordinationAgent()

    def run(self, patient):
        valid, message = self.quality_agent.check(patient)

        if not valid:
            return {
                "status": "Cannot predict",
                "data_quality": message,
            }

        probability = self.prediction_agent.predict(patient)

        return {
            "status": "Completed for research demonstration",
            "data_quality": message,
            "estimated_probability": probability,
            "explanation": self.explanation_agent.explain(patient),
            "care_task": self.care_agent.create_task(probability),
        }