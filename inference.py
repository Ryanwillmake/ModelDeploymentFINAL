from pathlib import Path
import joblib
import pandas as pd


class CreditScoreInferencer:
    """Loads the trained pipeline and label encoder to score new customer data."""

    def __init__(self, model_path="artifacts/credit_score_pipeline.pkl",
                 encoder_path="artifacts/label_encoder.pkl"):
        self.model = joblib.load(Path(model_path))
        self.label_encoder = joblib.load(Path(encoder_path))

    def predict(self, data: dict) -> str:
        df = pd.DataFrame([data])
        pred_encoded = self.model.predict(df)[0]
        return self.label_encoder.inverse_transform([pred_encoded])[0]

    def predict_batch(self, df: pd.DataFrame):
        preds_encoded = self.model.predict(df)
        return self.label_encoder.inverse_transform(preds_encoded)


if __name__ == "__main__":
    sample = {
        "Month": "August",
        "Age": 35, "Occupation": "Engineer", "Annual_Income": 50000,
        "Monthly_Inhand_Salary": 4000, "Num_Bank_Accounts": 3, "Num_Credit_Card": 4,
        "Interest_Rate": 12, "Num_of_Loan": 2, "Delay_from_due_date": 5,
        "Num_of_Delayed_Payment": 3, "Changed_Credit_Limit": 5.5, "Num_Credit_Inquiries": 2,
        "Credit_Mix": "Good", "Outstanding_Debt": 1200, "Credit_Utilization_Ratio": 30.5,
        "Credit_History_Age": "10 Years and 5 Months", "Payment_of_Min_Amount": "No",
        "Total_EMI_per_month": 150, "Amount_invested_monthly": 300,
        "Payment_Behaviour": "High_spent_Small_value_payments", "Monthly_Balance": 400
    }

    inferencer = CreditScoreInferencer()
    result = inferencer.predict(sample)
    print(f"Predicted Credit Score: {result}")