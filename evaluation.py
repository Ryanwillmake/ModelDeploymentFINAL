import mlflow
import mlflow.sklearn
import pandas as pd
from typing import Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class CreditModelEvaluator:
    """Pulls the selected pipeline back from MLflow tracking and validates it against test data."""

    def run(self, run_id: str, x_test: pd.DataFrame, y_test) -> Tuple[float, float, float, float]:
        print("--- Step 4: Evaluation ---")

        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)

        preds = model.predict(x_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='macro')
        rec = recall_score(y_test, preds, average='macro')
        f1 = f1_score(y_test, preds, average='macro')

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("final_accuracy", acc)
            mlflow.log_metric("final_precision", prec)
            mlflow.log_metric("final_recall", rec)
            mlflow.log_metric("final_f1_score", f1)

        print(f"Evaluation completed | Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f}")
        return acc, prec, rec, f1