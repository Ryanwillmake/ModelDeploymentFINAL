from pathlib import Path
from typing import Tuple
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocessor import CreditFeaturePreprocessor, DROP_COLS


class CreditModelTrainer:

    def __init__(self, experiment_name="Credit Score Prediction",
                 artifact_path="artifacts", test_size=0.2, random_state=42):
        self.experiment_name = experiment_name
        self.artifact_dir = Path(artifact_path)
        self.test_size = test_size
        self.random_state = random_state
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_experiment(self.experiment_name)

    def _split(self, data_path):
        df = pd.read_csv(data_path)
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
        X = df.drop(columns=['Credit_Score'])
        y = df['Credit_Score']
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)

    def run(self, data_path) -> Tuple[str, pd.DataFrame, np.ndarray]:
        print("--- Step 2 & 3: Preprocessing + Training ---")

        x_train, x_test, y_train, y_test = self._split(data_path)

        label_encoder = LabelEncoder()
        y_train_enc = label_encoder.fit_transform(y_train)
        y_test_enc = label_encoder.transform(y_test)

        candidate_models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state),
            'random_forest_tuned': RandomForestClassifier(
                n_estimators=300, max_depth=20, min_samples_split=2,
                min_samples_leaf=1, max_features='sqrt', random_state=self.random_state
            ),
        }

        best_run_id, best_pipeline, best_acc, best_name = None, None, -1, None

        for name, clf in candidate_models.items():
            pipeline = Pipeline([
                ('preprocessing', CreditFeaturePreprocessor()),
                ('classifier', clf)
            ])

            with mlflow.start_run(run_name=name) as run:
                mlflow.log_param("model_name", name)
                mlflow.log_params(clf.get_params())
                pipeline.fit(x_train, y_train_enc)
                preds = pipeline.predict(x_test)

                acc = accuracy_score(y_test_enc, preds)
                prec = precision_score(y_test_enc, preds, average='macro')
                rec = recall_score(y_test_enc, preds, average='macro')
                f1 = f1_score(y_test_enc, preds, average='macro')

                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1_score", f1)
                mlflow.sklearn.log_model(pipeline, name="model")

                print(f"[{name}] accuracy={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    best_run_id = run.info.run_id
                    best_pipeline = pipeline
                    best_name = name

        joblib.dump(best_pipeline, self.artifact_dir / "credit_score_pipeline.pkl", compress=9)
        joblib.dump(label_encoder, self.artifact_dir / "label_encoder.pkl", compress=9)

        print(f"Best model: {best_name} (accuracy={best_acc:.4f})")
        return best_run_id, x_test, y_test_enc