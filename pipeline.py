from pathlib import Path
from data_ingestion import CreditDataIngestion
from train import CreditModelTrainer
from evaluation import CreditModelEvaluator


class CreditScoringPipeline:
    """Master pipeline orchestrating ingestion, training, and evaluation."""

    def __init__(self, raw_data_path, accuracy_threshold=0.7):
        self.base_dir = Path(__file__).parent
        self.raw_data_path = Path(raw_data_path)
        self.ingested_dir = self.base_dir / "ingested"
        self.accuracy_threshold = accuracy_threshold

        self.ingestor = CreditDataIngestion(self.raw_data_path, self.ingested_dir)
        self.trainer = CreditModelTrainer()
        self.evaluator = CreditModelEvaluator()

    def execute(self):
        print("Executing Credit Scoring Pipeline...")

        ingested_file_path = self.ingestor.run()
        run_id, x_test, y_test = self.trainer.run(ingested_file_path)
        accuracy, precision, recall, f1 = self.evaluator.run(run_id, x_test, y_test)

        print("\n--- Deployment Approval Decision ---")
        if accuracy >= self.accuracy_threshold:
            print(f"Success: Model accuracy ({accuracy:.4f}) passes QA. Approved for deployment!")
        else:
            print(f"Rejected: Model accuracy ({accuracy:.4f}) falls short of threshold ({self.accuracy_threshold})")


if __name__ == "__main__":
    DATA_INPUT = Path(__file__).parent / "data_C.csv"
    credit_pipeline = CreditScoringPipeline(raw_data_path=DATA_INPUT, accuracy_threshold=0.7)
    credit_pipeline.execute()