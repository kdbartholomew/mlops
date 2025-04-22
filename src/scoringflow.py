from metaflow import FlowSpec, step
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import mlflow
import joblib
import os

class ScoringFlow(FlowSpec):

    @step
    def start(self):
        # Load new data for scoring (simulate by taking first 10 rows)
        print("📥 Loading new data...")
        data = load_breast_cancer()
        self.X_raw = data.data[:10]  # Simulated new incoming data to score
        self.next(self.clean_data)

    @step
    def clean_data(self):
        # Load the saved scaler and apply it to the new data
        print("🧹 Applying saved scaler...")
        self.scaler = joblib.load("scaler.pkl")  # Load previously saved scaler
        self.X = self.scaler.transform(self.X_raw)  # Scale the new data
        self.next(self.load_model)

    @step
    def load_model(self):
        # Load the trained model for prediction
        print("📦 Loading trained model...")
        self.model = joblib.load("best_model.pkl")  # Load previously saved model
        self.next(self.predict)

    @step
    def predict(self):
        # Use the model to predict on scaled new data
        print("🔮 Making predictions...")
        self.predictions = self.model.predict(self.X)
        print("✅ Predictions:", self.predictions.tolist())

        # Set MLflow tracking URI to local file path (replace with your path)
        mlflow.set_tracking_uri("file:///Users/catalinabartholomew/Documents/msdsMac/springMod2/specialTopicsInAI/mlops/mlruns")
        mlflow.set_experiment("ScoringFlow")  # Set the experiment name in MLflow

        # Log predictions as an artifact in MLflow under a run
        with mlflow.start_run(run_name="ScoringRun"):
            # Create artifacts directory if it doesn't exist
            os.makedirs("artifacts", exist_ok=True)

            # Save predictions to a text file
            predictions_path = "artifacts/predictions.txt"
            with open(predictions_path, "w") as f:
                f.write("\n".join(map(str, self.predictions.tolist())))

            # Log the predictions file to MLflow
            mlflow.log_artifact(predictions_path)
            print("📄 Predictions logged to MLflow")

        self.next(self.end)

    @step
    def end(self):
        # Final step indicating flow completion
        print("✅ Scoring flow completed.")

if __name__ == "__main__":
    ScoringFlow()  # Run the scoring flow
