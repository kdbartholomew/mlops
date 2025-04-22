from metaflow import FlowSpec, step, Parameter
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

class TrainingFlow(FlowSpec):

    # Define parameters with defaults that can be overridden at runtime
    seed = Parameter("seed", default=42)
    cv = Parameter("cv", default=5)  # Not used in this example but shows how to pass params

    @step
    def start(self):
        # Load the breast cancer dataset
        print("📥 Loading dataset...")
        data = load_breast_cancer()
        self.X_raw = data.data      # Features
        self.y = data.target        # Labels
        self.next(self.clean_data)  # Move to next step

    @step
    def clean_data(self):
        # Preprocess data by scaling features
        print("🧹 Preprocessing data...")
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X_raw)  # Fit and transform features
        joblib.dump(scaler, "scaler.pkl")           # Save scaler for later use (e.g., scoring)
        self.next(self.train)                        # Proceed to training step

    @step
    def train(self):
        # Train a RandomForest model using the preprocessed data
        print("🏋️ Training model...")
        model = RandomForestClassifier(random_state=self.seed)
        model.fit(self.X, self.y)
        self.model = model

        # Evaluate model accuracy on training data (just for demonstration)
        self.accuracy = accuracy_score(self.y, model.predict(self.X))
        self.next(self.register_model)              # Proceed to model registration

    @step
    def register_model(self):
        # Log the trained model and metrics to MLflow tracking server
        print("📝 Logging model to MLFlow...")
        mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Set MLflow tracking server URI

        with mlflow.start_run(run_name="TrainingFlow"):
            mlflow.sklearn.log_model(self.model, "model")  # Log the sklearn model artifact
            mlflow.log_metric("accuracy", self.accuracy)   # Log accuracy metric
            mlflow.log_params({"seed": self.seed})          # Log parameters used

        # Save the model locally for later use (e.g., in scoring flow)
        joblib.dump(self.model, "best_model.pkl")
        print("✅ Model saved and registered.")
        self.next(self.end)  # End flow

    @step
    def end(self):
        # Final step of the flow
        print("✅ Training flow completed.")

if __name__ == "__main__":
    TrainingFlow()  # Run the flow
