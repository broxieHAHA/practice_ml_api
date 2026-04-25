import os
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "iris_model"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("iris_experiment")

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

with mlflow.start_run() as run:
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    acc = accuracy_score(y_test, knn.predict(X_test))

    mlflow.log_param("n_neighbors", 5)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        sk_model=knn,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
    )

    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {acc:.4f}")

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
latest = client.get_latest_versions(MODEL_NAME, stages=["None"])
if latest:
    version = latest[0].version
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
    )
    print(f"Модель v{version} → Production")