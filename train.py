import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ① 最先设置，必须在 set_experiment / start_run 之前
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris_demo")

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = LogisticRegression(max_iter=200)

with mlflow.start_run() as run:
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metric("acc", model.score(X_test, y_test))
    print("run_id：", run.info.run_id)
