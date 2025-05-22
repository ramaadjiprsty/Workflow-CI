import argparse
import mlflow
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import dagshub

# Argument parser untuk hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--max_depth", type=int, default=3)
args = parser.parse_args()

# Inisialisasi MLflow & DagsHub
dagshub.init(repo_owner='ramaadjiprsty', repo_name='car-evaluation-classification', mlflow=True)
mlflow.set_experiment("Car Evaluation Classification")

# Load data
train_data = pd.read_csv("preprocessed_data/car_evaluation_train_preprocessed.csv")
test_data = pd.read_csv("preprocessed_data/car_evaluation_test_preprocessed.csv")

X_train = train_data.drop("class", axis=1)
y_train = train_data["class"]
X_test = test_data.drop("class", axis=1)
y_test = test_data["class"]

input_example = X_train.iloc[0:5]

# Definisi model dengan argumen
model = XGBClassifier(
    n_estimators=args.n_estimators,
    learning_rate=args.learning_rate,
    max_depth=args.max_depth,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

with mlflow.start_run():
    mlflow.autolog()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="macro")

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("f1_macro", f1)
    mlflow.log_metric("precision_macro", precision)
    mlflow.log_metric("recall_macro", recall)

    mlflow.log_artifact("car_evaluation_train_preprocessed.csv")
    mlflow.log_artifact("car_evaluation_test_preprocessed.csv")
