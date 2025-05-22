import mlflow
import pandas as pd
import argparse
import warnings
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def main(n_estimators, learning_rate, max_depth):
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Load dataset
    train_data = pd.read_csv("car_evaluation_train_preprocessed.csv")
    test_data = pd.read_csv("car_evaluation_test_preprocessed.csv")

    X_train = train_data.drop("class", axis=1)
    y_train = train_data["class"]

    X_test = test_data.drop("class", axis=1)
    y_test = test_data["class"]

    input_example = X_train.iloc[:5]

    with mlflow.start_run():
        mlflow.autolog()

        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Manual metric logging
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("precision_macro", precision)
        mlflow.log_metric("recall_macro", recall)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        mlflow.log_artifact("car_evaluation_train_preprocessed.csv")
        mlflow.log_artifact("car_evaluation_test_preprocessed.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=3)
    args = parser.parse_args()

    main(args.n_estimators, args.learning_rate, args.max_depth)