import os
import argparse
import pandas as pd
import mlflow
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from mlflow.tracking import MlflowClient

# -----------------------------------
# Argument Parser for CLI Flexibility
# -----------------------------------
parser = argparse.ArgumentParser(description="Train and register XGBoost model with MLflow and Unity Catalog")
parser.add_argument("--data_path", required=True, help="Path to input CSV file")
parser.add_argument("--model_name", required=True, help="Name of the model to register")
parser.add_argument("--catalog", default="ws_lg_bics", help="Unity Catalog name")
parser.add_argument("--schema", default="default", help="Schema within Unity Catalog")
args = parser.parse_args()

# -----------------------------------
# Load & Preprocess Data
# -----------------------------------
df = pd.read_csv(args.data_path).dropna()
df = df.drop(columns=["node_id"])
df_encoded = pd.get_dummies(df, columns=["region", "node_type"])

X = df_encoded.drop("fraud_detected", axis=1).astype("float64")
y = df_encoded["fraud_detected"].astype("int")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------
# MLflow Autologging & Training
# -----------------------------------
mlflow.set_registry_uri("databricks-uc")
mlflow.xgboost.autolog()

with mlflow.start_run():
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
        scale_pos_weight=9  # class imbalance
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))

    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path=args.model_name,
        input_example=X_train.iloc[:5]
    )

    run_id = mlflow.active_run().info.run_id

model_uri = f"runs:/{run_id}/{args.model_name}"
print(f"Model logged at: {model_uri}")

# -----------------------------------
# Register Model in Unity Catalog
# -----------------------------------
client = MlflowClient()
uc_model_name = f"{args.catalog}.{args.schema}.{args.model_name}"

try:
    client.create_registered_model(name=uc_model_name)
except Exception as e:
    print(f"Model may already exist: {e}")

client.create_model_version(
    name=uc_model_name,
    source=model_uri,
    run_id=run_id
)

print(f"Model version registered to Unity Catalog as: {uc_model_name}")
