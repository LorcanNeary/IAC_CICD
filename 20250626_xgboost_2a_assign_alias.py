import mlflow
from mlflow.tracking import MlflowClient

# -----------------------------------
# Configuration
# -----------------------------------
catalog = "ws_lg_bics"
schema = "default"
model_name = "telecom_fraud_model"
alias = "staging"

full_model_name = f"{catalog}.{schema}.{model_name}"

# Ensure Unity Catalog-aware client
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# -----------------------------------
# Resolve latest model version
# -----------------------------------
model_versions = client.search_model_versions(f"name='{full_model_name}'")
if not model_versions:
    raise ValueError(f"No model versions found for {full_model_name}")

latest_version = max(int(mv.version) for mv in model_versions)

# -----------------------------------
# Assign alias
# -----------------------------------
client.set_registered_model_alias(
    name=full_model_name,
    version=latest_version,
    alias=alias
)

print(f" Alias '{alias}' assigned to {full_model_name} version {latest_version}")