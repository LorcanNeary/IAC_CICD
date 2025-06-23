import mlflow
import os

server = "az-sql-lg-bics.database.windows.net"
database = "db_az_sql_20250619"
username = "sqladmin"
password = os.getenv("SQL_PASSWORD")  # Securely read from environment

if not password:
    raise ValueError("SQL_PASSWORD environment variable is not set")

mlflow.set_tracking_uri(
    f"mssql+pymssql://{username}:{password}@{server}:1433/{database}"
)

print("Tracking URI is now:", mlflow.get_tracking_uri())
