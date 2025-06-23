import mlflow
import os
import argparse

server = "az-sql-lg-bics.database.windows.net"
database = "db_az_sql_20250619"
username = "sqladmin"

parser = argparse.ArgumentParser()
parser.add_argument('--sql-password', required=True)
args = parser.parse_args()

password = args.sql_password

mlflow.set_tracking_uri(
    f"mssql+pymssql://{username}:{password}@{server}:1433/{database}"
)

print("Tracking URI is now:", mlflow.get_tracking_uri())

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sql-password', required=True)
args = parser.parse_args()

password = args.sql_password