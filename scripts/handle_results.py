import dagshub
import mlflow
import os
from dotenv import load_dotenv

load_dotenv("../config.env")

def store_results(experiment_name: str, params: dict, metrics: dict):
    dagshub.init(repo_owner=os.getenv("REPO_OWNER"), repo_name=("REPO_NAME"), mlflow=True)

    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(**params)
        mlflow.log_metrics(**metrics)
        print("Results saved successfully")

