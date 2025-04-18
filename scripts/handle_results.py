import dagshub
import mlflow
import os

def store_results(experiment_name: str, params: dict, metrics: dict, repo_owner:str, repo_name:str):
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(**params)
        mlflow.log_metrics(**metrics)
        print("Results saved successfully")

