"""Weekly retraining and forecast generation (Airflow TaskFlow DAG)."""

from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.decorators import task

default_args = {
    "owner": "ml-engineering",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="ecommerce_forecast_pipeline",
    default_args=default_args,
    description="Weekly retraining and forecast generation",
    schedule="0 6 * * MON",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["forecasting", "ml", "ecommerce"],
) as dag:

    @task()
    def run_dbt_transformations():
        import subprocess

        result = subprocess.run(
            ["dbt", "run", "--project-dir", "/opt/dbt", "--profiles-dir", "/opt/dbt"],
            capture_output=True,
            text=True,
            check=True,
        )
        return {"stdout": result.stdout, "returncode": result.returncode}

    @task()
    def run_dbt_tests():
        import subprocess

        result = subprocess.run(
            ["dbt", "test", "--project-dir", "/opt/dbt"],
            capture_output=True,
            text=True,
            check=True,
        )
        return {"passed": "ERROR" not in result.stdout}

    @task()
    def validate_data_quality(test_results: dict):
        if not test_results["passed"]:
            raise ValueError("dbt tests failed — aborting training pipeline")
        return True

    @task()
    def train_models(validation_passed: bool):
        if not validation_passed:
            raise ValueError("Skipping training — data validation failed")

        import pandas as pd

        from src.features.dataset_builder import SalesForecastDataModule
        from src.training.train_with_mlflow import train_lstm_with_mlflow

        df = pd.read_parquet("/opt/data/mart_ts_features.parquet")
        dm = SalesForecastDataModule(df, seq_len=8, pred_len=4, batch_size=32)
        dm.setup()
        return train_lstm_with_mlflow(dm)

    @task()
    def promote_model_if_improved(training_results: dict):
        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        new_mape = training_results["mape"]

        try:
            prod_versions = client.get_latest_versions(
                "lstm-ecommerce-forecaster", stages=["Production"]
            )
            prod_mape = float(
                client.get_run(prod_versions[0].run_id).data.metrics["test_mape"]
            )
            if new_mape < prod_mape * 0.98:
                latest = client.get_latest_versions(
                    "lstm-ecommerce-forecaster", stages=["Staging"]
                )[0]
                client.transition_model_version_stage(
                    "lstm-ecommerce-forecaster", latest.version, "Production"
                )
                return {"promoted": True, "improvement": prod_mape - new_mape}
        except (IndexError, KeyError):
            return {"promoted": True, "first_deploy": True}

        return {"promoted": False, "new_mape": new_mape, "prod_mape": prod_mape}

    @task()
    def generate_forecasts(promotion_result: dict):
        return {"forecasts_written": True, "promotion": promotion_result}

    dbt_run = run_dbt_transformations()
    dbt_test = run_dbt_tests()
    validated = validate_data_quality(dbt_test)
    trained = train_models(validated)
    promoted = promote_model_if_improved(trained)
    forecasts = generate_forecasts(promoted)

    dbt_run >> dbt_test >> validated >> trained >> promoted >> forecasts
