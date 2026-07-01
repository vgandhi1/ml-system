import pandas as pd
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import TestValueMAE, TestValueMAPE, TestValueRMSE


class ForecastDriftMonitor:
    """Monitors production forecast quality using Evidently AI."""

    def __init__(self, mape_threshold: float = 8.0, rmse_threshold: float = 300):
        self.mape_threshold = mape_threshold
        self.rmse_threshold = rmse_threshold

    def run_accuracy_tests(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> dict:
        test_suite = TestSuite(
            tests=[
                TestValueMAPE(lte=self.mape_threshold),
                TestValueMAE(lte=200),
                TestValueRMSE(lte=self.rmse_threshold),
            ]
        )
        test_suite.run(reference_data=reference_df, current_data=current_df)
        results = test_suite.as_dict()
        passed = all(t["status"] == "SUCCESS" for t in results["tests"])
        return {"passed": passed, "details": results["tests"]}

    def run_data_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        output_path: str = "reports/drift_report.html",
    ) -> None:
        report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
        report.run(reference_data=reference_df, current_data=current_df)
        report.save_html(output_path)
        print(f"Drift report saved to {output_path}")
