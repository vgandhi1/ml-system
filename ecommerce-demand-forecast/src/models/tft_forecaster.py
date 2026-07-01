import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


def build_tft_dataset(
    df: pd.DataFrame,
    max_encoder_length: int = 12,
    max_prediction_length: int = 4,
) -> TimeSeriesDataSet:
    """Build pytorch-forecasting TimeSeriesDataSet from the feature mart."""
    df = df.copy()
    df["series_id"] = "uk_ecommerce"
    df["time_idx"] = range(len(df))

    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="total_units",
        group_ids=["series_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx", "is_jan_effect", "is_valentines_season"],
        time_varying_unknown_reals=[
            "total_units",
            "total_revenue",
            "unique_customers",
            "avg_order_value",
            "rolling_4w_avg_units",
            "rolling_4w_std_units",
            "units_lag_1w",
            "units_lag_2w",
            "wow_growth_pct",
        ],
        target_normalizer=GroupNormalizer(groups=["series_id"], transformation="softplus"),
    )


def build_tft_model(dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
    """Instantiate TFT with quantile loss for probabilistic forecasting."""
    return TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=3e-3,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
