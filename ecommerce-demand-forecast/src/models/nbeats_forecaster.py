from pytorch_forecasting import NBeats, TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE


def build_nbeats_model(dataset: TimeSeriesDataSet) -> NBeats:
    """N-BEATS with trend + seasonality stacks."""
    return NBeats.from_dataset(
        dataset,
        learning_rate=4e-3,
        stack_types=["trend", "seasonality"],
        num_blocks=[3, 3],
        num_block_layers=[4, 4],
        expansion_coefficient_lengths=[3, 7],
        widths=[512, 512],
        sharing=[True, True],
        loss=SMAPE(),
        log_interval=5,
    )
