import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.dataset_builder import SalesForecastDataModule, load_sample_feature_mart


def test_dataset_length_and_shapes():
    df = load_sample_feature_mart()
    dm = SalesForecastDataModule(df, seq_len=8, pred_len=4, batch_size=4, num_workers=0)
    dm.setup()

    assert len(dm.train_ds) >= 1
    x, y = dm.train_ds[0]
    assert x.shape == (8, 13)
    assert y.shape == (4,)
