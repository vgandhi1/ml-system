# ml-system — ML Systems Pillar

Three production ML systems across batch, real-time, and privacy-preserving regimes:

| Sub-project | Regime | Stack |
|-------------|--------|-------|
| [`ecommerce-demand-forecast`](ecommerce-demand-forecast/) | Batch demand forecasting | scikit-learn, time-series |
| [`sentinel-fraud`](sentinel-fraud/) | Real-time fraud detection | Kafka/Faust streaming |
| [`Prism-Federated`](Prism-Federated/) | Federated privacy-preserving learning | PySyft |

**Note:** `sentinel-fraud/` was consolidated from the standalone repo `Sentinel-Stream`
(name kept as `sentinel-fraud` here). It is **not** the archived `SentinelFlow` (IIoT) — different project, FinTech ML.

_Consolidated 2026-06-30 from standalone repos `ecommerce-demand-forecast`, `Sentinel-Stream`, `Prism-Federated` (history preserved via git subtree; originals archived)._
