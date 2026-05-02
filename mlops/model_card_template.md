# Model card — Sentinel-Stream risk score (template)

> Complete this card before **staging** or **production** promotion. Remove instructional text in angle brackets.

## 1. Model identification

| Field | Value |
| --- | --- |
| Model name | \<e.g., sentinel_stream_xgb_v3\> |
| Version / run ID | \<MLflow run or git tag\> |
| Owner / team | \<name\> |
| Intended use | \<real-time authorization score / batch review queue / …\> |
| Deployment date | \<YYYY-MM-DD\> |

## 2. Business and risk context

- **Product / market:** \<cards, ACH, BNPL, …\>
- **Decision:** \<approve / route to step-up / hold — who consumes the score\>
- **Materiality:** \<expected volume, $ exposure, regulatory regime\>

## 3. Data

- **Training data source:** \<warehouse table, time window, refresh cadence\>
- **Label definition:** \<confirmed fraud vs disputes vs SAR-linked — be explicit\>
- **Population:** \<geographies, channels, exclusions\>
- **Known biases / gaps:** \<e.g., cold-start users under-represented\>
- **PII handling:** \<tokenization, retention, access controls — no raw PAN in features\>

## 4. Features

| Feature group | Description | staleness SLA |
| --- | --- | --- |
| \<group A\> | \<…\> | \<e.g., \< 1s from feature store\> |

## 5. Methodology

- **Algorithm:** \<XGBoost / ensemble / calibration method\>
- **Class balance:** \<scale_pos_weight, sampling, cost-sensitive loss\>
- **Validation:** \<temporal split, walk-forward, cross-validation\>
- **Primary metrics:** \<PR-AUC, recall @ fixed FPR, business $ catch\>

## 6. Performance (offline)

| Metric | Value | Dataset / window |
| --- | --- | --- |
| PR-AUC | \<\> | \<\> |
| ROC-AUC | \<\> | \<\> (interpret with care on rare fraud) |
| Operating point | \<threshold / FPR / approval impact\> | \<\> |

## 7. Performance (online) — post go-live

| Metric | Target | Dashboard / owner |
| --- | --- | --- |
| Fraud catch @ policy FPR | \<\> | \<\> |
| Latency p99 | \<e.g., \< 80ms\> | \<\> |
| Fallback rate | \<\> | \<\> |

## 8. Ethical & fairness considerations

- **Protected attributes:** \<not used directly / proxies audited\>
- **Fairness testing:** \<subgroups evaluated; known gaps\>

## 9. Monitoring & incident response

- **Data drift:** \<features monitored\>
- **Score drift:** \<population stability index, rank ordering\>
- **Incident:** \<rollback to prior model version; who is paged\>

## 10. Limitations

- \<e.g., not validated for crypto on-ramp flows\>
- \<e.g., scores are not legal adjudication of fraud\>

## 11. Approvals

| Role | Name | Date |
| --- | --- | --- |
| Model risk | | |
| Compliance | | |
| ML engineering | | |
