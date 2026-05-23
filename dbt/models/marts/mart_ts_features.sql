{{ config(materialized='table') }}

WITH weekly AS (
    SELECT * FROM {{ ref('int_weekly_sales') }}
),

feature_engineered AS (
    SELECT
        week_start,
        week_num,
        year_num,
        total_units,
        total_revenue,
        unique_customers,
        avg_order_value,

        AVG(total_units) OVER (
            ORDER BY week_start ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS rolling_4w_avg_units,
        STDDEV_SAMP(total_units) OVER (
            ORDER BY week_start ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS rolling_4w_std_units,
        MAX(total_units) OVER (
            ORDER BY week_start ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS rolling_4w_max_units,

        LAG(total_units, 1) OVER (ORDER BY week_start)  AS units_lag_1w,
        LAG(total_units, 2) OVER (ORDER BY week_start)  AS units_lag_2w,
        LAG(total_units, 4) OVER (ORDER BY week_start)  AS units_lag_4w,
        LAG(total_revenue, 1) OVER (ORDER BY week_start) AS revenue_lag_1w,

        ROUND(
            (total_units - LAG(total_units, 1) OVER (ORDER BY week_start))
            / NULLIF(LAG(total_units, 1) OVER (ORDER BY week_start), 0) * 100,
            2
        ) AS wow_growth_pct,

        ROUND(
            (total_revenue - MIN(total_revenue) OVER ())
            / NULLIF(MAX(total_revenue) OVER () - MIN(total_revenue) OVER (), 0),
            6
        ) AS norm_revenue,

        CASE WHEN week_num IN (1, 2) THEN 1 ELSE 0 END AS is_jan_effect,
        CASE WHEN week_num BETWEEN 6 AND 8 THEN 1 ELSE 0 END AS is_valentines_season

    FROM weekly
)

SELECT * FROM feature_engineered
