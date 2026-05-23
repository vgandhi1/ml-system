{{ config(materialized='table') }}

WITH transactions AS (
    SELECT * FROM {{ ref('stg_online_retail') }}
),

enriched AS (
    SELECT
        *,
        quantity * unit_price_usd                          AS line_revenue,
        DATE_TRUNC('week', invoiced_at)                   AS week_start,
        EXTRACT(WEEK FROM invoiced_at)::INTEGER           AS week_num,
        EXTRACT(YEAR FROM invoiced_at)::INTEGER           AS year_num,
        LAG(quantity * unit_price_usd, 1) OVER (
            PARTITION BY customer_id ORDER BY invoiced_at
        ) AS prev_order_revenue
    FROM transactions
),

weekly AS (
    SELECT
        week_start,
        week_num,
        year_num,
        SUM(quantity)                                      AS total_units,
        ROUND(SUM(line_revenue), 2)                       AS total_revenue,
        COUNT(DISTINCT customer_id)                        AS unique_customers,
        COUNT(DISTINCT invoice_id)                         AS total_orders,
        ROUND(
            SUM(line_revenue) / NULLIF(COUNT(DISTINCT invoice_id), 0), 2
        ) AS avg_order_value,
        COUNT(DISTINCT stock_code)                         AS unique_skus,
        ROUND(AVG(line_revenue), 2)                       AS avg_line_revenue
    FROM enriched
    GROUP BY week_start, week_num, year_num
)

SELECT * FROM weekly ORDER BY week_start
