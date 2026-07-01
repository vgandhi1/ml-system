{{ config(materialized='view') }}

WITH source AS (
    SELECT * FROM {{ source('raw', 'online_retail') }}
),

renamed AS (
    SELECT
        CAST(invoice_no AS VARCHAR)             AS invoice_id,
        stock_code,
        UPPER(TRIM(description))               AS product_name,
        CAST(quantity AS INTEGER)              AS quantity,
        CAST(invoice_date AS TIMESTAMP)        AS invoiced_at,
        CAST(unit_price AS DOUBLE)             AS unit_price_usd,
        CAST(customer_id AS VARCHAR)           AS customer_id,
        UPPER(TRIM(country))                   AS country
    FROM source
),

cleaned AS (
    SELECT *
    FROM renamed
    WHERE
        customer_id IS NOT NULL
        AND quantity > 0
        AND unit_price_usd > 0
        AND invoice_id NOT LIKE 'C%'
        AND country = 'UNITED KINGDOM'
)

SELECT * FROM cleaned
