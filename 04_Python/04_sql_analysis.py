import os
import sqlite3
import pandas as pd

# Format Pandas terminal output so columns don't truncate
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

DB_PATH = "../03_SQL/shopsmart_db.sqlite"
SQL_EXPORT_PATH = "../03_SQL/02_business_analysis.sql"

if not os.path.exists(DB_PATH):
    raise FileNotFoundError(
        "Database not found! Run 'py 03_load_to_sql.py' first to create shopsmart_db.sqlite."
    )

# ==================================
# DEFINE 6 BUSINESS ANALYSIS QUERIES
# ==================================
queries = {
    "1. EXECUTIVE KPI SUMMARY (COMPLETED & ACTIVE ORDERS)": """
        SELECT
            COUNT(DISTINCT o.order_id) AS total_orders,
            COUNT(DISTINCT c.customer_unique_id) AS unique_customers,
            ROUND(SUM(oi.price), 2) AS product_revenue,
            ROUND(SUM(oi.freight_value), 2) AS total_freight,
            ROUND(SUM(oi.price + oi.freight_value), 2) AS gross_merchandise_value,
            ROUND(SUM(oi.price) / COUNT(DISTINCT o.order_id), 2) AS avg_order_value,
            ROUND(SUM(oi.freight_value) * 100.0 / SUM(oi.price), 2) AS freight_to_sales_pct
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.order_status NOT IN ('canceled', 'unavailable');
    """,

    "2. MONTHLY REVENUE & MONTH-OVER-MONTH (MoM) GROWTH": """
        WITH monthly_sales AS (
            SELECT
                strftime('%Y-%m', o.order_purchase_timestamp) AS order_month,
                COUNT(DISTINCT o.order_id) AS total_orders,
                ROUND(SUM(oi.price), 2) AS monthly_revenue
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            WHERE o.order_status NOT IN ('canceled', 'unavailable')
            GROUP BY 1
        )
        SELECT
            order_month,
            total_orders,
            monthly_revenue,
            LAG(monthly_revenue) OVER (ORDER BY order_month) AS prev_month_revenue,
            ROUND(
                (monthly_revenue - LAG(monthly_revenue) OVER (ORDER BY order_month)) * 100.0
                / LAG(monthly_revenue) OVER (ORDER BY order_month),
                2
            ) AS mom_growth_pct
        FROM monthly_sales
        ORDER BY order_month;
    """,

    "3. TOP 10 PRODUCT CATEGORIES BY REVENUE & SHARE %": """
        WITH category_revenue AS (
            SELECT
                p.product_category_name AS category,
                COUNT(DISTINCT oi.order_id) AS total_orders,
                COUNT(*) AS units_sold,
                ROUND(SUM(oi.price), 2) AS total_revenue,
                ROUND(AVG(oi.price), 2) AS avg_item_price
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.order_id
            JOIN products p ON oi.product_id = p.product_id
            WHERE o.order_status NOT IN ('canceled', 'unavailable')
            GROUP BY p.product_category_name
        )
        SELECT
            category,
            total_orders,
            units_sold,
            total_revenue,
            avg_item_price,
            ROUND(total_revenue * 100.0 / SUM(total_revenue) OVER (), 2) AS revenue_share_pct
        FROM category_revenue
        ORDER BY total_revenue DESC
        LIMIT 10;
    """,

    "4. TOP 10 STATES BY REVENUE & LOGISTICS COST": """
        SELECT
            c.customer_state AS state,
            COUNT(DISTINCT o.order_id) AS total_orders,
            COUNT(DISTINCT c.customer_unique_id) AS unique_customers,
            ROUND(SUM(oi.price), 2) AS total_revenue,
            ROUND(AVG(oi.freight_value), 2) AS avg_freight_cost,
            ROUND(
                AVG(
                    CASE
                        WHEN o.order_status = 'delivered' AND o.is_date_anomaly = 0
                        THEN julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp)
                    END
                ),
                1
            ) AS avg_delivery_days
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.order_status NOT IN ('canceled', 'unavailable')
        GROUP BY c.customer_state
        ORDER BY total_revenue DESC
        LIMIT 10;
    """,

    "5. DELIVERY SLA & ON-TIME PERFORMANCE (CLEANED DELIVERED ORDERS)": """
        SELECT
            COUNT(*) AS valid_delivered_orders,
            ROUND(AVG(julianday(order_delivered_customer_date) - julianday(order_purchase_timestamp)), 2) AS avg_actual_delivery_days,
            ROUND(AVG(julianday(order_estimated_delivery_date) - julianday(order_purchase_timestamp)), 2) AS avg_estimated_days,
            SUM(CASE WHEN julianday(order_delivered_customer_date) > julianday(order_estimated_delivery_date) THEN 1 ELSE 0 END) AS late_orders,
            ROUND(
                SUM(CASE WHEN julianday(order_delivered_customer_date) > julianday(order_estimated_delivery_date) THEN 1 ELSE 0 END) * 100.0
                / COUNT(*),
                2
            ) AS late_delivery_pct
        FROM orders
        WHERE order_status = 'delivered'
          AND is_date_anomaly = 0;
    """,

    "6. PAYMENT METHOD BREAKDOWN & INSTALLMENT BEHAVIOR": """
        SELECT
            payment_type,
            COUNT(*) AS transaction_count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS transaction_share_pct,
            ROUND(SUM(payment_value), 2) AS total_payment_value,
            ROUND(AVG(payment_value), 2) AS avg_payment_value,
            ROUND(AVG(payment_installments), 1) AS avg_installments
        FROM payments
        WHERE payment_type != 'not_defined'
        GROUP BY payment_type
        ORDER BY total_payment_value DESC;
    """,

    "7. REPEAT CUSTOMER RATE & ONE-TIME VS. REPEAT BUYER REVENUE": """
        WITH customer_orders AS (
            SELECT
                c.customer_unique_id,
                COUNT(DISTINCT o.order_id) AS order_count,
                SUM(oi.price) AS customer_revenue
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            JOIN order_items oi ON o.order_id = oi.order_id
            WHERE o.order_status NOT IN ('canceled', 'unavailable')
            GROUP BY c.customer_unique_id
        ),
        segmented_customers AS (
            SELECT
                CASE
                    WHEN order_count = 1 THEN 'One-Time Buyer'
                    ELSE 'Repeat Buyer (2+ Orders)'
                END AS buyer_type,
                COUNT(*) AS customer_count,
                SUM(order_count) AS total_orders,
                ROUND(SUM(customer_revenue), 2) AS total_revenue,
                ROUND(AVG(customer_revenue), 2) AS avg_ltv
            FROM customer_orders
            GROUP BY 1
        )
        SELECT
            buyer_type,
            customer_count,
            ROUND(customer_count * 100.0 / SUM(customer_count) OVER (), 2) AS customer_share_pct,
            total_orders,
            total_revenue,
            ROUND(total_revenue * 100.0 / SUM(total_revenue) OVER (), 2) AS revenue_share_pct,
            avg_ltv
        FROM segmented_customers
        ORDER BY customer_count DESC;
    """,
}

# ==================================
# EXPORT .SQL FILE FOR GITHUB PORTFOLIO
# ==================================
with open(SQL_EXPORT_PATH, "w", encoding="utf-8") as f:
    f.write("-- ShopSmart E-Commerce Business Intelligence Queries\n")
    f.write("-- Compatible with SQLite 3.25+ (Window Functions & CTEs)\n\n")
    for title, sql in queries.items():
        f.write(f"-- {title}\n")
        f.write(sql.strip() + "\n\n")

print(f"Exported SQL portfolio script to: {SQL_EXPORT_PATH}\n")

# ==================================
# EXECUTE QUERIES & PRINT RESULTS
# ==================================
with sqlite3.connect(DB_PATH) as conn:
    for title, sql in queries.items():
        print("=" * 80)
        print(title)
        print("=" * 80)
        df_result = pd.read_sql_query(sql, conn)
        print(df_result.to_string(index=False))
        print("\n")