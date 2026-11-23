import sqlite3
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

DB_PATH = "../03_SQL/shopsmart_db.sqlite"
CLEAN_DIR = "../02_Cleaned_Data"

# ==================================
# 1. LOAD ENRICHED SALES DATA FROM SQL
# ==================================
query = """
SELECT
    o.order_id,
    o.customer_id,
    c.customer_unique_id,
    c.customer_city,
    c.customer_state,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    o.is_date_anomaly,
    oi.order_item_id,
    oi.product_id,
    p.product_category_name,
    p.product_weight_g,
    oi.price,
    oi.freight_value
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE o.order_status NOT IN ('canceled', 'unavailable');
"""

with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql_query(
        query,
        conn,
        parse_dates=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )

# ==================================
# 2. FEATURE ENGINEERING (LINE-ITEM & DELIVERY METRICS)
# ==================================
df["total_item_value"] = df["price"] + df["freight_value"]
df["freight_ratio_pct"] = (df["freight_value"] / df["price"]) * 100
df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
df["order_day_of_week"] = df["order_purchase_timestamp"].dt.day_name()
df["order_hour"] = df["order_purchase_timestamp"].dt.hour

# Calculate delivery duration & late flag only on valid delivered orders
valid_deliv = (df["order_status"] == "delivered") & (df["is_date_anomaly"] == 0)
df["actual_delivery_days"] = np.where(
    valid_deliv,
    (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400,
    np.nan,
)
df["is_late_delivery"] = np.where(
    valid_deliv,
    df["order_delivered_customer_date"] > df["order_estimated_delivery_date"],
    np.nan,
)

# ==================================
# 3. RFM CUSTOMER SEGMENTATION
# ==================================
# Reference date = 1 day after the latest purchase in the dataset
snapshot_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

rfm = (
    df.groupby("customer_unique_id")
    .agg(
        customer_state=("customer_state", "first"),
        last_purchase=("order_purchase_timestamp", "max"),
        frequency=("order_id", "nunique"),
        monetary=("price", "sum"),
        avg_freight=("freight_value", "mean"),
    )
    .reset_index()
)

rfm["recency_days"] = (snapshot_date - rfm["last_purchase"]).dt.days

# R & M use rank-based quintiles (1 to 5); R is reversed so lower recency = higher score
rfm["r_score"] = pd.qcut(rfm["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

# F uses explicit business thresholds because 96.96% of customers have frequency = 1
rfm["f_score"] = np.select(
    [
        rfm["frequency"] >= 4,
        rfm["frequency"] == 3,
        rfm["frequency"] == 2,
    ],
    [5, 4, 3],
    default=1,
)

rfm["rfm_score"] = rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)

def assign_segment(row):
    if row["f_score"] >= 3 and row["r_score"] >= 4 and row["m_score"] >= 4:
        return "Champions"
    elif row["f_score"] >= 3 and row["r_score"] >= 3:
        return "Loyal Repeaters"
    elif row["f_score"] >= 3 and row["r_score"] <= 2:
        return "At-Risk Repeaters"
    elif row["r_score"] >= 4 and row["m_score"] >= 4:
        return "High-Value New Buyers"
    elif row["r_score"] >= 4:
        return "Recent One-Time Buyers"
    elif row["r_score"] == 3:
        return "Needs Attention"
    else:
        return "Hibernating / Churned"

rfm["customer_segment"] = rfm.apply(assign_segment, axis=1)

# ==================================
# 4. PRINT EDA & SEGMENTATION INSIGHTS
# ==================================
print("=" * 80)
print("1. RFM CUSTOMER SEGMENTATION SUMMARY")
print("=" * 80)
seg_summary = (
    rfm.groupby("customer_segment")
    .agg(
        customers=("customer_unique_id", "count"),
        avg_recency_days=("recency_days", "mean"),
        avg_orders=("frequency", "mean"),
        avg_ltv=("monetary", "mean"),
        total_revenue=("monetary", "sum"),
    )
    .sort_values("total_revenue", ascending=False)
    .reset_index()
)
seg_summary["customer_share_pct"] = (seg_summary["customers"] / len(rfm) * 100).round(2)
seg_summary["revenue_share_pct"] = (seg_summary["total_revenue"] / seg_summary["total_revenue"].sum() * 100).round(2)
print(seg_summary.round(2).to_string(index=False))

print("\n" + "=" * 80)
print("2. PEAK SHOPPING HOURS & DAYS (ORDER VOLUME)")
print("=" * 80)
top_days = df.groupby("order_day_of_week")["order_id"].nunique().sort_values(ascending=False)
print("Orders by Day of Week:\n", top_days.to_string())

print("\n" + "=" * 80)
print("3. LOGISTICS FRICTION: ON-TIME VS. LATE DELIVERIES")
print("=" * 80)
logistics_check = (
    df.dropna(subset=["is_late_delivery"])
    .groupby("is_late_delivery")
    .agg(
        line_items=("order_id", "count"),
        avg_delivery_days=("actual_delivery_days", "mean"),
        avg_product_weight_g=("product_weight_g", "mean"),
        avg_freight_value=("freight_value", "mean"),
        avg_freight_ratio_pct=("freight_ratio_pct", "mean"),
    )
    .reset_index()
)
print(logistics_check.round(2).to_string(index=False))

# ==================================
# 5. EXPORT POWER BI & ML READY TABLES
# ==================================
rfm.drop(columns=["last_purchase"]).to_csv(f"{CLEAN_DIR}/rfm_customer_segments.csv", index=False)
df.to_csv(f"{CLEAN_DIR}/master_sales_enriched.csv", index=False)
print(f"\nExported 'rfm_customer_segments.csv' ({len(rfm):,} rows) and 'master_sales_enriched.csv' ({len(df):,} rows) to {CLEAN_DIR}/")