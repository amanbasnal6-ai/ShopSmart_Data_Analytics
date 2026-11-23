import os
import pandas as pd

# ==================================
# 1. LOAD RAW DATASETS
# ==================================
customers = pd.read_csv("../01_Dataset/olist_customers_dataset.csv")
orders = pd.read_csv("../01_Dataset/olist_orders_dataset.csv")
products = pd.read_csv("../01_Dataset/olist_products_dataset.csv")
payments = pd.read_csv("../01_Dataset/olist_order_payments_dataset.csv")
order_items = pd.read_csv("../01_Dataset/olist_order_items_dataset.csv")

CLEAN_DIR = "../02_Cleaned_Data"
os.makedirs(CLEAN_DIR, exist_ok=True)

# ==================================
# 2. CLEAN ORDERS
# ==================================
order_date_cols = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

for col in order_date_cols:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")

# Flag delivered orders missing critical approval or delivery timestamps
orders["is_date_anomaly"] = (
    (orders["order_status"] == "delivered")
    & (
        orders["order_delivered_customer_date"].isnull()
        | orders["order_approved_at"].isnull()
    )
)

# ==================================
# 3. CLEAN ORDER ITEMS
# ==================================
order_items["shipping_limit_date"] = pd.to_datetime(
    order_items["shipping_limit_date"], errors="coerce"
)

# ==================================
# 4. CLEAN PRODUCTS
# ==================================
# Fix source column typos
products = products.rename(columns={
    "product_name_lenght": "product_name_length",
    "product_description_lenght": "product_description_length",
})

# Preserve 179,535.28 in revenue by labeling missing categories
products["product_category_name"] = products["product_category_name"].fillna("unknown")

# Impute the 2 missing physical measurement rows using column medians
dim_cols = [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]
for col in dim_cols:
    products[col] = products[col].fillna(products[col].median())

# ==================================
# 5. STANDARDIZE CUSTOMERS TEXT
# ==================================
customers["customer_city"] = customers["customer_city"].str.strip().str.lower()
customers["customer_state"] = customers["customer_state"].str.strip().str.upper()

# ==================================
# 6. VERIFY & EXPORT CLEANED CSVs
# ==================================
cleaned_tables = {
    "customers_clean.csv": customers,
    "orders_clean.csv": orders,
    "products_clean.csv": products,
    "payments_clean.csv": payments,
    "order_items_clean.csv": order_items,
}

for filename, df in cleaned_tables.items():
    output_path = os.path.join(CLEAN_DIR, filename)
    df.to_csv(output_path, index=False)
    print(f"Saved {filename} -> Shape: {df.shape}")

print("\n--- POST-CLEANING AUDIT ---")
print("Orders date dtypes:\n", orders[order_date_cols].dtypes)
print("\nFlagged delivered date anomalies:", orders["is_date_anomaly"].sum())
print("Remaining missing product categories:", products["product_category_name"].isnull().sum())
print("Remaining missing product weights:", products["product_weight_g"].isnull().sum())