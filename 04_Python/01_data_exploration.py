import pandas as pd

# Load datasets

customers = pd.read_csv("../01_Dataset/olist_customers_dataset.csv")

orders = pd.read_csv("../01_Dataset/olist_orders_dataset.csv")

products = pd.read_csv("../01_Dataset/olist_products_dataset.csv")

payments = pd.read_csv("../01_Dataset/olist_order_payments_dataset.csv")

order_items = pd.read_csv("../01_Dataset/olist_order_items_dataset.csv")


# Display dataset information

print("CUSTOMERS DATA")
print(customers.head())

print("\nORDERS DATA")
print(orders.head())

print("\nPRODUCTS DATA")
print(products.head())

print("\nPAYMENTS DATA")
print(payments.head())

print("\nORDER ITEMS DATA")
print(order_items.head())



# ==================================
# DATA QUALITY ANALYSIS
# ==================================

datasets = {
    "Customers": customers,
    "Orders": orders,
    "Products": products,
    "Payments": payments,
    "Order Items": order_items
}

for name, df in datasets.items():

    print("\n================================")
    print("DATASET:", name)
    print("================================")

    # Total rows and columns
    print("Shape:", df.shape)

    # Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Duplicate records
    print("\nDuplicate Records:")
    print(df.duplicated().sum())

    # Data types
    print("\nColumn Data Types:")
    print(df.dtypes)

    # Dataset information
    print("\nDataset Information:")
    df.info()


    # ==================================
# INVESTIGATING MISSING VALUES
# ==================================

print("\nMISSING VALUES IN ORDERS")

print(
    orders[
        orders["order_delivered_customer_date"].isnull()
    ][["order_status", "order_purchase_timestamp"]]
    .head(10)
)


print("\nMISSING VALUES IN PRODUCTS")

print(
    products[
        products["product_category_name"].isnull()
    ].head(10)
)


# ==================================
# DAY 3: INVESTIGATING MISSING VALUES
# ==================================

print("\n--- 1. MISSING DELIVERY DATES BY ORDER STATUS ---")
missing_delivery = orders[orders["order_delivered_customer_date"].isnull()]
print(missing_delivery["order_status"].value_counts())

print("\n--- 2. MISSING APPROVAL DATES BY ORDER STATUS ---")
missing_approved = orders[orders["order_approved_at"].isnull()]
print(missing_approved["order_status"].value_counts())

print("\n--- 3. MISSING PRODUCT CATEGORIES: SAMPLE ROWS ---")
missing_cat = products[products["product_category_name"].isnull()]
print(missing_cat.head(5))

print("\n--- 4. DO MISSING-CATEGORY PRODUCTS HAVE SALES? ---")
missing_cat_sales = order_items[
    order_items["product_id"].isin(missing_cat["product_id"])
]
print("Total line items sold:", len(missing_cat_sales))
print("Total revenue at stake:", round(missing_cat_sales["price"].sum(), 2))

print("\n--- 5. PRODUCTS WITH MISSING WEIGHT/DIMENSIONS ---")
print(products[products["product_weight_g"].isnull()])