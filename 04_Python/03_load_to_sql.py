import os
import time
import pandas as pd
from sqlalchemy import create_engine, event, text

# ==================================
# 1. SQLITE DATABASE SETUP
# ==================================
SQL_DIR = "../03_SQL"
CLEAN_DIR = "../02_Cleaned_Data"
os.makedirs(SQL_DIR, exist_ok=True)

DB_URL = f"sqlite:///{SQL_DIR}/shopsmart_db.sqlite"
engine = create_engine(DB_URL)

# Enforce Foreign Key constraints on every SQLite connection
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

# ==================================
# 2. CREATE RELATIONAL SCHEMA
# ==================================
schema_statements = [
    # Drop child tables first so this script can be safely re-run anytime
    "DROP TABLE IF EXISTS order_items;",
    "DROP TABLE IF EXISTS payments;",
    "DROP TABLE IF EXISTS orders;",
    "DROP TABLE IF EXISTS products;",
    "DROP TABLE IF EXISTS customers;",
    """
    CREATE TABLE customers (
        customer_id VARCHAR(50) PRIMARY KEY,
        customer_unique_id VARCHAR(50) NOT NULL,
        customer_zip_code_prefix INT,
        customer_city VARCHAR(100),
        customer_state CHAR(2)
    );
    """,
    """
    CREATE TABLE products (
        product_id VARCHAR(50) PRIMARY KEY,
        product_category_name VARCHAR(100),
        product_name_length FLOAT NULL,
        product_description_length FLOAT NULL,
        product_photos_qty FLOAT NULL,
        product_weight_g FLOAT,
        product_length_cm FLOAT,
        product_height_cm FLOAT,
        product_width_cm FLOAT
    );
    """,
    """
    CREATE TABLE orders (
        order_id VARCHAR(50) PRIMARY KEY,
        customer_id VARCHAR(50) NOT NULL,
        order_status VARCHAR(20),
        order_purchase_timestamp DATETIME,
        order_approved_at DATETIME NULL,
        order_delivered_carrier_date DATETIME NULL,
        order_delivered_customer_date DATETIME NULL,
        order_estimated_delivery_date DATETIME,
        is_date_anomaly BOOLEAN,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    );
    """,
    """
    CREATE TABLE payments (
        order_id VARCHAR(50),
        payment_sequential INT,
        payment_type VARCHAR(30),
        payment_installments INT,
        payment_value DECIMAL(10, 2),
        PRIMARY KEY (order_id, payment_sequential),
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
    );
    """,
    """
    CREATE TABLE order_items (
        order_id VARCHAR(50),
        order_item_id INT,
        product_id VARCHAR(50),
        seller_id VARCHAR(50),
        shipping_limit_date DATETIME,
        price DECIMAL(10, 2),
        freight_value DECIMAL(10, 2),
        PRIMARY KEY (order_id, order_item_id),
        FOREIGN KEY (order_id) REFERENCES orders(order_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    );
    """,
]

with engine.begin() as conn:
    for stmt in schema_statements:
        conn.execute(text(stmt))
print("Relational schema created in 03_SQL/shopsmart_db.sqlite")

# ==================================
# 3. LOAD CLEANED CSVs IN PARENT-TO-CHILD ORDER
# ==================================
load_plan = [
    ("customers", "customers_clean.csv", []),
    ("products", "products_clean.csv", []),
    (
        "orders",
        "orders_clean.csv",
        [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    ),
    ("payments", "payments_clean.csv", []),
    ("order_items", "order_items_clean.csv", ["shipping_limit_date"]),
]

start_time = time.time()

for table_name, file_name, date_cols in load_plan:
    file_path = f"{CLEAN_DIR}/{file_name}"
    df = pd.read_csv(file_path, parse_dates=date_cols)

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="append",
        index=False,
        chunksize=1000,
    )
    print(f"Loaded {len(df):,} rows into '{table_name}'")

elapsed = round(time.time() - start_time, 2)
print(f"\nAll 5 tables loaded in {elapsed} seconds.")

# ==================================
# 4. VERIFY SQL ROW COUNTS
# ==================================
print("\n--- SQL DATABASE VERIFICATION ---")
with engine.connect() as conn:
    for table_name, _, _ in load_plan:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
        print(f"{table_name:<15} -> {count:,} rows")