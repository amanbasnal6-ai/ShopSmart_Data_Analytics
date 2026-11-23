import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# ==================================
# 1. SETUP PATHS & LOAD DATA
# ==================================
CLEAN_DIR = "../02_Cleaned_Data"
IMG_DIR = "../images"
os.makedirs(IMG_DIR, exist_ok=True)

sales_df = pd.read_csv(f"{CLEAN_DIR}/master_sales_enriched.csv")
rfm_df = pd.read_csv(f"{CLEAN_DIR}/rfm_customer_segments.csv")
sales_df["is_late_delivery"] = pd.to_numeric(sales_df["is_late_delivery"], errors="coerce")

# Filter to the 20 complete operational months (exclude partial boundary months)
monthly = (
    sales_df[
        (sales_df["order_month"] >= "2017-01")
        & (sales_df["order_month"] <= "2018-08")
    ]
    .groupby("order_month", as_index=False)
    .agg(
        revenue=("price", "sum"),
        orders=("order_id", "nunique"),
    )
    .sort_values("order_month")
    .reset_index(drop=True)
)

# ==================================
# 2. TIME-SERIES FEATURE ENGINEERING
# ==================================
def build_features(df_in):
    X = pd.DataFrame(index=df_in.index)
    X["t"] = df_in["t"]
    month_num = pd.to_datetime(df_in["order_month"] + "-01").dt.month
    X["sin_year"] = np.sin(2 * np.pi * month_num / 12)
    X["cos_year"] = np.cos(2 * np.pi * month_num / 12)
    X["is_nov_peak"] = (month_num == 11).astype(int)
    return X

monthly["t"] = np.arange(len(monthly))
X_all = build_features(monthly)
y_all = monthly["revenue"].values

# ==================================
# 3. HOLDOUT BACKTEST (LAST 3 MONTHS: JUN–AUG 2018)
# ==================================
train_idx = monthly["order_month"] < "2018-06"
test_idx = monthly["order_month"] >= "2018-06"

bt_model = Ridge(alpha=0.5)
bt_model.fit(X_all[train_idx], y_all[train_idx])
bt_preds = bt_model.predict(X_all[test_idx])

bt_mae = mean_absolute_error(y_all[test_idx], bt_preds)
bt_mape = mean_absolute_percentage_error(y_all[test_idx], bt_preds) * 100

print("=" * 80)
print("1. TIME-SERIES BACKTEST VALIDATION (JUN 2018 – AUG 2018)")
print("=" * 80)
bt_table = pd.DataFrame({
    "Month": monthly.loc[test_idx, "order_month"].values,
    "Actual_Revenue": np.round(y_all[test_idx], 2),
    "Predicted_Revenue": np.round(bt_preds, 2),
    "Error_Pct": np.round((bt_preds - y_all[test_idx]) / y_all[test_idx] * 100, 2),
})
print(bt_table.to_string(index=False))
print(f"\nBacktest MAE:  {bt_mae:,.2f}")
print(f"Backtest MAPE: {bt_mape:.2f}%")

# ==================================
# 4. RETRAIN ON FULL SERIES & FORECAST NEXT 6 MONTHS
# ==================================
final_model = Ridge(alpha=0.5)
final_model.fit(X_all, y_all)
fitted_vals = final_model.predict(X_all)

# Estimate 95% prediction interval from residual standard error
dof = len(y_all) - X_all.shape[1] - 1
resid_std = np.sqrt(np.sum((y_all - fitted_vals) ** 2) / dof)
margin_95 = 1.96 * resid_std

future_months = pd.date_range("2018-09-01", periods=6, freq="MS").strftime("%Y-%m")
future_df = pd.DataFrame({
    "order_month": future_months,
    "t": np.arange(len(monthly), len(monthly) + 6),
})
X_future = build_features(future_df)
future_preds = final_model.predict(X_future)

future_df["forecast_revenue"] = np.round(future_preds, 2)
future_df["lower_95"] = np.round(np.maximum(0, future_preds - margin_95), 2)
future_df["upper_95"] = np.round(future_preds + margin_95, 2)

# Save forecast CSV
forecast_csv_path = f"{CLEAN_DIR}/sales_forecast_6m.csv"
future_df[["order_month", "forecast_revenue", "lower_95", "upper_95"]].to_csv(
    forecast_csv_path, index=False
)

print("\n" + "=" * 80)
print("2. 6-MONTH REVENUE FORECAST (SEP 2018 – FEB 2019)")
print("=" * 80)
print(future_df[["order_month", "forecast_revenue", "lower_95", "upper_95"]].to_string(index=False))
print(f"\nProjected 6-Month Cumulative Revenue: {future_df['forecast_revenue'].sum():,.2f}")

# ==================================
# 5. EXPORT FORECAST VISUALIZATION FOR GITHUB & DASHBOARD
# ==================================
fig, ax = plt.subplots(figsize=(13, 5.8), dpi=300)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, axis="y", linestyle="--", color="#E2E8F0", alpha=0.9)
ax.set_axisbelow(True)

hist_x = np.arange(len(monthly))
fut_x = np.arange(len(monthly), len(monthly) + 6)
all_labels = list(monthly["order_month"]) + list(future_df["order_month"])

# Historical actuals & model fit
ax.plot(hist_x, y_all / 1e6, color="#2563EB", linewidth=2.4, marker="o", label="Actual Monthly Revenue")
ax.plot(hist_x, fitted_vals / 1e6, color="#64748B", linestyle=":", linewidth=1.6, label="Trend + Seasonality Fit")

# Bridge last actual point to first forecast point
bridge_x = [hist_x[-1], fut_x[0]]
bridge_y = [y_all[-1] / 1e6, future_preds[0] / 1e6]
ax.plot(bridge_x, bridge_y, color="#0D9488", linestyle="--", linewidth=2)

# 6-month forecast + 95% confidence band
ax.plot(fut_x, future_preds / 1e6, color="#0D9488", linestyle="--", linewidth=2.5, marker="s", label="6-Month Forecast")
ax.fill_between(
    fut_x,
    future_df["lower_95"] / 1e6,
    future_df["upper_95"] / 1e6,
    color="#0D9488",
    alpha=0.15,
    label="95% Confidence Interval",
)

# Annotate projected Nov 2018 Black Friday peak
nov18_idx = fut_x[2]
nov18_val = future_preds[2] / 1e6
ax.annotate(
    f"Projected Nov 2018 Peak\n{nov18_val:.2f}M",
    xy=(nov18_idx, nov18_val),
    xytext=(nov18_idx - 2.8, nov18_val + 0.14),
    fontsize=9.5,
    fontweight="bold",
    color="#1E293B",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F0FDFA", edgecolor="#0D9488", lw=1),
    arrowprops=dict(arrowstyle="->", color="#0D9488", lw=1.3),
)

ax.set_title(
    f"ShopSmart 6-Month Revenue Forecast (Backtest MAPE: {bt_mape:.1f}%)",
    fontsize=13,
    fontweight="bold",
    color="#1E293B",
    loc="left",
    pad=14,
)
ax.set_ylabel("Product Revenue (Millions)", color="#334155")
ax.set_xticks(np.arange(len(all_labels)))
ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8.5)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2fM"))
ax.set_ylim(0, (future_df["upper_95"].max() / 1e6) * 1.15)
ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="upper left")

plt.tight_layout()
chart_path = f"{IMG_DIR}/05_sales_forecast_6m.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved forecast chart to: {chart_path}")

# ==================================
# 6. AUTOMATED AI BUSINESS RECOMMENDATION ENGINE
# ==================================
total_rev = sales_df["price"].sum()
repeat_df = rfm_df[rfm_df["frequency"] >= 2]
onetime_df = rfm_df[rfm_df["frequency"] == 1]
repeat_rate = len(repeat_df) / len(rfm_df) * 100
ltv_lift = repeat_df["monetary"].mean() - onetime_df["monetary"].mean()

# Simulate converting 5% of one-time buyers into repeat buyers
retention_upside = (len(onetime_df) * 0.05) * ltv_lift

# Identify top-revenue category & worst-delay high-volume state
top_cat = sales_df.groupby("product_category_name")["price"].sum().idxmax()
top_cat_rev = sales_df.groupby("product_category_name")["price"].sum().max()

state_stats = sales_df.groupby("customer_state").agg(
    orders=("order_id", "nunique"),
    avg_freight=("freight_value", "mean"),
    avg_days=("actual_delivery_days", "mean"),
)
sp_freight = state_stats.loc["SP", "avg_freight"]
sp_days = state_stats.loc["SP", "avg_days"]
ba_freight = state_stats.loc["BA", "avg_freight"]
ba_days = state_stats.loc["BA", "avg_days"]

valid_deliv = sales_df.dropna(subset=["is_late_delivery"])
late_rate = (
    valid_deliv[valid_deliv["is_late_delivery"] == 1]["order_id"].nunique()
    / valid_deliv["order_id"].nunique()
    * 100
)

recommendations = [
    {
        "priority": "P1 — High Revenue Impact",
        "area": "Customer Retention & Lifecycle Marketing",
        "finding": (
            f"Only {repeat_rate:.2f}% of customers ({len(repeat_df):,}) place a 2nd order, "
            f"yet repeaters deliver +{ltv_lift:,.2f} higher LTV ({repeat_df['monetary'].mean():,.2f} vs. {onetime_df['monetary'].mean():,.2f})."
        ),
        "action": (
            "Launch an automated post-delivery 30-day win-back voucher targeting 'High-Value New Buyers' and 'Recent One-Time Buyers'."
        ),
        "quantified_impact": f"Converting just 5% of one-time buyers adds ~{retention_upside:,.2f} in incremental revenue.",
    },
    {
        "priority": "P2 — Margin & Conversion Protection",
        "area": "Regional Fulfillment & Freight Subsidies",
        "finding": (
            f"Northeast hubs like Bahia (BA) pay {ba_freight:,.2f} avg freight (+{((ba_freight/sp_freight)-1)*100:.0f}% vs. SP's {sp_freight:,.2f}) "
            f"and wait {ba_days:.1f} days vs. SP's {sp_days:.1f} days."
        ),
        "action": (
            "Position regional micro-fulfillment inventory for top categories ('beleza_saude', 'relogios_presentes') outside SP and cap freight fees on baskets above 150."
        ),
        "quantified_impact": "Reduces inter-state transit time by 4–6 days and lowers cart abandonment in high-freight states.",
    },
    {
        "priority": "P3 — Peak Inventory & SLA Readiness",
        "area": "Q4 Demand Scaling & Delivery Promises",
        "finding": (
            f"Model projects {future_df['forecast_revenue'].sum():,.2f} over the next 6 months, peaking in Nov 2018 at {future_df.loc[2, 'forecast_revenue']:,.2f}, "
            f"while current late delivery rate sits at {late_rate:.2f}%."
        ),
        "action": (
            f"Increase safety stock by 35% in '{top_cat}' ({top_cat_rev:,.2f} historical leader) by October and tighten padded delivery estimates (currently 23.7 days estimated vs. 12.6 actual)."
        ),
        "quantified_impact": "Prevents Q4 stockouts during peak demand while improving checkout conversion through realistic delivery dates.",
    },
]

# Export JSON for Streamlit integration
rec_json_path = f"{CLEAN_DIR}/ai_recommendations.json"
with open(rec_json_path, "w", encoding="utf-8") as f:
    json.dump(recommendations, f, indent=2)

print("\n" + "=" * 80)
print("3. AI-GENERATED EXECUTIVE RECOMMENDATIONS")
print("=" * 80)
for rec in recommendations:
    print(f"\n[{rec['priority']}] {rec['area']}")
    print(f"  - Data Signal: {rec['finding']}")
    print(f"  - Action Plan: {rec['action']}")
    print(f"  - Est. Impact: {rec['quantified_impact']}")