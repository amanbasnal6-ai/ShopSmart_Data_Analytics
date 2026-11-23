import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ==================================
# 1. SETUP PATHS & GLOBAL CHART STYLE
# ==================================
CLEAN_DIR = "../02_Cleaned_Data"
IMG_DIR = "../images"
os.makedirs(IMG_DIR, exist_ok=True)

sales_path = f"{CLEAN_DIR}/master_sales_enriched.csv"
rfm_path = f"{CLEAN_DIR}/rfm_customer_segments.csv"

if not (os.path.exists(sales_path) and os.path.exists(rfm_path)):
    raise FileNotFoundError(
        "Enriched CSVs not found! Run 'py 05_eda_and_rfm.py' first."
    )

df = pd.read_csv(sales_path)
rfm = pd.read_csv(rfm_path)

# Clean, publication-ready styling
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.8,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlepad": 14,
    "axes.labelsize": 10.5,
    "axes.labelcolor": "#334155",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
})

NAVY = "#1E293B"
BLUE = "#2563EB"
TEAL = "#0D9488"
AMBER = "#D97706"
CORAL = "#E11D48"
SLATE = "#64748B"


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.8)
    ax.set_axisbelow(True)


# ==================================
# CHART 1: MONTHLY REVENUE TREND (2017-01 TO 2018-08)
# ==================================
# Exclude partial boundary months (2016-09, 2016-10, 2016-12, 2018-09) for an unbroken series
monthly = (
    df[(df["order_month"] >= "2017-01") & (df["order_month"] <= "2018-08")]
    .groupby("order_month", as_index=False)
    .agg(revenue=("price", "sum"), orders=("order_id", "nunique"))
    .sort_values("order_month")
)

fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
style_axes(ax)

x = np.arange(len(monthly))
y = monthly["revenue"] / 1_000_000  # Convert to Millions

ax.plot(x, y, color=BLUE, linewidth=2.5, marker="o", markersize=6, zorder=3)
ax.fill_between(x, y, color=BLUE, alpha=0.10, zorder=2)

# Highlight Nov 2017 Black Friday Peak
peak_idx = int(np.argmax(y))
peak_val = y.iloc[peak_idx]
ax.scatter([peak_idx], [peak_val], color=CORAL, s=80, zorder=4)
ax.annotate(
    f"Black Friday Peak (Nov 2017)\n{peak_val:.2f}M ({monthly['orders'].iloc[peak_idx]:,} orders)",
    xy=(peak_idx, peak_val),
    xytext=(peak_idx - 4.3, peak_val - 0.14),
    fontsize=9.5,
    fontweight="bold",
    color=NAVY,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF1F2", edgecolor=CORAL, lw=1),
    arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.3),
)

ax.set_title("Monthly Product Revenue Trend (Jan 2017 – Aug 2018)", color=NAVY, loc="left")
ax.set_ylabel("Product Revenue (Millions)")
ax.set_xticks(x)
ax.set_xticklabels(monthly["order_month"], rotation=45, ha="right", fontsize=9)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2fM"))
ax.set_ylim(0, max(y) * 1.18)

fig.text(
    0.01, -0.02,
    "Note: Partial dataset boundary months (Sep–Dec 2016 and Sep 2018) excluded to preserve time-series continuity.",
    fontsize=8.5, color=SLATE
)

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/01_monthly_revenue_trend.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved: images/01_monthly_revenue_trend.png")


# ==================================
# CHART 2: TOP 10 PRODUCT CATEGORIES BY REVENUE
# ==================================
total_rev_all = df["price"].sum()
top_cats = (
    df.groupby("product_category_name", as_index=False)
    .agg(revenue=("price", "sum"), orders=("order_id", "nunique"))
    .sort_values("revenue", ascending=False)
    .head(10)
)
top_cats["share_pct"] = (top_cats["revenue"] / total_rev_all) * 100
top_cats = top_cats.sort_values("revenue", ascending=True)  # Ascending for horizontal bar

fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, axis="x", alpha=0.8)
ax.set_axisbelow(True)

colors = [TEAL if i >= 7 else BLUE for i in range(len(top_cats))]
bars = ax.barh(
    top_cats["product_category_name"],
    top_cats["revenue"] / 1_000_000,
    color=colors,
    height=0.65,
)

for bar, (_, row) in zip(bars, top_cats.iterrows()):
    val_m = row["revenue"] / 1_000_000
    ax.text(
        val_m + 0.02,
        bar.get_y() + bar.get_height() / 2,
        f"{val_m:.2f}M ({row['share_pct']:.1f}%)",
        va="center",
        ha="left",
        fontsize=9.5,
        fontweight="bold",
        color=NAVY,
    )

ax.set_title("Top 10 Product Categories by Revenue & Market Share", color=NAVY, loc="left")
ax.set_xlabel("Product Revenue (Millions)")
ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.2fM"))
ax.set_xlim(0, (top_cats["revenue"].max() / 1_000_000) * 1.24)

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/02_top_categories_revenue.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved: images/02_top_categories_revenue.png")


# ==================================
# CHART 3: RFM CUSTOMER SEGMENTS (HEADCOUNT SHARE VS. AVG LTV)
# ==================================
seg_df = (
    rfm.groupby("customer_segment", as_index=False)
    .agg(
        customers=("customer_unique_id", "count"),
        avg_ltv=("monetary", "mean"),
        total_revenue=("monetary", "sum"),
    )
    .sort_values("avg_ltv", ascending=True)
)
seg_df["cust_share_pct"] = (seg_df["customers"] / len(rfm)) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300, sharey=True)

# Panel 1: Customer Share (%)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(True, axis="x", alpha=0.8)
ax1.set_axisbelow(True)

bars1 = ax1.barh(seg_df["customer_segment"], seg_df["cust_share_pct"], color=SLATE, height=0.6)
for bar, (_, row) in zip(bars1, seg_df.iterrows()):
    ax1.text(
        row["cust_share_pct"] + 0.6,
        bar.get_y() + bar.get_height() / 2,
        f"{row['cust_share_pct']:.1f}% ({row['customers']:,})",
        va="center",
        fontsize=9,
        color=NAVY,
    )
ax1.set_title("Customer Base Share (%)", color=NAVY, loc="left")
ax1.set_xlabel("Share of Unique Customers (%)")
ax1.set_xlim(0, seg_df["cust_share_pct"].max() * 1.32)

# Panel 2: Average Lifetime Value (LTV)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(True, axis="x", alpha=0.8)
ax2.set_axisbelow(True)

ltv_colors = [
    CORAL if "Champions" in s or "Loyal" in s else (AMBER if "High-Value" in s else BLUE)
    for s in seg_df["customer_segment"]
]
bars2 = ax2.barh(seg_df["customer_segment"], seg_df["avg_ltv"], color=ltv_colors, height=0.6)
for bar, (_, row) in zip(bars2, seg_df.iterrows()):
    ax2.text(
        row["avg_ltv"] + 5,
        bar.get_y() + bar.get_height() / 2,
        f"{row['avg_ltv']:.2f}",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=NAVY,
    )
ax2.set_title("Average Customer Lifetime Value (Avg LTV)", color=NAVY, loc="left")
ax2.set_xlabel("Average Spend per Customer")
ax2.set_xlim(0, seg_df["avg_ltv"].max() * 1.18)

fig.suptitle("RFM Segmentation: Repeaters Are Small in Volume but ~2x Higher in LTV", fontsize=14, fontweight="bold", color=NAVY, y=1.02)
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/03_rfm_customer_segments.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved: images/03_rfm_customer_segments.png")


# ==================================
# CHART 4: STATE LOGISTICS FRICTION (DELIVERY DAYS VS. FREIGHT COST)
# ==================================
state_df = (
    df.groupby("customer_state", as_index=False)
    .agg(
        orders=("order_id", "nunique"),
        revenue=("price", "sum"),
        avg_freight=("freight_value", "mean"),
        avg_delivery=("actual_delivery_days", "mean"),
    )
)
# Filter to states with >= 300 orders so low-volume outliers don't skew the chart
state_df = state_df[state_df["orders"] >= 300].copy()

fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.8)
ax.set_axisbelow(True)

bubble_sizes = (state_df["revenue"] / 1_000_000) * 180 + 60
point_colors = [
    TEAL if st == "SP" else (CORAL if d > 18 else BLUE)
    for st, d in zip(state_df["customer_state"], state_df["avg_delivery"])
]

ax.scatter(
    state_df["avg_delivery"],
    state_df["avg_freight"],
    s=bubble_sizes,
    c=point_colors,
    alpha=0.82,
    edgecolors=NAVY,
    linewidth=1,
    zorder=3,
)

# Reference lines for cross-state averages
mean_deliv = state_df["avg_delivery"].mean()
mean_freight = state_df["avg_freight"].mean()
ax.axvline(mean_deliv, color=SLATE, linestyle=":", linewidth=1.2, zorder=2)
ax.axhline(mean_freight, color=SLATE, linestyle=":", linewidth=1.2, zorder=2)

# Label each state cleanly
for _, row in state_df.iterrows():
    offset_y = 0.55 if row["customer_state"] not in ["MG", "GO"] else -0.85
    ax.text(
        row["avg_delivery"],
        row["avg_freight"] + offset_y,
        row["customer_state"],
        fontsize=9,
        fontweight="bold",
        ha="center",
        color=NAVY,
        zorder=4,
    )

ax.set_title("Regional Logistics Friction: Avg Delivery Time vs. Freight Cost by State", color=NAVY, loc="left")
ax.set_xlabel("Average Actual Delivery Time (Days)")
ax.set_ylabel("Average Freight Cost per Item")
ax.set_ylim(12, state_df["avg_freight"].max() + 3.5)

fig.text(
    0.01, -0.02,
    "Bubble size represents total state revenue (states with 300+ orders shown). Dotted lines indicate cross-state averages.",
    fontsize=8.5, color=SLATE
)

plt.tight_layout()
plt.savefig(f"{IMG_DIR}/04_state_freight_vs_delivery.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved: images/04_state_freight_vs_delivery.png")