import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==================================
# 1. PAGE CONFIG & DATA LOADING
# ==================================
st.set_page_config(
    page_title="ShopSmart BI & Predictive Analytics Platform",
    page_icon="📊",
    layout="wide",
)

CLEAN_DIR = "../02_Cleaned_Data"


@st.cache_data
def load_core_data():
    sales = pd.read_csv(f"{CLEAN_DIR}/master_sales_enriched.csv")
    rfm = pd.read_csv(f"{CLEAN_DIR}/rfm_customer_segments.csv")
    sales["is_late_delivery"] = pd.to_numeric(sales["is_late_delivery"], errors="coerce")
    return sales, rfm


@st.cache_data
def load_ai_outputs():
    forecast_path = f"{CLEAN_DIR}/sales_forecast_6m.csv"
    rec_path = f"{CLEAN_DIR}/ai_recommendations.json"

    if not (os.path.exists(forecast_path) and os.path.exists(rec_path)):
        return None, None

    forecast_df = pd.read_csv(forecast_path)
    with open(rec_path, "r", encoding="utf-8") as f:
        recommendations = json.load(f)

    return forecast_df, recommendations


sales_df, rfm_df = load_core_data()
forecast_df, recommendations = load_ai_outputs()

# Merge RFM segment onto sales so sidebar filters slice all visuals together
merged_df = sales_df.merge(
    rfm_df[["customer_unique_id", "customer_segment"]],
    on="customer_unique_id",
    how="left",
)

# ==================================
# 2. SIDEBAR SLICERS (GLOBAL FILTERS)
# ==================================
st.sidebar.header("Dashboard Filters")

all_states = sorted(merged_df["customer_state"].dropna().unique())
selected_states = st.sidebar.multiselect(
    "Customer State",
    options=all_states,
    default=[],
    placeholder="All States (27)",
)

all_segments = sorted(rfm_df["customer_segment"].dropna().unique())
selected_segments = st.sidebar.multiselect(
    "RFM Customer Segment",
    options=all_segments,
    default=[],
    placeholder="All Segments (7)",
)

top_cat_list = (
    merged_df.groupby("product_category_name")["price"]
    .sum()
    .sort_values(ascending=False)
    .head(20)
    .index.tolist()
)
selected_categories = st.sidebar.multiselect(
    "Top 20 Product Categories",
    options=top_cat_list,
    default=[],
    placeholder="All Categories",
)

# Apply filters for Tabs 1–3
filtered_sales = merged_df.copy()
filtered_rfm = rfm_df.copy()

if selected_states:
    filtered_sales = filtered_sales[filtered_sales["customer_state"].isin(selected_states)]
    filtered_rfm = filtered_rfm[filtered_rfm["customer_state"].isin(selected_states)]

if selected_segments:
    filtered_sales = filtered_sales[filtered_sales["customer_segment"].isin(selected_segments)]
    filtered_rfm = filtered_rfm[filtered_rfm["customer_segment"].isin(selected_segments)]

if selected_categories:
    filtered_sales = filtered_sales[filtered_sales["product_category_name"].isin(selected_categories)]
    valid_custs = filtered_sales["customer_unique_id"].unique()
    filtered_rfm = filtered_rfm[filtered_rfm["customer_unique_id"].isin(valid_custs)]

# ==================================
# 3. HEADER & EXECUTIVE KPI CARDS
# ==================================
st.title("📊 ShopSmart AI-Powered Business Intelligence Platform")
st.caption("End-to-end analytics across 98,199 valid orders, 94,983 unique customers, and 6-month predictive forecasting.")

if filtered_sales.empty:
    st.warning("No records match your current filter selection. Clear one or more filters in the sidebar.")
    st.stop()

prod_rev = filtered_sales["price"].sum()
gmv = filtered_sales["total_item_value"].sum()
valid_orders = filtered_sales["order_id"].nunique()
aov = prod_rev / valid_orders if valid_orders > 0 else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Product Revenue", f"{prod_rev:,.2f}")
k2.metric("Gross Merchandise Value", f"{gmv:,.2f}")
k3.metric("Total Valid Orders", f"{valid_orders:,}")
k4.metric("Average Order Value (AOV)", f"{aov:,.2f}")

st.divider()

# ==================================
# 4. FOUR INTERACTIVE DASHBOARD TABS
# ==================================
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Revenue & Categories",
    "2. Customer & RFM Intelligence",
    "3. Logistics & Delivery SLA",
    "4. AI Forecast & Strategy",
])

# --- TAB 1: REVENUE & CATEGORIES ---
with tab1:
    c1, c2 = st.columns([1.3, 1])

    with c1:
        monthly = (
            filtered_sales[
                (filtered_sales["order_month"] >= "2017-01")
                & (filtered_sales["order_month"] <= "2018-08")
            ]
            .groupby("order_month", as_index=False)
            .agg(Revenue=("price", "sum"), Orders=("order_id", "nunique"))
            .sort_values("order_month")
        )
        fig_trend = px.area(
            monthly,
            x="order_month",
            y="Revenue",
            markers=True,
            hover_data={"Orders": ":,"},
            title="Monthly Product Revenue Trend (Jan 2017 – Aug 2018)",
        )
        fig_trend.update_layout(xaxis_title="Month", yaxis_title="Product Revenue", hovermode="x unified")
        st.plotly_chart(fig_trend, use_container_width=True)

    with c2:
        top_cats = (
            filtered_sales.groupby("product_category_name", as_index=False)
            .agg(Revenue=("price", "sum"), Orders=("order_id", "nunique"))
            .sort_values("Revenue", ascending=False)
            .head(10)
            .sort_values("Revenue", ascending=True)
        )
        fig_cats = px.bar(
            top_cats,
            x="Revenue",
            y="product_category_name",
            orientation="h",
            text_auto=".2s",
            hover_data={"Orders": ":,"},
            title="Top 10 Product Categories by Revenue",
        )
        fig_cats.update_layout(xaxis_title="Product Revenue", yaxis_title="")
        st.plotly_chart(fig_cats, use_container_width=True)

# --- TAB 2: CUSTOMER & RFM INTELLIGENCE ---
with tab2:
    unique_custs = len(filtered_rfm)
    repeat_custs = (filtered_rfm["frequency"] >= 2).sum()
    repeat_rate = (repeat_custs / unique_custs * 100) if unique_custs > 0 else 0
    avg_ltv = filtered_rfm["monetary"].mean() if unique_custs > 0 else 0

    r1, r2, r3 = st.columns(3)
    r1.metric("Unique Customers", f"{unique_custs:,}")
    r2.metric("Repeat Buyer Rate (2+ Orders)", f"{repeat_rate:.2f}%")
    r3.metric("Average Customer LTV", f"{avg_ltv:,.2f}")

    seg_summary = (
        filtered_rfm.groupby("customer_segment", as_index=False)
        .agg(
            Customers=("customer_unique_id", "count"),
            Total_Revenue=("monetary", "sum"),
            Avg_LTV=("monetary", "mean"),
            Avg_Recency_Days=("recency_days", "mean"),
        )
        .sort_values("Total_Revenue", ascending=False)
    )

    col_left, col_right = st.columns(2)
    with col_left:
        fig_seg_rev = px.bar(
            seg_summary,
            x="customer_segment",
            y="Total_Revenue",
            color="Avg_LTV",
            text_auto=".2s",
            title="Total Revenue & Avg LTV by RFM Segment",
        )
        fig_seg_rev.update_layout(xaxis_title="", yaxis_title="Total Revenue")
        st.plotly_chart(fig_seg_rev, use_container_width=True)

    with col_right:
        fig_seg_cnt = px.pie(
            seg_summary,
            names="customer_segment",
            values="Customers",
            hole=0.45,
            title="Customer Base Share by RFM Segment",
        )
        st.plotly_chart(fig_seg_cnt, use_container_width=True)

# --- TAB 3: LOGISTICS & DELIVERY SLA ---
with tab3:
    valid_deliv = filtered_sales.dropna(subset=["is_late_delivery"])
    avg_deliv_days = valid_deliv["actual_delivery_days"].mean()
    late_orders = valid_deliv[valid_deliv["is_late_delivery"] == 1]["order_id"].nunique()
    deliv_orders = valid_deliv["order_id"].nunique()
    late_rate = (late_orders / deliv_orders * 100) if deliv_orders > 0 else 0
    total_freight = filtered_sales["freight_value"].sum()

    l1, l2, l3 = st.columns(3)
    l1.metric("Avg Actual Delivery Time", f"{avg_deliv_days:.1f} days")
    l2.metric("Late Delivery Rate", f"{late_rate:.2f}%")
    l3.metric("Total Freight Cost", f"{total_freight:,.2f}")

    state_logistics = (
        filtered_sales.groupby("customer_state", as_index=False)
        .agg(
            Orders=("order_id", "nunique"),
            Revenue=("price", "sum"),
            Avg_Freight=("freight_value", "mean"),
            Avg_Delivery_Days=("actual_delivery_days", "mean"),
        )
    )
    state_logistics = state_logistics[state_logistics["Orders"] >= 50]

    fig_logistics = px.scatter(
        state_logistics,
        x="Avg_Delivery_Days",
        y="Avg_Freight",
        size="Revenue",
        color="customer_state",
        text="customer_state",
        hover_data={"Orders": ":,", "Revenue": ":,.2f"},
        title="Regional Logistics Friction: Delivery Days vs. Avg Freight Cost (States with 50+ Orders)",
    )
    fig_logistics.update_traces(textposition="top center")
    fig_logistics.update_layout(showlegend=False, xaxis_title="Avg Delivery Days", yaxis_title="Avg Freight Cost")
    st.plotly_chart(fig_logistics, use_container_width=True)

# --- TAB 4: AI FORECAST & EXECUTIVE STRATEGY ---
with tab4:
    if forecast_df is None or recommendations is None:
        st.info(
            "Forecast outputs not found yet. Run `py 08_forecast_and_ai_insights.py` inside `04_Python` first, then refresh this page."
        )
    else:
        st.caption("Company-wide predictive outlook based on complete monthly operational history (Jan 2017 – Aug 2018).")

        total_6m_proj = forecast_df["forecast_revenue"].sum()
        avg_6m_proj = forecast_df["forecast_revenue"].mean()
        peak_row = forecast_df.loc[forecast_df["forecast_revenue"].idxmax()]

        f1, f2, f3 = st.columns(3)
        f1.metric("6-Month Projected Revenue", f"{total_6m_proj:,.2f}")
        f2.metric("Projected Monthly Avg", f"{avg_6m_proj:,.2f}")
        f3.metric(
            f"Peak Forecast Month ({peak_row['order_month']})",
            f"{peak_row['forecast_revenue']:,.2f}",
        )

        hist_monthly = (
            sales_df[
                (sales_df["order_month"] >= "2017-01")
                & (sales_df["order_month"] <= "2018-08")
            ]
            .groupby("order_month", as_index=False)
            .agg(revenue=("price", "sum"))
            .sort_values("order_month")
        )

        fig_fc = go.Figure()

        fig_fc.add_trace(
            go.Scatter(
                x=hist_monthly["order_month"],
                y=hist_monthly["revenue"],
                mode="lines+markers",
                name="Actual Revenue",
                line=dict(color="#2563EB", width=2.5),
            )
        )

        fig_fc.add_trace(
            go.Scatter(
                x=[hist_monthly["order_month"].iloc[-1], forecast_df["order_month"].iloc[0]],
                y=[hist_monthly["revenue"].iloc[-1], forecast_df["forecast_revenue"].iloc[0]],
                mode="lines",
                line=dict(color="#0D9488", width=2, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig_fc.add_trace(
            go.Scatter(
                x=forecast_df["order_month"],
                y=forecast_df["upper_95"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_fc.add_trace(
            go.Scatter(
                x=forecast_df["order_month"],
                y=forecast_df["lower_95"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(13, 148, 136, 0.16)",
                name="95% Confidence Interval",
                hoverinfo="skip",
            )
        )

        fig_fc.add_trace(
            go.Scatter(
                x=forecast_df["order_month"],
                y=forecast_df["forecast_revenue"],
                mode="lines+markers",
                name="6-Month Forecast",
                line=dict(color="#0D9488", width=2.8, dash="dash"),
                marker=dict(symbol="square", size=7),
            )
        )

        fig_fc.update_layout(
            title="ShopSmart 6-Month Revenue Forecast (Sep 2018 – Feb 2019)",
            xaxis_title="Month",
            yaxis_title="Product Revenue",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        left_fc, right_ai = st.columns([1, 1.35])

        with left_fc:
            st.subheader("Monthly Forecast Schedule")
            display_fc = forecast_df.rename(
                columns={
                    "order_month": "Month",
                    "forecast_revenue": "Projected Revenue",
                    "lower_95": "Lower 95% CI",
                    "upper_95": "Upper 95% CI",
                }
            )
            st.dataframe(
                display_fc,
                column_config={
                    "Projected Revenue": st.column_config.NumberColumn(format="%.2f"),
                    "Lower 95% CI": st.column_config.NumberColumn(format="%.2f"),
                    "Upper 95% CI": st.column_config.NumberColumn(format="%.2f"),
                },
                use_container_width=True,
                hide_index=True,
            )

        with right_ai:
            st.subheader("AI-Generated Executive Recommendations")
            for rec in recommendations:
                with st.container(border=True):
                    st.markdown(f"**{rec['priority']} | {rec['area']}**")
                    st.markdown(f"**Data Signal:** {rec['finding']}")
                    st.markdown(f"**Action Plan:** {rec['action']}")
                    st.success(f"**Projected Impact:** {rec['quantified_impact']}")