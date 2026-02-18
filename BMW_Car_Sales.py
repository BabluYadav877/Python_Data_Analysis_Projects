
# ==========================================================
# 🚗 BMW CAR SALES DATA ANALYSIS + FORECASTING DASHBOARD
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------

st.set_page_config(
    page_title="BMW CAR SALES ANALYSIS",
    page_icon="🚗",
    layout="wide"
)

st.markdown("<h1 style='text-align: center;'>🚗 BMW CAR SALES DATA ANALYSIS DASHBOARD</h1>", unsafe_allow_html=True)
st.markdown("---")

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/Lenovo/Desktop/Power BI Projects/Data/BMW_DATA.csv")

    # Revenue
    df["Revenue"] = df["Price_USD"] * df["Sales_Volume"]

    # Assume cost = 70% of revenue (adjustable)
    df["Cost"] = df["Revenue"] * 0.70

    # Profit
    df["Profit"] = df["Revenue"] - df["Cost"]

    return df

df = load_data()

# ----------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------

st.sidebar.header("🔎 Advanced Filters")

region = st.sidebar.multiselect("Region", df["Region"].unique(), df["Region"].unique())
model = st.sidebar.multiselect("Model", df["Model"].unique(), df["Model"].unique())
fuel = st.sidebar.multiselect("Fuel Type", df["Fuel_Type"].unique(), df["Fuel_Type"].unique())

year_range = st.sidebar.slider(
    "Year Range",
    int(df["Year"].min()),
    int(df["Year"].max()),
    (int(df["Year"].min()), int(df["Year"].max()))
)

filtered_df = df[
    (df["Region"].isin(region)) &
    (df["Model"].isin(model)) &
    (df["Fuel_Type"].isin(fuel)) &
    (df["Year"].between(year_range[0], year_range[1]))
]

# ----------------------------------------------------------
# KPI SECTION
# ----------------------------------------------------------

st.subheader("📊 Executive KPIs")

total_revenue = filtered_df["Revenue"].sum()
total_sales = filtered_df["Sales_Volume"].sum()
avg_price = filtered_df["Price_USD"].mean()
total_profit = filtered_df["Profit"].sum()
profit_margin = (total_profit / total_revenue) * 100 if total_revenue != 0 else 0
#total_loss = filtered_df[filtered_df["Profit"] < 0]["Profit"].sum()

# YoY Growth
current_year = df["Year"].max()
prev_year = current_year - 1

current_sales = df[df["Year"] == current_year]["Sales_Volume"].sum()
prev_sales = df[df["Year"] == prev_year]["Sales_Volume"].sum()

growth = ((current_sales - prev_sales) / prev_sales) * 100 if prev_sales != 0 else 0

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

col1.metric("Total Revenue", f"${total_revenue:,.0f}")
col2.metric("Total Sales", f"{total_sales:,.0f}")
col3.metric("Avg Price", f"${avg_price:,.0f}")
col4.metric("Total Profit", f"${total_profit:,.0f}")
col5.metric("Profit Margin %", f"{profit_margin:.2f}%")
col7.metric("YoY Growth %", f"{growth:.2f}%")
#col6.metric("Total Loss", f"${total_loss:,.0f}")

st.markdown("---")

# ----------------------------------------------------------
# TABS
# ----------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📈 Performance", "📊 Market Analysis", "🔮 Forecasting"])

# ==========================================================
# TAB 1 – PERFORMANCE
# ==========================================================

with tab1:

    # Revenue by Region
    st.subheader("🌍 Revenue by Region")

    region_revenue = filtered_df.groupby("Region")["Revenue"].sum().reset_index()

    fig_region = px.bar(
        region_revenue,
        x="Region",
        y="Revenue",
        color="Revenue",
        text_auto=True
    )

    st.plotly_chart(fig_region, use_container_width=True)

    # Top 5 Regions
    st.subheader("🌎 Top 5 Regions by Revenue")

    top_regions = region_revenue.sort_values(
        by="Revenue", ascending=False
    ).head(5)

    fig_top_regions = px.bar(
        top_regions,
        x="Region",
        y="Revenue",
        color="Revenue",
        text_auto=True
    )

    st.plotly_chart(fig_top_regions, use_container_width=True)

    # Top 5 Models by Profit
    st.subheader("🏆 Top 5 Models by Profit")

    top_models_profit = (
        filtered_df.groupby("Model")["Profit"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )

    fig_top_models = px.bar(
        top_models_profit,
        x="Model",
        y="Profit",
        color="Profit",
        text_auto=True
    )

    st.plotly_chart(fig_top_models, use_container_width=True)

    # Profit Trend by Year
    st.subheader("💰 Profit Trend by Year")

    profit_year = filtered_df.groupby("Year")["Profit"].sum().reset_index()

    fig_profit = px.line(
        profit_year,
        x="Year",
        y="Profit",
        markers=True
    )

    st.plotly_chart(fig_profit, use_container_width=True)

    # Revenue vs Profit Comparison
    st.subheader("📊 Revenue vs Profit Comparison")

    rev_profit_year = filtered_df.groupby("Year")[["Revenue", "Profit"]].sum().reset_index()

    fig_compare = go.Figure()

    fig_compare.add_trace(go.Bar(
        x=rev_profit_year["Year"],
        y=rev_profit_year["Revenue"],
        name="Revenue"
    ))

    fig_compare.add_trace(go.Bar(
        x=rev_profit_year["Year"],
        y=rev_profit_year["Profit"],
        name="Profit"
    ))

    fig_compare.update_layout(barmode="group")

    st.plotly_chart(fig_compare, use_container_width=True)

# ==========================================================
# TAB 2 – MARKET ANALYSIS
# ==========================================================

with tab2:

    # Scatter
    st.subheader("🚘 Mileage vs Price")

    fig_scatter = px.scatter(
        filtered_df,
        x="Mileage_KM",
        y="Price_USD",
        color="Fuel_Type",
        size="Sales_Volume",
        hover_data=["Model"]
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    # Correlation Heatmap
    st.subheader("📊 Correlation Matrix")

    numeric_cols = [
        "Price_USD", "Mileage_KM", "Engine_Size_L",
        "Sales_Volume", "Composite_Health_Score",
        "Market_Saturation_Ratio", "Revenue", "Profit"
    ]

    corr = filtered_df[numeric_cols].corr()

    fig_heatmap = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

# ==========================================================
# TAB 3 – FORECASTING
# ==========================================================

with tab3:

    st.subheader("📈 5-Year Sales Forecast")

    forecast_df = df.groupby("Year")["Sales_Volume"].sum().reset_index()

    X = forecast_df[["Year"]]
    y = forecast_df["Sales_Volume"]

    model_lr = LinearRegression()
    model_lr.fit(X, y)

    future_years = pd.DataFrame({
        "Year": np.arange(forecast_df["Year"].max()+1,
                          forecast_df["Year"].max()+6)
    })

    future_years["Predicted_Sales"] = model_lr.predict(future_years)

    std_dev = y.std()

    fig_forecast = go.Figure()

    fig_forecast.add_trace(go.Scatter(
        x=forecast_df["Year"],
        y=forecast_df["Sales_Volume"],
        mode="lines+markers",
        name="Actual"
    ))

    fig_forecast.add_trace(go.Scatter(
        x=future_years["Year"],
        y=future_years["Predicted_Sales"],
        mode="lines+markers",
        name="Forecast"
    ))

    fig_forecast.add_trace(go.Scatter(
        x=future_years["Year"],
        y=future_years["Predicted_Sales"] + std_dev,
        mode="lines",
        line=dict(dash="dash"),
        name="Upper Bound"
    ))

    fig_forecast.add_trace(go.Scatter(
        x=future_years["Year"],
        y=future_years["Predicted_Sales"] - std_dev,
        mode="lines",
        line=dict(dash="dash"),
        name="Lower Bound"
    ))

    st.plotly_chart(fig_forecast, use_container_width=True)

# ----------------------------------------------------------
# DOWNLOAD
# ----------------------------------------------------------

st.sidebar.download_button(
    label="📥 Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="BMW_filtered_data.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown("Developed by Bablu Yadav 🚀 | BMW Enterprise Analytics Dashboard")

