import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(page_title="Pizza Sales Dashboard", layout="wide")

st.title("🍕 Pizza Sales Analytics Dashboard")

# -----------------------
# Load Dataset
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_excel("D:/DA_Project_Data/pizza_sales_excel_file.xlsx")
    return df

df = load_data()

# -----------------------
# Data Preparation
# -----------------------
df['order_date'] = pd.to_datetime(df['order_date'])
df['Month'] = df['order_date'].dt.month_name()
df['Hour'] = pd.to_datetime(df['order_time'], format='%H:%M:%S').dt.hour

# -----------------------
# KPIs
# -----------------------
total_revenue = df['total_price'].sum()
total_orders = df['order_id'].nunique()
total_quantity = df['quantity'].sum()
avg_order_value = total_revenue / total_orders

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Revenue", f"${total_revenue:,.2f}")
col2.metric("Total Orders", total_orders)
col3.metric("Total Quantity Sold", total_quantity)
col4.metric("Avg Order Value", f"${avg_order_value:,.2f}")

st.markdown("---")

# -----------------------
# Row 1
# -----------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Revenue by Category")
    fig1, ax1 = plt.subplots()
    df.groupby('pizza_category')['total_price'].sum().plot(
        kind='bar', ax=ax1)
    ax1.set_ylabel("Revenue")
    st.pyplot(fig1)

with col2:
    st.subheader("Monthly Revenue Trend")
    monthly = df.groupby('Month')['total_price'].sum()
    fig2, ax2 = plt.subplots()
    monthly.plot(kind='line', marker='o', ax=ax2)
    ax2.set_ylabel("Revenue")
    st.pyplot(fig2)

# -----------------------
# Row 2
# -----------------------
col3, col4 = st.columns(2)

with col3:
    st.subheader("Revenue by Pizza Size")
    size_data = df.groupby('pizza_size')['total_price'].sum()
    fig3, ax3 = plt.subplots()
    ax3.pie(size_data, labels=size_data.index, autopct='%1.1f%%')
    st.pyplot(fig3)

with col4:
    st.subheader("Orders by Hour")
    hourly = df.groupby('Hour')['order_id'].count()
    fig4, ax4 = plt.subplots()
    hourly.plot(kind='bar', ax=ax4)
    ax4.set_ylabel("Number of Orders")
    st.pyplot(fig4)

# -----------------------
# Row 3
# -----------------------
col5, col6 = st.columns(2)

with col5:
    st.subheader("Top 10 Best-Selling Pizzas")
    top10 = df.groupby('pizza_name')['quantity'].sum().sort_values(ascending=False).head(10)
    fig5, ax5 = plt.subplots()
    top10.sort_values().plot(kind='barh', ax=ax5)
    ax5.set_xlabel("Quantity Sold")
    st.pyplot(fig5)

with col6:
    st.subheader("Category vs Size Revenue")
    matrix = df.pivot_table(values='total_price',
                            index='pizza_category',
                            columns='pizza_size',
                            aggfunc='sum')
    fig6, ax6 = plt.subplots()
    sns.heatmap(matrix, annot=True, fmt=".0f", ax=ax6)
    st.pyplot(fig6)

st.markdown("---")
st.success("Dashboard Loaded Successfully 🚀")
