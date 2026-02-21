# ==========================================================
# 📊 ADVANCED CUSTOMER CHURN ANALYTICS DASHBOARD
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import shap

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📊 Customer Churn Analytics Dashboard")

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("D:\DA_Project_Data\customer_data.csv")
st.sidebar.header("Dashboard Filter")

country = st.sidebar.multiselect(
    "Select Country",
    options=df["country"].unique(),
    default=df["country"].unique()
)

df = df[df["country"].isin(country)]

# ==========================================================
# KPI SECTION
# ==========================================================

total_customer = df.shape[0]
churn_customer = df[df["churn"] == 1].shape[0]
active_customer = df[df["active_member"] == 1].shape[0]
avg_balance = int(df["balance"].mean())
churn_rate = (churn_customer / total_customer) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", total_customer)
col2.metric("Churn Customers", churn_customer)
col3.metric("Active Members", active_customer)
col4.metric("Average Balance", avg_balance)

st.metric("Churn Rate (%)", f"{churn_rate:.2f}%")

# ==========================================================
# SHOW DATA + DOWNLOAD
# ==========================================================

if st.checkbox("Show Raw Data"):
    st.dataframe(df)

csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
    "Download Filtered Data",
    csv,
    "Filtered_Customer_Data.csv",
    "text/csv"
)

# ==========================================================
# VISUALIZATION
# ==========================================================

st.subheader("Age Distribution by Churn")
fig1 = px.histogram(df, x="age", color="churn", barmode="overlay")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Tenure vs Churn")
fig2 = px.bar(df, x="tenure", color="churn")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Gender vs Churn")
fig3 = px.pie(df, names="gender", color="churn", hole=0.5)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Balance vs Products")
fig4 = px.scatter(df, x="products_number", y="balance", color="churn")
st.plotly_chart(fig4, use_container_width=True)

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

st.subheader("Correlation Heatmap")
numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr()
fig5 = px.imshow(corr, text_auto=True)
st.plotly_chart(fig5, use_container_width=True)

# ==========================================================
# MAP VISUALIZATION
# ==========================================================

st.subheader("Customer Distribution by Country")
country_df = df.groupby("country").size().reset_index(name='count')
fig9 = px.choropleth(country_df,
                     locations="country",
                     locationmode="country names",
                     color="count")
st.plotly_chart(fig9, use_container_width=True)

# ==========================================================
# MACHINE LEARNING MODEL
# ==========================================================

st.subheader("Churn Prediction Model")

if st.button("Train ML Model"):

    X = df.drop(["customer_id", "churn"], axis=1)
    y = df["churn"]

    X = pd.get_dummies(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.success(f"Model Accuracy: {accuracy*100:.2f}%")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

    importance = model.feature_importances_

    feature_df = pd.DataFrame({
        "Feature": pd.get_dummies(df.drop(["customer_id", "churn"], axis=1)).columns,
        "Importance": importance
    })

    fig7 = px.bar(feature_df.sort_values(by="Importance", ascending=False),
                  x="Importance",
                  y="Feature",
                  orientation='h')

    st.plotly_chart(fig7, use_container_width=True)

# ==========================================================
# ROC CURVE
# ==========================================================

    y_prob = model.predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    fig8 = px.area(
        x=fpr,
        y=tpr,
        title=f'ROC Curve (AUC={auc_score:.2f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate')
    )

    st.plotly_chart(fig8, use_container_width=True)

# ==========================================================
# SHAP EXPLAINABILITY
# ==========================================================

    st.subheader("Model Explainability using SHAP")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    st.set_option('deprecation.showPyplotGlobalUse', False)

    shap.summary_plot(shap_values, X_train, show=False)
    st.pyplot()

# ==========================================================
# CUSTOMER RISK PREDICTOR
# ==========================================================

st.subheader("Predict Customer Churn Risk")

age = st.slider("Age", 18, 80, 30)
balance = st.number_input("Balance", 0)
tenure = st.slider("Tenure", 0, 10, 2)
products = st.slider("Products Number", 1, 4, 1)

input_data = pd.DataFrame({
    'age':[age],
    'balance':[balance],
    'tenure':[tenure],
    'products_number':[products]
})

if st.button("Predict Churn Risk"):

    try:

        X = df.drop(["customer_id", "churn"], axis=1)
        y = df["churn"]

        X = pd.get_dummies(X)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model = RandomForestClassifier()
        model.fit(X, y)

        probability = model.predict_proba(scaler.transform(
            pd.get_dummies(input_data).reindex(columns=
            pd.get_dummies(df.drop(["customer_id","churn"],axis=1)).columns,
            fill_value=0)
        ))

        churn_prob = probability[0][1] * 100

        st.info(f"Churn Probability: {churn_prob:.2f}%")

        if churn_prob > 50:
            st.error("⚠️ High Risk Customer")
        else:
            st.success("✅ Low Risk Customer")

    except:
        st.warning("Train Model First!")

# ==========================================================
# BUSINESS INSIGHTS
# ==========================================================

st.subheader("Business Insights")

high_balance = df[df["balance"] > df["balance"].mean()].shape[0]
inactive = df[df["active_member"] == 0].shape[0]

st.write(f"🔹 {high_balance} customers have above average balance")
st.write(f"🔹 {inactive} customers are inactive members")
st.write(f"🔹 Churn Rate is {churn_rate:.2f}%")
st.write("🔹 Customers with low tenure are more likely to churn")
st.write("🔹 Inactive members have higher churn probability")