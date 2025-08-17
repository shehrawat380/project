import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# -----------------------
# App Title
# -----------------------
st.set_page_config(page_title="Smart Data Analyzer", layout="wide")
st.title("📊 Smart Data Analyzer & Forecaster")

# -----------------------
# File Upload
# -----------------------
st.sidebar.header("Upload your data file")
file = st.sidebar.file_uploader("Upload CSV / Excel / JSON", type=["csv", "xlsx", "json"])

if file is not None:
    # Detect file format
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
        file_type = "CSV"
    elif file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
        file_type = "Excel"
    elif file.name.endswith(".json"):
        df = pd.read_json(file)
        file_type = "JSON"
    else:
        st.error("Unsupported file format")
        st.stop()

    st.success(f"✅ File detected as {file_type}")
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # -----------------------
    # Data Summary
    # -----------------------
    st.subheader("🔎 Dataset Summary")
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.write(df.dtypes.value_counts().rename("count").to_frame("Column Types"))

    # Missing values
    st.write("🟠 Missing Values:")
    st.write(df.isnull().sum())

    # Duplicate rows
    dup_count = df.duplicated().sum()
    st.write(f"🟠 Duplicate Rows: {dup_count}")

    # -----------------------
    # Cleaning Options
    # -----------------------
    st.sidebar.header("🧹 Data Cleaning")
    if st.sidebar.button("Drop Missing Values"):
        df.dropna(inplace=True)
        st.sidebar.success("Dropped missing values!")

    if st.sidebar.button("Fill Missing with Mean"):
        df.fillna(df.mean(numeric_only=True), inplace=True)
        st.sidebar.success("Filled NA with mean!")

    if st.sidebar.button("Drop Duplicates"):
        df.drop_duplicates(inplace=True)
        st.sidebar.success("Dropped duplicate rows!")

    # -----------------------
    # Visualization Suggestions
    # -----------------------
    st.subheader("📊 Suggested Visualizations")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if num_cols:
        st.write("✅ Numerical Columns:", num_cols)
        col = st.selectbox("Choose a numerical column for distribution", num_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    if cat_cols:
        st.write("✅ Categorical Columns:", cat_cols)
        col = st.selectbox("Choose a categorical column for bar chart", cat_cols)
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    if len(num_cols) > 1:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # -----------------------
    # Forecasting Section
    # -----------------------
    st.subheader("📈 Forecasting (Time Series)")
    date_cols = df.select_dtypes(include=["datetime64", "object"]).columns.tolist()

    if date_cols:
        date_col = st.selectbox("Choose Date Column", date_cols)
        target_col = st.selectbox("Choose Target Column for Forecasting", num_cols)

        if target_col and date_col:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                ts_df = df[[date_col, target_col]].dropna()
                ts_df.columns = ["ds", "y"]

                st.write("🔮 Prophet Forecasting")
                model = Prophet()
                model.fit(ts_df)

                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)

                fig1 = model.plot(forecast)
                st.pyplot(fig1)

                st.write("📉 ARIMA Forecasting (last 100 points)")
                ts_series = ts_df.set_index("ds")["y"].tail(100)
                model_arima = ARIMA(ts_series, order=(2,1,2))
                results = model_arima.fit()
                forecast_arima = results.forecast(steps=30)

                fig, ax = plt.subplots()
                ts_series.plot(ax=ax, label="History")
                forecast_arima.plot(ax=ax, label="ARIMA Forecast")
                ax.legend()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Forecasting error: {e}")
    else:
        st.info("No date column detected for forecasting.")
else:
    st.info("👆 Upload a file to begin.")
