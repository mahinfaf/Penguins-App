import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(page_title="Data Explorer & Predictor", layout="wide")

# App Title & Overview
st.title("📊 Data Explorer & Linear Regression Predictor")
st.write("Explore dataset features, generate visualizations, and run predictions.")

# ---------------------------------------------------------
# Step 1: Load Data (Auto-load penguins.csv or Custom Upload)
# ---------------------------------------------------------
st.header("1. Dataset Preview")

uploaded_file = st.file_uploader("Upload a custom CSV file (optional)", type=["csv"])

df = None

# If user uploads a file, use it; otherwise, default to local penguins.csv
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Loaded uploaded CSV file.")
else:
    try:
        df = pd.read_csv("penguins.csv")
        st.info("Loaded default dataset: `penguins.csv`")
    except FileNotFoundError:
        st.warning("`penguins.csv` not found in folder. Please upload a CSV file above.")

if df is not None:
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Filter numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_cols) < 3:
        st.error("Please ensure the dataset contains at least 3 numeric columns.")
    else:
        # ---------------------------------------------------------
        # Step 2: Visualization
        # ---------------------------------------------------------
        st.header("2. Data Visualization")
        
        col1, col2 = st.columns(2)
        with col1:
            chart_type = st.selectbox("Select Chart Type", ["Scatter", "Line", "Histogram"])
            selected_col = st.selectbox("Select Primary Column (X-axis)", numeric_cols, index=0)
        
        with col2:
            if chart_type in ["Scatter", "Line"]:
                secondary_col = st.selectbox("Select Secondary Column (Y-axis)", numeric_cols, index=min(1, len(numeric_cols) - 1))

        fig, ax = plt.subplots(figsize=(8, 4))
        
        if chart_type == "Histogram":
            sns.histplot(df[selected_col].dropna(), kde=True, ax=ax, color="skyblue")
            ax.set_title(f"Histogram of {selected_col}")
        elif chart_type == "Line":
            ax.plot(df[selected_col], label=selected_col)
            if secondary_col:
                ax.plot(df[secondary_col], label=secondary_col)
            ax.legend()
            ax.set_title(f"Line Chart of {selected_col} vs {secondary_col}")
        elif chart_type == "Scatter":
            sns.scatterplot(data=df, x=selected_col, y=secondary_col, ax=ax, color="teal")
            ax.set_title(f"Scatter Plot: {selected_col} vs {secondary_col}")

        st.pyplot(fig)

        # ---------------------------------------------------------
        # Step 3: Linear Regression Model
        # ---------------------------------------------------------
        st.header("3. Linear Regression Predictor")

        c1, c2, c3 = st.columns(3)
        with c1:
            feature1 = st.selectbox("Feature 1 (X1)", numeric_cols, index=0)
        with c2:
            feature2 = st.selectbox("Feature 2 (X2)", numeric_cols, index=1)
        with c3:
            target = st.selectbox("Target (Y)", numeric_cols, index=min(2, len(numeric_cols) - 1))

        if feature1 == feature2 or feature1 == target or feature2 == target:
            st.warning("Please select three distinct columns for Feature 1, Feature 2, and Target.")
        else:
            # Clean missing values for training
            clean_df = df[[feature1, feature2, target]].dropna()

            X = clean_df[[feature1, feature2]]
            y = clean_df[target]

            model = LinearRegression()
            model.fit(X, y)

            st.subheader("Make Predictions")
            val1 = st.number_input(f"Enter value for {feature1}:", value=float(clean_df[feature1].mean()))
            val2 = st.number_input(f"Enter value for {feature2}:", value=float(clean_df[feature2].mean()))

            if st.button("Predict Target"):
                prediction = model.predict([[val1, val2]])
                st.success(f"Predicted **{target}**: `{prediction[0]:.2f}`")

        # ---------------------------------------------------------
        # Step 4: Explanation Sentence
        # ---------------------------------------------------------
        st.markdown("---")
        st.info(
            "**Model Usage & Risk:** This linear regression model predicts physical attributes of penguins (e.g., body mass) "
            "based on structural measurements; if the model is inaccurate, researchers risks misidentifying growth patterns, species classifications, or environmental health indicators."
        )
