import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, max_error
import os

# set the default parameters for the plots
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"

# Check if 'data' directory exists, create it if not
if not os.path.exists("./data"):
    os.makedirs("./data")

# Load the dataset if it exists
if os.path.exists("./data/dataset.csv"):
    df = pd.read_csv("./data/dataset.csv", encoding="latin-1")
else:
    df = pd.DataFrame()

# Sidebar and navigation
with st.sidebar:
    st.title(":bar_chart: ML Analysis and Training")
    st.markdown("By [Dipanshu Gupta](https://github.com/Voidd999)")
    st.markdown(
        "This app is an ML analysis and training app built with Streamlit. It allows you to upload a dataset, perform EDA, and build a model on **smaller datasets**."
    )
    choice = st.radio("Navigate", ["Upload", "EDA", "Model"])
    st.info(
        "This app is currently in experimental stage. Deployment is not recommended as data security concerns have not been addressed."
    )

if choice == "Upload":
    st.title("Upload A Dataset")
    st.markdown("Upload a dataset")
    file = st.file_uploader("Upload a CSV file")
    if file:
        df = pd.read_csv(file, encoding="latin-1")
        df.to_csv("./data/dataset.csv", index=False)
        st.success("File uploaded successfully!")
        st.subheader(f"{file.name}")
    else:
        with st.sidebar:
            st.info("Upload a CSV file to begin")
    if not df.empty:
        st.header(f"Dataset Loaded")
        st.write(df.head())

#
if choice == "EDA":
    st.title("Exploratory Data Analysis")
    if not df.empty:
        st.subheader("Summary Statistics:")
        st.dataframe(df)
        st.write(df.describe())
        st.subheader("Missing Values:")
        missing_values = df.isnull().sum()
        st.dataframe(missing_values)

        st.subheader("Correlation")
        correlation_plot_type = st.selectbox(
            "Select Plot Type",
            [
                "Heatmap",
                "Scatterplot",
                "Bubble Chart",
                "",
            ],  # "Heatmap", "Scatterplot", "Bubble Chart
        )

        if correlation_plot_type == "Heatmap":
            numeric_cols = df.select_dtypes(include=["number"]).columns
            corr_matrix = df[numeric_cols].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, cmap="crest", fmt=".2f", ax=ax)
            st.pyplot(fig)
        elif correlation_plot_type == "Scatterplot":
            st.subheader("Select Two Features:")
            feature1 = st.selectbox("Feature 1", df.columns)
            feature2 = st.selectbox("Feature 2", df.columns)

            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=feature1, y=feature2, ax=ax)
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, horizontalalignment="right"
            )
            st.pyplot(fig)
        elif correlation_plot_type == "Bubble Chart":
            st.subheader("Select Three Features:")
            feature1 = st.selectbox("Feature 1", df.columns)
            feature2 = st.selectbox("Feature 2", df.columns)
            feature3 = st.selectbox("Bubble Size (Feature 3)", df.columns)

            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=feature1, y=feature2, size=feature3, ax=ax)
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, horizontalalignment="right"
            )
            st.pyplot(fig)

        st.subheader("Data Visualization:")
        # Creating histograms for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        cols = st.selectbox("Choose a Numeric Column", numeric_cols)
        st.subheader(f"Histogram of {cols}")
        fig, ax = plt.subplots()
        sns.histplot(df[cols], ax=ax, kde=True, element="poly")
        st.pyplot(fig)

        # Creating bar charts for categorical columns
        st.subheader("Choose a Categorical Feature for Comparison")
        categorical_feature = st.selectbox(
            "Select a Categorical Feature", df.select_dtypes(include="object").columns
        )

        st.subheader(f"Comparison of {categorical_feature} with Other Features")
        numeric_columns = df.select_dtypes(include=["number"]).columns
        comparison_feature = st.selectbox(
            "Select a Numeric Feature for Comparison", numeric_columns
        )

        # Plot type selection
        plot_type = st.selectbox(
            "Select Plot Type", ["Bar Plot", "Box Plot", "Violin Plot"]
        )

        if plot_type == "Bar Plot":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=df, x=categorical_feature, y=comparison_feature, ax=ax)
            ax.set_title(f"{categorical_feature} vs {comparison_feature}")
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, horizontalalignment="right"
            )
            st.pyplot(fig)
        elif plot_type == "Box Plot":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, x=categorical_feature, y=comparison_feature, ax=ax)
            ax.set_title(f"{categorical_feature} vs {comparison_feature}")
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, horizontalalignment="right"
            )
            st.pyplot(fig)
        elif plot_type == "Violin Plot":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(data=df, x=categorical_feature, y=comparison_feature, ax=ax)
            ax.set_title(f"{categorical_feature} vs {comparison_feature}")
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, horizontalalignment="right"
            )
            st.pyplot(fig)

    else:
        st.info("Please upload a dataset to begin EDA.")

if choice == "Model":
    st.title("Model Building and Evaluation")
    if not df.empty:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        # target column
        chosen_target = st.selectbox(
            "Choose the Target Column(Numeric cols only)", numeric_cols
        )
        st.write(f"Chosen Target - **{chosen_target}**")
        categorical_columns = [col for col in df.columns if df[col].dtype == "object"]

        # Preprocess categorical columns by stripping whitespaces and converting to string
        for col in categorical_columns:
            df[col] = df[col].str.strip().astype(str)
        # model selection
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = st.selectbox(
            "Select a  Model",
            [
                "Random Forest",
                "Linear Regression",
                "Gradient Boosting",
                "HistGradient Boosting",
                "K-Nearest Neighbors",
            ],
        )
        categorical_columns = [col for col in df.columns if df[col].dtype == "object"]
        # Encode categorical features using one-hot encoding
        df_encoded = pd.get_dummies(df, columns=categorical_columns)

        impute_data = st.checkbox("Impute Missing Values Automatically", value=True)
        if impute_data:
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="mean")
            df_encoded_imputed = pd.DataFrame(
                imputer.fit_transform(df_encoded), columns=df_encoded.columns
            )
            st.info("Missing values imputed automatically.")
            show_imputed_df = st.checkbox("Show Imputed DataFrame")
            if show_imputed_df:
                st.subheader("Imputed DataFrame")
                st.dataframe(df_encoded_imputed)
            df_encoded = df_encoded_imputed
        else:
            df_encoded = df_encoded

        show_encoded_df = st.checkbox("Show Encoded DataFrame")
        if show_encoded_df:
            st.subheader("One Hot Encoded DataFrame")
            st.dataframe(df_encoded)
        # Spliting data into train and test sets
        X = df_encoded.drop(columns=[chosen_target])
        y = df_encoded[chosen_target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Create and train the selected model
        if model == "Linear Regression":
            model = LinearRegression()
        elif model == "Random Forest":
            model = RandomForestRegressor()
        elif model == "Gradient Boosting":
            model = GradientBoostingRegressor()
        elif model == "HistGradient Boosting":
            model = HistGradientBoostingRegressor()
        elif model == "K-Nearest Neighbors":
            model = KNeighborsRegressor()

        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        st.subheader("Model Evaluation")

        st.write(
            f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}"
        )
        st.write(f"Score: {model.score(X_test, y_test):.2f}")
        st.write(f"max_error: {max_error(y_test, y_pred):.2f}")
        loss = st.checkbox("Show Loss")
        if loss:
            st.subheader("Loss")
            fig, ax = plt.subplots()
            sns.histplot(
                y_test - y_pred, ax=ax, element="poly", label="Loss", color="green"
            )
            st.pyplot(fig)

        features = st.checkbox("Show Features in Model")
        if features:
            st.subheader("Feature Importance")
            if model in ["Random Forest", "Gradient Boosting"]:
                importances = model.feature_importances_
                feature_importance_df = pd.DataFrame(
                    {"Feature": X.columns, "Importance": importances}
                )
                st.dataframe(feature_importance_df)
            else:
                st.info("Feature importance not available for this model.")

        prediction = st.checkbox("Make Predictions on New Data (Experimental)")
        if prediction:
            st.title("Test Model Predictions")
            st.info("Enter values for the encoded features to get predictions.")

            # Create input values
            feature_values = {}
            for feature in X.columns:
                feature_values[feature] = st.number_input(
                    f"Enter {feature}",
                    min_value=float(X[feature].min()),  # Convert to float
                    max_value=float(X[feature].max()),
                )

            if st.button("Get Prediction"):
                input_data = pd.DataFrame([feature_values])
                prediction = model.predict(input_data)
                st.subheader(f"Predicted {chosen_target}")
                st.write(f"The predicted **{chosen_target}** is: {prediction[0]:.2f}")

                # Display feature values
                st.subheader("Feature Values")
                st.write(input_data.T)

        if st.button("Save Model"):
            import pickle

            pickle.dump(model, open("./Data/model.pkl", "wb"))
            st.success("Model saved successfully")
            download_path = "./Data/model.pkl"

            with open(download_path, "rb") as f:
                st.download_button(
                    "Download Model", f, file_name="model.pkl", mime="text/csv"
                )
    else:
        st.info("Please upload a dataset to begin model building.")
