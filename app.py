import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI & ML Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "original_df" not in st.session_state:
    st.session_state.original_df = None

if "filename" not in st.session_state:
    st.session_state.filename = ""


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_dataset(uploaded_file):
    """Load CSV or Excel file safely."""

    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower()

    try:
        if filename.endswith(".csv"):
            return pd.read_csv(uploaded_file)

        elif filename.endswith(".xlsx"):
            return pd.read_excel(uploaded_file, engine="openpyxl")

        elif filename.endswith(".xls"):
            return pd.read_excel(uploaded_file, engine="xlrd")

        else:
            st.error("Unsupported file type. Please upload CSV, XLSX, or XLS.")
            return None

    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return None


def missing_report(df):
    """Create missing value report."""

    report = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isna().sum().values,
        "Missing %": (
            df.isna().mean().values * 100
        ).round(2),
        "Data Type": df.dtypes.astype(str).values,
    })

    return report.sort_values(
        "Missing Count",
        ascending=False
    ).reset_index(drop=True)


def duplicate_report(df):
    """Create duplicate report."""

    total_duplicates = int(df.duplicated().sum())

    return pd.DataFrame({
        "Metric": [
            "Total Rows",
            "Duplicate Rows",
            "Duplicate %",
        ],
        "Value": [
            len(df),
            total_duplicates,
            round(
                (total_duplicates / len(df) * 100)
                if len(df) > 0 else 0,
                2
            ),
        ],
    })


def outlier_report(df):
    """IQR outlier report for numeric columns."""

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    results = []

    for column in numeric_columns:

        series = df[column].dropna()

        if len(series) == 0:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = (
            (series < lower) |
            (series > upper)
        ).sum()

        results.append({
            "Column": column,
            "Q1": round(q1, 4),
            "Q3": round(q3, 4),
            "IQR": round(iqr, 4),
            "Lower Bound": round(lower, 4),
            "Upper Bound": round(upper, 4),
            "Outliers": int(outliers),
            "Outlier %": round(
                outliers / len(series) * 100,
                2
            ),
        })

    if not results:
        return pd.DataFrame(
            columns=[
                "Column",
                "Q1",
                "Q3",
                "IQR",
                "Lower Bound",
                "Upper Bound",
                "Outliers",
                "Outlier %",
            ]
        )

    return pd.DataFrame(results)


def make_download_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def make_download_excel(df):
    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:
        df.to_excel(
            writer,
            index=False,
            sheet_name="Cleaned Data"
        )

    return output.getvalue()


def clean_missing_values(df, method, constant_value="0"):
    """Apply missing value strategy."""

    result = df.copy()

    numeric_columns = result.select_dtypes(
        include=np.number
    ).columns

    categorical_columns = result.select_dtypes(
        exclude=np.number
    ).columns

    if method == "Mean":
        for col in numeric_columns:
            result[col] = result[col].fillna(
                result[col].mean()
            )

    elif method == "Median":
        for col in numeric_columns:
            result[col] = result[col].fillna(
                result[col].median()
            )

    elif method == "Mode":
        for col in result.columns:
            mode = result[col].mode(dropna=True)

            if not mode.empty:
                result[col] = result[col].fillna(
                    mode.iloc[0]
                )

    elif method == "Constant":
        for col in result.columns:

            if pd.api.types.is_numeric_dtype(
                result[col]
            ):
                try:
                    value = float(constant_value)
                except Exception:
                    value = 0
            else:
                value = constant_value

            result[col] = result[col].fillna(value)

    elif method == "Forward Fill":
        result = result.ffill()

    elif method == "Backward Fill":
        result = result.bfill()

    elif method == "Drop Rows":
        result = result.dropna()

    return result


def prepare_ml_data(df, target_column):
    """Prepare dataset for machine learning."""

    data = df.copy()

    if target_column not in data.columns:
        raise ValueError("Target column not found.")

    data = data.dropna(subset=[target_column])

    if len(data) < 10:
        raise ValueError(
            "Not enough rows for machine learning."
        )

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Remove completely empty columns
    X = X.dropna(axis=1, how="all")

    if X.shape[1] == 0:
        raise ValueError(
            "No usable feature columns were found."
        )

    # Convert categorical columns into dummy variables
    X = pd.get_dummies(
        X,
        drop_first=True,
        dtype=float
    )

    # Fill remaining missing values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median(numeric_only=True)
    )

    X = X.fillna(0)

    return X, y


# ============================================================
# HEADER
# ============================================================

st.title("📊 AI & ML Analytics Dashboard")

st.markdown(
    """
    Upload your dataset and perform **EDA, data quality analysis,
    visualization, outlier detection, data cleaning, and machine learning**
    from one dashboard.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📁 Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"]
    )

    st.info(
        "Supported files: CSV, XLSX, XLS"
    )

    if st.session_state.df is not None:

        st.success(
            f"Loaded: {st.session_state.filename}"
        )

        if st.button(
            "🔄 Reset Dataset",
            use_container_width=True
        ):

            if st.session_state.original_df is not None:

                st.session_state.df = (
                    st.session_state.original_df.copy()
                )

                st.success("Dataset reset.")

                st.rerun()


# ============================================================
# LOAD DATASET
# ============================================================

if uploaded_file is not None:

    # Load only when a new file is uploaded
    if (
        st.session_state.filename
        != uploaded_file.name
    ):

        loaded_df = load_dataset(
            uploaded_file
        )

        if loaded_df is not None:

            st.session_state.df = (
                loaded_df.copy()
            )

            st.session_state.original_df = (
                loaded_df.copy()
            )

            st.session_state.filename = (
                uploaded_file.name
            )

            st.success(
                "Dataset uploaded successfully!"
            )


# ============================================================
# NO DATA
# ============================================================

if st.session_state.df is None:

    st.info(
        "👈 Upload a CSV or Excel file from the sidebar to begin."
    )

    st.markdown(
        """
        ### Dashboard Features

        - 📌 Dataset overview
        - 🔎 Data preview
        - 🧹 Missing-value analysis
        - 🔁 Duplicate detection
        - 📊 Statistical summary
        - 📈 Histograms
        - 📦 Box plots
        - 📊 Bar charts
        - 🔥 Correlation heatmap
        - 🔗 Scatter plots
        - 🚨 IQR outlier detection
        - 🧹 Missing-value cleaning
        - 🗑️ Duplicate removal
        - 🤖 Classification
        - 📉 Regression
        - 💾 CSV / Excel download
        """
    )

    st.stop()


# ============================================================
# CURRENT DATA
# ============================================================

df = st.session_state.df


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Rows",
        f"{df.shape[0]:,}"
    )

with col2:
    st.metric(
        "Columns",
        f"{df.shape[1]:,}"
    )

with col3:
    st.metric(
        "Missing Cells",
        f"{int(df.isna().sum().sum()):,}"
    )

with col4:
    st.metric(
        "Duplicate Rows",
        f"{int(df.duplicated().sum()):,}"
    )


# ============================================================
# TABS
# ============================================================

(
    overview_tab,
    quality_tab,
    eda_tab,
    outlier_tab,
    cleaning_tab,
    ml_tab,
    download_tab,
) = st.tabs(
    [
        "📋 Overview",
        "🔎 Data Quality",
        "📊 EDA & Visualization",
        "🚨 Outliers",
        "🧹 Cleaning",
        "🤖 Machine Learning",
        "💾 Download",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with overview_tab:

    st.header("📋 Dataset Overview")

    st.subheader("Data Preview")

    rows_to_show = st.slider(
        "Rows to display",
        min_value=5,
        max_value=min(100, max(5, len(df))),
        value=min(10, max(5, len(df))),
    )

    st.dataframe(
        df.head(rows_to_show),
        use_container_width=True,
    )

    st.subheader("Column Information")

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(dtype)
            for dtype in df.dtypes
        ],
        "Non-Null Count": [
            int(df[col].notna().sum())
            for col in df.columns
        ],
        "Missing Count": [
            int(df[col].isna().sum())
            for col in df.columns
        ],
        "Unique Values": [
            int(df[col].nunique(dropna=True))
            for col in df.columns
        ],
    })

    st.dataframe(
        column_info,
        use_container_width=True,
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(
            include="all"
        ).transpose(),
        use_container_width=True,
    )


# ============================================================
# DATA QUALITY
# ============================================================

with quality_tab:

    st.header("🔎 Data Quality")

    st.subheader("Missing Values")

    missing_df = missing_report(df)

    st.dataframe(
        missing_df,
        use_container_width=True,
    )

    missing_total = int(
        df.isna().sum().sum()
    )

    if missing_total == 0:
        st.success(
            "✅ No missing values found."
        )
    else:
        st.warning(
            f"⚠️ {missing_total:,} missing cells found."
        )

    st.subheader("Duplicate Analysis")

    duplicate_df = duplicate_report(df)

    st.dataframe(
        duplicate_df,
        use_container_width=True,
    )

    if int(df.duplicated().sum()) == 0:
        st.success(
            "✅ No duplicate rows found."
        )
    else:
        st.warning(
            "⚠️ Duplicate rows detected."
        )

    st.subheader("Value Counts")

    selected_value_column = st.selectbox(
        "Select a column",
        df.columns,
        key="value_count_column",
    )

    if selected_value_column:

        value_counts = (
            df[selected_value_column]
            .value_counts(dropna=False)
            .head(50)
            .reset_index()
        )

        value_counts.columns = [
            selected_value_column,
            "Count"
        ]

        st.dataframe(
            value_counts,
            use_container_width=True,
        )


# ============================================================
# EDA
# ============================================================

with eda_tab:

    st.header("📊 Exploratory Data Analysis")

    numeric_columns = list(
        df.select_dtypes(
            include=np.number
        ).columns
    )

    categorical_columns = list(
        df.select_dtypes(
            exclude=np.number
        ).columns
    )

    chart_type = st.selectbox(
        "Choose visualization",
        [
            "Histogram",
            "Box Plot",
            "Bar Chart",
            "Correlation Heatmap",
            "Scatter Plot",
        ],
    )

    if chart_type == "Histogram":

        if not numeric_columns:

            st.warning(
                "No numeric columns available."
            )

        else:

            column = st.selectbox(
                "Select numeric column",
                numeric_columns,
                key="hist_column",
            )

            bins = st.slider(
                "Number of bins",
                5,
                100,
                30,
            )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            ax.hist(
                df[column].dropna(),
                bins=bins,
            )

            ax.set_title(
                f"Histogram - {column}"
            )

            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")

            st.pyplot(fig)

            plt.close(fig)

    elif chart_type == "Box Plot":

        if not numeric_columns:

            st.warning(
                "No numeric columns available."
            )

        else:

            column = st.selectbox(
                "Select numeric column",
                numeric_columns,
                key="box_column",
            )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            sns.boxplot(
                x=df[column],
                ax=ax,
            )

            ax.set_title(
                f"Box Plot - {column}"
            )

            st.pyplot(fig)

            plt.close(fig)

    elif chart_type == "Bar Chart":

        if not categorical_columns:

            st.warning(
                "No categorical columns available."
            )

        else:

            column = st.selectbox(
                "Select categorical column",
                categorical_columns,
                key="bar_column",
            )

            top_n = st.slider(
                "Top categories",
                5,
                30,
                10,
            )

            counts = (
                df[column]
                .astype(str)
                .value_counts()
                .head(top_n)
            )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            counts.plot(
                kind="bar",
                ax=ax,
            )

            ax.set_title(
                f"Top Categories - {column}"
            )

            ax.set_xlabel(column)
            ax.set_ylabel("Count")

            plt.xticks(rotation=45)

            st.pyplot(fig)

            plt.close(fig)

    elif chart_type == "Correlation Heatmap":

        if len(numeric_columns) < 2:

            st.warning(
                "At least two numeric columns are required."
            )

        else:

            corr = df[numeric_columns].corr()

            fig, ax = plt.subplots(
                figsize=(12, 7)
            )

            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                ax=ax,
            )

            ax.set_title(
                "Correlation Heatmap"
            )

            st.pyplot(fig)

            plt.close(fig)

    elif chart_type == "Scatter Plot":

        if len(numeric_columns) < 2:

            st.warning(
                "At least two numeric columns are required."
            )

        else:

            col1, col2 = st.columns(2)

            with col1:
                x_column = st.selectbox(
                    "X-axis",
                    numeric_columns,
                    key="scatter_x",
                )

            with col2:
                y_column = st.selectbox(
                    "Y-axis",
                    numeric_columns,
                    index=1 if len(numeric_columns) > 1 else 0,
                    key="scatter_y",
                )

            plot_df = df[
                [x_column, y_column]
            ].dropna()

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            ax.scatter(
                plot_df[x_column],
                plot_df[y_column],
                alpha=0.7,
            )

            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)

            ax.set_title(
                f"{x_column} vs {y_column}"
            )

            st.pyplot(fig)

            plt.close(fig)


# ============================================================
# OUTLIERS
# ============================================================

with outlier_tab:

    st.header("🚨 Outlier Detection")

    st.write(
        "Outliers are detected using the IQR method."
    )

    outliers = outlier_report(df)

    if outliers.empty:

        st.info(
            "No numeric columns available for outlier analysis."
        )

    else:

        st.dataframe(
            outliers,
            use_container_width=True,
        )

        numeric_columns = list(
            df.select_dtypes(
                include=np.number
            ).columns
        )

        selected_outlier_column = st.selectbox(
            "Select column for outlier removal",
            numeric_columns,
            key="outlier_remove_column",
        )

        series = df[
            selected_outlier_column
        ].dropna()

        if len(series) > 0:

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outlier_count = int(
                (
                    (df[selected_outlier_column] < lower)
                    |
                    (df[selected_outlier_column] > upper)
                ).sum()
            )

            st.write(
                f"Detected outliers: **{outlier_count}**"
            )

            if st.button(
                "🗑️ Remove Outliers",
                key="remove_outliers",
            ):

                mask = (
                    (df[selected_outlier_column] >= lower)
                    &
                    (df[selected_outlier_column] <= upper)
                )

                # Keep missing values
                mask = (
                    mask
                    |
                    df[selected_outlier_column].isna()
                )

                st.session_state.df = (
                    df.loc[mask].copy()
                )

                st.success(
                    "Outliers removed successfully."
                )

                st.rerun()


# ============================================================
# CLEANING
# ============================================================

with cleaning_tab:

    st.header("🧹 Data Cleaning")

    st.subheader("Remove Duplicate Rows")

    duplicate_count = int(
        df.duplicated().sum()
    )

    st.write(
        f"Duplicate rows currently present: "
        f"**{duplicate_count}**"
    )

    if st.button(
        "🗑️ Remove Duplicate Rows"
    ):

        st.session_state.df = (
            df.drop_duplicates()
            .reset_index(drop=True)
        )

        st.success(
            "Duplicate rows removed."
        )

        st.rerun()

    st.divider()

    st.subheader(
        "Missing Value Treatment"
    )

    missing_methods = [
        "Mean",
        "Median",
        "Mode",
        "Constant",
        "Forward Fill",
        "Backward Fill",
        "Drop Rows",
    ]

    missing_method = st.selectbox(
        "Choose method",
        missing_methods,
    )

    constant_value = "0"

    if missing_method == "Constant":

        constant_value = st.text_input(
            "Constant value",
            value="0",
        )

    if st.button(
        "🧹 Apply Missing Value Treatment"
    ):

        before_missing = int(
            df.isna().sum().sum()
        )

        cleaned_df = clean_missing_values(
            df,
            missing_method,
            constant_value,
        )

        after_missing = int(
            cleaned_df.isna().sum().sum()
        )

        st.session_state.df = cleaned_df

        st.success(
            f"Cleaning complete. "
            f"Missing cells: "
            f"{before_missing:,} → {after_missing:,}"
        )

        st.rerun()

    st.divider()

    st.subheader("Current Data Preview")

    st.dataframe(
        st.session_state.df.head(100),
        use_container_width=True,
    )


# ============================================================
# MACHINE LEARNING
# ============================================================

with ml_tab:

    st.header("🤖 Machine Learning")

    st.write(
        "Build a simple classification or regression model "
        "using the current cleaned dataset."
    )

    if len(df.columns) < 2:

        st.warning(
            "At least two columns are required."
        )

    else:

        target_column = st.selectbox(
            "Select Target Column",
            df.columns,
            key="ml_target",
        )

        problem_type = st.radio(
            "Problem Type",
            [
                "Classification",
                "Regression",
            ],
            horizontal=True,
        )

        if problem_type == "Classification":

            model_name = st.selectbox(
                "Classification Model",
                [
                    "Random Forest",
                    "Logistic Regression",
                ],
            )

        else:

            model_name = st.selectbox(
                "Regression Model",
                [
                    "Random Forest",
                    "Linear Regression",
                ],
            )

        test_size = st.slider(
            "Test Size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
        )

        if st.button(
            "🚀 Train Model",
            type="primary",
        ):

            try:

                X, y = prepare_ml_data(
                    df,
                    target_column,
                )

                # Classification
                if problem_type == "Classification":

                    if y.nunique() < 2:

                        st.error(
                            "Classification requires at least "
                            "two target classes."
                        )

                    elif y.nunique() > 50:

                        st.error(
                            "The target has too many unique "
                            "values for simple classification."
                        )

                    else:

                        encoder = LabelEncoder()

                        y_encoded = encoder.fit_transform(
                            y.astype(str)
                        )

                        X_train, X_test, y_train, y_test = (
                            train_test_split(
                                X,
                                y_encoded,
                                test_size=test_size,
                                random_state=42,
                                stratify=y_encoded,
                            )
                        )

                        if model_name == "Random Forest":

                            model = RandomForestClassifier(
                                n_estimators=200,
                                random_state=42,
                                n_jobs=-1,
                            )

                        else:

                            model = LogisticRegression(
                                max_iter=2000,
                            )

                        model.fit(
                            X_train,
                            y_train,
                        )

                        predictions = model.predict(
                            X_test
                        )

                        accuracy = accuracy_score(
                            y_test,
                            predictions,
                        )

                        st.success(
                            f"Model trained successfully. "
                            f"Accuracy: {accuracy:.2%}"
                        )

                        st.subheader(
                            "Classification Metrics"
                        )

                        st.metric(
                            "Accuracy",
                            f"{accuracy:.2%}",
                        )

                        report = classification_report(
                            y_test,
                            predictions,
                            output_dict=True,
                            zero_division=0,
                        )

                        st.dataframe(
                            pd.DataFrame(report).transpose(),
                            use_container_width=True,
                        )

                        st.subheader(
                            "Confusion Matrix"
                        )

                        cm = confusion_matrix(
                            y_test,
                            predictions,
                        )

                        fig, ax = plt.subplots(
                            figsize=(7, 5)
                        )

                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt="d",
                            cmap="Blues",
                            ax=ax,
                        )

                        ax.set_xlabel(
                            "Predicted"
                        )

                        ax.set_ylabel(
                            "Actual"
                        )

                        st.pyplot(fig)

                        plt.close(fig)

                # Regression
                else:

                    y_numeric = pd.to_numeric(
                        y,
                        errors="coerce",
                    )

                    valid_mask = (
                        y_numeric.notna()
                    )

                    X = X.loc[
                        valid_mask
                    ]

                    y_numeric = y_numeric.loc[
                        valid_mask
                    ]

                    if len(y_numeric) < 10:

                        st.error(
                            "Not enough valid numeric target "
                            "values for regression."
                        )

                    else:

                        X_train, X_test, y_train, y_test = (
                            train_test_split(
                                X,
                                y_numeric,
                                test_size=test_size,
                                random_state=42,
                            )
                        )

                        if model_name == "Random Forest":

                            model = RandomForestRegressor(
                                n_estimators=200,
                                random_state=42,
                                n_jobs=-1,
                            )

                        else:

                            model = LinearRegression()

                        model.fit(
                            X_train,
                            y_train,
                        )

                        predictions = model.predict(
                            X_test
                        )

                        mae = mean_absolute_error(
                            y_test,
                            predictions,
                        )

                        rmse = np.sqrt(
                            mean_squared_error(
                                y_test,
                                predictions,
                            )
                        )

                        r2 = r2_score(
                            y_test,
                            predictions,
                        )

                        st.success(
                            "Regression model trained successfully."
                        )

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "MAE",
                                f"{mae:.4f}",
                            )

                        with col2:
                            st.metric(
                                "RMSE",
                                f"{rmse:.4f}",
                            )

                        with col3:
                            st.metric(
                                "R² Score",
                                f"{r2:.4f}",
                            )

                        st.subheader(
                            "Actual vs Predicted"
                        )

                        prediction_df = pd.DataFrame({
                            "Actual": y_test.values,
                            "Predicted": predictions,
                        })

                        st.dataframe(
                            prediction_df.head(100),
                            use_container_width=True,
                        )

                        fig, ax = plt.subplots(
                            figsize=(9, 6)
                        )

                        ax.scatter(
                            y_test,
                            predictions,
                            alpha=0.7,
                        )

                        minimum = min(
                            y_test.min(),
                            predictions.min(),
                        )

                        maximum = max(
                            y_test.max(),
                            predictions.max(),
                        )

                        ax.plot(
                            [minimum, maximum],
                            [minimum, maximum],
                            linestyle="--",
                        )

                        ax.set_xlabel(
                            "Actual"
                        )

                        ax.set_ylabel(
                            "Predicted"
                        )

                        ax.set_title(
                            "Actual vs Predicted"
                        )

                        st.pyplot(fig)

                        plt.close(fig)

            except Exception as e:

                st.error(
                    f"Machine learning could not be completed: {e}"
                )


# ============================================================
# DOWNLOAD
# ============================================================

with download_tab:

    st.header("💾 Download Cleaned Dataset")

    final_df = st.session_state.df

    st.write(
        f"Current dataset size: "
        f"**{final_df.shape[0]:,} rows × "
        f"{final_df.shape[1]:,} columns**"
    )

    st.subheader("Preview")

    st.dataframe(
        final_df.head(100),
        use_container_width=True,
    )

    st.divider()

    csv_data = make_download_csv(
        final_df
    )

    excel_data = make_download_excel(
        final_df
    )

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            label="⬇️ Download CSV",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:

        st.download_button(
            label="⬇️ Download Excel",
            data=excel_data,
            file_name="cleaned_dataset.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI & ML Analytics Dashboard • "
    "EDA • Data Quality • Cleaning • Visualization • Machine Learning"
)