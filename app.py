# app.py
# AI and ML Analytics Dashboard
# Streamlit application
# The cleaning pipeline always rebuilds the cleaned dataset from the
# original uploaded dataset, so changing a cleaning option and applying
# again works correctly.

import io
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="AI and ML Analytics Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>

    /* ---------------------------------------------------------------
       GLOBAL
       --------------------------------------------------------------- */

    * {
        font-family: "Times New Roman", Times, serif !important;
        box-sizing: border-box;
    }

    html, body, [class*="css"] {
        font-family: "Times New Roman", Times, serif !important;
    }

    .stApp {
        background: #f7f8fa;
        color: #20242a;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* ---------------------------------------------------------------
       HEADINGS
       --------------------------------------------------------------- */

    h1, h2, h3, h4, h5, h6 {
        font-family: "Times New Roman", Times, serif !important;
        color: #172033 !important;
        font-weight: 700 !important;
    }

    h1 {
        font-size: 34px !important;
        border-bottom: 2px solid #d8dde6;
        padding-bottom: 12px;
    }

    h2 {
        font-size: 27px !important;
        margin-top: 25px !important;
    }

    h3 {
        font-size: 22px !important;
        margin-top: 18px !important;
    }

    p, label, span, div {
        font-family: "Times New Roman", Times, serif !important;
    }

    /* ---------------------------------------------------------------
       REMOVE EMOJI / ICON-LIKE DECORATIONS FROM APP
       --------------------------------------------------------------- */

    .stApp svg {
        font-family: "Times New Roman", Times, serif !important;
    }

    /* ---------------------------------------------------------------
       FILE UPLOADER
       --------------------------------------------------------------- */

    [data-testid="stFileUploader"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #ffffff !important;
        border: 1px solid #cdd3dc !important;
        border-radius: 12px !important;
        padding: 22px !important;
        min-height: 145px !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        font-family: "Times New Roman", Times, serif !important;
        color: #333b49 !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: #ffffff !important;
        border: 1px solid #b9c1cc !important;
        border-radius: 8px !important;
        color: #172033 !important;
        font-family: "Times New Roman", Times, serif !important;
        font-size: 16px !important;
        padding: 8px 18px !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        border-color: #5b6675 !important;
        color: #000000 !important;
    }

    /* ---------------------------------------------------------------
       BUTTONS
       --------------------------------------------------------------- */

    .stButton > button,
    .stDownloadButton > button {
        width: 100%;
        min-height: 42px;
        border-radius: 7px;
        border: 1px solid #b9c1cc;
        background: #ffffff;
        color: #172033;
        font-family: "Times New Roman", Times, serif !important;
        font-size: 16px !important;
        font-weight: 600;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #3f4a5a;
        background: #f0f2f5;
        color: #000000;
    }

    /* Primary buttons */
    .primary-button button {
        background: #26364a !important;
        color: #ffffff !important;
        border: 1px solid #26364a !important;
    }

    .primary-button button:hover {
        background: #172536 !important;
        color: #ffffff !important;
    }

    /* ---------------------------------------------------------------
       INPUTS
       --------------------------------------------------------------- */

    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid #cbd1da !important;
        border-radius: 7px !important;
    }

    input, textarea {
        background: #ffffff !important;
        color: #20242a !important;
        border-radius: 7px !important;
    }

    /* ---------------------------------------------------------------
       METRIC CARDS
       --------------------------------------------------------------- */

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d4d9e1;
        border-radius: 10px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #5c6572 !important;
    }

    [data-testid="stMetricValue"] {
        color: #172033 !important;
    }

    /* ---------------------------------------------------------------
       DATAFRAME
       --------------------------------------------------------------- */

    [data-testid="stDataFrame"] {
        border: 1px solid #d5dbe3;
        border-radius: 8px;
    }

    /* ---------------------------------------------------------------
       INFO / SUCCESS / WARNING
       --------------------------------------------------------------- */

    .stAlert {
        font-family: "Times New Roman", Times, serif !important;
        border-radius: 8px !important;
    }

    /* ---------------------------------------------------------------
       SECTION CARDS
       --------------------------------------------------------------- */

    .section-box {
        background: #ffffff;
        border: 1px solid #d8dde5;
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #172033;
        margin-bottom: 8px;
    }

    .section-description {
        font-size: 15px;
        color: #606a78;
        margin-bottom: 14px;
    }

    /* ---------------------------------------------------------------
       CLEANING STATUS
       --------------------------------------------------------------- */

    .clean-status {
        background: #f3f5f8;
        border: 1px solid #d3d8e0;
        border-radius: 8px;
        padding: 14px 16px;
        margin: 10px 0 18px 0;
    }

    .clean-status-title {
        font-weight: 700;
        color: #172033;
        font-size: 17px;
    }

    .clean-status-text {
        color: #505967;
        font-size: 15px;
        margin-top: 5px;
    }

    /* ---------------------------------------------------------------
       TOP NAVIGATION
       --------------------------------------------------------------- */

    div[data-testid="stTabs"] button {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 16px !important;
        color: #303846 !important;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #172033 !important;
        font-weight: 700 !important;
    }

    /* ---------------------------------------------------------------
       FOOTER
       --------------------------------------------------------------- */

    .footer {
        border-top: 1px solid #d9dee6;
        margin-top: 35px;
        padding-top: 15px;
        text-align: center;
        color: #707987;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------

DEFAULT_STATE = {
    "original_df": None,
    "cleaned_df": None,
    "file_name": "",
    "audit_log": [],
    "training_result": None,
    "prediction_result": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def read_uploaded_file(uploaded_file):
    """Read CSV/XLSX/XLS safely."""
    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".csv"):
            try:
                return pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding="latin1")

        if file_name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file, engine="openpyxl")

        if file_name.endswith(".xls"):
            return pd.read_excel(uploaded_file, engine="xlrd")

        st.error("Unsupported file type. Please upload CSV, XLSX, or XLS.")
        return None

    except Exception as e:
        st.error(f"Unable to read the file: {str(e)}")
        return None


def numeric_columns(df):
    if df is None:
        return []
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df):
    if df is None:
        return []
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def make_download_file(df, file_format):
    """Create downloadable CSV/XLSX bytes."""
    if file_format == "CSV":
        return df.to_csv(index=False).encode("utf-8"), "csv", "text/csv"

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned_Data")

    output.seek(0)
    return (
        output.getvalue(),
        "xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def add_audit(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.audit_log.append(f"{timestamp} - {message}")


def dataset_metrics(df):
    if df is None:
        return {
            "rows": 0,
            "columns": 0,
            "numeric": 0,
            "categorical": 0,
            "missing": 0,
            "duplicates": 0,
        }

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric": len(numeric_columns(df)),
        "categorical": len(categorical_columns(df)),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


# ---------------------------------------------------------------------
# CLEANING ENGINE
# ---------------------------------------------------------------------

def apply_cleaning_pipeline(
    source_df,
    duplicate_treatment="Do nothing",
    categorical_treatment="Do nothing",
    missing_treatment="Do nothing",
    invalid_value_treatment="Do nothing",
    constant_value="0",
    numeric_treatment="Do nothing",
    outlier_treatment="Do nothing",
    outlier_column="",
    outlier_multiplier=1.5,
):
    """
    IMPORTANT:
    Every cleaning operation starts from source_df.

    This means:
        Original -> selected settings -> cleaned result

    It does NOT do:
        previous cleaned result -> new settings -> another result

    Therefore changing a dropdown and pressing Apply again always works.
    """

    if source_df is None:
        return None, []

    df = source_df.copy(deep=True)
    log = []

    original_rows = len(df)
    original_columns = len(df.columns)

    # ---------------------------------------------------------------
    # 1. DUPLICATES
    # ---------------------------------------------------------------

    if duplicate_treatment == "Remove duplicate rows":
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df)

        log.append(
            f"Duplicate Treatment: removed {removed:,} duplicate row(s)."
        )

    # ---------------------------------------------------------------
    # 2. MISSING VALUES
    # ---------------------------------------------------------------

    if missing_treatment == "Mean for numeric columns":
        cols = numeric_columns(df)

        for col in cols:
            if df[col].isna().any():
                value = df[col].mean()

                if pd.notna(value):
                    df[col] = df[col].fillna(value)

        log.append("Missing Values: numeric missing values filled using mean.")

    elif missing_treatment == "Median for numeric columns":
        cols = numeric_columns(df)

        for col in cols:
            if df[col].isna().any():
                value = df[col].median()

                if pd.notna(value):
                    df[col] = df[col].fillna(value)

        log.append("Missing Values: numeric missing values filled using median.")

    elif missing_treatment == "Mode for all columns":
        for col in df.columns:
            if df[col].isna().any():
                mode_values = df[col].mode(dropna=True)

                if len(mode_values) > 0:
                    df[col] = df[col].fillna(mode_values.iloc[0])

        log.append("Missing Values: missing values filled using mode.")

    elif missing_treatment == "Constant value":
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        value = float(constant_value)
                        if value.is_integer():
                            value = int(value)
                    except Exception:
                        value = 0
                else:
                    value = str(constant_value)

                df[col] = df[col].fillna(value)

        log.append(
            f"Missing Values: missing values filled using constant '{constant_value}'."
        )

    elif missing_treatment == "Forward fill":
        df = df.ffill()
        log.append("Missing Values: forward fill applied.")

    elif missing_treatment == "Backward fill":
        df = df.bfill()
        log.append("Missing Values: backward fill applied.")

    elif missing_treatment == "Drop rows containing missing values":
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        removed = before - len(df)

        log.append(
            f"Missing Values: removed {removed:,} row(s) containing missing values."
        )

    # ---------------------------------------------------------------
    # 3. CATEGORICAL VALUES
    # ---------------------------------------------------------------

    if categorical_treatment == "Convert categorical columns to string":
        for col in categorical_columns(df):
            df[col] = df[col].astype("string")

        log.append("Categorical Treatment: categorical columns converted to string.")

    elif categorical_treatment == "Fill categorical missing values with mode":
        for col in categorical_columns(df):
            if df[col].isna().any():
                modes = df[col].mode(dropna=True)

                if len(modes) > 0:
                    df[col] = df[col].fillna(modes.iloc[0])

        log.append(
            "Categorical Treatment: categorical missing values filled using mode."
        )

    # ---------------------------------------------------------------
    # 4. INVALID VALUES
    # ---------------------------------------------------------------

    if invalid_value_treatment == "Replace negative numeric values with zero":
        changed = 0

        for col in numeric_columns(df):
            mask = df[col] < 0
            changed += int(mask.sum())
            df.loc[mask, col] = 0

        log.append(
            f"Invalid Value Treatment: {changed:,} negative numeric value(s) replaced with zero."
        )

    elif invalid_value_treatment == "Replace infinite values with missing":
        numeric_cols = numeric_columns(df)

        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].replace(
                [np.inf, -np.inf],
                np.nan,
            )

        log.append(
            "Invalid Value Treatment: infinite numeric values replaced with missing values."
        )

    # ---------------------------------------------------------------
    # 5. NUMERIC TREATMENT
    # ---------------------------------------------------------------

    if numeric_treatment == "Convert numeric-looking text":
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                converted = pd.to_numeric(df[col], errors="coerce")

                original_non_null = df[col].notna().sum()

                if original_non_null > 0:
                    success_ratio = converted.notna().sum() / original_non_null

                    if success_ratio >= 0.80:
                        df[col] = converted

        log.append(
            "Numeric Treatment: numeric-looking text columns converted where appropriate."
        )

    # ---------------------------------------------------------------
    # 6. OUTLIERS
    # ---------------------------------------------------------------

    if (
        outlier_treatment == "Remove IQR outliers"
        and outlier_column
        and outlier_column in df.columns
        and pd.api.types.is_numeric_dtype(df[outlier_column])
    ):
        series = df[outlier_column]

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if pd.notna(iqr) and iqr > 0:
            lower = q1 - outlier_multiplier * iqr
            upper = q3 + outlier_multiplier * iqr

            before = len(df)

            mask = series.isna() | series.between(lower, upper)

            df = df.loc[mask].reset_index(drop=True)

            removed = before - len(df)

            log.append(
                f"Outlier Treatment: removed {removed:,} row(s) from '{outlier_column}' "
                f"using IQR bounds [{lower:.4f}, {upper:.4f}]."
            )

    # ---------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------

    df = df.reset_index(drop=True)

    final_rows = len(df)

    log.insert(
        0,
        f"Cleaning completed: {original_rows:,} rows -> {final_rows:,} rows; "
        f"{original_columns:,} columns retained.",
    )

    return df, log


# ---------------------------------------------------------------------
# CLEANING PREVIEW
# ---------------------------------------------------------------------

def get_cleaning_preview(
    source_df,
    duplicate_treatment,
    categorical_treatment,
    missing_treatment,
    invalid_value_treatment,
    constant_value,
    numeric_treatment,
    outlier_treatment,
    outlier_column,
    outlier_multiplier,
):
    if source_df is None:
        return None, "No dataset loaded."

    try:
        cleaned, log = apply_cleaning_pipeline(
            source_df=source_df,
            duplicate_treatment=duplicate_treatment,
            categorical_treatment=categorical_treatment,
            missing_treatment=missing_treatment,
            invalid_value_treatment=invalid_value_treatment,
            constant_value=constant_value,
            numeric_treatment=numeric_treatment,
            outlier_treatment=outlier_treatment,
            outlier_column=outlier_column,
            outlier_multiplier=outlier_multiplier,
        )

        if cleaned is None:
            return None, "No cleaning result available."

        before = dataset_metrics(source_df)
        after = dataset_metrics(cleaned)

        rows_removed = before["rows"] - after["rows"]
        missing_removed = before["missing"] - after["missing"]
        duplicate_removed = before["duplicates"] - after["duplicates"]

        message = (
            f"Preview result | "
            f"Rows: {before['rows']:,} -> {after['rows']:,} | "
            f"Rows removed: {rows_removed:,} | "
            f"Missing cells: {before['missing']:,} -> {after['missing']:,} | "
            f"Duplicates: {before['duplicates']:,} -> {after['duplicates']:,}"
        )

        return cleaned, message

    except Exception as e:
        return None, f"Cleaning preview failed: {str(e)}"


# ---------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------

def show_overview(df):
    st.subheader("Dataset Overview")

    metrics = dataset_metrics(df)

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Rows", f"{metrics['rows']:,}")
    c2.metric("Columns", f"{metrics['columns']:,}")
    c3.metric("Numeric Columns", f"{metrics['numeric']:,}")
    c4.metric("Categorical Columns", f"{metrics['categorical']:,}")
    c5.metric("Missing Cells", f"{metrics['missing']:,}")
    c6.metric("Duplicate Rows", f"{metrics['duplicates']:,}")

    st.markdown("### Data Preview")
    st.dataframe(
        df.head(100),
        use_container_width=True,
        height=420,
    )

    st.markdown("### Column Information")

    info_rows = []

    for col in df.columns:
        info_rows.append(
            {
                "Column": col,
                "Data Type": str(df[col].dtype),
                "Non-Null": int(df[col].notna().sum()),
                "Missing": int(df[col].isna().sum()),
                "Unique Values": int(df[col].nunique(dropna=True)),
            }
        )

    st.dataframe(
        pd.DataFrame(info_rows),
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------------------

def show_statistics(df):
    st.subheader("Statistics")

    st.markdown("### Statistical Summary")

    try:
        summary = df.describe(include="all").transpose()
        st.dataframe(summary, use_container_width=True)
    except Exception:
        st.dataframe(
            df.describe().transpose(),
            use_container_width=True,
        )

    st.markdown("### Value Counts")

    columns = list(df.columns)

    if columns:
        selected = st.selectbox(
            "Select column",
            columns,
            key="statistics_value_counts_column",
        )

        counts = (
            df[selected]
            .value_counts(dropna=False)
            .reset_index()
        )

        counts.columns = ["Value", "Count"]

        st.dataframe(
            counts.head(100),
            use_container_width=True,
        )


# ---------------------------------------------------------------------
# DATA QUALITY
# ---------------------------------------------------------------------

def show_data_quality(df):
    st.subheader("Data Quality")

    quality = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [str(df[c].dtype) for c in df.columns],
            "Missing Count": [int(df[c].isna().sum()) for c in df.columns],
            "Missing %": [
                round(float(df[c].isna().mean() * 100), 2)
                for c in df.columns
            ],
            "Unique Values": [
                int(df[c].nunique(dropna=True))
                for c in df.columns
            ],
            "Duplicate Values": [
                int(df[c].duplicated().sum())
                for c in df.columns
            ],
        }
    )

    st.dataframe(
        quality,
        use_container_width=True,
    )

    st.markdown("### Missing Value Report")

    missing = quality[quality["Missing Count"] > 0].copy()

    if missing.empty:
        st.success("No missing values were found.")
    else:
        st.dataframe(
            missing.sort_values("Missing Count", ascending=False),
            use_container_width=True,
        )

    st.markdown("### Duplicate Analysis")

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count == 0:
        st.success("No complete duplicate rows were found.")
    else:
        st.warning(
            f"{duplicate_count:,} duplicate row(s) were found."
        )

        st.dataframe(
            df[df.duplicated(keep=False)].head(100),
            use_container_width=True,
        )


# ---------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------

def show_eda(df):
    st.subheader("Exploratory Data Analysis")

    numeric_cols = numeric_columns(df)
    cat_cols = categorical_columns(df)

    st.markdown("### Histogram")

    if numeric_cols:
        col = st.selectbox(
            "Numeric column",
            numeric_cols,
            key="eda_hist_column",
        )

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.hist(
            df[col].dropna(),
            bins=30,
            edgecolor="black",
        )

        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No numeric columns are available for a histogram.")

    st.markdown("### Box Plot")

    if numeric_cols:
        col = st.selectbox(
            "Box plot column",
            numeric_cols,
            key="eda_box_column",
        )

        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.boxplot(
            df[col].dropna(),
            vert=False,
        )

        ax.set_title(f"Box Plot of {col}")
        ax.set_xlabel(col)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### Bar Chart")

    if cat_cols:
        col = st.selectbox(
            "Categorical column",
            cat_cols,
            key="eda_bar_column",
        )

        top_values = (
            df[col]
            .astype("string")
            .fillna("Missing")
            .value_counts()
            .head(15)
        )

        fig, ax = plt.subplots(figsize=(10, 4.5))
        top_values.sort_values().plot(
            kind="barh",
            ax=ax,
        )

        ax.set_title(f"Top Values in {col}")
        ax.set_xlabel("Count")
        ax.set_ylabel(col)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No categorical columns are available for a bar chart.")

    st.markdown("### Correlation Heatmap")

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots(
            figsize=(
                max(7, len(numeric_cols) * 0.75),
                max(5, len(numeric_cols) * 0.65),
            )
        )

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax,
        )

        ax.set_title("Numeric Correlation Matrix")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("At least two numeric columns are required.")

    st.markdown("### Scatter Plot")

    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            x_col = st.selectbox(
                "X-axis",
                numeric_cols,
                key="eda_scatter_x",
            )

        with col2:
            y_options = [c for c in numeric_cols if c != x_col]

            if not y_options:
                y_options = numeric_cols

            y_col = st.selectbox(
                "Y-axis",
                y_options,
                key="eda_scatter_y",
            )

        plot_df = df[[x_col, y_col]].dropna()

        fig, ax = plt.subplots(figsize=(9, 4.5))

        ax.scatter(
            plot_df[x_col],
            plot_df[y_col],
            alpha=0.6,
        )

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)


# ---------------------------------------------------------------------
# OUTLIERS
# ---------------------------------------------------------------------

def calculate_iqr_outliers(df, column, multiplier=1.5):
    if column not in df.columns:
        return None

    if not pd.api.types.is_numeric_dtype(df[column]):
        return None

    series = df[column].dropna()

    if series.empty:
        return None

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        lower = q1
        upper = q3
    else:
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

    mask = df[column].notna() & (
        (df[column] < lower) |
        (df[column] > upper)
    )

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower": lower,
        "upper": upper,
        "count": int(mask.sum()),
        "percentage": float(mask.mean() * 100),
    }


def show_outliers(df):
    st.subheader("Outlier Analysis")

    numeric_cols = numeric_columns(df)

    if not numeric_cols:
        st.info("No numeric columns are available.")
        return

    selected_col = st.selectbox(
        "Select numeric column",
        numeric_cols,
        key="outlier_analysis_column",
    )

    multiplier = st.slider(
        "IQR multiplier",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.1,
        key="outlier_analysis_multiplier",
    )

    result = calculate_iqr_outliers(
        df,
        selected_col,
        multiplier,
    )

    if result is None:
        st.warning("Unable to calculate outliers.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Q1", f"{result['q1']:.3f}")
    c2.metric("Q3", f"{result['q3']:.3f}")
    c3.metric("IQR", f"{result['iqr']:.3f}")
    c4.metric("Outliers", f"{result['count']:,}")
    c5.metric("Outlier %", f"{result['percentage']:.2f}%")

    st.write(
        f"Lower bound: **{result['lower']:.4f}**"
    )

    st.write(
        f"Upper bound: **{result['upper']:.4f}**"
    )

    series = df[selected_col].dropna()

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.boxplot(
        series,
        vert=False,
    )

    ax.set_title(f"IQR Outlier Analysis: {selected_col}")
    ax.set_xlabel(selected_col)

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    mask = (
        df[selected_col].notna()
        & (
            (df[selected_col] < result["lower"])
            | (df[selected_col] > result["upper"])
        )
    )

    if mask.any():
        st.markdown("### Detected Outlier Rows")

        st.dataframe(
            df.loc[mask].head(100),
            use_container_width=True,
        )


# ---------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------

def show_features(df):
    st.subheader("Feature Engineering")

    numeric_cols = numeric_columns(df)

    if not numeric_cols:
        st.info("No numeric columns are available.")
        return

    st.markdown("### Create Log Transformation")

    log_col = st.selectbox(
        "Select numeric column",
        numeric_cols,
        key="feature_log_column",
    )

    if st.button(
        "Create Log Feature",
        key="create_log_feature",
    ):
        new_df = st.session_state.cleaned_df.copy()

        safe_values = pd.to_numeric(
            new_df[log_col],
            errors="coerce",
        )

        new_df[f"{log_col}_log"] = np.log1p(
            safe_values.clip(lower=0)
        )

        st.session_state.cleaned_df = new_df

        add_audit(
            f"Created log feature for '{log_col}'."
        )

        st.success(
            f"Feature '{log_col}_log' created."
        )

    st.markdown("### Create Squared Feature")

    square_col = st.selectbox(
        "Select column for squared feature",
        numeric_cols,
        key="feature_square_column",
    )

    if st.button(
        "Create Squared Feature",
        key="create_square_feature",
    ):
        new_df = st.session_state.cleaned_df.copy()

        new_df[f"{square_col}_squared"] = (
            pd.to_numeric(
                new_df[square_col],
                errors="coerce",
            )
            ** 2
        )

        st.session_state.cleaned_df = new_df

        add_audit(
            f"Created squared feature for '{square_col}'."
        )

        st.success(
            f"Feature '{square_col}_squared' created."
        )

    st.markdown("### Create Ratio Feature")

    c1, c2 = st.columns(2)

    with c1:
        numerator = st.selectbox(
            "Numerator",
            numeric_cols,
            key="feature_ratio_num",
        )

    with c2:
        denominator = st.selectbox(
            "Denominator",
            numeric_cols,
            key="feature_ratio_den",
        )

    if st.button(
        "Create Ratio Feature",
        key="create_ratio_feature",
    ):
        new_df = st.session_state.cleaned_df.copy()

        denominator_values = pd.to_numeric(
            new_df[denominator],
            errors="coerce",
        )

        numerator_values = pd.to_numeric(
            new_df[numerator],
            errors="coerce",
        )

        new_df[
            f"{numerator}_div_{denominator}"
        ] = numerator_values.div(
            denominator_values.replace(0, np.nan)
        )

        st.session_state.cleaned_df = new_df

        add_audit(
            f"Created ratio feature '{numerator}_div_{denominator}'."
        )

        st.success("Ratio feature created.")


# ---------------------------------------------------------------------
# TARGET ANALYSIS
# ---------------------------------------------------------------------

def show_target(df):
    st.subheader("Target Analysis")

    if df is None or df.empty:
        return

    target = st.selectbox(
        "Select target column",
        list(df.columns),
        key="target_analysis_column",
    )

    series = df[target]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Data Type", str(series.dtype))
    c2.metric("Unique Values", f"{series.nunique(dropna=True):,}")
    c3.metric("Missing", f"{series.isna().sum():,}")
    c4.metric("Rows", f"{len(series):,}")

    st.markdown("### Target Distribution")

    if pd.api.types.is_numeric_dtype(series):
        fig, ax = plt.subplots(figsize=(9, 4))

        ax.hist(
            series.dropna(),
            bins=30,
            edgecolor="black",
        )

        ax.set_title(f"Target Distribution: {target}")
        ax.set_xlabel(target)
        ax.set_ylabel("Frequency")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    else:
        counts = (
            series.astype("string")
            .fillna("Missing")
            .value_counts()
            .head(20)
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        counts.sort_values().plot(
            kind="barh",
            ax=ax,
        )

        ax.set_title(f"Target Distribution: {target}")
        ax.set_xlabel("Count")
        ax.set_ylabel(target)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            counts.rename("Count").reset_index(),
            use_container_width=True,
        )


# ---------------------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------------------

def prepare_ml_data(df, target):
    data = df.copy()

    data = data.dropna(subset=[target])

    X = data.drop(columns=[target])
    y = data[target]

    # Remove completely empty columns
    X = X.dropna(axis=1, how="all")

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    __import__(
                        "sklearn.preprocessing",
                        fromlist=["OneHotEncoder"],
                    ).OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    if not transformers:
        return None, None, None

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    return X, y, preprocessor


def show_training(df):
    st.subheader("Model Training")

    if df is None or df.empty:
        st.info("Load a dataset first.")
        return

    target = st.selectbox(
        "Target column",
        list(df.columns),
        key="training_target",
    )

    problem_type = st.radio(
        "Problem type",
        ["Classification", "Regression"],
        horizontal=True,
        key="training_problem",
    )

    if problem_type == "Classification":
        models = [
            "Random Forest",
            "Logistic Regression",
            "Decision Tree",
            "K Nearest Neighbors",
            "Gradient Boosting",
        ]
    else:
        models = [
            "Random Forest",
            "Linear Regression",
            "Decision Tree",
            "Gradient Boosting",
        ]

    model_name = st.selectbox(
        "Model",
        models,
        key="training_model",
    )

    test_size = st.slider(
        "Test size",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05,
        key="training_test_size",
    )

    random_state = st.number_input(
        "Random state",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
        key="training_random_state",
    )

    if problem_type == "Classification":
        if df[target].nunique(dropna=True) < 2:
            st.error(
                "Classification requires at least two target classes."
            )
            return

    if st.button(
        "Train Model",
        key="train_model_button",
    ):

        try:
            X, y, preprocessor = prepare_ml_data(
                df,
                target,
            )

            if X is None:
                st.error(
                    "No usable feature columns were found."
                )
                return

            if problem_type == "Classification":

                if y.dtype == "object" or str(y.dtype) in [
                    "category",
                    "string",
                    "bool",
                ]:
                    encoder = LabelEncoder()
                    y_encoded = encoder.fit_transform(
                        y.astype(str)
                    )
                    class_names = list(
                        encoder.classes_
                    )
                else:
                    encoder = None
                    y_encoded = y.values
                    class_names = sorted(
                        pd.Series(y).dropna().unique()
                    )

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y_encoded,
                    test_size=test_size,
                    random_state=int(random_state),
                    stratify=y_encoded
                    if len(np.unique(y_encoded)) > 1
                    else None,
                )

                if model_name == "Random Forest":
                    model = RandomForestClassifier(
                        n_estimators=200,
                        random_state=int(random_state),
                    )

                elif model_name == "Logistic Regression":
                    model = LogisticRegression(
                        max_iter=2000,
                    )

                elif model_name == "Decision Tree":
                    model = DecisionTreeClassifier(
                        random_state=int(random_state),
                    )

                elif model_name == "K Nearest Neighbors":
                    model = KNeighborsClassifier(
                        n_neighbors=5,
                    )

                else:
                    model = GradientBoostingClassifier(
                        random_state=int(random_state),
                    )

                pipeline = Pipeline(
                    steps=[
                        (
                            "preprocessor",
                            preprocessor,
                        ),
                        (
                            "model",
                            model,
                        ),
                    ]
                )

                pipeline.fit(
                    X_train,
                    y_train,
                )

                predictions = pipeline.predict(
                    X_test
                )

                accuracy = accuracy_score(
                    y_test,
                    predictions,
                )

                precision = precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )

                recall = recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )

                f1 = f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )

                st.session_state.training_result = {
                    "pipeline": pipeline,
                    "target": target,
                    "problem_type": problem_type,
                    "model_name": model_name,
                    "encoder": encoder,
                    "class_names": class_names,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "X_test": X_test,
                    "y_test": y_test,
                    "predictions": predictions,
                }

                add_audit(
                    f"Trained {model_name} classification model for target '{target}'."
                )

                st.success(
                    "Classification model trained successfully."
                )

                m1, m2, m3, m4 = st.columns(4)

                m1.metric(
                    "Accuracy",
                    f"{accuracy:.4f}",
                )

                m2.metric(
                    "Precision",
                    f"{precision:.4f}",
                )

                m3.metric(
                    "Recall",
                    f"{recall:.4f}",
                )

                m4.metric(
                    "F1 Score",
                    f"{f1:.4f}",
                )

                st.markdown("### Confusion Matrix")

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

                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")

                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

            else:
                y_numeric = pd.to_numeric(
                    y,
                    errors="coerce",
                )

                valid = y_numeric.notna()

                X = X.loc[valid]
                y_numeric = y_numeric.loc[valid]

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y_numeric,
                    test_size=test_size,
                    random_state=int(random_state),
                )

                if model_name == "Random Forest":
                    model = RandomForestRegressor(
                        n_estimators=200,
                        random_state=int(random_state),
                    )

                elif model_name == "Linear Regression":
                    model = LinearRegression()

                elif model_name == "Decision Tree":
                    model = DecisionTreeRegressor(
                        random_state=int(random_state),
                    )

                else:
                    model = GradientBoostingRegressor(
                        random_state=int(random_state),
                    )

                pipeline = Pipeline(
                    steps=[
                        (
                            "preprocessor",
                            preprocessor,
                        ),
                        (
                            "model",
                            model,
                        ),
                    ]
                )

                pipeline.fit(
                    X_train,
                    y_train,
                )

                predictions = pipeline.predict(
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

                st.session_state.training_result = {
                    "pipeline": pipeline,
                    "target": target,
                    "problem_type": problem_type,
                    "model_name": model_name,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "X_test": X_test,
                    "y_test": y_test,
                    "predictions": predictions,
                }

                add_audit(
                    f"Trained {model_name} regression model for target '{target}'."
                )

                st.success(
                    "Regression model trained successfully."
                )

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "MAE",
                    f"{mae:.4f}",
                )

                m2.metric(
                    "RMSE",
                    f"{rmse:.4f}",
                )

                m3.metric(
                    "R2 Score",
                    f"{r2:.4f}",
                )

                fig, ax = plt.subplots(
                    figsize=(8, 5)
                )

                ax.scatter(
                    y_test,
                    predictions,
                    alpha=0.65,
                )

                min_value = min(
                    y_test.min(),
                    predictions.min(),
                )

                max_value = max(
                    y_test.max(),
                    predictions.max(),
                )

                ax.plot(
                    [min_value, max_value],
                    [min_value, max_value],
                    linestyle="--",
                )

                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title(
                    "Actual vs Predicted"
                )

                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

        except Exception as e:
            st.error(
                f"Model training could not be completed: {str(e)}"
            )


# ---------------------------------------------------------------------
# MODEL PREDICTION
# ---------------------------------------------------------------------

def show_prediction(df):
    st.subheader("Model Prediction")

    result = st.session_state.training_result

    if result is None:
        st.info(
            "Train a model first in the Training section."
        )
        return

    pipeline = result["pipeline"]

    target = result["target"]

    feature_columns = [
        c for c in df.columns
        if c != target
    ]

    st.write(
        f"Prediction model: **{result['model_name']}**"
    )

    input_data = {}

    cols = st.columns(2)

    for i, col in enumerate(feature_columns):

        with cols[i % 2]:

            series = df[col]

            if pd.api.types.is_numeric_dtype(series):

                default_value = (
                    float(series.median())
                    if series.notna().any()
                    else 0.0
                )

                input_data[col] = st.number_input(
                    col,
                    value=default_value,
                    key=f"prediction_{col}",
                )

            else:

                values = (
                    series
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                values = values[:200]

                if not values:
                    values = [""]

                input_data[col] = st.selectbox(
                    col,
                    values,
                    key=f"prediction_{col}",
                )

    if st.button(
        "Generate Prediction",
        key="generate_prediction",
    ):

        try:
            input_df = pd.DataFrame(
                [input_data]
            )

            prediction = pipeline.predict(
                input_df
            )

            if result["problem_type"] == "Classification":

                if result.get("encoder") is not None:
                    prediction_display = result[
                        "encoder"
                    ].inverse_transform(
                        prediction.astype(int)
                    )[0]
                else:
                    prediction_display = prediction[0]

            else:
                prediction_display = prediction[0]

            st.session_state.prediction_result = (
                prediction_display
            )

            st.success(
                f"Prediction: {prediction_display}"
            )

        except Exception as e:
            st.error(
                f"Prediction failed: {str(e)}"
            )


# ---------------------------------------------------------------------
# AUDIT LOG
# ---------------------------------------------------------------------

def show_audit_log():
    st.subheader("Audit Log")

    if not st.session_state.audit_log:
        st.info("No actions have been recorded yet.")
        return

    for item in reversed(
        st.session_state.audit_log
    ):
        st.write(item)


# ---------------------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------------------

st.title("AI and ML Analytics Dashboard")

st.write(
    "Data analysis, data cleaning, visualization, feature engineering, "
    "outlier analysis, target analysis and machine learning in one dashboard."
)

# ---------------------------------------------------------------------
# UPLOAD
# ---------------------------------------------------------------------

st.subheader("Upload Dataset")

st.write(
    "Upload a CSV, XLSX or XLS file to begin."
)

uploaded_file = st.file_uploader(
    "Choose a dataset",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=False,
    key="dataset_uploader",
)

if uploaded_file is not None:

    # Detect a genuinely new uploaded file.
    new_file = (
        st.session_state.file_name
        != uploaded_file.name
    )

    if new_file:

        df = read_uploaded_file(
            uploaded_file
        )

        if df is not None:

            st.session_state.original_df = (
                df.copy(deep=True)
            )

            st.session_state.cleaned_df = (
                df.copy(deep=True)
            )

            st.session_state.file_name = (
                uploaded_file.name
            )

            st.session_state.audit_log = []

            st.session_state.training_result = None
            st.session_state.prediction_result = None

            add_audit(
                f"Uploaded dataset '{uploaded_file.name}' "
                f"with {len(df):,} rows and {len(df.columns):,} columns."
            )

            st.success(
                f"Dataset loaded successfully: {uploaded_file.name}"
            )

# ---------------------------------------------------------------------
# DATASET AVAILABLE
# ---------------------------------------------------------------------

if st.session_state.original_df is not None:

    original_df = st.session_state.original_df

    if st.session_state.cleaned_df is None:
        st.session_state.cleaned_df = (
            original_df.copy(deep=True)
        )

    cleaned_df = st.session_state.cleaned_df

    st.markdown("---")

    st.subheader("Dataset Summary")

    metrics = dataset_metrics(
        cleaned_df
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        "Rows",
        f"{metrics['rows']:,}",
    )

    c2.metric(
        "Columns",
        f"{metrics['columns']:,}",
    )

    c3.metric(
        "Numeric Columns",
        f"{metrics['numeric']:,}",
    )

    c4.metric(
        "Categorical Columns",
        f"{metrics['categorical']:,}",
    )

    c5.metric(
        "Missing Values",
        f"{metrics['missing']:,}",
    )

    c6.metric(
        "Duplicate Rows",
        f"{metrics['duplicates']:,}",
    )

    # -----------------------------------------------------------------
    # RESET
    # -----------------------------------------------------------------

    reset_col1, reset_col2, reset_col3 = st.columns(
        [1, 2, 1]
    )

    with reset_col1:

        if st.button(
            "Reset to Original",
            key="reset_original_button",
        ):

            # IMPORTANT:
            # This creates a completely new dataframe object.
            # It does not modify the original dataset.
            st.session_state.cleaned_df = (
                st.session_state.original_df.copy(
                    deep=True
                )
            )

            st.session_state.training_result = None
            st.session_state.prediction_result = None

            add_audit(
                "Reset cleaned dataset to the original uploaded dataset."
            )

            st.success(
                "Dataset has been reset to the original uploaded data."
            )

            st.rerun()

    # -----------------------------------------------------------------
    # TABS
    # -----------------------------------------------------------------

    tabs = st.tabs(
        [
            "Overview",
            "Statistics",
            "Data Quality",
            "EDA",
            "Cleaning",
            "Features",
            "Outliers",
            "Target Analysis",
            "Model Training",
            "Model Prediction",
            "Audit Log",
        ]
    )

    # -----------------------------------------------------------------
    # OVERVIEW
    # -----------------------------------------------------------------

    with tabs[0]:

        show_overview(
            st.session_state.cleaned_df
        )

    # -----------------------------------------------------------------
    # STATISTICS
    # -----------------------------------------------------------------

    with tabs[1]:

        show_statistics(
            st.session_state.cleaned_df
        )

    # -----------------------------------------------------------------
    # DATA QUALITY
    # -----------------------------------------------------------------

    with tabs[2]:

        show_data_quality(
            st.session_state.cleaned_df
        )

    # -----------------------------------------------------------------
    # EDA
    # -----------------------------------------------------------------

    with tabs[3]:

        show_eda(
            st.session_state.cleaned_df
        )

    # -----------------------------------------------------------------
    # CLEANING
    # -----------------------------------------------------------------

    with tabs[4]:

        st.subheader("Data Cleaning")

        st.write(
            "Select the operations you want to apply. "
            "The dashboard calculates the result from the original "
            "uploaded dataset every time you press Apply Cleaning. "
            "Changing a selection and applying again will therefore "
            "rebuild the result correctly."
        )

        st.markdown(
            '<div class="clean-status">'
            '<div class="clean-status-title">'
            'Cleaning Workflow'
            '</div>'
            '<div class="clean-status-text">'
            'Select options → review the expected result → '
            'apply cleaning → review the cleaned dataset → '
            'download the result.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # -------------------------------------------------------------
        # CURRENT STATUS
        # -------------------------------------------------------------

        original_metrics = dataset_metrics(
            st.session_state.original_df
        )

        current_metrics = dataset_metrics(
            st.session_state.cleaned_df
        )

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "Original Rows",
            f"{original_metrics['rows']:,}",
        )

        s2.metric(
            "Current Rows",
            f"{current_metrics['rows']:,}",
        )

        s3.metric(
            "Current Missing",
            f"{current_metrics['missing']:,}",
        )

        s4.metric(
            "Current Duplicates",
            f"{current_metrics['duplicates']:,}",
        )

        st.markdown("### Cleaning Options")

        left, right = st.columns(2)

        with left:

            duplicate_treatment = st.selectbox(
                "Duplicate Treatment",
                [
                    "Do nothing",
                    "Remove duplicate rows",
                ],
                key="clean_duplicate_treatment",
            )

            missing_treatment = st.selectbox(
                "Missing Value Treatment",
                [
                    "Do nothing",
                    "Mean for numeric columns",
                    "Median for numeric columns",
                    "Mode for all columns",
                    "Constant value",
                    "Forward fill",
                    "Backward fill",
                    "Drop rows containing missing values",
                ],
                key="clean_missing_treatment",
            )

            constant_value = st.text_input(
                "Constant Value",
                value="0",
                key="clean_constant_value",
            )

            numeric_treatment = st.selectbox(
                "Numeric Data Treatment",
                [
                    "Do nothing",
                    "Convert numeric-looking text",
                ],
                key="clean_numeric_treatment",
            )

        with right:

            categorical_treatment = st.selectbox(
                "Categorical Data Treatment",
                [
                    "Do nothing",
                    "Convert categorical columns to string",
                    "Fill categorical missing values with mode",
                ],
                key="clean_categorical_treatment",
            )

            invalid_value_treatment = st.selectbox(
                "Invalid Value Treatment",
                [
                    "Do nothing",
                    "Replace negative numeric values with zero",
                    "Replace infinite values with missing",
                ],
                key="clean_invalid_treatment",
            )

            outlier_treatment = st.selectbox(
                "Outlier Treatment",
                [
                    "Do nothing",
                    "Remove IQR outliers",
                ],
                key="clean_outlier_treatment",
            )

            clean_numeric_cols = numeric_columns(
                st.session_state.original_df
            )

            if clean_numeric_cols:
                outlier_column = st.selectbox(
                    "Outlier Column",
                    clean_numeric_cols,
                    key="clean_outlier_column",
                )

                outlier_multiplier = st.number_input(
                    "IQR Multiplier",
                    min_value=0.5,
                    max_value=5.0,
                    value=1.5,
                    step=0.1,
                    key="clean_outlier_multiplier",
                )
            else:
                outlier_column = ""
                outlier_multiplier = 1.5
                st.info(
                    "No numeric column is available for outlier treatment."
                )

        # -------------------------------------------------------------
        # PREVIEW BUTTON
        # -------------------------------------------------------------

        st.markdown("### Review Cleaning Result")

        preview_col1, preview_col2 = st.columns(2)

        with preview_col1:

            if st.button(
                "Preview Cleaning",
                key="preview_cleaning_button",
            ):

                (
                    preview_df,
                    preview_message,
                ) = get_cleaning_preview(
                    source_df=st.session_state.original_df,
                    duplicate_treatment=duplicate_treatment,
                    categorical_treatment=categorical_treatment,
                    missing_treatment=missing_treatment,
                    invalid_value_treatment=invalid_value_treatment,
                    constant_value=constant_value,
                    numeric_treatment=numeric_treatment,
                    outlier_treatment=outlier_treatment,
                    outlier_column=outlier_column,
                    outlier_multiplier=outlier_multiplier,
                )

                if preview_df is not None:

                    st.session_state.clean_preview_df = (
                        preview_df
                    )

                    st.session_state.clean_preview_message = (
                        preview_message
                    )

        with preview_col2:

            if st.button(
                "Clear Preview",
                key="clear_cleaning_preview",
            ):

                if "clean_preview_df" in st.session_state:
                    del st.session_state.clean_preview_df

                if "clean_preview_message" in st.session_state:
                    del st.session_state.clean_preview_message

        if (
            "clean_preview_df" in st.session_state
            and "clean_preview_message" in st.session_state
        ):

            st.info(
                st.session_state.clean_preview_message
            )

            preview_metrics = dataset_metrics(
                st.session_state.clean_preview_df
            )

            p1, p2, p3, p4 = st.columns(4)

            p1.metric(
                "Preview Rows",
                f"{preview_metrics['rows']:,}",
            )

            p2.metric(
                "Rows Removed",
                f"{original_metrics['rows'] - preview_metrics['rows']:,}",
            )

            p3.metric(
                "Preview Missing",
                f"{preview_metrics['missing']:,}",
            )

            p4.metric(
                "Preview Duplicates",
                f"{preview_metrics['duplicates']:,}",
            )

            st.dataframe(
                st.session_state.clean_preview_df.head(100),
                use_container_width=True,
                height=350,
            )

        # -------------------------------------------------------------
        # APPLY CLEANING
        # -------------------------------------------------------------

        st.markdown("### Apply Cleaning")

        st.markdown(
            '<div class="primary-button">',
            unsafe_allow_html=True,
        )

        apply_clicked = st.button(
            "Apply Cleaning to Current Dataset",
            key="apply_cleaning_button",
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        if apply_clicked:

            try:

                before_df = (
                    st.session_state.original_df.copy(
                        deep=True
                    )
                )

                cleaned_result, cleaning_log = (
                    apply_cleaning_pipeline(
                        source_df=before_df,
                        duplicate_treatment=duplicate_treatment,
                        categorical_treatment=categorical_treatment,
                        missing_treatment=missing_treatment,
                        invalid_value_treatment=invalid_value_treatment,
                        constant_value=constant_value,
                        numeric_treatment=numeric_treatment,
                        outlier_treatment=outlier_treatment,
                        outlier_column=outlier_column,
                        outlier_multiplier=outlier_multiplier,
                    )
                )

                # CRITICAL:
                # Always replace the cleaned dataframe with a NEW result.
                st.session_state.cleaned_df = (
                    cleaned_result.copy(deep=True)
                )

                for log_item in cleaning_log:
                    add_audit(log_item)

                before = dataset_metrics(
                    before_df
                )

                after = dataset_metrics(
                    cleaned_result
                )

                removed_rows = (
                    before["rows"]
                    - after["rows"]
                )

                st.success(
                    f"Cleaning applied successfully. "
                    f"Rows: {before['rows']:,} -> {after['rows']:,}. "
                    f"Rows removed: {removed_rows:,}."
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Cleaning could not be applied: {str(e)}"
                )

        # -------------------------------------------------------------
        # CURRENT CLEANED DATA
        # -------------------------------------------------------------

        st.markdown("### Current Cleaned Dataset")

        latest_df = (
            st.session_state.cleaned_df
        )

        latest_metrics = dataset_metrics(
            latest_df
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Rows",
            f"{latest_metrics['rows']:,}",
        )

        c2.metric(
            "Columns",
            f"{latest_metrics['columns']:,}",
        )

        c3.metric(
            "Missing Values",
            f"{latest_metrics['missing']:,}",
        )

        c4.metric(
            "Duplicate Rows",
            f"{latest_metrics['duplicates']:,}",
        )

        st.dataframe(
            latest_df.head(100),
            use_container_width=True,
            height=430,
        )

        # -------------------------------------------------------------
        # DOWNLOAD
        # -------------------------------------------------------------

        st.markdown("### Download Cleaned Dataset")

        st.write(
            "Choose the file format and download the currently displayed cleaned dataset."
        )

        download_format = st.radio(
            "Download Format",
            ["CSV", "Excel"],
            horizontal=True,
            key="clean_download_format",
        )

        download_bytes = None
        download_extension = ""
        download_mime = ""

        if download_format == "CSV":

            download_bytes = latest_df.to_csv(
                index=False
            ).encode("utf-8")

            download_extension = "csv"
            download_mime = "text/csv"

        else:

            excel_buffer = io.BytesIO()

            with pd.ExcelWriter(
                excel_buffer,
                engine="openpyxl",
            ) as writer:

                latest_df.to_excel(
                    writer,
                    index=False,
                    sheet_name="Cleaned_Data",
                )

            excel_buffer.seek(0)

            download_bytes = (
                excel_buffer.getvalue()
            )

            download_extension = "xlsx"

            download_mime = (
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )

        base_name = (
            st.session_state.file_name
            .rsplit(".", 1)[0]
            if st.session_state.file_name
            else "dataset"
        )

        st.download_button(
            label=f"Download Cleaned Dataset as {download_format}",
            data=download_bytes,
            file_name=f"{base_name}_cleaned.{download_extension}",
            mime=download_mime,
            key="download_cleaned_dataset",
        )

    # -----------------------------------------------------------------
    # FEATURES
    # -----------------------------------------------------------------

    with tabs[5]:

        show_features(
            st.session_state.cleaned_df
        )

    # -----------------------------------------------------------------
    # OUTLIERS
    # -----------------------------------------------------------------

    with tabs[6]:

        show_outliers(
            st.session_state.cleaned_df
        )

    # -----------------------------------------------------------------
    # TARGET
    # -----------------------------------------------------------------

    with tabs[7]:

        show_target(
            st.session_state.cleaned_df
        )

    # -----------------------------------------------------------------
    # TRAINING
    # -----------------------------------------------------------------

    with tabs[8]:

        show_training(
            st.session_state.cleaned_df
        )

    # -----------------------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------------------

    with tabs[9]:

        show_prediction(
            st.session_state.cleaned_df
        )

    # -----------------------------------------------------------------
    # AUDIT
    # -----------------------------------------------------------------

    with tabs[10]:

        show_audit_log()

    # -----------------------------------------------------------------
    # FOOTER
    # -----------------------------------------------------------------

    st.markdown(
        """
        <div class="footer">
        AI and ML Analytics Dashboard | Data Analysis | Data Cleaning |
        Exploratory Data Analysis | Feature Engineering | Outlier Analysis |
        Machine Learning
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.info(
        "Upload a CSV, XLSX, or XLS dataset to start the dashboard."
    )