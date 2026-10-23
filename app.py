# ============================================================
# AI AND ML ANALYTICS DASHBOARD
# Complete Streamlit Application
# ============================================================

import io
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
)
from sklearn.neighbors import KNeighborsClassifier


warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI and ML Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GLOBAL FONT
       -------------------------------------------------------- */

    html,
    body,
    [class*="css"],
    .stApp,
    .stMarkdown,
    .stText,
    .stButton,
    .stSelectbox,
    .stRadio,
    .stCheckbox,
    .stNumberInput,
    .stTextInput,
    .stFileUploader,
    .stDataFrame,
    .stDownloadButton {
        font-family: "Times New Roman", Times, serif !important;
    }

    .stApp {
        background: #f5f7fb;
        color: #1f2937;
    }

    /* --------------------------------------------------------
       MAIN CONTAINER
       -------------------------------------------------------- */

    .main .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* --------------------------------------------------------
       MAIN TITLE
       -------------------------------------------------------- */

    .dashboard-title {
        background: linear-gradient(
            135deg,
            #12355b 0%,
            #176b87 50%,
            #159a9c 100%
        );
        padding: 28px 35px;
        border-radius: 16px;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(18, 53, 91, 0.18);
    }

    .dashboard-title h1 {
        color: white;
        font-size: 38px;
        margin: 0;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .dashboard-title p {
        color: #e8f6f7;
        font-size: 17px;
        margin: 8px 0 0 0;
    }

    /* --------------------------------------------------------
       SECTION HEADINGS
       -------------------------------------------------------- */

    .section-heading {
        color: #12355b;
        font-size: 26px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 12px;
        padding-bottom: 7px;
        border-bottom: 2px solid #d9e4ef;
    }

    .sub-heading {
        color: #176b87;
        font-size: 20px;
        font-weight: 700;
        margin-top: 16px;
        margin-bottom: 8px;
    }

    /* --------------------------------------------------------
       CARDS
       -------------------------------------------------------- */

    .metric-card {
        background: white;
        border: 1px solid #dbe4ee;
        border-radius: 12px;
        padding: 18px;
        min-height: 115px;
        box-shadow: 0 3px 12px rgba(31, 41, 55, 0.06);
    }

    .metric-label {
        color: #607086;
        font-size: 15px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #12355b;
        font-size: 28px;
        font-weight: 700;
    }

    .info-box {
        background: #eaf4fb;
        border-left: 5px solid #176b87;
        padding: 13px 17px;
        border-radius: 8px;
        margin: 10px 0;
        color: #243b53;
    }

    .success-box {
        background: #eaf7f1;
        border-left: 5px solid #159a72;
        padding: 13px 17px;
        border-radius: 8px;
        margin: 10px 0;
        color: #14532d;
    }

    .warning-box {
        background: #fff7e6;
        border-left: 5px solid #d97706;
        padding: 13px 17px;
        border-radius: 8px;
        margin: 10px 0;
        color: #78350f;
    }

    .danger-box {
        background: #fff0f0;
        border-left: 5px solid #dc3545;
        padding: 13px 17px;
        border-radius: 8px;
        margin: 10px 0;
        color: #7f1d1d;
    }

    /* --------------------------------------------------------
       BUTTONS
       -------------------------------------------------------- */

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px !important;
        border: 1px solid #176b87 !important;
        background: #176b87 !important;
        color: white !important;
        font-family: "Times New Roman", Times, serif !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #12355b !important;
        border-color: #12355b !important;
    }

    /* --------------------------------------------------------
       INPUTS
       -------------------------------------------------------- */

    div[data-baseweb="select"] > div {
        border-radius: 8px;
        border-color: #ccd7e3;
    }

    input,
    textarea {
        border-radius: 8px !important;
    }

    /* --------------------------------------------------------
       TABS
       -------------------------------------------------------- */

    button[data-baseweb="tab"] {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #40566f !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #176b87 !important;
    }

    /* --------------------------------------------------------
       DATAFRAME
       -------------------------------------------------------- */

    [data-testid="stDataFrame"] {
        border: 1px solid #d7e0ea;
        border-radius: 10px;
        overflow: hidden;
    }

    /* --------------------------------------------------------
       SIDEBAR
       -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: #edf3f8;
        border-right: 1px solid #d5e0ea;
    }

    section[data-testid="stSidebar"] h2 {
        color: #12355b;
    }

    /* --------------------------------------------------------
       FILE UPLOADER
       -------------------------------------------------------- */

    [data-testid="stFileUploader"] {
        background: white;
        border: 1px solid #ccd7e3;
        border-radius: 12px;
        padding: 8px;
    }

    /* --------------------------------------------------------
       ALERTS
       -------------------------------------------------------- */

    .stAlert {
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "original_df": None,
    "current_df": None,
    "file_name": "",
    "audit_log": [],
    "target_column": None,
    "trained_model": None,
    "trained_pipeline": None,
    "training_features": None,
    "training_target": None,
    "problem_type": None,
    "label_encoder": None,
    "model_metrics": None,
    "prediction_input": None,
}


for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def add_audit(message):
    """Add an action to the audit log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.audit_log.append(
        f"{timestamp} - {message}"
    )


def safe_copy(df):
    """Return a safe DataFrame copy."""
    if df is None:
        return None
    return df.copy(deep=True)


def reset_dataset():
    """Reset current dataset to original uploaded dataset."""
    if st.session_state.original_df is not None:
        st.session_state.current_df = safe_copy(
            st.session_state.original_df
        )

        st.session_state.trained_model = None
        st.session_state.trained_pipeline = None
        st.session_state.training_features = None
        st.session_state.training_target = None
        st.session_state.model_metrics = None

        add_audit("Dataset reset to original uploaded data.")


def read_uploaded_file(uploaded_file):
    """Read CSV, XLSX or XLS safely."""
    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".csv"):
            try:
                return pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                return pd.read_csv(
                    uploaded_file,
                    encoding="latin1"
                )

        if file_name.endswith(".xlsx"):
            return pd.read_excel(
                uploaded_file,
                engine="openpyxl"
            )

        if file_name.endswith(".xls"):
            return pd.read_excel(
                uploaded_file,
                engine="xlrd"
            )

        raise ValueError(
            "Unsupported file format. Please upload CSV, XLSX or XLS."
        )

    except Exception as exc:
        raise ValueError(
            f"Unable to read the uploaded file: {exc}"
        )


def numeric_columns(df):
    if df is None:
        return []
    return df.select_dtypes(
        include=np.number
    ).columns.tolist()


def categorical_columns(df):
    if df is None:
        return []

    return df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()


def datetime_columns(df):
    if df is None:
        return []

    return df.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()


def convert_possible_dates(df):
    """Try to identify object columns containing dates."""
    result = df.copy()

    converted = []

    for col in result.columns:
        if result[col].dtype == "object":
            sample = result[col].dropna()

            if len(sample) == 0:
                continue

            try:
                parsed = pd.to_datetime(
                    sample,
                    errors="coerce"
                )

                valid_ratio = parsed.notna().mean()

                if valid_ratio >= 0.80:
                    result[col] = pd.to_datetime(
                        result[col],
                        errors="coerce"
                    )
                    converted.append(col)

            except Exception:
                pass

    return result, converted


def format_number(value):
    try:
        return f"{value:,}"
    except Exception:
        return str(value)


def calculate_iqr_bounds(series, multiplier=1.5):
    series = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if series.empty:
        return None, None

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    return lower, upper


def create_download_file(df, file_format):
    """Create CSV or XLSX bytes."""
    if file_format == "CSV":
        return (
            df.to_csv(index=False).encode("utf-8"),
            "cleaned_dataset.csv",
            "text/csv",
        )

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:
        df.to_excel(
            writer,
            index=False,
            sheet_name="Cleaned_Data"
        )

    return (
        output.getvalue(),
        "cleaned_dataset.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def make_summary(df):
    summary = pd.DataFrame(
        {
            "Metric": [
                "Rows",
                "Columns",
                "Numeric Columns",
                "Categorical Columns",
                "Date Columns",
                "Missing Cells",
                "Duplicate Rows",
                "Memory Usage (MB)",
            ],
            "Value": [
                len(df),
                len(df.columns),
                len(numeric_columns(df)),
                len(categorical_columns(df)),
                len(datetime_columns(df)),
                int(df.isna().sum().sum()),
                int(df.duplicated().sum()),
                round(
                    df.memory_usage(deep=True).sum()
                    / (1024 * 1024),
                    2,
                ),
            ],
        }
    )

    return summary


def column_information(df):
    records = []

    for col in df.columns:
        series = df[col]

        records.append(
            {
                "Column": col,
                "Data Type": str(series.dtype),
                "Non-Null": int(series.notna().sum()),
                "Missing": int(series.isna().sum()),
                "Missing %": round(
                    series.isna().mean() * 100,
                    2
                ),
                "Unique Values": int(
                    series.nunique(dropna=True)
                ),
                "Sample Value": (
                    series.dropna().iloc[0]
                    if series.notna().any()
                    else ""
                ),
            }
        )

    return pd.DataFrame(records)


def missing_report(df):
    report = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing Count": [
                int(df[col].isna().sum())
                for col in df.columns
            ],
            "Missing %": [
                round(
                    df[col].isna().mean() * 100,
                    2
                )
                for col in df.columns
            ],
            "Data Type": [
                str(df[col].dtype)
                for col in df.columns
            ],
        }
    )

    return report.sort_values(
        "Missing Count",
        ascending=False
    ).reset_index(drop=True)


def duplicate_report(df):
    duplicate_count = int(df.duplicated().sum())

    return pd.DataFrame(
        {
            "Metric": [
                "Total Rows",
                "Duplicate Rows",
                "Unique Rows",
                "Duplicate Percentage",
            ],
            "Value": [
                len(df),
                duplicate_count,
                len(df) - duplicate_count,
                round(
                    duplicate_count
                    / max(len(df), 1)
                    * 100,
                    2,
                ),
            ],
        }
    )


def outlier_report(df):
    nums = numeric_columns(df)

    rows = []

    for col in nums:
        lower, upper = calculate_iqr_bounds(
            df[col]
        )

        if lower is None:
            continue

        series = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        mask = (
            (series < lower)
            | (series > upper)
        )

        count = int(mask.sum())

        rows.append(
            {
                "Column": col,
                "Q1": round(
                    series.quantile(0.25),
                    4
                ),
                "Q3": round(
                    series.quantile(0.75),
                    4
                ),
                "IQR": round(
                    series.quantile(0.75)
                    - series.quantile(0.25),
                    4,
                ),
                "Lower Bound": round(
                    lower,
                    4
                ),
                "Upper Bound": round(
                    upper,
                    4
                ),
                "Outlier Count": count,
                "Outlier %": round(
                    count
                    / max(len(series), 1)
                    * 100,
                    2,
                ),
            }
        )

    return pd.DataFrame(rows)


def skewness_report(df):
    nums = numeric_columns(df)

    rows = []

    for col in nums:
        series = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if len(series) >= 3:
            rows.append(
                {
                    "Column": col,
                    "Skewness": round(
                        series.skew(),
                        4
                    ),
                    "Kurtosis": round(
                        series.kurtosis(),
                        4
                    ),
                }
            )

    return pd.DataFrame(rows)


def detect_constant_columns(df):
    rows = []

    for col in df.columns:
        unique = df[col].nunique(
            dropna=False
        )

        if unique <= 1:
            rows.append(
                {
                    "Column": col,
                    "Unique Values": unique,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def apply_cleaning_operations(
    df,
    duplicate_treatment,
    missing_treatment,
    missing_scope,
    missing_constant,
    outlier_treatment,
    outlier_column,
    iqr_multiplier,
    data_type_treatment,
    date_column,
):
    """
    Apply all selected cleaning operations to the CURRENT
    dataset.

    Operations are cumulative. Every time the user presses
    Apply Cleaning, the current dataset becomes the starting
    point for the next cleaning operation.
    """

    result = df.copy()

    changes = []

    rows_before = len(result)

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    if duplicate_treatment == "Remove Duplicate Rows":
        before = len(result)

        result = result.drop_duplicates(
            keep="first"
        ).reset_index(drop=True)

        removed = before - len(result)

        if removed > 0:
            changes.append(
                f"Removed {removed} duplicate row(s)."
            )

    elif duplicate_treatment == "Keep Last Duplicate":
        before = len(result)

        result = result.drop_duplicates(
            keep="last"
        ).reset_index(drop=True)

        removed = before - len(result)

        if removed > 0:
            changes.append(
                f"Removed {removed} duplicate row(s), keeping last records."
            )

    # --------------------------------------------------------
    # DATA TYPE TREATMENT
    # --------------------------------------------------------

    if data_type_treatment == "Convert Numeric-Like Columns":

        converted = []

        for col in result.columns:
            if result[col].dtype == "object":

                converted_series = pd.to_numeric(
                    result[col],
                    errors="coerce"
                )

                original_non_null = result[col].notna().sum()

                if original_non_null > 0:
                    ratio = (
                        converted_series.notna().sum()
                        / original_non_null
                    )

                    if ratio >= 0.90:
                        result[col] = converted_series
                        converted.append(col)

        if converted:
            changes.append(
                "Converted numeric-like columns: "
                + ", ".join(converted)
            )

    elif data_type_treatment == "Convert Date-Like Columns":

        result, converted = convert_possible_dates(
            result
        )

        if converted:
            changes.append(
                "Converted date-like columns: "
                + ", ".join(converted)
            )

    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    if missing_treatment != "Do Nothing":

        if missing_scope == "All Columns":
            columns_to_process = list(
                result.columns
            )
        else:
            columns_to_process = [
                missing_scope
            ]

        processed = []

        for col in columns_to_process:

            if col not in result.columns:
                continue

            if result[col].isna().sum() == 0:
                continue

            series = result[col]

            if missing_treatment == "Mean":
                if pd.api.types.is_numeric_dtype(series):
                    result[col] = series.fillna(
                        series.mean()
                    )
                    processed.append(col)

            elif missing_treatment == "Median":
                if pd.api.types.is_numeric_dtype(series):
                    result[col] = series.fillna(
                        series.median()
                    )
                    processed.append(col)

            elif missing_treatment == "Mode":
                modes = series.mode(
                    dropna=True
                )

                if not modes.empty:
                    result[col] = series.fillna(
                        modes.iloc[0]
                    )
                    processed.append(col)

            elif missing_treatment == "Constant":
                result[col] = series.fillna(
                    missing_constant
                )
                processed.append(col)

            elif missing_treatment == "Forward Fill":
                result[col] = series.ffill()
                processed.append(col)

            elif missing_treatment == "Backward Fill":
                result[col] = series.bfill()
                processed.append(col)

        if processed:
            changes.append(
                f"Missing values treated using '{missing_treatment}' "
                f"for: {', '.join(processed)}"
            )

    # --------------------------------------------------------
    # OUTLIERS
    # --------------------------------------------------------

    if (
        outlier_treatment == "Remove IQR Outliers"
        and outlier_column
        and outlier_column in result.columns
    ):

        series = pd.to_numeric(
            result[outlier_column],
            errors="coerce"
        )

        lower, upper = calculate_iqr_bounds(
            series,
            iqr_multiplier
        )

        if lower is not None:

            before = len(result)

            mask = (
                series.isna()
                | (
                    (series >= lower)
                    & (series <= upper)
                )
            )

            result = result.loc[
                mask
            ].reset_index(drop=True)

            removed = before - len(result)

            changes.append(
                f"Removed {removed} outlier row(s) from "
                f"'{outlier_column}' using IQR multiplier "
                f"{iqr_multiplier}."
            )

    rows_after = len(result)

    changes.append(
        f"Rows before cleaning: {rows_before}"
    )

    changes.append(
        f"Rows after cleaning: {rows_after}"
    )

    return result, changes


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_histogram(df, column, bins):
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(
        df[column].dropna(),
        bins=bins,
        kde=True,
        ax=ax,
    )

    ax.set_title(
        f"Distribution of {column}",
        fontsize=16
    )

    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    plt.tight_layout()

    return fig


def plot_boxplot(df, column):
    fig, ax = plt.subplots(figsize=(10, 4))

    sns.boxplot(
        x=df[column],
        ax=ax
    )

    ax.set_title(
        f"Box Plot of {column}",
        fontsize=16
    )

    ax.set_xlabel(column)

    plt.tight_layout()

    return fig


def plot_bar_chart(df, column, top_n):
    counts = (
        df[column]
        .astype(str)
        .value_counts()
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        f"Top {top_n} Values in {column}",
        fontsize=16
    )

    ax.set_xlabel(column)
    ax.set_ylabel("Count")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    return fig


def plot_scatter(df, x_col, y_col):
    temp = df[
        [x_col, y_col]
    ].dropna()

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.scatterplot(
        data=temp,
        x=x_col,
        y=y_col,
        ax=ax
    )

    ax.set_title(
        f"{x_col} vs {y_col}",
        fontsize=16
    )

    plt.tight_layout()

    return fig


def plot_correlation(df):
    nums = numeric_columns(df)

    if len(nums) < 2:
        return None

    corr = df[nums].corr()

    fig, ax = plt.subplots(
        figsize=(
            max(8, len(nums) * 0.7),
            max(6, len(nums) * 0.6),
        )
    )

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
    )

    ax.set_title(
        "Correlation Heatmap",
        fontsize=16
    )

    plt.tight_layout()

    return fig


# ============================================================
# MACHINE LEARNING FUNCTIONS
# ============================================================

def prepare_ml_data(df, target, problem_type):
    data = df.copy()

    if target not in data.columns:
        raise ValueError(
            "Target column does not exist."
        )

    data = data.dropna(
        subset=[target]
    )

    X = data.drop(
        columns=[target]
    )

    y = data[target]

    # Remove completely empty columns
    empty_columns = [
        col
        for col in X.columns
        if X[col].isna().all()
    ]

    if empty_columns:
        X = X.drop(
            columns=empty_columns
        )

    # Remove constant columns
    constant_columns = [
        col
        for col in X.columns
        if X[col].nunique(
            dropna=False
        ) <= 1
    ]

    if constant_columns:
        X = X.drop(
            columns=constant_columns
        )

    if X.shape[1] == 0:
        raise ValueError(
            "No usable feature columns remain."
        )

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    transformers = []

    if numeric_features:

        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                ),
                numeric_features,
            )
        )

    if categorical_features:

        transformers.append(
            (
                "categorical",
                Pipeline(
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
                                "sklearn"
                            ).preprocessing.OneHotEncoder(
                                handle_unknown="ignore"
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    if problem_type == "Classification":

        if y.dtype == "object" or str(
            y.dtype
        ).startswith("category"):

            label_encoder = LabelEncoder()

            y_encoded = label_encoder.fit_transform(
                y.astype(str)
            )

        else:
            label_encoder = None
            y_encoded = y

        return (
            X,
            y_encoded,
            preprocessor,
            label_encoder,
        )

    return (
        X,
        pd.to_numeric(
            y,
            errors="coerce"
        ),
        preprocessor,
        None,
    )


def build_model(problem_type, model_name):
    if problem_type == "Classification":

        if model_name == "Logistic Regression":
            return LogisticRegression(
                max_iter=2000
            )

        if model_name == "K-Nearest Neighbors":
            return KNeighborsClassifier(
                n_neighbors=5
            )

        if model_name == "Gradient Boosting":
            return GradientBoostingClassifier(
                random_state=42
            )

        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )

    if model_name == "Linear Regression":
        return LinearRegression()

    if model_name == "Gradient Boosting":
        return GradientBoostingRegressor(
            random_state=42
        )

    return RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )


def train_model(
    df,
    target,
    problem_type,
    model_name,
    test_size,
):
    (
        X,
        y,
        preprocessor,
        label_encoder,
    ) = prepare_ml_data(
        df,
        target,
        problem_type
    )

    if problem_type == "Regression":
        valid = pd.notna(y)

        X = X.loc[valid]
        y = y.loc[valid]

    if len(X) < 10:
        raise ValueError(
            "At least 10 usable rows are recommended for training."
        )

    if problem_type == "Classification":

        unique_classes = pd.Series(
            y
        ).nunique()

        if unique_classes < 2:
            raise ValueError(
                "Classification requires at least two target classes."
            )

    stratify_value = None

    if problem_type == "Classification":

        value_counts = pd.Series(
            y
        ).value_counts()

        if (
            len(value_counts) > 1
            and value_counts.min() >= 2
        ):
            stratify_value = y

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify_value,
        )
    )

    model = build_model(
        problem_type,
        model_name
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    if problem_type == "Classification":

        accuracy = accuracy_score(
            y_test,
            predictions
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

        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Predictions": predictions,
            "Actual": y_test,
        }

    else:

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2,
            "Predictions": predictions,
            "Actual": y_test,
        }

    return (
        pipeline,
        label_encoder,
        metrics,
        X.columns.tolist(),
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## Dataset"
    )

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV, XLSX and XLS",
    )

    if uploaded_file is not None:

        if (
            st.session_state.file_name
            != uploaded_file.name
        ):

            try:
                loaded_df = read_uploaded_file(
                    uploaded_file
                )

                if loaded_df is None:
                    st.error(
                        "The file could not be loaded."
                    )

                elif loaded_df.empty:
                    st.error(
                        "The uploaded dataset is empty."
                    )

                else:

                    st.session_state.original_df = (
                        loaded_df.copy()
                    )

                    st.session_state.current_df = (
                        loaded_df.copy()
                    )

                    st.session_state.file_name = (
                        uploaded_file.name
                    )

                    st.session_state.audit_log = []

                    st.session_state.trained_model = None
                    st.session_state.trained_pipeline = None
                    st.session_state.training_features = None
                    st.session_state.training_target = None
                    st.session_state.model_metrics = None

                    add_audit(
                        f"Loaded dataset: {uploaded_file.name}"
                    )

                    st.success(
                        "Dataset loaded successfully."
                    )

            except Exception as exc:
                st.error(
                    f"File loading failed: {exc}"
                )

    if st.session_state.current_df is not None:

        st.markdown("---")

        st.markdown(
            "### Current Dataset"
        )

        st.write(
            st.session_state.file_name
        )

        st.write(
            f"Rows: {len(st.session_state.current_df):,}"
        )

        st.write(
            f"Columns: {len(st.session_state.current_df.columns):,}"
        )

        st.markdown("---")

        if st.button(
            "Reset to Original",
            use_container_width=True,
        ):

            reset_dataset()

            st.success(
                "Dataset restored to the original uploaded data."
            )

            st.rerun()


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    """
    <div class="dashboard-title">
        <h1>AI and ML Analytics Dashboard</h1>
        <p>
            Data Analysis, Data Cleaning, Visualization,
            Feature Engineering, Outlier Detection and Machine Learning
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# NO DATA SCREEN
# ============================================================

if st.session_state.current_df is None:

    st.markdown(
        '<div class="section-heading">Dataset</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Upload a CSV, XLSX or XLS dataset from the left panel to begin."
    )

    st.markdown(
        """
        <div class="info-box">
        The dashboard will provide dataset overview, statistics,
        data-quality analysis, EDA, cleaning, feature engineering,
        outlier analysis, target analysis, machine learning,
        model comparison and cleaned-data download.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# CURRENT DATA
# ============================================================

df = st.session_state.current_df


# ============================================================
# TOP DATASET STATUS
# ============================================================

st.markdown(
    '<div class="section-heading">Dataset Summary</div>',
    unsafe_allow_html=True,
)

summary = make_summary(df)

metric_cols = st.columns(5)

summary_values = {
    "Rows": len(df),
    "Columns": len(df.columns),
    "Numeric Columns": len(
        numeric_columns(df)
    ),
    "Categorical Columns": len(
        categorical_columns(df)
    ),
    "Missing Cells": int(
        df.isna().sum().sum()
    ),
}

for index, (
    label,
    value,
) in enumerate(
    summary_values.items()
):

    with metric_cols[index]:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{format_number(value)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "Overview",
        "Statistics",
        "Data Quality",
        "EDA",
        "Cleaning",
        "Features",
        "Outliers",
        "Target",
        "Training",
        "Model Comparison",
        "Prediction",
        "Download",
        "Audit Log",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:

    st.markdown(
        '<div class="section-heading">Dataset Overview</div>',
        unsafe_allow_html=True,
    )

    st.write(
        f"Loaded file: **{st.session_state.file_name}**"
    )

    st.markdown(
        '<div class="sub-heading">Dataset Information</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        column_information(df),
        use_container_width=True,
        height=420,
    )

    st.markdown(
        '<div class="sub-heading">Dataset Viewer</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Use the controls below to inspect any section of the dataset."
    )

    view_type = st.selectbox(
        "Rows to Display",
        [
            "First 50",
            "First 100",
            "First 147",
            "First 500",
            "Last 50",
            "Last 100",
            "Last 147",
            "Last 500",
            "Head",
            "Tail",
            "Middle",
            "All Rows",
            "Custom Range",
        ],
        key="overview_view_type",
    )

    if view_type == "First 50":
        display_df = df.head(50)

    elif view_type == "First 100":
        display_df = df.head(100)

    elif view_type == "First 147":
        display_df = df.head(147)

    elif view_type == "First 500":
        display_df = df.head(500)

    elif view_type == "Last 50":
        display_df = df.tail(50)

    elif view_type == "Last 100":
        display_df = df.tail(100)

    elif view_type == "Last 147":
        display_df = df.tail(147)

    elif view_type == "Last 500":
        display_df = df.tail(500)

    elif view_type == "Head":
        display_df = df.head()

    elif view_type == "Tail":
        display_df = df.tail()

    elif view_type == "Middle":

        middle = len(df) // 2

        start = max(
            0,
            middle - 25
        )

        display_df = df.iloc[
            start:start + 50
        ]

    elif view_type == "All Rows":
        display_df = df

    else:

        custom_start = st.number_input(
            "Starting Row",
            min_value=1,
            max_value=max(
                1,
                len(df)
            ),
            value=1,
            step=1,
            key="custom_start",
        )

        custom_end = st.number_input(
            "Ending Row",
            min_value=1,
            max_value=max(
                1,
                len(df)
            ),
            value=min(
                50,
                len(df)
            ),
            step=1,
            key="custom_end",
        )

        if custom_end < custom_start:
            custom_end = custom_start

        display_df = df.iloc[
            int(custom_start) - 1:
            int(custom_end)
        ]

    st.dataframe(
        display_df,
        use_container_width=True,
        height=500,
    )


# ============================================================
# STATISTICS
# ============================================================

with tabs[1]:

    st.markdown(
        '<div class="section-heading">Statistical Analysis</div>',
        unsafe_allow_html=True,
    )

    nums = numeric_columns(df)

    if nums:

        st.markdown(
            '<div class="sub-heading">Descriptive Statistics</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            df[nums].describe(
                include="all"
            ).T,
            use_container_width=True,
        )

        st.markdown(
            '<div class="sub-heading">Skewness and Kurtosis</div>',
            unsafe_allow_html=True,
        )

        skew_report = skewness_report(
            df
        )

        if not skew_report.empty:
            st.dataframe(
                skew_report,
                use_container_width=True,
            )

        st.markdown(
            '<div class="sub-heading">Value Counts</div>',
            unsafe_allow_html=True,
        )

        value_column = st.selectbox(
            "Select Column",
            df.columns.tolist(),
            key="statistics_value_column",
        )

        value_limit = st.number_input(
            "Number of Values",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="statistics_value_limit",
        )

        counts = (
            df[value_column]
            .astype(str)
            .value_counts()
            .head(int(value_limit))
            .reset_index()
        )

        counts.columns = [
            "Value",
            "Count"
        ]

        st.dataframe(
            counts,
            use_container_width=True,
        )

    else:

        st.info(
            "No numeric columns are available for statistical analysis."
        )


# ============================================================
# DATA QUALITY
# ============================================================

with tabs[2]:

    st.markdown(
        '<div class="section-heading">Data Quality</div>',
        unsafe_allow_html=True,
    )

    quality_tabs = st.tabs(
        [
            "Missing Values",
            "Duplicates",
            "Column Types",
            "Constant Columns",
        ]
    )

    with quality_tabs[0]:

        missing = missing_report(df)

        total_missing = int(
            missing["Missing Count"].sum()
        )

        if total_missing == 0:

            st.markdown(
                """
                <div class="success-box">
                No missing values were detected in the current dataset.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.warning(
                f"{total_missing:,} missing cells detected."
            )

            st.dataframe(
                missing,
                use_container_width=True,
            )

    with quality_tabs[1]:

        duplicates = duplicate_report(
            df
        )

        st.dataframe(
            duplicates,
            use_container_width=True,
        )

        duplicate_rows = df[
            df.duplicated(
                keep=False
            )
        ]

        if not duplicate_rows.empty:

            st.markdown(
                '<div class="sub-heading">Duplicate Records</div>',
                unsafe_allow_html=True,
            )

            st.dataframe(
                duplicate_rows,
                use_container_width=True,
                height=350,
            )

    with quality_tabs[2]:

        st.dataframe(
            column_information(df),
            use_container_width=True,
        )

    with quality_tabs[3]:

        constants = detect_constant_columns(
            df
        )

        if constants.empty:
            st.success(
                "No constant columns detected."
            )
        else:
            st.dataframe(
                constants,
                use_container_width=True,
            )


# ============================================================
# EDA
# ============================================================

with tabs[3]:

    st.markdown(
        '<div class="section-heading">Exploratory Data Analysis</div>',
        unsafe_allow_html=True,
    )

    eda_type = st.selectbox(
        "Visualization Type",
        [
            "Histogram",
            "Box Plot",
            "Bar Chart",
            "Scatter Plot",
            "Correlation Heatmap",
        ],
        key="eda_type",
    )

    if eda_type == "Histogram":

        nums = numeric_columns(df)

        if nums:

            col = st.selectbox(
                "Numeric Column",
                nums,
                key="eda_hist_column",
            )

            bins = st.slider(
                "Number of Bins",
                min_value=5,
                max_value=100,
                value=30,
                key="eda_hist_bins",
            )

            fig = plot_histogram(
                df,
                col,
                bins
            )

            st.pyplot(
                fig,
                clear_figure=True
            )

        else:
            st.info(
                "No numeric columns available."
            )

    elif eda_type == "Box Plot":

        nums = numeric_columns(df)

        if nums:

            col = st.selectbox(
                "Numeric Column",
                nums,
                key="eda_box_column",
            )

            fig = plot_boxplot(
                df,
                col
            )

            st.pyplot(
                fig,
                clear_figure=True
            )

        else:
            st.info(
                "No numeric columns available."
            )

    elif eda_type == "Bar Chart":

        cols = df.columns.tolist()

        col = st.selectbox(
            "Column",
            cols,
            key="eda_bar_column",
        )

        top_n = st.slider(
            "Top Categories",
            min_value=5,
            max_value=50,
            value=10,
            key="eda_bar_top",
        )

        fig = plot_bar_chart(
            df,
            col,
            top_n
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

    elif eda_type == "Scatter Plot":

        nums = numeric_columns(df)

        if len(nums) >= 2:

            col1, col2 = st.columns(2)

            with col1:
                x_col = st.selectbox(
                    "X Column",
                    nums,
                    key="eda_scatter_x",
                )

            with col2:
                y_col = st.selectbox(
                    "Y Column",
                    nums,
                    index=1,
                    key="eda_scatter_y",
                )

            if x_col == y_col:
                st.warning(
                    "Please select two different columns."
                )
            else:

                fig = plot_scatter(
                    df,
                    x_col,
                    y_col
                )

                st.pyplot(
                    fig,
                    clear_figure=True
                )

        else:
            st.info(
                "At least two numeric columns are required."
            )

    else:

        fig = plot_correlation(
            df
        )

        if fig is None:
            st.info(
                "At least two numeric columns are required."
            )
        else:
            st.pyplot(
                fig,
                clear_figure=True
            )


# ============================================================
# CLEANING
# ============================================================

with tabs[4]:

    st.markdown(
        '<div class="section-heading">Data Cleaning</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-box">
        Cleaning is cumulative. The first cleaning operation works
        on the current dataset. After it is applied, the next cleaning
        operation works on the already-cleaned dataset. Use
        "Reset to Original" whenever you want to start again.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # CURRENT STATUS
    # --------------------------------------------------------

    clean_status_cols = st.columns(4)

    with clean_status_cols[0]:

        st.metric(
            "Current Rows",
            f"{len(df):,}"
        )

    with clean_status_cols[1]:

        st.metric(
            "Current Columns",
            f"{len(df.columns):,}"
        )

    with clean_status_cols[2]:

        st.metric(
            "Missing Cells",
            f"{int(df.isna().sum().sum()):,}"
        )

    with clean_status_cols[3]:

        st.metric(
            "Duplicate Rows",
            f"{int(df.duplicated().sum()):,}"
        )

    # --------------------------------------------------------
    # CLEANING CONTROLS
    # --------------------------------------------------------

    st.markdown(
        '<div class="sub-heading">Cleaning Operations</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:

        duplicate_treatment = st.selectbox(
            "Duplicate Treatment",
            [
                "Do Nothing",
                "Remove Duplicate Rows",
                "Keep Last Duplicate",
            ],
            key="clean_duplicate",
        )

        missing_treatment = st.selectbox(
            "Missing Value Treatment",
            [
                "Do Nothing",
                "Mean",
                "Median",
                "Mode",
                "Constant",
                "Forward Fill",
                "Backward Fill",
            ],
            key="clean_missing_treatment",
        )

        if missing_treatment == "Do Nothing":

            missing_scope = "All Columns"

        else:

            missing_scope_options = [
                "All Columns"
            ] + df.columns.tolist()

            missing_scope = st.selectbox(
                "Missing Value Scope",
                missing_scope_options,
                key="clean_missing_scope",
            )

        missing_constant = st.text_input(
            "Constant Value",
            value="0",
            key="clean_missing_constant",
        )

    with right:

        outlier_treatment = st.selectbox(
            "Outlier Treatment",
            [
                "Do Nothing",
                "Remove IQR Outliers",
            ],
            key="clean_outlier_treatment",
        )

        nums = numeric_columns(df)

        if nums:

            outlier_column = st.selectbox(
                "Outlier Column",
                nums,
                key="clean_outlier_column",
            )

            iqr_multiplier = st.number_input(
                "IQR Multiplier",
                min_value=0.5,
                max_value=5.0,
                value=1.5,
                step=0.1,
                key="clean_iqr_multiplier",
            )

        else:

            outlier_column = None
            iqr_multiplier = 1.5

        data_type_treatment = st.selectbox(
            "Data Type Treatment",
            [
                "Do Nothing",
                "Convert Numeric-Like Columns",
                "Convert Date-Like Columns",
            ],
            key="clean_type_treatment",
        )

    # --------------------------------------------------------
    # APPLY BUTTON
    # --------------------------------------------------------

    st.markdown(
        '<div class="sub-heading">Apply Cleaning</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "Apply Selected Cleaning Operations",
        use_container_width=True,
        key="apply_cleaning_button",
    ):

        try:

            before_rows = len(
                st.session_state.current_df
            )

            cleaned_df, changes = (
                apply_cleaning_operations(
                    st.session_state.current_df,
                    duplicate_treatment,
                    missing_treatment,
                    missing_scope,
                    missing_constant,
                    outlier_treatment,
                    outlier_column,
                    iqr_multiplier,
                    data_type_treatment,
                    None,
                )
            )

            st.session_state.current_df = (
                cleaned_df
            )

            after_rows = len(
                cleaned_df
            )

            add_audit(
                "Cleaning operation applied."
            )

            for change in changes:
                add_audit(change)

            st.success(
                f"Cleaning completed. Rows changed from "
                f"{before_rows:,} to {after_rows:,}."
            )

            st.rerun()

        except Exception as exc:

            st.error(
                f"Cleaning could not be completed: {exc}"
            )

    # --------------------------------------------------------
    # CURRENT CLEANED DATASET
    # --------------------------------------------------------

    st.markdown(
        '<div class="sub-heading">Current Cleaned Dataset</div>',
        unsafe_allow_html=True,
    )

    st.write(
        f"Current rows: {len(st.session_state.current_df):,}"
    )

    st.dataframe(
        st.session_state.current_df.head(100),
        use_container_width=True,
        height=450,
    )

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    if st.button(
        "Reset Current Dataset to Original",
        use_container_width=True,
        key="clean_reset_button",
    ):

        reset_dataset()

        st.success(
            "Current dataset has been restored to the original uploaded dataset."
        )

        st.rerun()


# ============================================================
# FEATURES
# ============================================================

with tabs[5]:

    st.markdown(
        '<div class="section-heading">Feature Analysis and Engineering</div>',
        unsafe_allow_html=True,
    )

    feature_action = st.selectbox(
        "Feature Operation",
        [
            "View Numeric Features",
            "View Categorical Features",
            "View Date Features",
            "Create Log Feature",
            "Create Ratio Feature",
            "Create Date Components",
            "Remove Selected Column",
        ],
        key="feature_action",
    )

    if feature_action == "View Numeric Features":

        nums = numeric_columns(df)

        st.write(
            f"Numeric features: {len(nums)}"
        )

        if nums:
            st.dataframe(
                df[nums].describe().T,
                use_container_width=True,
            )

    elif feature_action == "View Categorical Features":

        cats = categorical_columns(df)

        st.write(
            f"Categorical features: {len(cats)}"
        )

        if cats:
            st.dataframe(
                column_information(
                    df[cats]
                ),
                use_container_width=True,
            )

    elif feature_action == "View Date Features":

        dates = datetime_columns(df)

        if dates:
            st.dataframe(
                df[dates].head(100),
                use_container_width=True,
            )
        else:
            st.info(
                "No datetime columns detected."
            )

    elif feature_action == "Create Log Feature":

        nums = numeric_columns(df)

        if nums:

            source_col = st.selectbox(
                "Source Numeric Column",
                nums,
                key="feature_log_source",
            )

            new_name = st.text_input(
                "New Feature Name",
                value=f"log_{source_col}",
                key="feature_log_name",
            )

            if st.button(
                "Create Log Feature",
                key="create_log_feature",
            ):

                try:

                    values = pd.to_numeric(
                        st.session_state.current_df[
                            source_col
                        ],
                        errors="coerce",
                    )

                    shifted = (
                        values - values.min()
                    )

                    st.session_state.current_df[
                        new_name
                    ] = np.log1p(
                        shifted
                    )

                    add_audit(
                        f"Created log feature '{new_name}'."
                    )

                    st.success(
                        f"Feature '{new_name}' created."
                    )

                    st.rerun()

                except Exception as exc:
                    st.error(
                        f"Feature creation failed: {exc}"
                    )

        else:
            st.info(
                "Numeric columns are required."
            )

    elif feature_action == "Create Ratio Feature":

        nums = numeric_columns(df)

        if len(nums) >= 2:

            c1, c2 = st.columns(2)

            with c1:
                numerator = st.selectbox(
                    "Numerator",
                    nums,
                    key="ratio_numerator",
                )

            with c2:
                denominator = st.selectbox(
                    "Denominator",
                    nums,
                    index=1,
                    key="ratio_denominator",
                )

            ratio_name = st.text_input(
                "New Feature Name",
                value="ratio_feature",
                key="ratio_name",
            )

            if st.button(
                "Create Ratio Feature",
                key="create_ratio_feature",
            ):

                try:

                    denominator_values = (
                        pd.to_numeric(
                            st.session_state.current_df[
                                denominator
                            ],
                            errors="coerce",
                        )
                    )

                    numerator_values = (
                        pd.to_numeric(
                            st.session_state.current_df[
                                numerator
                            ],
                            errors="coerce",
                        )
                    )

                    denominator_values = (
                        denominator_values.replace(
                            0,
                            np.nan
                        )
                    )

                    st.session_state.current_df[
                        ratio_name
                    ] = (
                        numerator_values
                        / denominator_values
                    )

                    add_audit(
                        f"Created ratio feature '{ratio_name}'."
                    )

                    st.success(
                        f"Feature '{ratio_name}' created."
                    )

                    st.rerun()

                except Exception as exc:
                    st.error(
                        f"Ratio feature creation failed: {exc}"
                    )

        else:
            st.info(
                "At least two numeric columns are required."
            )

    elif feature_action == "Create Date Components":

        date_candidates = []

        for col in df.columns:

            if pd.api.types.is_datetime64_any_dtype(
                df[col]
            ):
                date_candidates.append(col)

        if date_candidates:

            date_col = st.selectbox(
                "Date Column",
                date_candidates,
                key="date_component_column",
            )

            if st.button(
                "Create Year, Month and Day Features",
                key="create_date_components",
            ):

                try:

                    series = pd.to_datetime(
                        st.session_state.current_df[
                            date_col
                        ],
                        errors="coerce",
                    )

                    st.session_state.current_df[
                        f"{date_col}_year"
                    ] = series.dt.year

                    st.session_state.current_df[
                        f"{date_col}_month"
                    ] = series.dt.month

                    st.session_state.current_df[
                        f"{date_col}_day"
                    ] = series.dt.day

                    st.session_state.current_df[
                        f"{date_col}_dayofweek"
                    ] = series.dt.dayofweek

                    add_audit(
                        f"Created date components from '{date_col}'."
                    )

                    st.success(
                        "Date features created."
                    )

                    st.rerun()

                except Exception as exc:
                    st.error(
                        f"Date feature creation failed: {exc}"
                    )

        else:
            st.info(
                "Convert a date-like column to datetime in Cleaning first."
            )

    else:

        column_to_remove = st.selectbox(
            "Column to Remove",
            df.columns.tolist(),
            key="feature_remove_column",
        )

        if st.button(
            "Remove Column",
            key="remove_feature_column",
        ):

            if len(df.columns) <= 1:

                st.error(
                    "At least one column must remain."
                )

            else:

                del st.session_state.current_df[
                    column_to_remove
                ]

                add_audit(
                    f"Removed column '{column_to_remove}'."
                )

                st.success(
                    f"Column '{column_to_remove}' removed."
                )

                st.rerun()


# ============================================================
# OUTLIERS
# ============================================================

with tabs[6]:

    st.markdown(
        '<div class="section-heading">Outlier Analysis</div>',
        unsafe_allow_html=True,
    )

    report = outlier_report(
        df
    )

    if report.empty:

        st.info(
            "No numeric columns are available for IQR outlier analysis."
        )

    else:

        st.dataframe(
            report,
            use_container_width=True,
        )

        nums = numeric_columns(df)

        selected_outlier_col = st.selectbox(
            "Select Column for Detailed Outlier Analysis",
            nums,
            key="outlier_analysis_column",
        )

        multiplier = st.number_input(
            "IQR Multiplier",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.1,
            key="outlier_analysis_multiplier",
        )

        lower, upper = calculate_iqr_bounds(
            df[selected_outlier_col],
            multiplier
        )

        if lower is not None:

            series = pd.to_numeric(
                df[selected_outlier_col],
                errors="coerce"
            )

            outlier_mask = (
                (series < lower)
                | (series > upper)
            )

            outlier_rows = df[
                outlier_mask
            ]

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Lower Bound",
                    f"{lower:.4f}"
                )

            with c2:
                st.metric(
                    "Upper Bound",
                    f"{upper:.4f}"
                )

            with c3:
                st.metric(
                    "Outlier Rows",
                    f"{int(outlier_mask.sum()):,}"
                )

            st.markdown(
                '<div class="sub-heading">Detected Outlier Rows</div>',
                unsafe_allow_html=True,
            )

            st.dataframe(
                outlier_rows,
                use_container_width=True,
                height=350,
            )


# ============================================================
# TARGET ANALYSIS
# ============================================================

with tabs[7]:

    st.markdown(
        '<div class="section-heading">Target Analysis</div>',
        unsafe_allow_html=True,
    )

    target = st.selectbox(
        "Select Target Column",
        df.columns.tolist(),
        key="target_analysis_column",
    )

    st.session_state.target_column = target

    target_series = df[target]

    if pd.api.types.is_numeric_dtype(
        target_series
    ):

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Mean",
                f"{target_series.mean():.4f}"
            )

        with c2:
            st.metric(
                "Median",
                f"{target_series.median():.4f}"
            )

        with c3:
            st.metric(
                "Minimum",
                f"{target_series.min():.4f}"
            )

        with c4:
            st.metric(
                "Maximum",
                f"{target_series.max():.4f}"
            )

        fig = plot_histogram(
            df,
            target,
            30
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

    else:

        counts = (
            target_series
            .astype(str)
            .value_counts()
            .head(30)
        )

        st.dataframe(
            counts.reset_index(
                name="Count"
            ),
            use_container_width=True,
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            f"Target Distribution: {target}"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )


# ============================================================
# TRAINING
# ============================================================

with tabs[8]:

    st.markdown(
        '<div class="section-heading">Machine Learning Training</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-box">
        Select the target, problem type and model. The dashboard
        automatically handles missing values, numerical scaling and
        categorical encoding during training.
        </div>
        """,
        unsafe_allow_html=True,
    )

    target_options = df.columns.tolist()

    selected_target = st.selectbox(
        "Target Column",
        target_options,
        key="training_target_column",
    )

    problem_type = st.radio(
        "Problem Type",
        [
            "Classification",
            "Regression",
        ],
        horizontal=True,
        key="training_problem_type",
    )

    if problem_type == "Classification":

        model_options = [
            "Random Forest",
            "Logistic Regression",
            "K-Nearest Neighbors",
            "Gradient Boosting",
        ]

    else:

        model_options = [
            "Random Forest",
            "Linear Regression",
            "Gradient Boosting",
        ]

    model_name = st.selectbox(
        "Model",
        model_options,
        key="training_model",
    )

    test_size_percent = st.slider(
        "Test Data Percentage",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        key="training_test_size",
    )

    if st.button(
        "Train Model",
        use_container_width=True,
        key="train_model_button",
    ):

        try:

            (
                pipeline,
                label_encoder,
                metrics,
                feature_columns,
            ) = train_model(
                df,
                selected_target,
                problem_type,
                model_name,
                test_size_percent / 100,
            )

            st.session_state.trained_model = (
                model_name
            )

            st.session_state.trained_pipeline = (
                pipeline
            )

            st.session_state.training_features = (
                feature_columns
            )

            st.session_state.training_target = (
                selected_target
            )

            st.session_state.problem_type = (
                problem_type
            )

            st.session_state.label_encoder = (
                label_encoder
            )

            st.session_state.model_metrics = (
                metrics
            )

            add_audit(
                f"Trained {model_name} for {problem_type} using target '{selected_target}'."
            )

            st.success(
                "Model trained successfully."
            )

        except Exception as exc:

            st.error(
                f"Model training failed: {exc}"
            )

    if st.session_state.trained_pipeline is not None:

        metrics = (
            st.session_state.model_metrics
        )

        st.markdown(
            '<div class="sub-heading">Training Results</div>',
            unsafe_allow_html=True,
        )

        if (
            st.session_state.problem_type
            == "Classification"
        ):

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Accuracy",
                    f"{metrics['Accuracy']:.4f}"
                )

            with c2:
                st.metric(
                    "Precision",
                    f"{metrics['Precision']:.4f}"
                )

            with c3:
                st.metric(
                    "Recall",
                    f"{metrics['Recall']:.4f}"
                )

            with c4:
                st.metric(
                    "F1 Score",
                    f"{metrics['F1 Score']:.4f}"
                )

            cm = confusion_matrix(
                metrics["Actual"],
                metrics["Predictions"]
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

            ax.set_title(
                "Confusion Matrix"
            )

            ax.set_xlabel(
                "Predicted"
            )

            ax.set_ylabel(
                "Actual"
            )

            plt.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True
            )

            report_dict = classification_report(
                metrics["Actual"],
                metrics["Predictions"],
                output_dict=True,
                zero_division=0,
            )

            st.dataframe(
                pd.DataFrame(
                    report_dict
                ).T,
                use_container_width=True,
            )

        else:

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "MAE",
                    f"{metrics['MAE']:.4f}"
                )

            with c2:
                st.metric(
                    "RMSE",
                    f"{metrics['RMSE']:.4f}"
                )

            with c3:
                st.metric(
                    "R2 Score",
                    f"{metrics['R2 Score']:.4f}"
                )

            comparison = pd.DataFrame(
                {
                    "Actual": metrics["Actual"],
                    "Predicted": metrics[
                        "Predictions"
                    ],
                }
            )

            st.dataframe(
                comparison.head(100),
                use_container_width=True,
            )

            fig, ax = plt.subplots(
                figsize=(8, 6)
            )

            ax.scatter(
                metrics["Actual"],
                metrics["Predictions"]
            )

            min_value = min(
                np.min(metrics["Actual"]),
                np.min(
                    metrics["Predictions"]
                ),
            )

            max_value = max(
                np.max(metrics["Actual"]),
                np.max(
                    metrics["Predictions"]
                ),
            )

            ax.plot(
                [min_value, max_value],
                [min_value, max_value],
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

            plt.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True
            )


# ============================================================
# MODEL COMPARISON
# ============================================================

with tabs[9]:

    st.markdown(
        '<div class="section-heading">Model Comparison</div>',
        unsafe_allow_html=True,
    )

    comparison_target = st.selectbox(
        "Target Column",
        df.columns.tolist(),
        key="comparison_target",
    )

    comparison_problem = st.radio(
        "Problem Type",
        [
            "Classification",
            "Regression",
        ],
        horizontal=True,
        key="comparison_problem",
    )

    comparison_test_size = st.slider(
        "Test Data Percentage",
        10,
        40,
        20,
        5,
        key="comparison_test",
    )

    if st.button(
        "Compare Models",
        use_container_width=True,
        key="compare_models_button",
    ):

        if comparison_problem == "Classification":

            comparison_models = [
                "Random Forest",
                "Logistic Regression",
                "K-Nearest Neighbors",
                "Gradient Boosting",
            ]

        else:

            comparison_models = [
                "Random Forest",
                "Linear Regression",
                "Gradient Boosting",
            ]

        results = []

        progress = st.progress(
            0
        )

        for index, model_name_compare in enumerate(
            comparison_models
        ):

            try:

                (
                    _pipeline,
                    _encoder,
                    metrics,
                    _features,
                ) = train_model(
                    df,
                    comparison_target,
                    comparison_problem,
                    model_name_compare,
                    comparison_test_size
                    / 100,
                )

                if (
                    comparison_problem
                    == "Classification"
                ):

                    results.append(
                        {
                            "Model": model_name_compare,
                            "Accuracy": round(
                                metrics["Accuracy"],
                                4,
                            ),
                            "Precision": round(
                                metrics["Precision"],
                                4,
                            ),
                            "Recall": round(
                                metrics["Recall"],
                                4,
                            ),
                            "F1 Score": round(
                                metrics["F1 Score"],
                                4,
                            ),
                        }
                    )

                else:

                    results.append(
                        {
                            "Model": model_name_compare,
                            "MAE": round(
                                metrics["MAE"],
                                4,
                            ),
                            "RMSE": round(
                                metrics["RMSE"],
                                4,
                            ),
                            "R2 Score": round(
                                metrics["R2 Score"],
                                4,
                            ),
                        }
                    )

            except Exception as exc:

                results.append(
                    {
                        "Model": model_name_compare,
                        "Error": str(exc),
                    }
                )

            progress.progress(
                (index + 1)
                / len(comparison_models)
            )

        comparison_df = pd.DataFrame(
            results
        )

        st.session_state["model_comparison"] = (
            comparison_df
        )

        add_audit(
            f"Compared models using target '{comparison_target}'."
        )

    if (
        "model_comparison"
        in st.session_state
    ):

        st.dataframe(
            st.session_state[
                "model_comparison"
            ],
            use_container_width=True,
        )


# ============================================================
# PREDICTION
# ============================================================

with tabs[10]:

    st.markdown(
        '<div class="section-heading">Prediction</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.trained_pipeline is None:

        st.info(
            "Train a model first in the Training section."
        )

    else:

        st.write(
            f"Model: {st.session_state.trained_model}"
        )

        st.write(
            f"Target: {st.session_state.training_target}"
        )

        feature_columns = (
            st.session_state.training_features
        )

        prediction_values = {}

        input_columns = st.columns(
            2
        )

        for index, column in enumerate(
            feature_columns
        ):

            with input_columns[
                index % 2
            ]:

                series = df[column]

                if pd.api.types.is_numeric_dtype(
                    series
                ):

                    default_value = (
                        float(
                            series.median()
                        )
                        if series.notna().any()
                        else 0.0
                    )

                    prediction_values[
                        column
                    ] = st.number_input(
                        column,
                        value=default_value,
                        key=f"prediction_{column}",
                    )

                else:

                    choices = (
                        series.dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    if choices:

                        prediction_values[
                            column
                        ] = st.selectbox(
                            column,
                            choices,
                            key=f"prediction_{column}",
                        )

                    else:

                        prediction_values[
                            column
                        ] = ""

        if st.button(
            "Make Prediction",
            use_container_width=True,
            key="make_prediction",
        ):

            try:

                input_df = pd.DataFrame(
                    [prediction_values]
                )

                prediction = (
                    st.session_state.trained_pipeline
                    .predict(input_df)
                )

                result = prediction[0]

                encoder = (
                    st.session_state.label_encoder
                )

                if (
                    encoder is not None
                    and st.session_state.problem_type
                    == "Classification"
                ):

                    try:
                        result = (
                            encoder.inverse_transform(
                                [int(result)]
                            )[0]
                        )
                    except Exception:
                        pass

                st.success(
                    f"Prediction: {result}"
                )

                add_audit(
                    "Made a new model prediction."
                )

            except Exception as exc:

                st.error(
                    f"Prediction failed: {exc}"
                )


# ============================================================
# DOWNLOAD
# ============================================================

with tabs[11]:

    st.markdown(
        '<div class="section-heading">Download Dataset</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-box">
        Select the required file format. The download contains
        the current dataset after all cleaning and feature
        operations that have been applied.
        </div>
        """,
        unsafe_allow_html=True,
    )

    download_format = st.radio(
        "Download Format",
        [
            "CSV",
            "Excel (XLSX)",
        ],
        horizontal=True,
        key="download_format",
    )

    if download_format == "CSV":

        data_bytes = (
            df.to_csv(
                index=False
            ).encode("utf-8")
        )

        file_name = (
            "cleaned_dataset.csv"
        )

        mime = "text/csv"

    else:

        excel_buffer = io.BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl"
        ) as writer:

            df.to_excel(
                writer,
                index=False,
                sheet_name="Cleaned_Data"
            )

        data_bytes = (
            excel_buffer.getvalue()
        )

        file_name = (
            "cleaned_dataset.xlsx"
        )

        mime = (
            "application/vnd.openxmlformats-officedocument"
            ".spreadsheetml.sheet"
        )

    st.write(
        f"Rows available for download: {len(df):,}"
    )

    st.write(
        f"Columns available for download: {len(df.columns):,}"
    )

    st.download_button(
        "Download Current Dataset",
        data=data_bytes,
        file_name=file_name,
        mime=mime,
        use_container_width=True,
        key="download_current_dataset",
    )


# ============================================================
# AUDIT LOG
# ============================================================

with tabs[12]:

    st.markdown(
        '<div class="section-heading">Audit Log</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "This section records important actions performed in the dashboard."
    )

    if st.session_state.audit_log:

        audit_df = pd.DataFrame(
            {
                "Action": st.session_state.audit_log
            }
        )

        st.dataframe(
            audit_df,
            use_container_width=True,
            height=450,
        )

        audit_text = "\n".join(
            st.session_state.audit_log
        )

        st.download_button(
            "Download Audit Log",
            data=audit_text,
            file_name="analytics_audit_log.txt",
            mime="text/plain",
            use_container_width=True,
            key="download_audit",
        )

    else:

        st.info(
            "No actions have been recorded yet."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <hr>
    <div style="
        text-align:center;
        color:#607086;
        font-size:14px;
        padding:10px;
    ">
        AI and ML Analytics Dashboard
        | Data Analysis
        | Data Cleaning
        | EDA
        | Feature Engineering
        | Outlier Analysis
        | Machine Learning
    </div>
    """,
    unsafe_allow_html=True,
)