# app.py
# AI and ML Analytics Dashboard
# Streamlit version
# No Gradio
# No notebook/Colab commands
# Professional dashboard with EDA, cleaning, outliers, features, target analysis,
# machine learning, dataset viewing, reset, and CSV/XLSX downloads.

import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
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
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI and ML Analytics Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# PROFESSIONAL CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    * {
        font-family: "Times New Roman", Times, serif !important;
    }

    html, body, [class*="css"] {
        font-family: "Times New Roman", Times, serif !important;
    }

    .stApp {
        background: #f4f7fb;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: "Times New Roman", Times, serif !important;
        color: #183153 !important;
    }

    p, div, span, label {
        font-family: "Times New Roman", Times, serif !important;
    }

    .dashboard-header {
        background: linear-gradient(
            135deg,
            #17365d 0%,
            #24527a 55%,
            #2f75a8 100%
        );
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 22px;
        color: white;
        box-shadow: 0 8px 25px rgba(23, 54, 93, 0.18);
    }

    .dashboard-header h1 {
        color: white !important;
        margin: 0;
        font-size: 34px;
    }

    .dashboard-header p {
        color: #eaf4ff !important;
        margin: 8px 0 0 0;
        font-size: 16px;
    }

    .section-title {
        background: #eaf2f8;
        border-left: 5px solid #2f75a8;
        padding: 10px 15px;
        border-radius: 6px;
        margin: 15px 0;
        color: #183153;
        font-size: 22px;
        font-weight: bold;
    }

    .info-box {
        background: #eaf4ff;
        border: 1px solid #c6dff5;
        border-radius: 9px;
        padding: 12px 15px;
        margin-bottom: 12px;
        color: #183153;
    }

    .success-box {
        background: #eaf7ef;
        border: 1px solid #b7dfc5;
        border-radius: 9px;
        padding: 12px 15px;
        color: #1e5631;
    }

    .warning-box {
        background: #fff7e6;
        border: 1px solid #efd49a;
        border-radius: 9px;
        padding: 12px 15px;
        color: #795600;
    }

    .metric-card {
        background: white;
        border: 1px solid #d9e2ec;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.04);
    }

    .metric-title {
        color: #66788a;
        font-size: 14px;
    }

    .metric-value {
        color: #17365d;
        font-size: 26px;
        font-weight: bold;
        margin-top: 5px;
    }

    .stButton > button {
        background: #24527a;
        color: white;
        border: none;
        border-radius: 7px;
        padding: 9px 18px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background: #17365d;
        color: white;
    }

    .stDownloadButton > button {
        background: #2e7d5b;
        color: white;
        border: none;
        border-radius: 7px;
        font-weight: bold;
    }

    .stDownloadButton > button:hover {
        background: #256b4d;
        color: white;
    }

    div[data-testid="stFileUploader"] {
        background: white;
        border: 1px solid #d9e2ec;
        border-radius: 12px;
        padding: 10px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #d9e2ec;
        border-radius: 8px;
    }

    .footer {
        text-align: center;
        color: #718096;
        font-size: 13px;
        padding-top: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "original_df" not in st.session_state:
    st.session_state.original_df = None

if "working_df" not in st.session_state:
    st.session_state.working_df = None

if "filename" not in st.session_state:
    st.session_state.filename = ""

if "cleaning_log" not in st.session_state:
    st.session_state.cleaning_log = []

if "model_result" not in st.session_state:
    st.session_state.model_result = None


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def reset_dataset():
    if st.session_state.original_df is not None:
        st.session_state.working_df = st.session_state.original_df.copy(deep=True)
        st.session_state.cleaning_log = []
        st.session_state.model_result = None
        return True
    return False


def load_dataset(uploaded_file):
    try:
        extension = uploaded_file.name.lower().split(".")[-1]

        if extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif extension == "xlsx":
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        elif extension == "xls":
            df = pd.read_excel(uploaded_file, engine="xlrd")
        else:
            return None, "Unsupported file format."

        if df.empty:
            return None, "The uploaded dataset is empty."

        df.columns = make_unique_columns(df.columns)

        return df, None

    except Exception as exc:
        return None, f"Could not read the file: {str(exc)}"


def make_unique_columns(columns):
    result = []
    counts = {}

    for col in columns:
        col = str(col).strip()

        if col == "":
            col = "Unnamed"

        if col not in counts:
            counts[col] = 0
            result.append(col)
        else:
            counts[col] += 1
            result.append(f"{col}_{counts[col]}")

    return result


def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df):
    return df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()


def datetime_columns(df):
    return df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()


def safe_numeric_conversion(df, columns):
    result = df.copy()

    for col in columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    return result


def dataset_summary(df):
    return pd.DataFrame(
        {
            "Metric": [
                "Rows",
                "Columns",
                "Numeric Columns",
                "Categorical Columns",
                "Date/Time Columns",
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
                round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            ],
        }
    )


def column_information(df):
    rows = []

    for col in df.columns:
        rows.append(
            {
                "Column": col,
                "Data Type": str(df[col].dtype),
                "Non-Null": int(df[col].notna().sum()),
                "Missing": int(df[col].isna().sum()),
                "Missing %": round(df[col].isna().mean() * 100, 2),
                "Unique": int(df[col].nunique(dropna=True)),
                "Sample Value": (
                    str(df[col].dropna().iloc[0])
                    if df[col].notna().any()
                    else ""
                ),
            }
        )

    return pd.DataFrame(rows)


def missing_report(df):
    result = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing Count": df.isna().sum().values,
            "Missing %": (
                df.isna().mean().values * 100
            ).round(2),
            "Non-Missing": df.notna().sum().values,
        }
    )

    return result.sort_values(
        "Missing Count",
        ascending=False,
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
                    duplicate_count / len(df) * 100,
                    2,
                )
                if len(df) > 0
                else 0,
            ],
        }
    )


def statistics_report(df):
    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:
        return pd.DataFrame()

    stats = numeric.describe().T

    stats["missing"] = numeric.isna().sum()
    stats["missing_%"] = numeric.isna().mean() * 100
    stats["skewness"] = numeric.skew()
    stats["kurtosis"] = numeric.kurt()

    return stats.round(4)


def value_count_report(df, column):
    if not column:
        return pd.DataFrame()

    return (
        df[column]
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="Count")
    )


def outlier_report(df):
    numeric = numeric_columns(df)

    rows = []

    for col in numeric:
        series = df[col].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (df[col] < lower) | (df[col] > upper)

        rows.append(
            {
                "Column": col,
                "Q1": round(q1, 4),
                "Q3": round(q3, 4),
                "IQR": round(iqr, 4),
                "Lower Bound": round(lower, 4),
                "Upper Bound": round(upper, 4),
                "Outlier Rows": int(mask.sum()),
                "Outlier %": round(mask.mean() * 100, 2),
            }
        )

    return pd.DataFrame(rows)


def correlation_table(df, method="pearson"):
    numeric = df.select_dtypes(include=np.number)

    if numeric.shape[1] < 2:
        return pd.DataFrame()

    return numeric.corr(method=method).round(3)


def cramers_v(x, y):
    table = pd.crosstab(x, y)

    if table.empty:
        return np.nan

    observed = table.values.astype(float)

    total = observed.sum()

    if total == 0:
        return np.nan

    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)

    expected = row_sum @ col_sum / total

    expected = np.where(expected == 0, 1e-12, expected)

    chi2 = ((observed - expected) ** 2 / expected).sum()

    phi2 = chi2 / total

    rows, cols = observed.shape

    phi2corr = max(
        0,
        phi2
        - ((cols - 1) * (rows - 1)) / max(total - 1, 1),
    )

    rcorr = rows - (
        (rows - 1) ** 2
    ) / max(total - 1, 1)

    kcorr = cols - (
        (cols - 1) ** 2
    ) / max(total - 1, 1)

    denominator = min(kcorr - 1, rcorr - 1)

    if denominator <= 0:
        return np.nan

    return np.sqrt(phi2corr / denominator)


def categorical_association(df):
    cats = categorical_columns(df)

    if len(cats) < 2:
        return pd.DataFrame()

    rows = []

    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            try:
                value = cramers_v(
                    df[cats[i]].fillna("Missing"),
                    df[cats[j]].fillna("Missing"),
                )

                rows.append(
                    {
                        "Column 1": cats[i],
                        "Column 2": cats[j],
                        "Cramer's V": (
                            round(value, 4)
                            if pd.notna(value)
                            else np.nan
                        ),
                    }
                )
            except Exception:
                pass

    return pd.DataFrame(rows).sort_values(
        "Cramer's V",
        ascending=False,
    )


def convert_possible_dates(df):
    result = df.copy()

    converted = []

    for col in result.columns:
        if result[col].dtype == "object":
            sample = result[col].dropna().astype(str).head(200)

            if len(sample) == 0:
                continue

            parsed = pd.to_datetime(
                sample,
                errors="coerce",
            )

            success_rate = parsed.notna().mean()

            if success_rate >= 0.80:
                full_parsed = pd.to_datetime(
                    result[col],
                    errors="coerce",
                )

                if full_parsed.notna().sum() > 0:
                    result[col] = full_parsed
                    converted.append(col)

    return result, converted


def create_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def create_excel_bytes(df):
    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:
        df.to_excel(
            writer,
            index=False,
            sheet_name="Cleaned_Data",
        )

    output.seek(0)

    return output.getvalue()


def apply_missing_strategy(df, strategy, constant_value="0"):
    result = df.copy()

    if strategy == "Do nothing":
        return result

    if strategy == "Mean":
        for col in numeric_columns(result):
            result[col] = result[col].fillna(
                result[col].mean()
            )

    elif strategy == "Median":
        for col in numeric_columns(result):
            result[col] = result[col].fillna(
                result[col].median()
            )

    elif strategy == "Mode":
        for col in result.columns:
            mode = result[col].mode(dropna=True)

            if not mode.empty:
                result[col] = result[col].fillna(
                    mode.iloc[0]
                )

    elif strategy == "Constant":
        result = result.fillna(constant_value)

    elif strategy == "Forward Fill":
        result = result.ffill()

    elif strategy == "Backward Fill":
        result = result.bfill()

    return result


def apply_outlier_treatment(
    df,
    treatment,
    selected_columns,
):
    result = df.copy()

    if treatment == "Do nothing":
        return result, 0

    if not selected_columns:
        return result, 0

    original_rows = len(result)

    if treatment == "Remove outlier rows":
        keep_mask = pd.Series(
            True,
            index=result.index,
        )

        for col in selected_columns:
            if col not in result.columns:
                continue

            series = result[col]

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            keep_mask &= (
                series.isna()
                | (
                    (series >= lower)
                    & (series <= upper)
                )
            )

        result = result.loc[keep_mask].copy()

    elif treatment == "Cap to IQR bounds":
        for col in selected_columns:
            if col not in result.columns:
                continue

            series = result[col]

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            result[col] = series.clip(
                lower=lower,
                upper=upper,
            )

    removed = original_rows - len(result)

    return result, removed


def build_preprocessor(df, scaling=True):
    numeric = numeric_columns(df)
    categorical = categorical_columns(df)

    transformers = []

    if numeric:
        numeric_steps = [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]

        if scaling:
            numeric_steps.append(
                (
                    "scaler",
                    StandardScaler(),
                )
            )

        numeric_pipe = Pipeline(
            numeric_steps
        )

        transformers.append(
            (
                "numeric",
                numeric_pipe,
                numeric,
            )
        )

    if categorical:
        categorical_pipe = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipe,
                categorical,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def train_model(
    df,
    target,
    problem_type,
    model_name,
    test_size,
    scaling,
):
    if target not in df.columns:
        return None, "Target column was not found."

    if df[target].nunique(dropna=True) < 2:
        return None, "Target must contain at least two different values."

    data = df.dropna(subset=[target]).copy()

    X = data.drop(columns=[target])
    y = data[target]

    if X.shape[1] == 0:
        return None, "No feature columns are available."

    if problem_type == "Classification":

        if y.dtype == "object" or str(y.dtype) == "category":
            encoder = LabelEncoder()
            y = encoder.fit_transform(
                y.astype(str)
            )
            target_classes = list(
                encoder.classes_
            )
        else:
            target_classes = list(
                pd.Series(y).dropna().unique()
            )

        if len(np.unique(y)) < 2:
            return None, "Classification requires at least two classes."

        if model_name == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=250,
                random_state=42,
                n_jobs=-1,
            )

        elif model_name == "Gradient Boosting":
            model = GradientBoostingClassifier(
                random_state=42
            )

        else:
            model = LogisticRegression(
                max_iter=2000
            )

    else:

        y = pd.to_numeric(
            y,
            errors="coerce",
        )

        valid = y.notna()

        X = X.loc[valid]
        y = y.loc[valid]

        if model_name == "Random Forest":
            model = RandomForestRegressor(
                n_estimators=250,
                random_state=42,
                n_jobs=-1,
            )

        elif model_name == "Gradient Boosting":
            model = GradientBoostingRegressor(
                random_state=42
            )

        elif model_name == "Ridge":
            model = Ridge(alpha=1.0)

        else:
            model = LinearRegression()

        target_classes = None

    if len(X) < 10:
        return None, "At least 10 usable rows are recommended for training."

    stratify = None

    if (
        problem_type == "Classification"
        and len(np.unique(y)) > 1
    ):
        class_counts = pd.Series(y).value_counts()

        if class_counts.min() >= 2:
            stratify = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
        )

    preprocessor = build_preprocessor(
        X,
        scaling=scaling,
    )

    pipeline = Pipeline(
        [
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

    predictions = pipeline.predict(X_test)

    result = {
        "pipeline": pipeline,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "target_classes": target_classes,
        "problem_type": problem_type,
        "target": target,
        "model_name": model_name,
    }

    return result, None


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="dashboard-header">
        <h1>AI and ML Analytics Dashboard</h1>
        <p>
            Data Analysis, Data Cleaning, Visualization, Feature Analysis,
            Outlier Detection, Target Analysis and Machine Learning
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SIDEBAR - DATASET
# ---------------------------------------------------------

with st.sidebar:

    st.header("Dataset")

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx", "xls"],
        help="Upload CSV, XLSX or XLS files.",
    )

    if uploaded_file is not None:

        if (
            st.session_state.filename
            != uploaded_file.name
        ):
            df_loaded, error = load_dataset(
                uploaded_file
            )

            if error:
                st.error(error)
            else:
                st.session_state.original_df = (
                    df_loaded.copy(deep=True)
                )

                st.session_state.working_df = (
                    df_loaded.copy(deep=True)
                )

                st.session_state.filename = (
                    uploaded_file.name
                )

                st.session_state.cleaning_log = []

                st.session_state.model_result = None

                st.success(
                    "Dataset loaded successfully."
                )

    if st.session_state.working_df is not None:

        st.markdown(
            f"""
            <div class="success-box">
                <b>Loaded file:</b><br>
                {st.session_state.filename}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            f"Current rows: {len(st.session_state.working_df):,}"
        )

        st.write(
            f"Current columns: {len(st.session_state.working_df.columns):,}"
        )

        if st.button(
            "Reset to Original Dataset",
            use_container_width=True,
        ):
            if reset_dataset():
                st.success(
                    "Dataset restored to the original uploaded version."
                )
                st.rerun()

# ---------------------------------------------------------
# NO DATA MESSAGE
# ---------------------------------------------------------

if st.session_state.working_df is None:

    st.markdown(
        """
        <div class="info-box">
            <b>Start here:</b>
            Upload a CSV, XLSX or XLS dataset from the left panel.
            The complete dashboard will become available after the dataset loads.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


df = st.session_state.working_df

# ---------------------------------------------------------
# TOP DATASET SUMMARY
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Dataset Summary</div>',
    unsafe_allow_html=True,
)

summary_cols = st.columns(8)

summary_values = [
    ("Rows", f"{len(df):,}"),
    ("Columns", f"{len(df.columns):,}"),
    ("Numeric", f"{len(numeric_columns(df)):,}"),
    ("Categorical", f"{len(categorical_columns(df)):,}"),
    ("Date/Time", f"{len(datetime_columns(df)):,}"),
    ("Missing", f"{int(df.isna().sum().sum()):,}"),
    ("Duplicates", f"{int(df.duplicated().sum()):,}"),
    (
        "Memory MB",
        f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}",
    ),
]

for column, (title, value) in zip(
    summary_cols,
    summary_values,
):
    with column:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

# ---------------------------------------------------------
# MAIN NAVIGATION
# ---------------------------------------------------------

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
        "Model Comparison",
        "Download",
        "Audit Log",
    ]
)

# =========================================================
# OVERVIEW
# =========================================================

with tabs[0]:

    st.markdown(
        '<div class="section-title">Overview</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Dataset Information")

    c1, c2 = st.columns(2)

    with c1:
        st.dataframe(
            dataset_summary(df),
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        st.dataframe(
            column_information(df),
            use_container_width=True,
            hide_index=True,
            height=330,
        )

    st.subheader("Dataset Viewer")

    view_mode = st.selectbox(
        "Choose rows to display",
        [
            "First 5",
            "First 10",
            "First 25",
            "First 50",
            "First 100",
            "First 147",
            "First 500",
            "Last 5",
            "Last 10",
            "Last 25",
            "Last 50",
            "Last 100",
            "Last 147",
            "Last 500",
            "Middle 50",
            "Middle 100",
            "Middle 147",
            "Middle 500",
            "All Rows",
            "Custom Range",
        ],
    )

    if view_mode == "All Rows":
        view_df = df

    elif view_mode == "Custom Range":

        start_row = st.number_input(
            "Start row",
            min_value=1,
            max_value=max(len(df), 1),
            value=1,
            step=1,
        )

        number_of_rows = st.number_input(
            "Number of rows",
            min_value=1,
            max_value=max(len(df), 1),
            value=min(50, len(df)),
            step=1,
        )

        start_index = int(start_row) - 1

        view_df = df.iloc[
            start_index : start_index
            + int(number_of_rows)
        ]

    elif view_mode.startswith("First"):

        n = int(
            view_mode.split()[1]
        )

        view_df = df.head(n)

    elif view_mode.startswith("Last"):

        n = int(
            view_mode.split()[1]
        )

        view_df = df.tail(n)

    else:

        n = int(
            view_mode.split()[1]
        )

        start = max(
            (len(df) - n) // 2,
            0,
        )

        view_df = df.iloc[
            start : start + n
        ]

    st.dataframe(
        view_df,
        use_container_width=True,
        height=450,
    )

# =========================================================
# STATISTICS
# =========================================================

with tabs[1]:

    st.markdown(
        '<div class="section-title">Statistics</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Descriptive Statistics")

    stats = statistics_report(df)

    if stats.empty:
        st.info(
            "No numeric columns are available for statistical analysis."
        )
    else:
        st.dataframe(
            stats,
            use_container_width=True,
        )

    st.subheader("Value Counts")

    if len(df.columns) > 0:

        selected_value_column = st.selectbox(
            "Select a column",
            df.columns,
            key="value_count_column",
        )

        counts = value_count_report(
            df,
            selected_value_column,
        )

        st.dataframe(
            counts,
            use_container_width=True,
            hide_index=True,
        )

# =========================================================
# DATA QUALITY
# =========================================================

with tabs[2]:

    st.markdown(
        '<div class="section-title">Data Quality</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Missing Value Analysis")

    missing = missing_report(df)

    st.dataframe(
        missing,
        use_container_width=True,
        hide_index=True,
    )

    missing_nonzero = missing[
        missing["Missing Count"] > 0
    ]

    if not missing_nonzero.empty:

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        sns.barplot(
            data=missing_nonzero,
            x="Missing %",
            y="Column",
            ax=ax,
        )

        ax.set_title(
            "Missing Values by Column"
        )

        ax.set_xlabel(
            "Missing Percentage"
        )

        ax.set_ylabel(
            "Column"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    else:
        st.success(
            "No missing values were found."
        )

    st.subheader("Duplicate Analysis")

    duplicate_info = duplicate_report(df)

    st.dataframe(
        duplicate_info,
        use_container_width=True,
        hide_index=True,
    )

    if df.duplicated().sum() > 0:

        st.warning(
            f"{int(df.duplicated().sum()):,} duplicate rows are currently present."
        )

    else:
        st.success(
            "No duplicate rows are currently present."
        )

    st.subheader("Column Information")

    st.dataframe(
        column_information(df),
        use_container_width=True,
        hide_index=True,
    )

# =========================================================
# EDA
# =========================================================

with tabs[3]:

    st.markdown(
        '<div class="section-title">Exploratory Data Analysis</div>',
        unsafe_allow_html=True,
    )

    chart_type = st.selectbox(
        "Select visualization",
        [
            "Histogram",
            "Box Plot",
            "Bar Chart",
            "Scatter Plot",
            "Correlation Heatmap",
            "Missing Values",
        ],
    )

    numeric = numeric_columns(df)
    all_columns = df.columns.tolist()

    if chart_type == "Histogram":

        if numeric:

            column = st.selectbox(
                "Numeric column",
                numeric,
                key="eda_hist",
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

            sns.histplot(
                df[column].dropna(),
                bins=bins,
                kde=True,
                ax=ax,
            )

            ax.set_title(
                f"Distribution of {column}"
            )

            st.pyplot(fig)

            plt.close(fig)

        else:
            st.info(
                "No numeric columns available."
            )

    elif chart_type == "Box Plot":

        if numeric:

            column = st.selectbox(
                "Numeric column",
                numeric,
                key="eda_box",
            )

            fig, ax = plt.subplots(
                figsize=(10, 4)
            )

            sns.boxplot(
                x=df[column],
                ax=ax,
            )

            ax.set_title(
                f"Box Plot of {column}"
            )

            st.pyplot(fig)

            plt.close(fig)

        else:
            st.info(
                "No numeric columns available."
            )

    elif chart_type == "Bar Chart":

        column = st.selectbox(
            "Column",
            all_columns,
            key="eda_bar",
        )

        top_n = st.slider(
            "Top categories",
            5,
            50,
            15,
        )

        counts = (
            df[column]
            .value_counts(dropna=False)
            .head(top_n)
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.barplot(
            x=counts.values,
            y=counts.index.astype(str),
            ax=ax,
        )

        ax.set_title(
            f"Top Values in {column}"
        )

        ax.set_xlabel("Count")
        ax.set_ylabel(column)

        st.pyplot(fig)

        plt.close(fig)

    elif chart_type == "Scatter Plot":

        if len(numeric) >= 2:

            c1, c2 = st.columns(2)

            with c1:
                x_col = st.selectbox(
                    "X-axis",
                    numeric,
                    key="eda_scatter_x",
                )

            with c2:
                y_col = st.selectbox(
                    "Y-axis",
                    [
                        c
                        for c in numeric
                        if c != x_col
                    ],
                    key="eda_scatter_y",
                )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            sns.scatterplot(
                data=df,
                x=x_col,
                y=y_col,
                ax=ax,
            )

            ax.set_title(
                f"{x_col} vs {y_col}"
            )

            st.pyplot(fig)

            plt.close(fig)

        else:
            st.info(
                "At least two numeric columns are required."
            )

    elif chart_type == "Correlation Heatmap":

        corr = correlation_table(df)

        if corr.empty:
            st.info(
                "At least two numeric columns are required."
            )
        else:

            fig, ax = plt.subplots(
                figsize=(12, 7)
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
                "Correlation Heatmap"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

            st.subheader(
                "Pearson Correlation Table"
            )

            st.dataframe(
                corr,
                use_container_width=True,
            )

    elif chart_type == "Missing Values":

        missing_plot = missing[
            missing["Missing Count"] > 0
        ]

        if missing_plot.empty:
            st.success(
                "No missing values found."
            )
        else:

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            sns.barplot(
                data=missing_plot,
                x="Missing Count",
                y="Column",
                ax=ax,
            )

            ax.set_title(
                "Missing Value Distribution"
            )

            st.pyplot(fig)

            plt.close(fig)

    st.subheader(
        "Categorical Association"
    )

    association = categorical_association(
        df
    )

    if association.empty:
        st.info(
            "At least two categorical columns are required."
        )
    else:
        st.dataframe(
            association,
            use_container_width=True,
            hide_index=True,
        )

# =========================================================
# CLEANING
# =========================================================

with tabs[4]:

    st.markdown(
        '<div class="section-title">Data Cleaning</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-box">
            Cleaning operations are applied to the current dataset.
            You can apply one operation, inspect the result, and then apply another operation.
            The Reset button always returns to the original uploaded dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_rows_before = len(df)
    current_columns_before = len(df.columns)

    st.write(
        f"Current dataset: {current_rows_before:,} rows × "
        f"{current_columns_before:,} columns"
    )

    st.subheader("Duplicate Treatment")

    duplicate_action = st.selectbox(
        "Duplicate treatment",
        [
            "Do nothing",
            "Remove duplicate rows",
        ],
        key="clean_duplicates",
    )

    st.subheader("Missing Value Treatment")

    missing_strategy = st.selectbox(
        "Missing value strategy",
        [
            "Do nothing",
            "Mean",
            "Median",
            "Mode",
            "Constant",
            "Forward Fill",
            "Backward Fill",
        ],
        key="clean_missing",
    )

    constant_value = "0"

    if missing_strategy == "Constant":

        constant_value = st.text_input(
            "Constant value",
            value="0",
            key="constant_value",
        )

    st.subheader("Data Type Treatment")

    datatype_action = st.selectbox(
        "Data type action",
        [
            "Do nothing",
            "Convert numeric-looking columns",
            "Detect and convert date columns",
        ],
        key="clean_datatype",
    )

    st.subheader("Outlier Treatment")

    outlier_action = st.selectbox(
        "Outlier action",
        [
            "Do nothing",
            "Remove outlier rows",
            "Cap to IQR bounds",
        ],
        key="clean_outlier",
    )

    if numeric_columns(df):

        outlier_columns = st.multiselect(
            "Numeric columns for outlier treatment",
            numeric_columns(df),
            default=[],
            key="clean_outlier_columns",
        )

    else:

        outlier_columns = []

        st.info(
            "No numeric columns are available for outlier treatment."
        )

    st.subheader("Numeric Precision")

    numeric_precision = st.number_input(
        "Decimal places",
        min_value=0,
        max_value=10,
        value=4,
        step=1,
        key="clean_precision",
    )

    st.subheader("Apply Cleaning")

    if st.button(
        "Apply Selected Cleaning Operations",
        type="primary",
        use_container_width=True,
        key="apply_cleaning",
    ):

        working = st.session_state.working_df.copy(
            deep=True
        )

        log_entries = []

        # Duplicate cleaning
        if (
            duplicate_action
            == "Remove duplicate rows"
        ):

            before = len(working)

            working = working.drop_duplicates().copy()

            removed = before - len(working)

            log_entries.append(
                f"Duplicate treatment: removed {removed:,} duplicate rows."
            )

        # Missing treatment
        if missing_strategy != "Do nothing":

            before_missing = int(
                working.isna().sum().sum()
            )

            working = apply_missing_strategy(
                working,
                missing_strategy,
                constant_value,
            )

            after_missing = int(
                working.isna().sum().sum()
            )

            log_entries.append(
                f"Missing value treatment: {missing_strategy}. "
                f"Missing cells changed from {before_missing:,} to {after_missing:,}."
            )

        # Data type treatment
        if (
            datatype_action
            == "Convert numeric-looking columns"
        ):

            converted_columns = []

            for col in working.columns:

                if working[col].dtype == "object":

                    original_non_null = (
                        working[col].notna().sum()
                    )

                    if original_non_null == 0:
                        continue

                    converted = pd.to_numeric(
                        working[col],
                        errors="coerce",
                    )

                    converted_non_null = (
                        converted.notna().sum()
                    )

                    if (
                        converted_non_null
                        >= original_non_null * 0.90
                    ):
                        working[col] = converted
                        converted_columns.append(
                            col
                        )

            log_entries.append(
                "Numeric conversion completed. "
                f"Converted columns: {', '.join(converted_columns) if converted_columns else 'None'}."
            )

        elif (
            datatype_action
            == "Detect and convert date columns"
        ):

            working, converted_dates = (
                convert_possible_dates(
                    working
                )
            )

            log_entries.append(
                "Date detection completed. "
                f"Converted columns: {', '.join(converted_dates) if converted_dates else 'None'}."
            )

        # Outlier treatment
        if (
            outlier_action != "Do nothing"
            and outlier_columns
        ):

            working, removed = (
                apply_outlier_treatment(
                    working,
                    outlier_action,
                    outlier_columns,
                )
            )

            if (
                outlier_action
                == "Remove outlier rows"
            ):
                log_entries.append(
                    f"Outlier treatment: removed {removed:,} rows "
                    f"from the selected columns."
                )

            else:
                log_entries.append(
                    "Outlier treatment: values were capped to IQR bounds."
                )

        # Numeric rounding
        if numeric_precision >= 0:

            numeric_now = numeric_columns(
                working
            )

            if numeric_now:

                working[
                    numeric_now
                ] = working[
                    numeric_now
                ].round(
                    int(numeric_precision)
                )

                log_entries.append(
                    f"Numeric columns rounded to {int(numeric_precision)} decimal places."
                )

        # Save
        st.session_state.working_df = (
            working.copy(deep=True)
        )

        st.session_state.cleaning_log.extend(
            log_entries
        )

        st.success(
            "Cleaning operation completed successfully."
        )

        st.rerun()

    st.subheader("Current Cleaned Dataset")

    st.dataframe(
        st.session_state.working_df,
        use_container_width=True,
        height=420,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Current Rows",
            f"{len(st.session_state.working_df):,}",
        )

    with c2:
        st.metric(
            "Current Columns",
            f"{len(st.session_state.working_df.columns):,}",
        )

    with c3:
        st.metric(
            "Missing Cells",
            f"{int(st.session_state.working_df.isna().sum().sum()):,}",
        )

    st.subheader("Reset Dataset")

    if st.button(
        "Reset Everything to Original Dataset",
        use_container_width=True,
        key="clean_reset",
    ):

        if reset_dataset():
            st.success(
                "All cleaning changes have been removed."
            )
            st.rerun()

# =========================================================
# FEATURES
# =========================================================

with tabs[5]:

    st.markdown(
        '<div class="section-title">Feature Analysis</div>',
        unsafe_allow_html=True,
    )

    numeric = numeric_columns(df)
    categorical = categorical_columns(df)

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Numeric Features")

        if numeric:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Numeric Feature": numeric
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No numeric features."
            )

    with c2:

        st.subheader("Categorical Features")

        if categorical:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Categorical Feature": categorical
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No categorical features."
            )

    st.subheader("Feature Cardinality")

    cardinality = pd.DataFrame(
        {
            "Column": df.columns,
            "Unique Values": [
                df[col].nunique(
                    dropna=True
                )
                for col in df.columns
            ],
            "Unique %": [
                round(
                    df[col].nunique(
                        dropna=True
                    )
                    / max(len(df), 1)
                    * 100,
                    2,
                )
                for col in df.columns
            ],
        }
    ).sort_values(
        "Unique Values",
        ascending=False,
    )

    st.dataframe(
        cardinality,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Numeric Feature Distribution")

    if numeric:

        feature = st.selectbox(
            "Select numeric feature",
            numeric,
            key="feature_distribution",
        )

        series = df[feature].dropna()

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        feature_info = pd.DataFrame(
            {
                "Metric": [
                    "Mean",
                    "Median",
                    "Minimum",
                    "Maximum",
                    "Standard Deviation",
                    "Skewness",
                    "Kurtosis",
                    "Q1",
                    "Q3",
                ],
                "Value": [
                    series.mean(),
                    series.median(),
                    series.min(),
                    series.max(),
                    series.std(),
                    series.skew(),
                    series.kurt(),
                    q1,
                    q3,
                ],
            }
        )

        st.dataframe(
            feature_info.round(4),
            use_container_width=True,
            hide_index=True,
        )

# =========================================================
# OUTLIERS
# =========================================================

with tabs[6]:

    st.markdown(
        '<div class="section-title">Outlier Analysis</div>',
        unsafe_allow_html=True,
    )

    report = outlier_report(df)

    if report.empty:

        st.info(
            "No numeric columns are available."
        )

    else:

        st.dataframe(
            report,
            use_container_width=True,
            hide_index=True,
        )

        selected_outlier_column = st.selectbox(
            "Select column for detailed outlier view",
            numeric_columns(df),
            key="detailed_outlier",
        )

        series = df[
            selected_outlier_column
        ].dropna()

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        detailed = df[
            (
                df[selected_outlier_column]
                < lower
            )
            | (
                df[selected_outlier_column]
                > upper
            )
        ]

        st.write(
            f"Lower bound: {lower:.4f}"
        )

        st.write(
            f"Upper bound: {upper:.4f}"
        )

        st.write(
            f"Outlier rows: {len(detailed):,}"
        )

        if not detailed.empty:
            st.dataframe(
                detailed,
                use_container_width=True,
                height=350,
            )

# =========================================================
# TARGET ANALYSIS
# =========================================================

with tabs[7]:

    st.markdown(
        '<div class="section-title">Target Analysis</div>',
        unsafe_allow_html=True,
    )

    target = st.selectbox(
        "Select target column",
        df.columns,
        key="target_analysis_column",
    )

    target_series = df[target]

    st.subheader("Target Summary")

    target_summary = pd.DataFrame(
        {
            "Metric": [
                "Data Type",
                "Rows",
                "Missing",
                "Unique Values",
            ],
            "Value": [
                str(target_series.dtype),
                len(target_series),
                int(target_series.isna().sum()),
                int(
                    target_series.nunique(
                        dropna=True
                    )
                ),
            ],
        }
    )

    st.dataframe(
        target_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Target Distribution")

    counts = (
        target_series
        .value_counts(dropna=False)
        .head(30)
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        x=counts.values,
        y=counts.index.astype(str),
        ax=ax,
    )

    ax.set_title(
        f"Distribution of Target: {target}"
    )

    ax.set_xlabel("Count")
    ax.set_ylabel(target)

    st.pyplot(fig)

    plt.close(fig)

    if pd.api.types.is_numeric_dtype(
        target_series
    ):

        st.subheader(
            "Target Correlation with Numeric Features"
        )

        numeric = numeric_columns(df)

        if len(numeric) > 1:

            target_corr = (
                df[numeric]
                .corr()[target]
                .drop(target)
                .sort_values(
                    key=lambda x: x.abs(),
                    ascending=False,
                )
                .to_frame(
                    "Correlation"
                )
            )

            st.dataframe(
                target_corr.round(4),
                use_container_width=True,
            )

# =========================================================
# MODEL TRAINING
# =========================================================

with tabs[8]:

    st.markdown(
        '<div class="section-title">Model Training</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-box">
            The model pipeline automatically handles missing values,
            categorical variables and numeric features.
        </div>
        """,
        unsafe_allow_html=True,
    )

    problem_type = st.radio(
        "Problem type",
        [
            "Classification",
            "Regression",
        ],
        horizontal=True,
        key="problem_type",
    )

    target_column = st.selectbox(
        "Target column",
        df.columns,
        key="training_target",
    )

    if problem_type == "Classification":

        model_name = st.selectbox(
            "Model",
            [
                "Random Forest",
                "Gradient Boosting",
                "Logistic Regression",
            ],
            key="classification_model",
        )

    else:

        model_name = st.selectbox(
            "Model",
            [
                "Random Forest",
                "Gradient Boosting",
                "Linear Regression",
                "Ridge",
            ],
            key="regression_model",
        )

    test_size_percent = st.slider(
        "Test size (%)",
        10,
        40,
        20,
        key="test_size",
    )

    scaling = st.checkbox(
        "Apply feature scaling",
        value=True,
        key="training_scaling",
    )

    if st.button(
        "Train Model",
        type="primary",
        use_container_width=True,
        key="train_model",
    ):

        with st.spinner(
            "Training model..."
        ):

            result, error = train_model(
                df=df,
                target=target_column,
                problem_type=problem_type,
                model_name=model_name,
                test_size=test_size_percent
                / 100,
                scaling=scaling,
            )

        if error:

            st.error(error)

        else:

            st.session_state.model_result = result

            st.success(
                "Model trained successfully."
            )

    result = st.session_state.model_result

    if result is not None:

        st.subheader("Model Results")

        y_test = result["y_test"]
        predictions = result["predictions"]

        if result["problem_type"] == "Classification":

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

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.metric(
                    "Accuracy",
                    f"{accuracy:.4f}",
                )

            with m2:
                st.metric(
                    "Precision",
                    f"{precision:.4f}",
                )

            with m3:
                st.metric(
                    "Recall",
                    f"{recall:.4f}",
                )

            with m4:
                st.metric(
                    "F1 Score",
                    f"{f1:.4f}",
                )

            st.subheader(
                "Confusion Matrix"
            )

            cm = confusion_matrix(
                y_test,
                predictions,
            )

            fig, ax = plt.subplots(
                figsize=(7, 6)
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

        else:

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

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric(
                    "MAE",
                    f"{mae:.4f}",
                )

            with m2:
                st.metric(
                    "RMSE",
                    f"{rmse:.4f}",
                )

            with m3:
                st.metric(
                    "R2 Score",
                    f"{r2:.4f}",
                )

            st.subheader(
                "Actual vs Predicted"
            )

            prediction_df = pd.DataFrame(
                {
                    "Actual": y_test.values,
                    "Predicted": predictions,
                    "Error": (
                        y_test.values
                        - predictions
                    ),
                }
            )

            st.dataframe(
                prediction_df.head(100),
                use_container_width=True,
            )

            fig, ax = plt.subplots(
                figsize=(9, 5)
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

# =========================================================
# MODEL COMPARISON
# =========================================================

with tabs[9]:

    st.markdown(
        '<div class="section-title">Model Comparison</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Compare multiple models using the same target and train/test split."
    )

    comparison_problem = st.radio(
        "Problem type",
        [
            "Classification",
            "Regression",
        ],
        horizontal=True,
        key="comparison_problem",
    )

    comparison_target = st.selectbox(
        "Target",
        df.columns,
        key="comparison_target",
    )

    comparison_test_size = st.slider(
        "Test size (%)",
        10,
        40,
        20,
        key="comparison_test",
    )

    if st.button(
        "Run Model Comparison",
        type="primary",
        use_container_width=True,
        key="run_comparison",
    ):

        if comparison_problem == "Classification":

            models = [
                "Random Forest",
                "Gradient Boosting",
                "Logistic Regression",
            ]

        else:

            models = [
                "Random Forest",
                "Gradient Boosting",
                "Linear Regression",
                "Ridge",
            ]

        comparison_rows = []

        progress = st.progress(0)

        for index, model in enumerate(models):

            result, error = train_model(
                df=df,
                target=comparison_target,
                problem_type=comparison_problem,
                model_name=model,
                test_size=comparison_test_size
                / 100,
                scaling=True,
            )

            if result is not None:

                y_test = result["y_test"]
                predictions = result[
                    "predictions"
                ]

                if (
                    comparison_problem
                    == "Classification"
                ):

                    comparison_rows.append(
                        {
                            "Model": model,
                            "Accuracy": round(
                                accuracy_score(
                                    y_test,
                                    predictions,
                                ),
                                4,
                            ),
                            "Precision": round(
                                precision_score(
                                    y_test,
                                    predictions,
                                    average="weighted",
                                    zero_division=0,
                                ),
                                4,
                            ),
                            "Recall": round(
                                recall_score(
                                    y_test,
                                    predictions,
                                    average="weighted",
                                    zero_division=0,
                                ),
                                4,
                            ),
                            "F1": round(
                                f1_score(
                                    y_test,
                                    predictions,
                                    average="weighted",
                                    zero_division=0,
                                ),
                                4,
                            ),
                        }
                    )

                else:

                    comparison_rows.append(
                        {
                            "Model": model,
                            "MAE": round(
                                mean_absolute_error(
                                    y_test,
                                    predictions,
                                ),
                                4,
                            ),
                            "RMSE": round(
                                np.sqrt(
                                    mean_squared_error(
                                        y_test,
                                        predictions,
                                    )
                                ),
                                4,
                            ),
                            "R2": round(
                                r2_score(
                                    y_test,
                                    predictions,
                                ),
                                4,
                            ),
                        }
                    )

            progress.progress(
                (index + 1) / len(models)
            )

        if comparison_rows:

            comparison_df = pd.DataFrame(
                comparison_rows
            )

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True,
            )

            if comparison_problem == "Classification":

                metric = st.selectbox(
                    "Comparison metric",
                    [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1",
                    ],
                )

            else:

                metric = st.selectbox(
                    "Comparison metric",
                    [
                        "MAE",
                        "RMSE",
                        "R2",
                    ],
                )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            sns.barplot(
                data=comparison_df,
                x="Model",
                y=metric,
                ax=ax,
            )

            ax.set_title(
                f"Model Comparison - {metric}"
            )

            ax.tick_params(
                axis="x",
                rotation=20,
            )

            st.pyplot(fig)

            plt.close(fig)

        else:

            st.error(
                "No model could be trained with the current dataset."
            )

# =========================================================
# DOWNLOAD
# =========================================================

with tabs[10]:

    st.markdown(
        '<div class="section-title">Download Dataset</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-box">
            Download the current dataset after all applied cleaning operations.
            The downloaded file contains the current version, not the original version.
        </div>
        """,
        unsafe_allow_html=True,
    )

    download_df = st.session_state.working_df

    st.subheader("Current Dataset")

    st.write(
        f"Rows: {len(download_df):,}"
    )

    st.write(
        f"Columns: {len(download_df.columns):,}"
    )

    st.dataframe(
        download_df.head(50),
        use_container_width=True,
        height=350,
    )

    st.subheader("Choose Download Format")

    download_format = st.radio(
        "File format",
        [
            "CSV",
            "Excel XLSX",
        ],
        horizontal=True,
        key="download_format",
    )

    base_name = (
        st.session_state.filename
        .rsplit(".", 1)[0]
        if st.session_state.filename
        else "cleaned_dataset"
    )

    if download_format == "CSV":

        st.download_button(
            label="Download Cleaned CSV",
            data=create_csv_bytes(
                download_df
            ),
            file_name=f"{base_name}_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_csv",
        )

    else:

        st.download_button(
            label="Download Cleaned Excel",
            data=create_excel_bytes(
                download_df
            ),
            file_name=f"{base_name}_cleaned.xlsx",
            mime=(
                "application/"
                "vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
            key="download_excel",
        )

    st.subheader("Original Dataset")

    original_df = st.session_state.original_df

    if original_df is not None:

        if st.button(
            "Download Original CSV",
            use_container_width=True,
            key="download_original_csv",
        ):

            st.download_button(
                label="Download Original CSV File",
                data=create_csv_bytes(
                    original_df
                ),
                file_name=f"{base_name}_original.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_original_csv_actual",
            )

# =========================================================
# AUDIT LOG
# =========================================================

with tabs[11]:

    st.markdown(
        '<div class="section-title">Cleaning Audit Log</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.cleaning_log:

        log_df = pd.DataFrame(
            {
                "Step": range(
                    1,
                    len(
                        st.session_state.cleaning_log
                    )
                    + 1,
                ),
                "Operation": st.session_state.cleaning_log,
            }
        )

        st.dataframe(
            log_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No cleaning operations have been applied yet."
        )

    st.subheader("Dataset State")

    state_df = pd.DataFrame(
        {
            "Metric": [
                "Original Rows",
                "Current Rows",
                "Rows Changed",
                "Original Columns",
                "Current Columns",
                "Missing Cells",
                "Duplicate Rows",
            ],
            "Value": [
                len(
                    st.session_state.original_df
                ),
                len(
                    st.session_state.working_df
                ),
                len(
                    st.session_state.original_df
                )
                - len(
                    st.session_state.working_df
                ),
                len(
                    st.session_state.original_df.columns
                ),
                len(
                    st.session_state.working_df.columns
                ),
                int(
                    st.session_state.working_df
                    .isna()
                    .sum()
                    .sum()
                ),
                int(
                    st.session_state.working_df
                    .duplicated()
                    .sum()
                ),
            ],
        }
    )

    st.dataframe(
        state_df,
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        AI and ML Analytics Dashboard | Data Analysis | Data Cleaning |
        Exploratory Data Analysis | Feature Analysis | Outlier Detection |
        Target Analysis | Machine Learning
    </div>
    """,
    unsafe_allow_html=True,
)