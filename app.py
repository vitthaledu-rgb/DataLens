# app.py
# AI and ML Analytics Dashboard
# Streamlit application
# No Gradio required.
# Designed for CSV, XLSX and XLS datasets.

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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
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
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI and ML Analytics Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Global font */
    html, body, [class*="css"], .stApp {
        font-family: "Times New Roman", Times, serif !important;
    }

    .stApp {
        background-color: #f7f8fa;
    }

    /* Main container */
    .main .block-container {
        max-width: 1450px;
        padding-top: 25px;
        padding-bottom: 40px;
    }

    /* Remove Streamlit decoration */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Main title */
    .dashboard-title {
        font-family: "Times New Roman", Times, serif;
        font-size: 38px;
        font-weight: 700;
        text-align: center;
        color: #172033;
        margin-bottom: 4px;
    }

    .dashboard-subtitle {
        font-family: "Times New Roman", Times, serif;
        font-size: 17px;
        text-align: center;
        color: #687386;
        margin-bottom: 25px;
    }

    /* Section headings */
    .section-title {
        font-family: "Times New Roman", Times, serif;
        font-size: 25px;
        font-weight: 700;
        color: #172033;
        border-bottom: 1px solid #d8dde6;
        padding-bottom: 8px;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .sub-title {
        font-family: "Times New Roman", Times, serif;
        font-size: 20px;
        font-weight: 700;
        color: #26344d;
        margin-top: 16px;
        margin-bottom: 10px;
    }

    /* Cards */
    .metric-card {
        background: white;
        border: 1px solid #dfe4ec;
        border-radius: 8px;
        padding: 18px 12px;
        text-align: center;
        min-height: 105px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    .metric-label {
        font-size: 14px;
        color: #687386;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 27px;
        font-weight: 700;
        color: #172033;
    }

    /* Information boxes */
    .info-box {
        background: #eef5ff;
        border: 1px solid #cdddf5;
        border-radius: 7px;
        padding: 12px 15px;
        color: #25344d;
        margin-bottom: 15px;
    }

    .success-box {
        background: #edf8f0;
        border: 1px solid #c8e6cf;
        border-radius: 7px;
        padding: 12px 15px;
        color: #245a31;
        margin-bottom: 15px;
    }

    .warning-box {
        background: #fff8e8;
        border: 1px solid #f0dfae;
        border-radius: 7px;
        padding: 12px 15px;
        color: #6c5518;
        margin-bottom: 15px;
    }

    /* Buttons */
    .stButton > button {
        font-family: "Times New Roman", Times, serif !important;
        border-radius: 6px !important;
        min-height: 42px;
        font-size: 15px !important;
    }

    .stDownloadButton > button {
        font-family: "Times New Roman", Times, serif !important;
        border-radius: 6px !important;
        min-height: 42px;
        font-size: 15px !important;
    }

    /* Inputs */
    .stSelectbox label,
    .stMultiSelect label,
    .stNumberInput label,
    .stSlider label,
    .stTextInput label,
    .stRadio label,
    .stCheckbox label,
    .stFileUploader label {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 15px !important;
        color: #26344d !important;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #dfe4ec;
        border-radius: 6px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        border-bottom: 1px solid #d5dbe5;
        overflow-x: auto;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 14px !important;
        padding: 10px 14px !important;
        color: #26344d !important;
    }

    .stTabs [aria-selected="true"] {
        color: #c62828 !important;
        font-weight: 700 !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border: 1px solid #dfe4ec;
        border-radius: 8px;
        padding: 10px;
    }

    /* Horizontal rule */
    hr {
        border: none;
        border-top: 1px solid #dfe4ec;
        margin: 20px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

def initialize_state():
    defaults = {
        "original_df": None,
        "working_df": None,
        "file_name": "",
        "audit_log": [],
        "target_column": None,
        "trained_model": None,
        "model_scaler": None,
        "model_features": [],
        "model_type": None,
        "model_results": None,
        "label_encoder": None,
        "prediction_info": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def add_log(action, before_rows=None, after_rows=None, details=""):
    st.session_state.audit_log.append(
        {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Action": action,
            "Rows Before": before_rows,
            "Rows After": after_rows,
            "Details": details,
        }
    )


def reset_application():
    st.session_state.original_df = None
    st.session_state.working_df = None
    st.session_state.file_name = ""
    st.session_state.audit_log = []
    st.session_state.target_column = None
    st.session_state.trained_model = None
    st.session_state.model_scaler = None
    st.session_state.model_features = []
    st.session_state.model_type = None
    st.session_state.model_results = None
    st.session_state.label_encoder = None
    st.session_state.prediction_info = None


def load_dataset(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin1")

    if filename.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, engine="openpyxl")

    if filename.endswith(".xls"):
        try:
            return pd.read_excel(uploaded_file)
        except Exception:
            st.error(
                "XLS files require the xlrd package. "
                "Use XLSX or add xlrd to requirements.txt."
            )
            return None

    st.error("Unsupported file format. Please upload CSV, XLSX or XLS.")
    return None


def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df):
    return df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()


def date_columns(df):
    result = []

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            result.append(col)
        else:
            if df[col].dtype == "object":
                sample = df[col].dropna().head(100)

                if len(sample) > 0:
                    converted = pd.to_datetime(
                        sample,
                        errors="coerce"
                    )

                    if converted.notna().mean() >= 0.8:
                        result.append(col)

    return result


def make_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def make_excel(df):
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

    output.seek(0)
    return output.getvalue()


def safe_numeric(df, columns):
    result = df[columns].copy()

    for col in columns:
        result[col] = pd.to_numeric(
            result[col],
            errors="coerce"
        )

    return result


def detect_constant_columns(df):
    return [
        col for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    ]


def detect_high_cardinality_columns(df, threshold=0.8):
    result = []

    for col in categorical_columns(df):
        unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)

        if unique_ratio >= threshold:
            result.append(col)

    return result


def skewness_table(df):
    nums = numeric_columns(df)

    if not nums:
        return pd.DataFrame()

    rows = []

    for col in nums:
        series = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if len(series) > 2:
            rows.append(
                {
                    "Column": col,
                    "Skewness": series.skew(),
                    "Kurtosis": series.kurtosis(),
                }
            )

    return pd.DataFrame(rows)


def missing_report(df):
    result = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [str(df[c].dtype) for c in df.columns],
            "Missing Count": [df[c].isna().sum() for c in df.columns],
            "Missing %": [
                round(df[c].isna().mean() * 100, 2)
                for c in df.columns
            ],
            "Unique Values": [
                df[c].nunique(dropna=True)
                for c in df.columns
            ],
        }
    )

    return result.sort_values(
        "Missing Count",
        ascending=False
    ).reset_index(drop=True)


def duplicate_report(df):
    duplicated = df.duplicated()

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
                int(duplicated.sum()),
                int((~duplicated).sum()),
                round(duplicated.mean() * 100, 2),
            ],
        }
    )


def outlier_report(df):
    nums = numeric_columns(df)

    rows = []

    for col in nums:
        s = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if len(s) < 4:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(((s < lower) | (s > upper)).sum())

        rows.append(
            {
                "Column": col,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Lower Bound": lower,
                "Upper Bound": upper,
                "Outlier Count": count,
                "Outlier %": round(
                    count / len(s) * 100,
                    2
                ),
            }
        )

    return pd.DataFrame(rows)


def value_counts_table(df, column):
    vc = (
        df[column]
        .value_counts(dropna=False)
        .reset_index()
    )

    vc.columns = [
        column,
        "Count"
    ]

    vc["Percentage"] = (
        vc["Count"] / len(df) * 100
    ).round(2)

    return vc


def type_report(df):
    rows = []

    for col in df.columns:
        rows.append(
            {
                "Column": col,
                "Current Type": str(df[col].dtype),
                "Non-Null": int(df[col].notna().sum()),
                "Unique": int(df[col].nunique(dropna=True)),
                "Sample": str(df[col].dropna().iloc[0])
                if df[col].notna().any()
                else "",
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# DATA VALIDATION
# ============================================================

def validation_report(df):
    rows = []

    for col in df.columns:
        series = df[col]

        negative_count = 0

        if pd.api.types.is_numeric_dtype(series):
            negative_count = int((series < 0).sum())

        rows.append(
            {
                "Column": col,
                "Missing": int(series.isna().sum()),
                "Infinite": (
                    int(np.isinf(series).sum())
                    if pd.api.types.is_numeric_dtype(series)
                    else 0
                ),
                "Negative Values": negative_count,
                "Unique Values": int(series.nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# CLEANING ENGINE
# ============================================================

def apply_cleaning(
    df,
    duplicate_action,
    missing_action,
    constant_value,
    outlier_column,
    outlier_action,
    numeric_conversion,
    categorical_conversion,
):
    new_df = df.copy()

    before_rows = len(new_df)

    # ----------------------------------------
    # Duplicate treatment
    # ----------------------------------------

    if duplicate_action == "Remove duplicate rows":
        new_df = new_df.drop_duplicates()

    # ----------------------------------------
    # Numeric conversion
    # ----------------------------------------

    if numeric_conversion:
        for col in numeric_conversion:
            if col in new_df.columns:
                new_df[col] = pd.to_numeric(
                    new_df[col],
                    errors="coerce"
                )

    # ----------------------------------------
    # Categorical conversion
    # ----------------------------------------

    if categorical_conversion:
        for col in categorical_conversion:
            if col in new_df.columns:
                new_df[col] = new_df[col].astype("string")

    # ----------------------------------------
    # Missing values
    # ----------------------------------------

    if missing_action != "Do nothing":

        nums = numeric_columns(new_df)
        cats = categorical_columns(new_df)

        if missing_action == "Mean":
            for col in nums:
                if new_df[col].isna().any():
                    new_df[col] = new_df[col].fillna(
                        new_df[col].mean()
                    )

        elif missing_action == "Median":
            for col in nums:
                if new_df[col].isna().any():
                    new_df[col] = new_df[col].fillna(
                        new_df[col].median()
                    )

        elif missing_action == "Mode":
            for col in new_df.columns:
                mode = new_df[col].mode(dropna=True)

                if not mode.empty:
                    new_df[col] = new_df[col].fillna(
                        mode.iloc[0]
                    )

        elif missing_action == "Constant":
            new_df = new_df.fillna(constant_value)

        elif missing_action == "Forward Fill":
            new_df = new_df.ffill()

        elif missing_action == "Backward Fill":
            new_df = new_df.bfill()

        elif missing_action == "Drop rows with missing values":
            new_df = new_df.dropna()

        elif missing_action == "Drop columns with missing values":
            new_df = new_df.dropna(axis=1)

        elif missing_action == "Mean + Mode":
            for col in nums:
                new_df[col] = new_df[col].fillna(
                    new_df[col].mean()
                )

            for col in cats:
                mode = new_df[col].mode(dropna=True)

                if not mode.empty:
                    new_df[col] = new_df[col].fillna(
                        mode.iloc[0]
                    )

    # ----------------------------------------
    # Outlier treatment
    # ----------------------------------------

    if (
        outlier_action != "Do nothing"
        and outlier_column
        and outlier_column in new_df.columns
        and pd.api.types.is_numeric_dtype(
            new_df[outlier_column]
        )
    ):
        s = pd.to_numeric(
            new_df[outlier_column],
            errors="coerce"
        )

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        if outlier_action == "Remove IQR outliers":
            mask = (
                (s >= lower) &
                (s <= upper)
            )

            new_df = new_df[
                mask | s.isna()
            ].copy()

        elif outlier_action == "Cap IQR outliers":
            new_df[outlier_column] = s.clip(
                lower,
                upper
            )

        elif outlier_action == "Replace outliers with median":
            mask = (
                (s < lower) |
                (s > upper)
            )

            new_df.loc[mask, outlier_column] = s.median()

    new_df = new_df.reset_index(drop=True)

    return new_df, before_rows, len(new_df)


# ============================================================
# MODEL PREPARATION
# ============================================================

def prepare_ml_data(df, target):
    data = df.copy()

    if target not in data.columns:
        raise ValueError("Target column not found.")

    y = data[target].copy()
    X = data.drop(columns=[target])

    # Remove obviously unsuitable columns
    drop_cols = []

    for col in X.columns:
        if X[col].nunique(dropna=True) <= 1:
            drop_cols.append(col)

        elif X[col].dtype == "object":
            unique_ratio = (
                X[col].nunique(dropna=True) /
                max(len(X), 1)
            )

            if unique_ratio > 0.9:
                drop_cols.append(col)

    X = X.drop(columns=drop_cols, errors="ignore")

    numeric = numeric_columns(X)
    categorical = categorical_columns(X)

    # Numeric values
    if numeric:
        num_imputer = SimpleImputer(
            strategy="median"
        )

        X[numeric] = num_imputer.fit_transform(
            X[numeric]
        )

    # Categorical values
    for col in categorical:
        X[col] = X[col].fillna("Missing").astype(str)

    # One-hot encoding
    X = pd.get_dummies(
        X,
        columns=categorical,
        drop_first=False,
        dtype=float,
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    # Target
    label_encoder = None

    if y.dtype == "object" or str(y.dtype) == "category":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(
            y.fillna("Missing").astype(str)
        )
    else:
        y = pd.to_numeric(
            y,
            errors="coerce"
        )

        valid = y.notna()

        X = X.loc[valid]
        y = y.loc[valid]

    return X, y, label_encoder


def infer_problem_type(y):
    if y.dtype == "object":
        return "Classification"

    if len(np.unique(y)) <= 10:
        return "Classification"

    return "Regression"


def get_models(problem_type):
    if problem_type == "Classification":
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=2000
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                random_state=42
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42
            ),
            "KNN": KNeighborsClassifier(),
        }

    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        ),
        "KNN": KNeighborsRegressor(),
    }


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">AI and ML Analytics Dashboard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Data Analysis, Data Cleaning, Visualization, Machine Learning and Prediction'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# DATASET UPLOAD
# ============================================================

if st.session_state.working_df is None:

    st.markdown(
        '<div class="section-title">Dataset</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sub-title">Upload Dataset</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload CSV, XLSX or XLS file",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file is not None:

        if (
            st.session_state.file_name
            != uploaded_file.name
        ):
            with st.spinner("Loading dataset..."):
                loaded = load_dataset(
                    uploaded_file
                )

            if loaded is not None:

                loaded = loaded.copy()

                st.session_state.original_df = (
                    loaded.copy()
                )

                st.session_state.working_df = (
                    loaded.copy()
                )

                st.session_state.file_name = (
                    uploaded_file.name
                )

                st.session_state.audit_log = []

                st.session_state.target_column = (
                    loaded.columns[-1]
                    if len(loaded.columns) > 0
                    else None
                )

                add_log(
                    "Dataset uploaded",
                    len(loaded),
                    len(loaded),
                    uploaded_file.name,
                )

                st.rerun()

    st.stop()


# ============================================================
# CURRENT DATASET
# ============================================================

df = st.session_state.working_df


# ============================================================
# TOP DATASET INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">Dataset Summary</div>',
    unsafe_allow_html=True,
)

summary_cols = st.columns(6)

summary_values = [
    ("Rows", len(df)),
    ("Columns", len(df.columns)),
    ("Numeric Columns", len(numeric_columns(df))),
    ("Categorical Columns", len(categorical_columns(df))),
    ("Missing Values", int(df.isna().sum().sum())),
    ("Duplicate Rows", int(df.duplicated().sum())),
]

for column, (label, value) in zip(
    summary_cols,
    summary_values
):
    with column:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.write("")

st.markdown(
    f"""
    <div class="info-box">
        <b>Loaded file:</b> {st.session_state.file_name}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Current rows:</b> {len(df):,}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Current columns:</b> {len(df.columns):,}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RESET ORIGINAL DATASET
# ============================================================

reset_col1, reset_col2 = st.columns([1, 5])

with reset_col1:

    if st.button(
        "Reset to Original",
        use_container_width=True,
    ):

        if st.session_state.original_df is not None:

            st.session_state.working_df = (
                st.session_state.original_df.copy()
            )

            st.session_state.trained_model = None
            st.session_state.model_results = None
            st.session_state.prediction_info = None

            add_log(
                "Reset to original dataset",
                len(df),
                len(st.session_state.working_df),
                "All cleaning changes were removed.",
            )

            st.success(
                "Dataset successfully restored to the original version."
            )

            st.rerun()


# ============================================================
# MAIN TABS
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
        "Target Analysis",
        "Model Training",
        "Model Comparison",
        "Model Evaluation",
        "Prediction",
        "Explainability",
        "Audit Log",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:

    st.markdown(
        '<div class="section-title">Overview</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        df.head(100),
        use_container_width=True,
        height=420,
    )

    st.markdown(
        '<div class="sub-title">Column Information</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        type_report(df),
        use_container_width=True,
        height=350,
    )

    st.markdown(
        '<div class="sub-title">Dataset Shape</div>',
        unsafe_allow_html=True,
    )

    st.write(
        f"Rows: {df.shape[0]:,} | "
        f"Columns: {df.shape[1]:,}"
    )


# ============================================================
# STATISTICS
# ============================================================

with tabs[1]:

    st.markdown(
        '<div class="section-title">Statistics</div>',
        unsafe_allow_html=True,
    )

    nums = numeric_columns(df)

    if nums:

        st.markdown(
            '<div class="sub-title">Descriptive Statistics</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            df[nums].describe(
                include="all"
            ).T,
            use_container_width=True,
        )

        st.markdown(
            '<div class="sub-title">Skewness and Kurtosis</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            skewness_table(df),
            use_container_width=True,
        )

    else:
        st.info(
            "No numeric columns are available."
        )

    st.markdown(
        '<div class="sub-title">Value Counts</div>',
        unsafe_allow_html=True,
    )

    if len(df.columns) > 0:

        selected_value_col = st.selectbox(
            "Select a column",
            df.columns,
            key="statistics_value_column",
        )

        st.dataframe(
            value_counts_table(
                df,
                selected_value_col
            ),
            use_container_width=True,
        )


# ============================================================
# DATA QUALITY
# ============================================================

with tabs[2]:

    st.markdown(
        '<div class="section-title">Data Quality</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sub-title">Missing Value Report</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        missing_report(df),
        use_container_width=True,
    )

    st.markdown(
        '<div class="sub-title">Duplicate Analysis</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        duplicate_report(df),
        use_container_width=True,
    )

    st.markdown(
        '<div class="sub-title">Data Validation</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        validation_report(df),
        use_container_width=True,
    )

    constants = detect_constant_columns(df)
    high_cardinality = detect_high_cardinality_columns(df)

    st.markdown(
        '<div class="sub-title">Potential Data Problems</div>',
        unsafe_allow_html=True,
    )

    if constants:
        st.warning(
            "Constant columns: "
            + ", ".join(constants)
        )
    else:
        st.success(
            "No constant columns detected."
        )

    if high_cardinality:
        st.warning(
            "High-cardinality categorical columns: "
            + ", ".join(high_cardinality)
        )
    else:
        st.success(
            "No high-cardinality categorical columns detected."
        )


# ============================================================
# EDA
# ============================================================

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
            "Line Chart",
        ],
        key="eda_chart_type",
    )

    nums = numeric_columns(df)
    cats = categorical_columns(df)

    if chart_type == "Histogram":

        if nums:

            col = st.selectbox(
                "Numeric column",
                nums,
                key="hist_column",
            )

            bins = st.slider(
                "Number of bins",
                5,
                100,
                30,
                key="hist_bins",
            )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            ax.hist(
                df[col].dropna(),
                bins=bins,
                edgecolor="black",
            )

            ax.set_title(
                f"Distribution of {col}"
            )

            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        else:
            st.info(
                "No numeric columns available."
            )

    elif chart_type == "Box Plot":

        if nums:

            col = st.selectbox(
                "Numeric column",
                nums,
                key="box_column",
            )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            sns.boxplot(
                x=df[col],
                ax=ax,
            )

            ax.set_title(
                f"Box Plot of {col}"
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        else:
            st.info(
                "No numeric columns available."
            )

    elif chart_type == "Bar Chart":

        if cats:

            col = st.selectbox(
                "Categorical column",
                cats,
                key="bar_column",
            )

            limit = st.slider(
                "Categories to display",
                5,
                50,
                15,
                key="bar_limit",
            )

            counts = (
                df[col]
                .value_counts(dropna=False)
                .head(limit)
            )

            fig, ax = plt.subplots(
                figsize=(11, 5)
            )

            counts.plot.bar(
                ax=ax
            )

            ax.set_title(
                f"Category Frequency: {col}"
            )

            ax.set_xlabel(col)
            ax.set_ylabel("Count")

            plt.xticks(rotation=45)

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        else:
            st.info(
                "No categorical columns available."
            )

    elif chart_type == "Scatter Plot":

        if len(nums) >= 2:

            x_col = st.selectbox(
                "X axis",
                nums,
                key="scatter_x",
            )

            y_col = st.selectbox(
                "Y axis",
                nums,
                index=1 if len(nums) > 1 else 0,
                key="scatter_y",
            )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            ax.scatter(
                df[x_col],
                df[y_col],
                alpha=0.6,
            )

            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(
                f"{x_col} vs {y_col}"
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        else:
            st.info(
                "At least two numeric columns are required."
            )

    elif chart_type == "Correlation Heatmap":

        if len(nums) >= 2:

            corr = df[nums].corr(
                method="pearson"
            )

            fig, ax = plt.subplots(
                figsize=(11, 7)
            )

            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                ax=ax,
            )

            ax.set_title(
                "Pearson Correlation Matrix"
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

            st.markdown(
                '<div class="sub-title">Spearman Correlation</div>',
                unsafe_allow_html=True,
            )

            st.dataframe(
                df[nums].corr(
                    method="spearman"
                ),
                use_container_width=True,
            )

        else:
            st.info(
                "At least two numeric columns are required."
            )

    elif chart_type == "Line Chart":

        if nums:

            y_col = st.selectbox(
                "Y column",
                nums,
                key="line_column",
            )

            fig, ax = plt.subplots(
                figsize=(11, 5)
            )

            ax.plot(
                df[y_col].reset_index(drop=True)
            )

            ax.set_title(
                f"Line Chart: {y_col}"
            )

            ax.set_xlabel("Row Index")
            ax.set_ylabel(y_col)

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        else:
            st.info(
                "No numeric columns available."
            )


# ============================================================
# CLEANING
# ============================================================

with tabs[4]:

    st.markdown(
        '<div class="section-title">Data Cleaning</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-box">
        Select the operations you want to perform.
        Nothing changes until you press "Apply Cleaning".
        Every new cleaning operation starts from the current cleaned dataset,
        so multiple cleaning operations can be performed one after another.
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_rows = len(df)

    c1, c2 = st.columns(2)

    with c1:

        duplicate_action = st.selectbox(
            "Duplicate Treatment",
            [
                "Do nothing",
                "Remove duplicate rows",
            ],
            key="clean_duplicate",
        )

        missing_action = st.selectbox(
            "Missing Value Treatment",
            [
                "Do nothing",
                "Mean",
                "Median",
                "Mode",
                "Mean + Mode",
                "Constant",
                "Forward Fill",
                "Backward Fill",
                "Drop rows with missing values",
                "Drop columns with missing values",
            ],
            key="clean_missing",
        )

        constant_value = st.text_input(
            "Constant Value",
            value="0",
            key="clean_constant",
        )

        numeric_conversion = st.multiselect(
            "Convert to Numeric",
            numeric_columns(df)
            + [
                c for c in df.columns
                if c not in numeric_columns(df)
            ],
            key="clean_numeric_conversion",
        )

    with c2:

        numeric_for_outlier = numeric_columns(df)

        if numeric_for_outlier:

            outlier_column = st.selectbox(
                "Outlier Column",
                ["None"] + numeric_for_outlier,
                key="clean_outlier_column",
            )

        else:

            outlier_column = "None"

        outlier_action = st.selectbox(
            "Outlier Treatment",
            [
                "Do nothing",
                "Remove IQR outliers",
                "Cap IQR outliers",
                "Replace outliers with median",
            ],
            key="clean_outlier_action",
        )

        categorical_conversion = st.multiselect(
            "Convert to Categorical/Text",
            [
                c for c in df.columns
                if c not in numeric_columns(df)
            ],
            key="clean_categorical_conversion",
        )

    st.markdown(
        '<div class="sub-title">Cleaning Preview</div>',
        unsafe_allow_html=True,
    )

    preview_col1, preview_col2, preview_col3 = st.columns(3)

    with preview_col1:
        st.metric(
            "Rows Before",
            f"{current_rows:,}"
        )

    with preview_col2:
        st.metric(
            "Columns Before",
            f"{len(df.columns):,}"
        )

    with preview_col3:
        st.metric(
            "Missing Cells",
            f"{int(df.isna().sum().sum()):,}"
        )

    apply_button = st.button(
        "Apply Cleaning to Current Dataset",
        type="primary",
        use_container_width=True,
        key="apply_cleaning_button",
    )

    if apply_button:

        try:

            outlier_column_value = (
                None
                if outlier_column == "None"
                else outlier_column
            )

            new_df, before_rows, after_rows = (
                apply_cleaning(
                    df=df,
                    duplicate_action=duplicate_action,
                    missing_action=missing_action,
                    constant_value=constant_value,
                    outlier_column=outlier_column_value,
                    outlier_action=outlier_action,
                    numeric_conversion=numeric_conversion,
                    categorical_conversion=categorical_conversion,
                )
            )

            st.session_state.working_df = (
                new_df
            )

            details = []

            if duplicate_action != "Do nothing":
                details.append(
                    f"Duplicate treatment: {duplicate_action}"
                )

            if missing_action != "Do nothing":
                details.append(
                    f"Missing treatment: {missing_action}"
                )

            if outlier_action != "Do nothing":
                details.append(
                    f"Outlier treatment: {outlier_action}"
                )

            if numeric_conversion:
                details.append(
                    "Numeric conversion applied"
                )

            if categorical_conversion:
                details.append(
                    "Categorical conversion applied"
                )

            add_log(
                "Cleaning applied",
                before_rows,
                after_rows,
                "; ".join(details)
                if details
                else "No operation selected.",
            )

            st.success(
                f"Cleaning completed successfully. "
                f"Rows changed from {before_rows:,} to {after_rows:,}."
            )

            st.rerun()

        except Exception as error:

            st.error(
                "Cleaning could not be completed."
            )

            st.exception(error)

    st.markdown(
        '<div class="sub-title">Current Cleaned Dataset</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        st.session_state.working_df.head(200),
        use_container_width=True,
        height=400,
    )

    st.markdown(
        '<div class="sub-title">Download Cleaned Dataset</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Choose the required format and download the current cleaned dataset."
    )

    download_format = st.radio(
        "Download Format",
        ["CSV", "Excel"],
        horizontal=True,
        key="download_format_cleaning",
    )

    if download_format == "CSV":

        st.download_button(
            "Download Cleaned CSV",
            data=make_csv(
                st.session_state.working_df
            ),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_clean_csv",
        )

    else:

        st.download_button(
            "Download Cleaned Excel",
            data=make_excel(
                st.session_state.working_df
            ),
            file_name="cleaned_dataset.xlsx",
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
            use_container_width=True,
            key="download_clean_excel",
        )


# ============================================================
# FEATURES
# ============================================================

with tabs[5]:

    st.markdown(
        '<div class="section-title">Feature Analysis</div>',
        unsafe_allow_html=True,
    )

    nums = numeric_columns(df)

    if nums:

        feature_stats = []

        for col in nums:

            s = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            feature_stats.append(
                {
                    "Feature": col,
                    "Mean": s.mean(),
                    "Median": s.median(),
                    "Std": s.std(),
                    "Min": s.min(),
                    "Max": s.max(),
                    "Unique": s.nunique(),
                    "Missing": s.isna().sum(),
                    "Skewness": s.skew(),
                    "Kurtosis": s.kurtosis(),
                }
            )

        feature_table = pd.DataFrame(
            feature_stats
        )

        st.dataframe(
            feature_table,
            use_container_width=True,
        )

    st.markdown(
        '<div class="sub-title">Potential Feature Columns</div>',
        unsafe_allow_html=True,
    )

    potential_features = []

    for col in df.columns:

        if df[col].nunique(dropna=True) > 1:

            potential_features.append(
                {
                    "Column": col,
                    "Type": str(df[col].dtype),
                    "Unique": int(
                        df[col].nunique(
                            dropna=True
                        )
                    ),
                    "Missing %": round(
                        df[col].isna().mean() * 100,
                        2
                    ),
                }
            )

    st.dataframe(
        pd.DataFrame(
            potential_features
        ),
        use_container_width=True,
    )


# ============================================================
# OUTLIERS
# ============================================================

with tabs[6]:

    st.markdown(
        '<div class="section-title">Outlier Analysis</div>',
        unsafe_allow_html=True,
    )

    report = outlier_report(df)

    if not report.empty:

        st.dataframe(
            report,
            use_container_width=True,
        )

        nums = numeric_columns(df)

        selected_outlier_col = st.selectbox(
            "Select column for detailed outlier analysis",
            nums,
            key="outlier_analysis_column",
        )

        s = pd.to_numeric(
            df[selected_outlier_col],
            errors="coerce"
        )

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = df[
            (s < lower) |
            (s > upper)
        ]

        st.write(
            f"Lower Bound: {lower:.4f}"
        )

        st.write(
            f"Upper Bound: {upper:.4f}"
        )

        st.write(
            f"Outlier Rows: {len(outliers):,}"
        )

        if len(outliers) > 0:
            st.dataframe(
                outliers,
                use_container_width=True,
            )

    else:
        st.info(
            "No numeric columns are available for outlier analysis."
        )


# ============================================================
# TARGET ANALYSIS
# ============================================================

with tabs[7]:

    st.markdown(
        '<div class="section-title">Target Analysis</div>',
        unsafe_allow_html=True,
    )

    target = st.selectbox(
        "Select Target Column",
        df.columns,
        index=(
            list(df.columns).index(
                st.session_state.target_column
            )
            if st.session_state.target_column
            in df.columns
            else 0
        ),
        key="target_analysis_column",
    )

    st.session_state.target_column = target

    st.markdown(
        '<div class="sub-title">Target Information</div>',
        unsafe_allow_html=True,
    )

    target_series = df[target]

    target_info = pd.DataFrame(
        {
            "Metric": [
                "Data Type",
                "Rows",
                "Missing Values",
                "Unique Values",
            ],
            "Value": [
                str(target_series.dtype),
                len(target_series),
                int(target_series.isna().sum()),
                int(target_series.nunique(dropna=True)),
            ],
        }
    )

    st.dataframe(
        target_info,
        use_container_width=True,
    )

    if pd.api.types.is_numeric_dtype(
        target_series
    ):

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.hist(
            target_series.dropna(),
            bins=30,
            edgecolor="black",
        )

        ax.set_title(
            f"Target Distribution: {target}"
        )

        ax.set_xlabel(target)
        ax.set_ylabel("Frequency")

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    else:

        counts = (
            target_series
            .value_counts(dropna=False)
            .head(30)
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        counts.plot.bar(
            ax=ax
        )

        ax.set_title(
            f"Target Frequency: {target}"
        )

        ax.set_xlabel(target)
        ax.set_ylabel("Count")

        plt.xticks(rotation=45)

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


# ============================================================
# MODEL TRAINING
# ============================================================

with tabs[8]:

    st.markdown(
        '<div class="section-title">Model Training</div>',
        unsafe_allow_html=True,
    )

    target = st.selectbox(
        "Target Column",
        df.columns,
        index=(
            list(df.columns).index(
                st.session_state.target_column
            )
            if st.session_state.target_column
            in df.columns
            else len(df.columns) - 1
        ),
        key="training_target",
    )

    st.session_state.target_column = target

    try:

        X_temp, y_temp, encoder_temp = (
            prepare_ml_data(
                df,
                target
            )
        )

        problem_type = infer_problem_type(
            y_temp
        )

        st.markdown(
            f"""
            <div class="success-box">
                Detected problem type: <b>{problem_type}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        model_options = list(
            get_models(problem_type).keys()
        )

        selected_model_name = st.selectbox(
            "Model",
            model_options,
            key="training_model",
        )

        test_size = st.slider(
            "Test Size",
            0.10,
            0.40,
            0.20,
            0.05,
            key="training_test_size",
        )

        scaling = st.selectbox(
            "Feature Scaling",
            [
                "Standard Scaling",
                "No Scaling",
            ],
            key="training_scaling",
        )

        random_state = st.number_input(
            "Random State",
            min_value=0,
            max_value=9999,
            value=42,
            step=1,
            key="training_random_state",
        )

        train_button = st.button(
            "Train Selected Model",
            type="primary",
            use_container_width=True,
            key="train_model_button",
        )

        if train_button:

            if len(X_temp) < 10:
                st.error(
                    "The dataset is too small for reliable model training."
                )

            else:

                stratify = None

                if (
                    problem_type == "Classification"
                    and len(np.unique(y_temp)) > 1
                ):
                    try:
                        stratify = y_temp
                    except Exception:
                        stratify = None

                X_train, X_test, y_train, y_test = (
                    train_test_split(
                        X_temp,
                        y_temp,
                        test_size=test_size,
                        random_state=random_state,
                        stratify=stratify,
                    )
                )

                scaler = None

                if scaling == "Standard Scaling":

                    scaler = StandardScaler()

                    X_train = scaler.fit_transform(
                        X_train
                    )

                    X_test = scaler.transform(
                        X_test
                    )

                models = get_models(
                    problem_type
                )

                model = models[
                    selected_model_name
                ]

                with st.spinner(
                    "Training model..."
                ):

                    model.fit(
                        X_train,
                        y_train
                    )

                    predictions = model.predict(
                        X_test
                    )

                st.session_state.trained_model = (
                    model
                )

                st.session_state.model_scaler = (
                    scaler
                )

                st.session_state.model_features = (
                    X_temp.columns.tolist()
                )

                st.session_state.model_type = (
                    problem_type
                )

                st.session_state.label_encoder = (
                    encoder_temp
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

                    results = {
                        "Problem Type": problem_type,
                        "Model": selected_model_name,
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1 Score": f1,
                        "y_test": y_test,
                        "predictions": predictions,
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

                    results = {
                        "Problem Type": problem_type,
                        "Model": selected_model_name,
                        "MAE": mae,
                        "RMSE": rmse,
                        "R2": r2,
                        "y_test": y_test,
                        "predictions": predictions,
                    }

                st.session_state.model_results = (
                    results
                )

                add_log(
                    "Model trained",
                    len(df),
                    len(df),
                    f"{selected_model_name}; {problem_type}",
                )

                st.success(
                    "Model trained successfully."
                )

    except Exception as error:

        st.error(
            "Model training could not be completed."
        )

        st.exception(error)


# ============================================================
# MODEL COMPARISON
# ============================================================

with tabs[9]:

    st.markdown(
        '<div class="section-title">Model Comparison</div>',
        unsafe_allow_html=True,
    )

    target = st.selectbox(
        "Target Column",
        df.columns,
        key="comparison_target",
    )

    if st.button(
        "Run Model Comparison",
        type="primary",
        use_container_width=True,
        key="compare_models_button",
    ):

        try:

            X, y, encoder = prepare_ml_data(
                df,
                target
            )

            problem_type = infer_problem_type(y)

            if len(X) < 10:
                st.error(
                    "The dataset is too small."
                )
            else:

                models = get_models(
                    problem_type
                )

                stratify = (
                    y
                    if problem_type == "Classification"
                    else None
                )

                X_train, X_test, y_train, y_test = (
                    train_test_split(
                        X,
                        y,
                        test_size=0.2,
                        random_state=42,
                        stratify=stratify,
                    )
                )

                rows = []

                for name, model in models.items():

                    try:

                        model.fit(
                            X_train,
                            y_train
                        )

                        pred = model.predict(
                            X_test
                        )

                        if problem_type == "Classification":

                            rows.append(
                                {
                                    "Model": name,
                                    "Accuracy": accuracy_score(
                                        y_test,
                                        pred
                                    ),
                                    "Precision": precision_score(
                                        y_test,
                                        pred,
                                        average="weighted",
                                        zero_division=0,
                                    ),
                                    "Recall": recall_score(
                                        y_test,
                                        pred,
                                        average="weighted",
                                        zero_division=0,
                                    ),
                                    "F1": f1_score(
                                        y_test,
                                        pred,
                                        average="weighted",
                                        zero_division=0,
                                    ),
                                }
                            )

                        else:

                            rows.append(
                                {
                                    "Model": name,
                                    "MAE": mean_absolute_error(
                                        y_test,
                                        pred
                                    ),
                                    "RMSE": np.sqrt(
                                        mean_squared_error(
                                            y_test,
                                            pred
                                        )
                                    ),
                                    "R2": r2_score(
                                        y_test,
                                        pred
                                    ),
                                }
                            )

                    except Exception as model_error:

                        rows.append(
                            {
                                "Model": name,
                                "Error": str(
                                    model_error
                                ),
                            }
                        )

                comparison_df = pd.DataFrame(
                    rows
                )

                st.session_state.model_results = {
                    "comparison": comparison_df,
                    "Problem Type": problem_type,
                }

                st.dataframe(
                    comparison_df,
                    use_container_width=True,
                )

        except Exception as error:

            st.error(
                "Model comparison failed."
            )

            st.exception(error)


# ============================================================
# MODEL EVALUATION
# ============================================================

with tabs[10]:

    st.markdown(
        '<div class="section-title">Model Evaluation</div>',
        unsafe_allow_html=True,
    )

    results = st.session_state.model_results

    if (
        results is None
        or "y_test" not in results
    ):

        st.info(
            "Train a model first to view detailed evaluation."
        )

    else:

        problem_type = results[
            "Problem Type"
        ]

        y_test = results[
            "y_test"
        ]

        predictions = results[
            "predictions"
        ]

        if problem_type == "Classification":

            evaluation_cols = st.columns(4)

            metrics = [
                (
                    "Accuracy",
                    accuracy_score(
                        y_test,
                        predictions
                    ),
                ),
                (
                    "Precision",
                    precision_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                ),
                (
                    "Recall",
                    recall_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                ),
                (
                    "F1 Score",
                    f1_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                ),
            ]

            for col, (name, value) in zip(
                evaluation_cols,
                metrics
            ):

                with col:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">{name}</div>
                            <div class="metric-value">
                                {value:.4f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown(
                '<div class="sub-title">Confusion Matrix</div>',
                unsafe_allow_html=True,
            )

            cm = confusion_matrix(
                y_test,
                predictions
            )

            fig, ax = plt.subplots(
                figsize=(8, 6)
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

            ax.set_title(
                "Confusion Matrix"
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

            st.markdown(
                '<div class="sub-title">Classification Report</div>',
                unsafe_allow_html=True,
            )

            report = classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0,
            )

            st.dataframe(
                pd.DataFrame(report).T,
                use_container_width=True,
            )

        else:

            evaluation_cols = st.columns(3)

            metrics = [
                (
                    "MAE",
                    mean_absolute_error(
                        y_test,
                        predictions
                    ),
                ),
                (
                    "RMSE",
                    np.sqrt(
                        mean_squared_error(
                            y_test,
                            predictions
                        )
                    ),
                ),
                (
                    "R2",
                    r2_score(
                        y_test,
                        predictions
                    ),
                ),
            ]

            for col, (name, value) in zip(
                evaluation_cols,
                metrics
            ):

                with col:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">{name}</div>
                            <div class="metric-value">
                                {value:.4f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            fig, ax = plt.subplots(
                figsize=(9, 6)
            )

            ax.scatter(
                y_test,
                predictions,
                alpha=0.6,
            )

            min_value = min(
                np.min(y_test),
                np.min(predictions)
            )

            max_value = max(
                np.max(y_test),
                np.max(predictions)
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

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


# ============================================================
# PREDICTION
# ============================================================

with tabs[11]:

    st.markdown(
        '<div class="section-title">Prediction</div>',
        unsafe_allow_html=True,
    )

    model = st.session_state.trained_model

    if model is None:

        st.info(
            "Train a model first."
        )

    else:

        st.markdown(
            f"""
            <div class="success-box">
                Trained model:
                <b>{type(model).__name__}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        features = st.session_state.model_features

        if not features:

            st.warning(
                "No model features are available."
            )

        else:

            st.write(
                "Enter values for the model features."
            )

            input_values = {}

            feature_columns = st.columns(
                3
            )

            for i, feature in enumerate(
                features
            ):

                with feature_columns[
                    i % 3
                ]:

                    input_values[feature] = (
                        st.number_input(
                            feature,
                            value=0.0,
                            key=f"prediction_{feature}",
                        )
                    )

            if st.button(
                "Generate Prediction",
                type="primary",
                use_container_width=True,
                key="prediction_button",
            ):

                try:

                    X_input = pd.DataFrame(
                        [input_values]
                    )

                    scaler = (
                        st.session_state.model_scaler
                    )

                    if scaler is not None:

                        X_input = scaler.transform(
                            X_input
                        )

                    prediction = model.predict(
                        X_input
                    )

                    value = prediction[0]

                    encoder = (
                        st.session_state.label_encoder
                    )

                    if (
                        encoder is not None
                        and hasattr(
                            encoder,
                            "inverse_transform"
                        )
                    ):

                        try:
                            value = (
                                encoder.inverse_transform(
                                    [int(value)]
                                )[0]
                            )
                        except Exception:
                            pass

                    st.session_state.prediction_info = (
                        value
                    )

                    st.success(
                        f"Prediction: {value}"
                    )

                except Exception as error:

                    st.error(
                        "Prediction could not be generated."
                    )

                    st.exception(error)


# ============================================================
# EXPLAINABILITY
# ============================================================

with tabs[12]:

    st.markdown(
        '<div class="section-title">Model Explainability</div>',
        unsafe_allow_html=True,
    )

    model = st.session_state.trained_model
    features = st.session_state.model_features

    if model is None:

        st.info(
            "Train a model first."
        )

    elif not features:

        st.info(
            "No feature information is available."
        )

    else:

        importance = None

        if hasattr(
            model,
            "feature_importances_"
        ):

            importance = (
                model.feature_importances_
            )

        elif hasattr(
            model,
            "coef_"
        ):

            coef = np.asarray(
                model.coef_
            )

            if coef.ndim == 1:
                importance = np.abs(
                    coef
                )
            else:
                importance = np.mean(
                    np.abs(coef),
                    axis=0
                )

        if importance is not None:

            importance_df = pd.DataFrame(
                {
                    "Feature": features,
                    "Importance": importance,
                }
            ).sort_values(
                "Importance",
                ascending=False
            )

            st.dataframe(
                importance_df,
                use_container_width=True,
            )

            top_n = st.slider(
                "Number of features to display",
                5,
                min(
                    50,
                    len(importance_df)
                ),
                min(
                    15,
                    len(importance_df)
                ),
                key="importance_top_n",
            )

            plot_df = (
                importance_df
                .head(top_n)
                .sort_values(
                    "Importance"
                )
            )

            fig, ax = plt.subplots(
                figsize=(10, 7)
            )

            ax.barh(
                plot_df["Feature"],
                plot_df["Importance"]
            )

            ax.set_title(
                "Feature Importance"
            )

            ax.set_xlabel(
                "Importance"
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        else:

            st.info(
                "This model does not provide direct feature importance."
            )


# ============================================================
# AUDIT LOG
# ============================================================

with tabs[13]:

    st.markdown(
        '<div class="section-title">Audit Log</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.audit_log:

        audit_df = pd.DataFrame(
            st.session_state.audit_log
        )

        st.dataframe(
            audit_df,
            use_container_width=True,
            height=450,
        )

        st.download_button(
            "Download Audit Log",
            data=make_csv(
                audit_df
            ),
            file_name="audit_log.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_audit",
        )

    else:

        st.info(
            "No actions have been recorded yet."
        )


# ============================================================
# FINAL DOWNLOAD AREA
# ============================================================

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">Download Current Dataset</div>',
    unsafe_allow_html=True,
)

download_left, download_right = st.columns(2)

with download_left:

    st.download_button(
        "Download Current CSV",
        data=make_csv(
            st.session_state.working_df
        ),
        file_name="analytics_dashboard_dataset.csv",
        mime="text/csv",
        use_container_width=True,
        key="final_csv_download",
    )

with download_right:

    st.download_button(
        "Download Current Excel",
        data=make_excel(
            st.session_state.working_df
        ),
        file_name="analytics_dashboard_dataset.xlsx",
        mime=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        ),
        use_container_width=True,
        key="final_excel_download",
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#7a8496;
        font-family:'Times New Roman', Times, serif;
        font-size:13px;
        margin-top:30px;
        padding-top:15px;
        border-top:1px solid #dfe4ec;
    ">
        AI and ML Analytics Dashboard |
        Data Analysis | Data Cleaning | Visualization |
        Machine Learning | Model Evaluation
    </div>
    """,
    unsafe_allow_html=True,
)