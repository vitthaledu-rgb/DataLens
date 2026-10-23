# ============================================================
# AI & ML ANALYTICS STUDIO
# Complete Google Colab single-cell application
# Run this entire cell in Google Colab.
# Do NOT run this code as a Streamlit app.
# ============================================================

# ============================================================
# 1. INSTALLATION
# ============================================================

import sys
import subprocess
import os
import warnings

warnings.filterwarnings("ignore")

REQUIRED_PACKAGES = [
    "gradio==5.50.0",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scipy",
    "scikit-learn",
    "openpyxl",
    "xlrd",
]

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check"]
    + REQUIRED_PACKAGES
)

# ============================================================
# 2. IMPORTS
# ============================================================

import re
import json
import math
import tempfile
import traceback
from pathlib import Path
from datetime import datetime

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import (
    pearsonr,
    spearmanr,
    kendalltau,
    zscore,
    probplot,
)

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    KFold,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR

# Optional XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ============================================================
# 3. GLOBAL STATE
# ============================================================

original_df = None
processed_df = None
viewed_df = None

feature_df = None
feature_base_df = None
feature_history = []

cleaning_history = []

trained_pipeline = None
trained_model = None
target_column = None
task_type = None

X_train = None
X_test = None
y_train = None
y_test = None

predictions = None
prediction_probabilities = None

model_results = pd.DataFrame()
evaluation_metrics = pd.DataFrame()
feature_importance_df = pd.DataFrame()

trained_feature_names = []
trained_numeric_features = []
trained_categorical_features = []

last_statistics_df = pd.DataFrame()
last_missing_df = pd.DataFrame()
last_outlier_df = pd.DataFrame()
last_quality_df = pd.DataFrame()
last_duplicate_df = pd.DataFrame()

current_filename = ""
last_view_description = ""


# ============================================================
# 4. APPLICATION TEMP DIRECTORY
# ============================================================

APP_DIR = Path(tempfile.gettempdir()) / "ai_ml_analytics_studio"
APP_DIR.mkdir(parents=True, exist_ok=True)


def unique_file(name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))
    return str(APP_DIR / f"{timestamp}_{safe}")


# ============================================================
# 5. GENERAL HELPERS
# ============================================================

def safe_df(df):
    if df is None:
        return pd.DataFrame()
    return df.copy()


def loaded():
    return original_df is not None and not original_df.empty


def numeric_columns(df):
    if df is None or df.empty:
        return []
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df):
    if df is None or df.empty:
        return []
    return df.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()


def all_columns(df):
    if df is None:
        return []
    return df.columns.astype(str).tolist()


def safe_numeric_series(series):
    return pd.to_numeric(series, errors="coerce")


def total_missing(df):
    if df is None:
        return 0
    return int(df.isna().sum().sum())


def duplicate_count(df):
    if df is None:
        return 0
    return int(df.duplicated().sum())


def count_outliers(df):
    if df is None or df.empty:
        return 0

    count = 0

    for col in numeric_columns(df):
        s = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(s) < 4:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count += int(((s < lower) | (s > upper)).sum())

    return count


def basic_metrics(df):
    if df is None:
        return {
            "Rows": 0,
            "Columns": 0,
            "Numeric": 0,
            "Categorical": 0,
            "Missing": 0,
            "Duplicates": 0,
            "Outliers": 0,
        }

    return {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Numeric": len(numeric_columns(df)),
        "Categorical": len(categorical_columns(df)),
        "Missing": total_missing(df),
        "Duplicates": duplicate_count(df),
        "Outliers": count_outliers(df),
    }


def metrics_comparison(before, after):
    b = basic_metrics(before)
    a = basic_metrics(after)

    rows = []

    for key in [
        "Rows",
        "Columns",
        "Missing",
        "Duplicates",
        "Numeric",
        "Categorical",
        "Outliers",
    ]:
        rows.append(
            {
                "Metric": key,
                "Before": b[key],
                "After": a[key],
                "Change": a[key] - b[key],
            }
        )

    return pd.DataFrame(rows)


def make_download(df, filename, excel=False):
    try:
        if df is None:
            return None

        df = safe_df(df)

        if excel:
            path = unique_file(filename.replace(".csv", ".xlsx"))
            df.to_excel(path, index=False)
        else:
            path = unique_file(filename.replace(".xlsx", ".csv"))
            df.to_csv(path, index=False)

        if os.path.exists(path):
            return path

        return None

    except Exception:
        return None


def html_message(text, level="info"):
    colors = {
        "info": "#1d4ed8",
        "success": "#15803d",
        "warning": "#b45309",
        "danger": "#b91c1c",
    }

    bg = {
        "info": "#eff6ff",
        "success": "#f0fdf4",
        "warning": "#fffbeb",
        "danger": "#fef2f2",
    }

    return f"""
    <div style="
        padding:12px 15px;
        border-radius:10px;
        border:1px solid {colors[level]}33;
        background:{bg[level]};
        color:{colors[level]};
        font-family:'Times New Roman',serif;
        font-size:15px;">
        {text}
    </div>
    """


# ============================================================
# 6. DATASET LOADING
# ============================================================

def load_dataset(file):
    global original_df
    global processed_df
    global feature_df
    global feature_base_df
    global cleaning_history
    global feature_history
    global current_filename
    global trained_pipeline
    global trained_model
    global model_results
    global evaluation_metrics
    global feature_importance_df

    try:
        if file is None:
            return (
                html_message("Please upload a CSV, XLSX or XLS file.", "warning"),
                pd.DataFrame(),
                pd.DataFrame(),
                "No dataset loaded.",
                gr.update(choices=[], value=None),
            )

        filepath = file if isinstance(file, str) else getattr(file, "name", None)

        if not filepath or not os.path.exists(filepath):
            return (
                html_message("The uploaded file could not be found.", "danger"),
                pd.DataFrame(),
                pd.DataFrame(),
                "No dataset loaded.",
                gr.update(choices=[], value=None),
            )

        suffix = Path(filepath).suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(filepath)
        elif suffix == ".xlsx":
            df = pd.read_excel(filepath, engine="openpyxl")
        elif suffix == ".xls":
            df = pd.read_excel(filepath, engine="xlrd")
        else:
            return (
                html_message(
                    "Unsupported file format. Use CSV, XLSX or XLS.",
                    "warning",
                ),
                pd.DataFrame(),
                pd.DataFrame(),
                "No dataset loaded.",
                gr.update(choices=[], value=None),
            )

        if df is None or df.empty:
            return (
                html_message("The dataset is empty.", "warning"),
                pd.DataFrame(),
                pd.DataFrame(),
                "Dataset contains no rows.",
                gr.update(choices=[], value=None),
            )

        df.columns = make_unique_columns(df.columns)

        original_df = df.copy()
        processed_df = df.copy()

        feature_base_df = df.copy()
        feature_df = df.copy()

        cleaning_history = []
        feature_history = []

        current_filename = Path(filepath).name

        trained_pipeline = None
        trained_model = None
        model_results = pd.DataFrame()
        evaluation_metrics = pd.DataFrame()
        feature_importance_df = pd.DataFrame()

        metrics = basic_metrics(df)

        summary = f"""
        <div class="upload-success">
            <b>Dataset loaded successfully</b><br>
            File: {current_filename}<br>
            Rows: {metrics["Rows"]:,} |
            Columns: {metrics["Columns"]:,}<br>
            Numeric columns: {metrics["Numeric"]} |
            Categorical columns: {metrics["Categorical"]}<br>
            Missing values: {metrics["Missing"]:,} |
            Duplicate rows: {metrics["Duplicates"]:,}
        </div>
        """

        return (
            summary,
            df.head(50),
            df.head(50),
            f"Loaded {current_filename} with {len(df):,} rows and {len(df.columns):,} columns.",
            gr.update(
                choices=df.columns.tolist(),
                value=df.columns.tolist()[0] if len(df.columns) else None,
            ),
        )

    except Exception as e:
        return (
            html_message(f"Dataset loading failed: {str(e)}", "danger"),
            pd.DataFrame(),
            pd.DataFrame(),
            "Dataset loading failed.",
            gr.update(choices=[], value=None),
        )


def make_unique_columns(columns):
    result = []
    used = {}

    for col in columns:
        name = str(col).strip()

        if name == "":
            name = "Unnamed"

        if name not in used:
            used[name] = 0
            result.append(name)
        else:
            used[name] += 1
            result.append(f"{name}_{used[name]}")

    return result


# ============================================================
# 7. OVERVIEW
# ============================================================

def overview():
    if not loaded():
        return (
            "No dataset loaded.",
            0, 0, 0, 0, 0, 0, 0, "None",
            pd.DataFrame(),
            pd.DataFrame(),
        )

    df = processed_df

    metrics = basic_metrics(df)

    target_guess = detect_target(df)

    dtype_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [str(x) for x in df.dtypes],
            "Non-Null": [int(df[c].notna().sum()) for c in df.columns],
            "Null": [int(df[c].isna().sum()) for c in df.columns],
            "Unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )

    describe = df.describe(include="all").transpose().reset_index()
    describe = describe.rename(columns={"index": "Column"})

    return (
        f"Current processed dataset: {len(df):,} rows × {len(df.columns):,} columns.",
        metrics["Rows"],
        metrics["Columns"],
        metrics["Numeric"],
        metrics["Categorical"],
        metrics["Missing"],
        metrics["Duplicates"],
        metrics["Outliers"],
        target_guess if target_guess else "None detected",
        dtype_df,
        describe,
    )


def detect_target(df):
    if df is None or df.empty:
        return None

    priority_words = [
        "target",
        "label",
        "outcome",
        "class",
        "diagnosis",
        "disease",
        "condition",
        "status",
        "result",
        "response",
        "survived",
        "readmitted",
        "heartdisease",
    ]

    for col in df.columns:
        low = str(col).lower().replace("_", "").replace(" ", "")
        if any(word in low for word in priority_words):
            return col

    return df.columns[-1] if len(df.columns) else None


# ============================================================
# 8. DATASET VIEWER
# ============================================================

def update_view_controls(view_type):
    needs_rows = view_type in ["Head", "Tail", "Random"]
    needs_range = view_type == "Custom Range"

    return (
        gr.update(visible=needs_rows),
        gr.update(visible=needs_range),
        gr.update(visible=needs_range),
    )


def get_viewed_dataset(dataset_choice, view_type, rows, start, end, search_text):
    global viewed_df
    global last_view_description

    try:
        if not loaded():
            return (
                pd.DataFrame(),
                "Please load a dataset first.",
                0,
                0,
                0,
            )

        source = original_df if dataset_choice == "Original" else processed_df

        if source is None or source.empty:
            return pd.DataFrame(), "Dataset is empty.", 0, 0, 0

        df = source.copy()

        if search_text and str(search_text).strip():
            term = str(search_text).strip().lower()

            mask = pd.Series(False, index=df.index)

            for col in df.columns:
                mask = mask | df[col].astype(str).str.lower().str.contains(
                    re.escape(term),
                    na=False,
                )

            df = df.loc[mask]

        if view_type == "Head":
            n = min(int(rows), len(df))
            viewed_df = df.head(n).copy()
            description = f"Showing first {n:,} rows of {len(source):,}."

        elif view_type == "Tail":
            n = min(int(rows), len(df))
            viewed_df = df.tail(n).copy()
            description = f"Showing last {n:,} rows of {len(source):,}."

        elif view_type == "Random":
            n = min(int(rows), len(df))
            viewed_df = df.sample(n=n, random_state=42).copy()
            description = f"Showing random {n:,} rows from {len(source):,}."

        elif view_type == "Custom Range":
            s = max(0, int(start or 0))
            e = min(len(df), int(end or 0))

            if e <= s:
                return (
                    pd.DataFrame(),
                    "Custom range is invalid. End must be greater than Start.",
                    0,
                    len(source),
                    len(source.columns),
                )

            viewed_df = df.iloc[s:e].copy()
            description = (
                f"Showing rows {s + 1:,}–{e:,} of {len(source):,}."
            )

        else:
            safe_limit = 2000

            if len(df) > safe_limit:
                viewed_df = df.head(safe_limit).copy()
                description = (
                    f"Showing first {safe_limit:,} rows of {len(df):,} "
                    f"for performance. Use Head/Custom Range for smaller views."
                )
            else:
                viewed_df = df.copy()
                description = f"Showing entire dataset — {len(df):,} rows."

        if search_text and str(search_text).strip():
            description += f" Search: '{search_text}'."

        last_view_description = description

        return (
            viewed_df,
            description,
            len(viewed_df),
            len(source),
            len(source.columns),
        )

    except Exception as e:
        return pd.DataFrame(), f"Viewer error: {str(e)}", 0, 0, 0


def clear_search():
    return "", pd.DataFrame(), "Search cleared."


# ============================================================
# 9. STATISTICS
# ============================================================

def calculate_statistics(selected_columns, percentile):
    global last_statistics_df

    try:
        if not loaded():
            return pd.DataFrame(), "Please load a dataset first."

        df = processed_df

        if not selected_columns:
            selected_columns = numeric_columns(df)

        selected_columns = [
            c for c in selected_columns
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not selected_columns:
            return pd.DataFrame(), "No numerical columns selected."

        p = float(percentile)

        rows = []

        for col in selected_columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()

            if s.empty:
                continue

            mode_values = s.mode()
            mode_value = mode_values.iloc[0] if len(mode_values) else np.nan

            q1 = s.quantile(0.25)
            median = s.quantile(0.50)
            q3 = s.quantile(0.75)

            mad = np.mean(np.abs(s - s.mean()))

            rows.append(
                {
                    "Column": col,
                    "Count": int(s.count()),
                    "Mean": s.mean(),
                    "Median": median,
                    "Mode": mode_value,
                    "Minimum": s.min(),
                    "Maximum": s.max(),
                    "Range": s.max() - s.min(),
                    "Variance": s.var(),
                    "Std Dev": s.std(),
                    "Q1": q1,
                    "Q3": q3,
                    "IQR": q3 - q1,
                    f"Percentile {p:g}": s.quantile(p / 100),
                    "Skewness": s.skew(),
                    "Kurtosis": s.kurt(),
                    "MAD": mad,
                }
            )

        result = pd.DataFrame(rows)
        last_statistics_df = result.copy()

        return result, f"Statistics calculated for {len(result)} numerical columns."

    except Exception as e:
        return pd.DataFrame(), f"Statistics error: {str(e)}"


# ============================================================
# 10. MISSING VALUES
# ============================================================

def missing_report():
    global last_missing_df

    if not loaded():
        return pd.DataFrame(), "Please load a dataset first."

    df = processed_df

    result = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing Count": df.isna().sum().values,
            "Missing %": (df.isna().mean() * 100).round(2).values,
            "Non-Missing": df.notna().sum().values,
            "Data Type": [str(x) for x in df.dtypes],
        }
    )

    result = result.sort_values(
        "Missing Count",
        ascending=False,
    ).reset_index(drop=True)

    last_missing_df = result.copy()

    total = int(result["Missing Count"].sum())

    if total == 0:
        return result, "No missing values found."

    return result, f"Total missing values: {total:,}."


# ============================================================
# 11. DUPLICATE REPORT
# ============================================================

def duplicate_report():
    global last_duplicate_df

    if not loaded():
        return pd.DataFrame(), "Please load a dataset first."

    df = processed_df

    duplicate_rows = df[df.duplicated(keep=False)].copy()

    summary = pd.DataFrame(
        {
            "Metric": [
                "Total Rows",
                "Duplicate Rows",
                "Unique Rows",
            ],
            "Value": [
                len(df),
                int(df.duplicated().sum()),
                len(df.drop_duplicates()),
            ],
        }
    )

    last_duplicate_df = summary.copy()

    if duplicate_rows.empty:
        return summary, "No duplicate rows found."

    return duplicate_rows.head(500), (
        f"Found {df.duplicated().sum():,} duplicate rows. "
        f"Showing up to 500 duplicate records."
    )


# ============================================================
# 12. DATA QUALITY
# ============================================================

def healthcare_warnings(df):
    warnings_list = []

    if df is None or df.empty:
        return warnings_list

    for col in df.columns:
        name = str(col).lower().replace("_", "").replace(" ", "")

        numeric = pd.to_numeric(df[col], errors="coerce")

        if "age" in name:
            bad = ((numeric < 0) | (numeric > 120)).sum()
            if bad:
                warnings_list.append(
                    f"{col}: {int(bad)} suspicious age values outside 0–120."
                )

        if "bloodpressure" in name or "bloodpress" in name:
            bad = (numeric <= 0).sum()
            if bad:
                warnings_list.append(
                    f"{col}: {int(bad)} non-positive blood-pressure values."
                )

        if "heartrate" in name or "pulse" in name:
            bad = (numeric <= 0).sum()
            if bad:
                warnings_list.append(
                    f"{col}: {int(bad)} non-positive heart-rate values."
                )

        if "cholesterol" in name:
            bad = (numeric < 0).sum()
            if bad:
                warnings_list.append(
                    f"{col}: {int(bad)} negative cholesterol values."
                )

        if "bmi" in name:
            bad = ((numeric <= 0) | (numeric > 100)).sum()
            if bad:
                warnings_list.append(
                    f"{col}: {int(bad)} suspicious BMI values."
                )

    return warnings_list


def quality_analysis():
    global last_quality_df

    if not loaded():
        return (
            pd.DataFrame(),
            0,
            "Please load a dataset first.",
            "",
        )

    df = processed_df

    rows = []

    duplicate_columns = int(df.columns.duplicated().sum())
    missing_by_col = df.isna().sum()

    for col in df.columns:
        s = df[col]

        numeric_version = pd.to_numeric(s, errors="coerce")
        non_null = s.notna().sum()

        numeric_looking_text = (
            non_null > 0
            and str(s.dtype) == "object"
            and numeric_version.notna().sum() / max(non_null, 1) >= 0.8
        )

        unique_ratio = s.nunique(dropna=True) / max(len(df), 1)

        rows.append(
            {
                "Column": col,
                "Data Type": str(s.dtype),
                "Missing": int(s.isna().sum()),
                "Missing %": round(s.isna().mean() * 100, 2),
                "Unique": int(s.nunique(dropna=True)),
                "Unique %": round(unique_ratio * 100, 2),
                "Constant": bool(s.nunique(dropna=False) <= 1),
                "Near Constant": bool(
                    s.value_counts(dropna=False, normalize=True).iloc[0] >= 0.95
                ) if len(s) else False,
                "Empty": bool(s.isna().all()),
                "Numeric-Looking Text": numeric_looking_text,
                "High Cardinality": bool(
                    str(s.dtype) in ["object", "category"]
                    and unique_ratio >= 0.5
                ),
                "Infinite": int(
                    np.isinf(numeric_version.dropna()).sum()
                ) if numeric_version.notna().any() else 0,
            }
        )

    result = pd.DataFrame(rows)
    last_quality_df = result.copy()

    total_cells = max(df.shape[0] * df.shape[1], 1)

    completeness = 100 * (
        1 - total_missing(df) / total_cells
    )

    consistency_penalty = (
        result["Constant"].sum()
        + result["Near Constant"].sum()
        + result["Numeric-Looking Text"].sum()
    )

    consistency = max(
        0,
        100 - (consistency_penalty / max(len(result), 1)) * 100
    )

    validity_penalty = result["Infinite"].sum()

    validity = max(
        0,
        100 - (validity_penalty / max(len(result), 1)) * 100
    )

    uniqueness = max(
        0,
        100 - (duplicate_count(df) / max(len(df), 1)) * 100
    )

    score = round(
        0.30 * completeness
        + 0.25 * consistency
        + 0.25 * validity
        + 0.20 * uniqueness
    )

    warnings_list = healthcare_warnings(df)

    healthcare_text = (
        "<b>Healthcare-aware warnings</b><br>"
        + "<br>".join(
            f"Warning: {x}" for x in warnings_list[:30]
        )
        if warnings_list
        else "<b>Healthcare-aware warnings</b><br>No suspicious healthcare values detected."
    )

    text = f"""
    <b>Data Quality Score: {score}/100</b><br>
    This is a dashboard heuristic, not an official statistical quality score.<br><br>
    Completeness: {completeness:.1f}%<br>
    Consistency: {consistency:.1f}%<br>
    Validity: {validity:.1f}%<br>
    Uniqueness: {uniqueness:.1f}%<br><br>
    Duplicate columns: {duplicate_columns}<br>
    Constant columns: {int(result["Constant"].sum())}<br>
    Near-constant columns: {int(result["Near Constant"].sum())}<br>
    Empty columns: {int(result["Empty"].sum())}<br>
    Numeric-looking text columns: {int(result["Numeric-Looking Text"].sum())}<br>
    High-cardinality categorical columns: {int(result["High Cardinality"].sum())}
    """

    return result, score, text, healthcare_text


# ============================================================
# 13. OUTLIER REPORT
# ============================================================

def outlier_report():
    global last_outlier_df

    if not loaded():
        return pd.DataFrame(), "Please load a dataset first."

    df = processed_df

    rows = []

    for col in numeric_columns(df):
        s = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(s) < 4:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            iqr_count = 0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            iqr_count = int(((s < lower) | (s > upper)).sum())

        if len(s) >= 3 and s.std() > 0:
            z_count = int((np.abs(stats.zscore(s)) > 3).sum())
        else:
            z_count = 0

        rows.append(
            {
                "Column": col,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "IQR Lower": q1 - 1.5 * iqr,
                "IQR Upper": q3 + 1.5 * iqr,
                "IQR Outliers": iqr_count,
                "Z-Score Outliers": z_count,
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        result = pd.DataFrame(
            columns=[
                "Column",
                "Q1",
                "Q3",
                "IQR",
                "IQR Lower",
                "IQR Upper",
                "IQR Outliers",
                "Z-Score Outliers",
            ]
        )

    last_outlier_df = result.copy()

    total = int(result["IQR Outliers"].sum()) if not result.empty else 0

    return result, f"Detected {total:,} IQR outlier observations."


# ============================================================
# 14. CLEANING
# ============================================================

CLEANING_OPERATIONS = [
    "Drop Missing Rows",
    "Drop Missing Columns",
    "Drop Columns Above Missing %",
    "Mean Imputation",
    "Median Imputation",
    "Mode Imputation",
    "Constant Imputation",
    "Forward Fill",
    "Backward Fill",
    "Interpolation",
    "Remove Exact Duplicates",
    "Strip Whitespace",
    "Lowercase Text",
    "Uppercase Text",
    "Title Case Text",
    "Replace Values",
    "Convert Numeric",
    "Convert Category",
    "Convert Date",
    "Replace Negative Numeric Values",
    "Replace Infinite Values",
    "Replace Empty Strings",
    "Replace Common Null Strings",
    "IQR Remove Outliers",
    "IQR Cap Outliers",
    "Z-Score Remove Outliers",
    "Z-Score Cap Outliers",
]


def update_cleaning_controls(operation, apply_to):
    missing_param = operation in [
        "Drop Columns Above Missing %",
    ]

    constant_param = operation == "Constant Imputation"

    replace_param = operation == "Replace Values"

    column_needed = (
        apply_to == "Selected Column"
        and operation not in [
            "Remove Exact Duplicates",
            "Drop Missing Columns",
            "Drop Columns Above Missing %",
        ]
    )

    help_text = cleaning_operation_help(operation)

    return (
        gr.update(visible=column_needed),
        gr.update(visible=constant_param),
        gr.update(visible=replace_param),
        gr.update(visible=replace_param),
        gr.update(visible=missing_param),
        help_text,
    )


def cleaning_operation_help(operation):
    help_map = {
        "Drop Missing Rows":
            "Removes rows containing at least one missing value.",
        "Drop Missing Columns":
            "Removes columns containing missing values.",
        "Drop Columns Above Missing %":
            "Removes columns whose missing percentage exceeds the selected threshold.",
        "Mean Imputation":
            "Replaces numerical missing values with the column mean.",
        "Median Imputation":
            "Replaces numerical missing values with the column median.",
        "Mode Imputation":
            "Replaces missing values with the most frequent value.",
        "Constant Imputation":
            "Replaces missing values with your supplied constant.",
        "Forward Fill":
            "Fills missing values using the previous available value.",
        "Backward Fill":
            "Fills missing values using the next available value.",
        "Interpolation":
            "Interpolates missing numerical values.",
        "Remove Exact Duplicates":
            "Removes exact duplicate rows.",
        "Strip Whitespace":
            "Removes leading and trailing whitespace from text.",
        "Lowercase Text":
            "Converts selected text columns to lowercase.",
        "Uppercase Text":
            "Converts selected text columns to uppercase.",
        "Title Case Text":
            "Converts selected text columns to title case.",
        "Replace Values":
            "Replaces one exact value with another.",
        "Convert Numeric":
            "Attempts to convert selected columns to numeric values.",
        "Convert Category":
            "Converts selected columns to pandas categorical dtype.",
        "Convert Date":
            "Attempts to convert selected columns to datetime.",
        "Replace Negative Numeric Values":
            "Replaces negative numeric values with missing values.",
        "Replace Infinite Values":
            "Replaces positive and negative infinity with missing values.",
        "Replace Empty Strings":
            "Converts empty/whitespace-only strings to missing values.",
        "Replace Common Null Strings":
            "Converts common text null markers such as NA, N/A and null to missing.",
        "IQR Remove Outliers":
            "Removes observations outside the IQR fences.",
        "IQR Cap Outliers":
            "Caps numerical observations at IQR lower/upper fences.",
        "Z-Score Remove Outliers":
            "Removes numerical observations with absolute z-score above 3.",
        "Z-Score Cap Outliers":
            "Caps extreme numerical values using z-score thresholds.",
    }

    return help_map.get(
        operation,
        "Select a cleaning operation."
    )


def selected_clean_columns(df, apply_to, columns):
    if apply_to == "Entire Dataset":
        return list(df.columns)

    if not columns:
        return []

    return [c for c in columns if c in df.columns]


def perform_cleaning_operation(
    df,
    operation,
    apply_to,
    columns,
    constant_value,
    replace_old,
    replace_new,
    missing_threshold,
):
    result = df.copy()

    selected = selected_clean_columns(
        result,
        apply_to,
        columns,
    )

    if operation == "Drop Missing Rows":
        result = result.dropna()

    elif operation == "Drop Missing Columns":
        result = result.dropna(axis=1)

    elif operation == "Drop Columns Above Missing %":
        threshold = float(missing_threshold or 50)

        drop_cols = [
            c for c in result.columns
            if result[c].isna().mean() * 100 > threshold
        ]

        result = result.drop(columns=drop_cols)

    elif operation in [
        "Mean Imputation",
        "Median Imputation",
        "Mode Imputation",
        "Constant Imputation",
    ]:
        targets = selected if selected else numeric_columns(result)

        for col in targets:
            if col not in result.columns:
                continue

            if operation == "Mean Imputation":
                if pd.api.types.is_numeric_dtype(result[col]):
                    value = result[col].mean()
                else:
                    continue

            elif operation == "Median Imputation":
                if pd.api.types.is_numeric_dtype(result[col]):
                    value = result[col].median()
                else:
                    continue

            elif operation == "Mode Imputation":
                mode = result[col].mode(dropna=True)
                if mode.empty:
                    continue
                value = mode.iloc[0]

            else:
                value = constant_value

            result[col] = result[col].fillna(value)

    elif operation == "Forward Fill":
        targets = selected if selected else list(result.columns)
        result[targets] = result[targets].ffill()

    elif operation == "Backward Fill":
        targets = selected if selected else list(result.columns)
        result[targets] = result[targets].bfill()

    elif operation == "Interpolation":
        targets = selected if selected else numeric_columns(result)
        for col in targets:
            if col in result.columns:
                result[col] = pd.to_numeric(
                    result[col],
                    errors="coerce",
                ).interpolate()

    elif operation == "Remove Exact Duplicates":
        result = result.drop_duplicates()

    elif operation == "Strip Whitespace":
        targets = selected if selected else categorical_columns(result)

        for col in targets:
            result[col] = result[col].astype("string").str.strip()

    elif operation == "Lowercase Text":
        targets = selected if selected else categorical_columns(result)

        for col in targets:
            result[col] = result[col].astype("string").str.lower()

    elif operation == "Uppercase Text":
        targets = selected if selected else categorical_columns(result)

        for col in targets:
            result[col] = result[col].astype("string").str.upper()

    elif operation == "Title Case Text":
        targets = selected if selected else categorical_columns(result)

        for col in targets:
            result[col] = result[col].astype("string").str.title()

    elif operation == "Replace Values":
        targets = selected if selected else list(result.columns)

        for col in targets:
            result[col] = result[col].replace(
                replace_old,
                replace_new,
            )

    elif operation == "Convert Numeric":
        targets = selected if selected else list(result.columns)

        for col in targets:
            result[col] = pd.to_numeric(
                result[col],
                errors="coerce",
            )

    elif operation == "Convert Category":
        targets = selected if selected else categorical_columns(result)

        for col in targets:
            result[col] = result[col].astype("category")

    elif operation == "Convert Date":
        targets = selected if selected else list(result.columns)

        for col in targets:
            result[col] = pd.to_datetime(
                result[col],
                errors="coerce",
            )

    elif operation == "Replace Negative Numeric Values":
        targets = selected if selected else numeric_columns(result)

        for col in targets:
            if col in result.columns:
                numeric = pd.to_numeric(
                    result[col],
                    errors="coerce",
                )

                result.loc[numeric < 0, col] = np.nan

    elif operation == "Replace Infinite Values":
        for col in selected if selected else numeric_columns(result):
            if col in result.columns:
                result[col] = result[col].replace(
                    [np.inf, -np.inf],
                    np.nan,
                )

    elif operation == "Replace Empty Strings":
        targets = selected if selected else categorical_columns(result)

        for col in targets:
            result[col] = result[col].replace(
                r"^\s*$",
                np.nan,
                regex=True,
            )

    elif operation == "Replace Common Null Strings":
        targets = selected if selected else categorical_columns(result)

        null_values = [
            "",
            " ",
            "na",
            "n/a",
            "NA",
            "N/A",
            "null",
            "NULL",
            "none",
            "None",
            "missing",
            "Missing",
            "?",
            "-",
        ]

        for col in targets:
            result[col] = result[col].replace(
                null_values,
                np.nan,
            )

    elif operation in [
        "IQR Remove Outliers",
        "IQR Cap Outliers",
    ]:
        targets = selected if selected else numeric_columns(result)

        rows_to_keep = pd.Series(True, index=result.index)

        for col in targets:
            if not pd.api.types.is_numeric_dtype(result[col]):
                continue

            s = result[col]
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1

            if pd.isna(iqr) or iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            if operation == "IQR Remove Outliers":
                rows_to_keep &= (
                    s.isna() | ((s >= lower) & (s <= upper))
                )

            else:
                result[col] = s.clip(
                    lower=lower,
                    upper=upper,
                )

        if operation == "IQR Remove Outliers":
            result = result.loc[rows_to_keep].copy()

    elif operation in [
        "Z-Score Remove Outliers",
        "Z-Score Cap Outliers",
    ]:
        targets = selected if selected else numeric_columns(result)

        rows_to_keep = pd.Series(True, index=result.index)

        for col in targets:
            if not pd.api.types.is_numeric_dtype(result[col]):
                continue

            s = result[col]
            valid = s.dropna()

            if len(valid) < 3 or valid.std() == 0:
                continue

            z = pd.Series(
                np.abs(stats.zscore(valid)),
                index=valid.index,
            )

            if operation == "Z-Score Remove Outliers":
                rows_to_keep.loc[z.index] &= z <= 3

            else:
                mean = valid.mean()
                std = valid.std()

                lower = mean - 3 * std
                upper = mean + 3 * std

                result[col] = s.clip(
                    lower=lower,
                    upper=upper,
                )

        if operation == "Z-Score Remove Outliers":
            result = result.loc[rows_to_keep].copy()

    return result


def cleaning_preview(
    operation,
    apply_to,
    columns,
    constant_value,
    replace_old,
    replace_new,
    missing_threshold,
):
    try:
        if not loaded():
            return (
                pd.DataFrame(),
                pd.DataFrame(),
                "Please load a dataset first.",
                pd.DataFrame(),
            )

        before = processed_df.copy()

        preview = perform_cleaning_operation(
            before,
            operation,
            apply_to,
            columns,
            constant_value,
            replace_old,
            replace_new,
            missing_threshold,
        )

        comparison = metrics_comparison(
            before,
            preview,
        )

        description = create_cleaning_preview_description(
            before,
            preview,
            operation,
            columns,
            constant_value,
            replace_old,
            replace_new,
        )

        return (
            before.head(50),
            preview.head(50),
            description,
            comparison,
        )

    except Exception as e:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            f"Cleaning preview failed: {str(e)}",
            pd.DataFrame(),
        )


def create_cleaning_preview_description(
    before,
    after,
    operation,
    columns,
    constant_value,
    replace_old,
    replace_new,
):
    selected_text = (
        ", ".join(columns)
        if columns
        else "All applicable columns"
    )

    before_missing = total_missing(before)
    after_missing = total_missing(after)

    before_duplicates = duplicate_count(before)
    after_duplicates = duplicate_count(after)

    text = f"""
    <div class="preview-card">
        <b>What will change?</b><br><br>
        Operation: <b>{operation}</b><br>
        Columns: <b>{selected_text}</b><br><br>

        Rows: {len(before):,} → {len(after):,}<br>
        Columns: {len(before.columns):,} → {len(after.columns):,}<br>
        Missing values: {before_missing:,} → {after_missing:,}<br>
        Duplicate rows: {before_duplicates:,} → {after_duplicates:,}<br>
    """

    if operation == "Constant Imputation":
        text += f"<br>Replacement value: <b>{constant_value}</b>"

    if operation == "Replace Values":
        text += (
            f"<br>Replace: <b>{replace_old}</b>"
            f" → <b>{replace_new}</b>"
        )

    if operation in [
        "IQR Remove Outliers",
        "IQR Cap Outliers",
        "Z-Score Remove Outliers",
        "Z-Score Cap Outliers",
    ]:
        text += (
            f"<br>Outlier observations before: "
            f"<b>{count_outliers(before):,}</b><br>"
            f"Outlier observations after: "
            f"<b>{count_outliers(after):,}</b>"
        )

    text += "</div>"

    return text


def apply_cleaning(
    operation,
    apply_to,
    columns,
    constant_value,
    replace_old,
    replace_new,
    missing_threshold,
):
    global processed_df
    global cleaning_history

    try:
        if not loaded():
            return (
                pd.DataFrame(),
                pd.DataFrame(),
                "Please load a dataset first.",
                pd.DataFrame(),
                "No cleaning applied.",
            )

        before = processed_df.copy()

        after = perform_cleaning_operation(
            before,
            operation,
            apply_to,
            columns,
            constant_value,
            replace_old,
            replace_new,
            missing_threshold,
        )

        if after.empty:
            return (
                before.head(50),
                before.head(50),
                "The selected operation would produce an empty dataset. No change was applied.",
                metrics_comparison(before, before),
                "No cleaning applied.",
            )

        processed_df = after.copy()

        selected_text = (
            ", ".join(columns)
            if columns
            else "All applicable columns"
        )

        entry = {
            "Step": len(cleaning_history) + 1,
            "Operation": operation,
            "Column": selected_text,
            "Before": f"{len(before):,} rows / {total_missing(before):,} missing",
            "After": f"{len(after):,} rows / {total_missing(after):,} missing",
            "Status": "Applied",
        }

        cleaning_history.append(entry)

        comparison = metrics_comparison(
            before,
            after,
        )

        message = f"""
        <div class="success-card">
            <b>Cleaning operation applied successfully.</b><br>
            {operation}<br>
            Rows: {len(before):,} → {len(after):,}<br>
            Missing: {total_missing(before):,} → {total_missing(after):,}<br>
            Duplicates: {duplicate_count(before):,} → {duplicate_count(after):,}
        </div>
        """

        return (
            before.head(50),
            after.head(50),
            message,
            comparison,
            pd.DataFrame(cleaning_history),
        )

    except Exception as e:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            f"Cleaning failed: {str(e)}",
            pd.DataFrame(),
            pd.DataFrame(cleaning_history),
        )


def reset_to_original():
    global processed_df
    global feature_df
    global feature_base_df
    global cleaning_history
    global feature_history

    if original_df is None:
        return (
            pd.DataFrame(),
            "No original dataset is available.",
            pd.DataFrame(),
        )

    processed_df = original_df.copy()
    feature_base_df = processed_df.copy()
    feature_df = processed_df.copy()

    cleaning_history = []
    feature_history = []

    return (
        processed_df.head(50),
        html_message(
            "Processed dataset has been restored to the original uploaded dataset.",
            "success",
        ),
        pd.DataFrame(),
    )


# ============================================================
# 15. EDA
# ============================================================

EDA_TYPES = [
    "Histogram",
    "KDE",
    "Box Plot",
    "Violin Plot",
    "Count Plot",
    "Bar Chart",
    "Pie Chart",
    "Scatter Plot",
    "Line Chart",
    "Area Chart",
    "Correlation Heatmap",
    "Covariance Heatmap",
    "Pair Plot",
    "Q-Q Plot",
    "ECDF",
    "Missing Values",
    "3D Scatter",
]


def update_eda_controls(graph_type):
    x_needed = graph_type in [
        "Histogram",
        "KDE",
        "Box Plot",
        "Violin Plot",
        "Count Plot",
        "Bar Chart",
        "Pie Chart",
        "Scatter Plot",
        "Line Chart",
        "Area Chart",
        "Q-Q Plot",
        "ECDF",
        "3D Scatter",
    ]

    y_needed = graph_type in [
        "Scatter Plot",
        "Line Chart",
        "Area Chart",
        "Bar Chart",
        "3D Scatter",
    ]

    z_needed = graph_type == "3D Scatter"

    hue_needed = graph_type in [
        "Scatter Plot",
        "Box Plot",
        "Violin Plot",
        "Count Plot",
    ]

    return (
        gr.update(visible=x_needed),
        gr.update(visible=y_needed),
        gr.update(visible=z_needed),
        gr.update(visible=hue_needed),
    )


def create_eda_graph(graph_type, x_col, y_col, z_col, hue_col):
    try:
        if not loaded():
            return None, "Please load a dataset first."

        df = processed_df.copy()

        sns.set_theme(
            style="whitegrid",
            font_scale=0.9,
        )

        fig = None

        if graph_type == "Histogram":
            if not x_col or x_col not in df.columns:
                return None, "Histogram requires an X column."

            fig, ax = plt.subplots(figsize=(9, 5))
            sns.histplot(
                data=df,
                x=x_col,
                kde=True,
                ax=ax,
            )
            ax.set_title(f"Histogram: {x_col}")
            plt.tight_layout()

        elif graph_type == "KDE":
            if not x_col or x_col not in df.columns:
                return None, "KDE requires an X column."

            if not pd.api.types.is_numeric_dtype(df[x_col]):
                return None, "KDE requires a numerical column."

            fig, ax = plt.subplots(figsize=(9, 5))
            sns.kdeplot(
                data=df,
                x=x_col,
                fill=True,
                ax=ax,
            )
            ax.set_title(f"KDE: {x_col}")
            plt.tight_layout()

        elif graph_type == "Box Plot":
            if not x_col or x_col not in df.columns:
                return None, "Box plot requires an X column."

            fig, ax = plt.subplots(figsize=(9, 5))

            if hue_col and hue_col in df.columns:
                sns.boxplot(
                    data=df,
                    x=x_col,
                    hue=hue_col,
                    ax=ax,
                )
            else:
                sns.boxplot(
                    data=df,
                    x=x_col,
                    ax=ax,
                )

            ax.set_title(f"Box Plot: {x_col}")
            plt.tight_layout()

        elif graph_type == "Violin Plot":
            if not x_col or x_col not in df.columns:
                return None, "Violin plot requires an X column."

            fig, ax = plt.subplots(figsize=(9, 5))

            if hue_col and hue_col in df.columns:
                sns.violinplot(
                    data=df,
                    x=x_col,
                    hue=hue_col,
                    ax=ax,
                )
            else:
                sns.violinplot(
                    data=df,
                    x=x_col,
                    ax=ax,
                )

            ax.set_title(f"Violin Plot: {x_col}")
            plt.tight_layout()

        elif graph_type == "Count Plot":
            if not x_col or x_col not in df.columns:
                return None, "Count plot requires an X column."

            fig, ax = plt.subplots(figsize=(9, 5))

            sns.countplot(
                data=df,
                x=x_col,
                hue=hue_col if hue_col in df.columns else None,
                ax=ax,
            )

            ax.tick_params(axis="x", rotation=45)
            ax.set_title(f"Count Plot: {x_col}")
            plt.tight_layout()

        elif graph_type == "Bar Chart":
            if not x_col or not y_col:
                return None, "Bar chart requires X and Y."

            if x_col not in df.columns or y_col not in df.columns:
                return None, "Selected columns are not available."

            grouped = df.groupby(x_col)[y_col].mean().sort_values(
                ascending=False
            ).head(30)

            fig, ax = plt.subplots(figsize=(10, 5))
            grouped.plot(kind="bar", ax=ax)
            ax.set_title(f"Bar Chart: Mean {y_col} by {x_col}")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()

        elif graph_type == "Pie Chart":
            if not x_col or x_col not in df.columns:
                return None, "Pie chart requires a categorical X column."

            counts = df[x_col].astype(str).value_counts().head(15)

            if counts.empty:
                return None, "No categorical values available."

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%",
            )
            ax.set_title(f"Pie Chart: {x_col}")
            plt.tight_layout()

        elif graph_type == "Scatter Plot":
            if not x_col or not y_col:
                return None, "Scatter plot requires X and Y."

            if not pd.api.types.is_numeric_dtype(df[x_col]):
                return None, "Scatter X must be numerical."

            if not pd.api.types.is_numeric_dtype(df[y_col]):
                return None, "Scatter Y must be numerical."

            fig, ax = plt.subplots(figsize=(9, 5))

            sns.scatterplot(
                data=df,
                x=x_col,
                y=y_col,
                hue=hue_col if hue_col in df.columns else None,
                ax=ax,
            )

            ax.set_title(f"Scatter: {x_col} vs {y_col}")
            plt.tight_layout()

        elif graph_type in ["Line Chart", "Area Chart"]:
            if not x_col or not y_col:
                return None, f"{graph_type} requires X and Y."

            temp = df[[x_col, y_col]].copy()

            if pd.api.types.is_numeric_dtype(temp[y_col]):
                temp = temp.dropna()

                fig, ax = plt.subplots(figsize=(10, 5))

                if graph_type == "Line Chart":
                    ax.plot(
                        temp[x_col].astype(str).values[:500],
                        temp[y_col].values[:500],
                    )
                else:
                    ax.fill_between(
                        range(min(len(temp), 500)),
                        temp[y_col].values[:500],
                        alpha=0.3,
                    )

                ax.set_title(f"{graph_type}: {x_col} / {y_col}")
                ax.tick_params(axis="x", rotation=45)
                plt.tight_layout()

            else:
                return None, "Y must be numerical."

        elif graph_type == "Correlation Heatmap":
            num = df.select_dtypes(include=np.number)

            if num.shape[1] < 2:
                return None, "Correlation requires at least two numerical columns."

            fig, ax = plt.subplots(
                figsize=(
                    max(7, num.shape[1] * 0.7),
                    max(5, num.shape[1] * 0.6),
                )
            )

            sns.heatmap(
                num.corr(),
                annot=num.shape[1] <= 12,
                fmt=".2f",
                cmap="coolwarm",
                ax=ax,
            )

            ax.set_title("Pearson Correlation Heatmap")
            plt.tight_layout()

        elif graph_type == "Covariance Heatmap":
            num = df.select_dtypes(include=np.number)

            if num.shape[1] < 2:
                return None, "Covariance requires at least two numerical columns."

            fig, ax = plt.subplots(
                figsize=(
                    max(7, num.shape[1] * 0.7),
                    max(5, num.shape[1] * 0.6),
                )
            )

            sns.heatmap(
                num.cov(),
                annot=num.shape[1] <= 10,
                fmt=".2f",
                cmap="vlag",
                ax=ax,
            )

            ax.set_title("Covariance Heatmap")
            plt.tight_layout()

        elif graph_type == "Pair Plot":
            cols = numeric_columns(df)[:6]

            if len(cols) < 2:
                return None, "Pair plot requires at least two numerical columns."

            sample = df[cols].dropna().head(500)

            g = sns.pairplot(sample)
            g.fig.suptitle("Pair Plot", y=1.02)

            fig = g.fig

        elif graph_type == "Q-Q Plot":
            if not x_col or x_col not in df.columns:
                return None, "Q-Q plot requires an X column."

            s = pd.to_numeric(
                df[x_col],
                errors="coerce",
            ).dropna()

            if len(s) < 3:
                return None, "Not enough numerical observations."

            fig, ax = plt.subplots(figsize=(8, 5))
            probplot(s, dist="norm", plot=ax)
            ax.set_title(f"Q-Q Plot: {x_col}")
            plt.tight_layout()

        elif graph_type == "ECDF":
            if not x_col or x_col not in df.columns:
                return None, "ECDF requires an X column."

            s = pd.to_numeric(
                df[x_col],
                errors="coerce",
            ).dropna()

            if s.empty:
                return None, "ECDF requires numerical data."

            values = np.sort(s.values)
            y = np.arange(1, len(values) + 1) / len(values)

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.step(values, y, where="post")
            ax.set_xlabel(x_col)
            ax.set_ylabel("ECDF")
            ax.set_title(f"ECDF: {x_col}")
            plt.tight_layout()

        elif graph_type == "Missing Values":
            missing = df.isna().sum()
            missing = missing[missing > 0].sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(10, 5))

            if missing.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No missing values",
                    ha="center",
                    va="center",
                    fontsize=18,
                )
                ax.axis("off")
            else:
                missing.head(30).plot(
                    kind="bar",
                    ax=ax,
                )
                ax.set_title("Missing Values by Column")
                ax.set_ylabel("Missing Count")
                ax.tick_params(axis="x", rotation=45)

            plt.tight_layout()

        elif graph_type == "3D Scatter":
            if not x_col or not y_col or not z_col:
                return None, "3D scatter requires X, Y and Z."

            from mpl_toolkits.mplot3d import Axes3D  # noqa

            cols = [x_col, y_col, z_col]

            if not all(
                pd.api.types.is_numeric_dtype(df[c])
                for c in cols
            ):
                return None, "3D scatter requires numerical columns."

            temp = df[cols].dropna().head(1000)

            fig = plt.figure(figsize=(9, 6))
            ax = fig.add_subplot(111, projection="3d")

            ax.scatter(
                temp[x_col],
                temp[y_col],
                temp[z_col],
                alpha=0.7,
            )

            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_zlabel(z_col)
            ax.set_title("3D Scatter")
            plt.tight_layout()

        else:
            return None, "Select a graph type."

        return fig, f"{graph_type} generated successfully."

    except Exception as e:
        return None, f"Graph generation failed: {str(e)}"


# ============================================================
# 16. ADVANCED CORRELATION
# ============================================================

def advanced_correlation():
    if not loaded():
        return pd.DataFrame(), "Please load a dataset first."

    df = processed_df.select_dtypes(include=np.number)

    if df.shape[1] < 2:
        return pd.DataFrame(), "At least two numerical columns are required."

    pearson = df.corr(method="pearson")
    spearman = df.corr(method="spearman")
    kendall = df.corr(method="kendall")
    covariance = df.cov()

    rows = []

    cols = df.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            x = df[cols[i]]
            y = df[cols[j]]

            valid = pd.concat([x, y], axis=1).dropna()

            if len(valid) < 3:
                continue

            p = pearson.loc[cols[i], cols[j]]
            s = spearman.loc[cols[i], cols[j]]
            k = kendall.loc[cols[i], cols[j]]
            c = covariance.loc[cols[i], cols[j]]

            rows.append(
                {
                    "Feature 1": cols[i],
                    "Feature 2": cols[j],
                    "Pearson": p,
                    "Spearman": s,
                    "Kendall": k,
                    "Covariance": c,
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        return result, "No valid feature pairs found."

    result = result.sort_values(
        "Pearson",
        key=lambda s: s.abs(),
        ascending=False,
    )

    return result, "Pearson, Spearman, Kendall and covariance calculated."


# ============================================================
# 17. TARGET ANALYSIS
# ============================================================

def target_analysis(target):
    try:
        if not loaded():
            return (
                "Please load a dataset first.",
                pd.DataFrame(),
                None,
            )

        df = processed_df

        if not target or target not in df.columns:
            return (
                "Select a valid target column.",
                pd.DataFrame(),
                None,
            )

        s = df[target]

        if (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_categorical_dtype(s)
            or s.nunique(dropna=True) <= 20
        ):
            task = "Classification"

            counts = s.astype(str).value_counts(dropna=False)
            percentages = (
                counts / max(counts.sum(), 1) * 100
            ).round(2)

            result = pd.DataFrame(
                {
                    "Class": counts.index,
                    "Count": counts.values,
                    "Percentage": percentages.values,
                }
            )

            if len(result) == 2:
                imbalance = (
                    result["Percentage"].max()
                    / max(result["Percentage"].min(), 0.01)
                )
            else:
                imbalance = (
                    result["Percentage"].max()
                    / max(result["Percentage"].min(), 0.01)
                )

            text = f"""
            Target: <b>{target}</b><br>
            Task suggestion: <b>{task}</b><br>
            Classes: {len(result)}<br>
            Imbalance ratio: {imbalance:.2f}<br>
            Largest class: {result.iloc[0]["Class"]}
            """

            return text, result, task

        numeric = pd.to_numeric(s, errors="coerce").dropna()

        if numeric.empty:
            return (
                "Target cannot be analysed.",
                pd.DataFrame(),
                None,
            )

        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1

        outliers = (
            ((numeric < q1 - 1.5 * iqr)
             | (numeric > q3 + 1.5 * iqr)).sum()
            if iqr != 0 else 0
        )

        result = pd.DataFrame(
            {
                "Metric": [
                    "Count",
                    "Mean",
                    "Median",
                    "Std Dev",
                    "Minimum",
                    "Maximum",
                    "Skewness",
                    "Kurtosis",
                    "IQR",
                    "Outliers",
                ],
                "Value": [
                    len(numeric),
                    numeric.mean(),
                    numeric.median(),
                    numeric.std(),
                    numeric.min(),
                    numeric.max(),
                    numeric.skew(),
                    numeric.kurt(),
                    iqr,
                    int(outliers),
                ],
            }
        )

        text = f"""
        Target: <b>{target}</b><br>
        Task suggestion: <b>Regression</b><br>
        Mean: {numeric.mean():.4f}<br>
        Median: {numeric.median():.4f}<br>
        Skewness: {numeric.skew():.4f}<br>
        Kurtosis: {numeric.kurt():.4f}<br>
        IQR outliers: {int(outliers):,}
        """

        return text, result, "Regression"

    except Exception as e:
        return f"Target analysis failed: {str(e)}", pd.DataFrame(), None


# ============================================================
# 18. FEATURE-TARGET ANALYSIS
# ============================================================

def feature_target_analysis(target):
    try:
        if not loaded():
            return pd.DataFrame(), "Please load a dataset first."

        if not target or target not in processed_df.columns:
            return pd.DataFrame(), "Select a valid target."

        df = processed_df
        rows = []

        target_numeric = pd.to_numeric(
            df[target],
            errors="coerce",
        )

        for col in numeric_columns(df):
            if col == target:
                continue

            x = pd.to_numeric(
                df[col],
                errors="coerce",
            )

            valid = pd.concat(
                [x, target_numeric],
                axis=1,
            ).dropna()

            if len(valid) < 3:
                continue

            correlation = valid.iloc[:, 0].corr(
                valid.iloc[:, 1]
            )

            rows.append(
                {
                    "Feature": col,
                    "Pearson with Target": correlation,
                    "Absolute Correlation": abs(correlation),
                }
            )

        result = pd.DataFrame(rows)

        if result.empty:
            return result, (
                "Feature-target numeric correlation requires "
                "a numerical target and numerical features."
            )

        return (
            result.sort_values(
                "Absolute Correlation",
                ascending=False,
            ),
            "Feature-target analysis completed.",
        )

    except Exception as e:
        return pd.DataFrame(), str(e)


# ============================================================
# 19. FEATURE ENGINEERING
# ============================================================

FEATURE_OPERATIONS = [
    "Log Transform",
    "Square Transform",
    "Square Root Transform",
    "Standardization",
    "MinMax Scaling",
    "Robust Scaling",
    "One-Hot Encoding",
    "Ordinal Encoding",
    "Frequency Encoding",
    "Variance Filtering",
    "Correlation Filtering",
    "SelectKBest",
    "PCA",
]


def update_feature_controls(operation):
    k_visible = operation == "SelectKBest"
    threshold_visible = operation in [
        "Variance Filtering",
        "Correlation Filtering",
    ]

    return (
        gr.update(visible=k_visible),
        gr.update(visible=threshold_visible),
    )


def apply_feature_engineering(
    operation,
    columns,
    k_value,
    threshold,
    target,
):
    global feature_df
    global feature_base_df
    global feature_history

    try:
        if not loaded():
            return (
                pd.DataFrame(),
                "Please load a dataset first.",
                pd.DataFrame(),
            )

        if feature_base_df is None:
            feature_base_df = processed_df.copy()

        if feature_df is None:
            feature_df = feature_base_df.copy()

        df = feature_df.copy()

        cols = [
            c for c in (columns or [])
            if c in df.columns and c != target
        ]

        if not cols:
            cols = numeric_columns(df)

        if operation == "Log Transform":
            for col in cols:
                s = pd.to_numeric(df[col], errors="coerce")
                shift = 1 - s.min() if s.min() <= 0 else 0
                df[f"{col}_log"] = np.log1p(s + shift)

        elif operation == "Square Transform":
            for col in cols:
                s = pd.to_numeric(df[col], errors="coerce")
                df[f"{col}_square"] = s ** 2

        elif operation == "Square Root Transform":
            for col in cols:
                s = pd.to_numeric(df[col], errors="coerce")
                shift = -s.min() if s.min() < 0 else 0
                df[f"{col}_sqrt"] = np.sqrt(s + shift)

        elif operation in [
            "Standardization",
            "MinMax Scaling",
            "Robust Scaling",
        ]:
            for col in cols:
                s = pd.to_numeric(df[col], errors="coerce")

                if operation == "Standardization":
                    scaler = StandardScaler()
                elif operation == "MinMax Scaling":
                    scaler = MinMaxScaler()
                else:
                    scaler = RobustScaler()

                valid = s.notna()

                if valid.sum() > 1:
                    transformed = np.full(len(s), np.nan)
                    transformed[valid] = scaler.fit_transform(
                        s[valid].to_numpy().reshape(-1, 1)
                    ).ravel()

                    df[f"{col}_{operation.replace(' ', '_').lower()}"] = transformed

        elif operation == "One-Hot Encoding":
            selected = cols or categorical_columns(df)

            if selected:
                encoded = pd.get_dummies(
                    df[selected],
                    dummy_na=True,
                    dtype=int,
                )

                df = pd.concat(
                    [
                        df.drop(columns=selected),
                        encoded,
                    ],
                    axis=1,
                )

        elif operation == "Ordinal Encoding":
            selected = [
                c for c in cols
                if c in categorical_columns(df)
            ]

            if selected:
                enc = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )

                temp = df[selected].astype(str).fillna("Missing")

                transformed = enc.fit_transform(temp)

                for i, col in enumerate(selected):
                    df[f"{col}_ordinal"] = transformed[:, i]

        elif operation == "Frequency Encoding":
            selected = [
                c for c in cols
                if c in df.columns
            ]

            for col in selected:
                freq = df[col].value_counts(
                    normalize=True,
                    dropna=False,
                )

                df[f"{col}_frequency"] = df[col].map(freq)

        elif operation == "Variance Filtering":
            threshold_value = float(threshold or 0.0)

            nums = df.select_dtypes(include=np.number)

            variances = nums.var()

            remove = variances[
                variances <= threshold_value
            ].index.tolist()

            if remove:
                df = df.drop(
                    columns=[
                        c for c in remove
                        if c != target
                    ],
                    errors="ignore",
                )

        elif operation == "Correlation Filtering":
            threshold_value = float(threshold or 0.95)

            nums = df.select_dtypes(include=np.number)

            if nums.shape[1] > 1:
                corr = nums.corr().abs()

                upper = corr.where(
                    np.triu(
                        np.ones(corr.shape),
                        k=1,
                    ).astype(bool)
                )

                remove = [
                    col
                    for col in upper.columns
                    if any(
                        upper[col] > threshold_value
                    )
                    and col != target
                ]

                df = df.drop(
                    columns=remove,
                    errors="ignore",
                )

        elif operation == "SelectKBest":
            k = max(1, int(k_value or 10))

            nums = df.select_dtypes(include=np.number).copy()

            if target in nums.columns:
                X = nums.drop(columns=[target])
            else:
                X = nums

            if X.shape[1] == 0:
                return (
                    feature_df,
                    "No numerical features available.",
                    pd.DataFrame(feature_history),
                )

            if target in df.columns:
                y = df[target]

                if (
                    pd.api.types.is_numeric_dtype(y)
                    and y.nunique() > 10
                ):
                    score_func = f_regression
                    y_use = pd.to_numeric(
                        y,
                        errors="coerce",
                    )
                else:
                    score_func = f_classif
                    y_use = pd.factorize(y)[0]

                temp = pd.concat(
                    [
                        X,
                        pd.Series(
                            y_use,
                            index=df.index,
                            name="__target__",
                        ),
                    ],
                    axis=1,
                ).dropna()

                if len(temp) >= 5:
                    X_use = temp.drop(
                        columns=["__target__"]
                    )
                    y_use = temp["__target__"]

                    selector = SelectKBest(
                        score_func=score_func,
                        k=min(k, X_use.shape[1]),
                    )

                    selector.fit(
                        X_use,
                        y_use,
                    )

                    selected_features = X_use.columns[
                        selector.get_support()
                    ].tolist()

                    keep = [
                        c for c in df.columns
                        if c == target
                        or c in selected_features
                        or c not in X.columns
                    ]

                    df = df[keep]

        elif operation == "PCA":
            nums = df.select_dtypes(include=np.number)

            if nums.shape[1] < 2:
                return (
                    feature_df,
                    "PCA requires at least two numerical columns.",
                    pd.DataFrame(feature_history),
                )

            temp = nums.fillna(
                nums.median()
            )

            n_components = min(
                3,
                temp.shape[1],
                temp.shape[0],
            )

            pca = PCA(
                n_components=n_components
            )

            transformed = pca.fit_transform(temp)

            pca_df = pd.DataFrame(
                transformed,
                index=df.index,
                columns=[
                    f"PC{i + 1}"
                    for i in range(n_components)
                ],
            )

            non_numeric = df.drop(
                columns=nums.columns,
                errors="ignore",
            )

            df = pd.concat(
                [non_numeric, pca_df],
                axis=1,
            )

        feature_df = df.copy()

        feature_history.append(
            {
                "Step": len(feature_history) + 1,
                "Operation": operation,
                "Columns": ", ".join(cols[:10]),
                "Rows": len(df),
                "Columns After": len(df.columns),
            }
        )

        return (
            df.head(100),
            f"Feature engineering applied: {operation}.",
            pd.DataFrame(feature_history),
        )

    except Exception as e:
        return (
            feature_df if feature_df is not None else pd.DataFrame(),
            f"Feature engineering failed: {str(e)}",
            pd.DataFrame(feature_history),
        )


def reset_features():
    global feature_df
    global feature_base_df
    global feature_history

    if processed_df is None:
        return pd.DataFrame(), "No dataset loaded.", pd.DataFrame()

    feature_base_df = processed_df.copy()
    feature_df = processed_df.copy()
    feature_history = []

    return (
        feature_df.head(100),
        "Feature engineering workspace reset.",
        pd.DataFrame(),
    )


# ============================================================
# 20. ML PREPROCESSING
# ============================================================

CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "KNN",
    "Naive Bayes",
    "Decision Tree",
    "Random Forest",
    "SVM",
    "Gradient Boosting",
]

REGRESSION_MODELS = [
    "Linear Regression",
    "Polynomial Regression",
    "Ridge",
    "Lasso",
    "Decision Tree",
    "Random Forest",
    "SVR",
    "Gradient Boosting",
]

if XGBOOST_AVAILABLE:
    CLASSIFICATION_MODELS.append("XGBoost")
    REGRESSION_MODELS.append("XGBoost")


def get_model(task, model_name):
    if task == "Classification":
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
            ),
            "KNN": KNeighborsClassifier(
                n_neighbors=5,
            ),
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(
                random_state=42,
                max_depth=8,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                n_jobs=-1,
            ),
            "SVM": SVC(
                probability=True,
                random_state=42,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42,
            ),
        }

        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                random_state=42,
                eval_metric="logloss",
                n_jobs=2,
            )

    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Polynomial Regression": Pipeline(
                [
                    (
                        "poly",
                        PolynomialFeatures(
                            degree=2,
                            include_bias=False,
                        ),
                    ),
                    (
                        "model",
                        LinearRegression(),
                    ),
                ]
            ),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.01),
            "Decision Tree": DecisionTreeRegressor(
                random_state=42,
                max_depth=8,
            ),
            "Random Forest": RandomForestRegressor(
                n_estimators=150,
                random_state=42,
                n_jobs=-1,
            ),
            "SVR": SVR(),
            "Gradient Boosting": GradientBoostingRegressor(
                random_state=42,
            ),
        }

        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                random_state=42,
                n_jobs=2,
            )

    return models.get(model_name)


def prepare_training_data(
    df,
    target,
    scaling,
    encoding,
):
    if target not in df.columns:
        raise ValueError("Selected target does not exist.")

    work = df.copy()

    work = work.dropna(
        subset=[target]
    )

    if len(work) < 10:
        raise ValueError(
            "At least 10 usable rows are recommended for training."
        )

    X = work.drop(
        columns=[target]
    )

    y = work[target]

    if y.nunique(dropna=True) < 2:
        raise ValueError(
            "Target must contain at least two distinct values."
        )

    num_cols = numeric_columns(X)
    cat_cols = categorical_columns(X)

    transformers = []

    if num_cols:
        if scaling == "Standard":
            scaler = StandardScaler()
        elif scaling == "MinMax":
            scaler = MinMaxScaler()
        elif scaling == "Robust":
            scaler = RobustScaler()
        else:
            scaler = "passthrough"

        numeric_pipe = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    scaler,
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipe,
                num_cols,
            )
        )

    if cat_cols:
        if encoding == "One-Hot":
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        else:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )

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
                    encoder,
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipe,
                cat_cols,
            )
        )

    if not transformers:
        raise ValueError(
            "No usable feature columns were found."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return X, y, preprocessor, num_cols, cat_cols


# ============================================================
# 21. TRAIN MODEL
# ============================================================

def train_model(
    target,
    task,
    model_name,
    test_size,
    scaling,
    encoding,
    cv_folds,
):
    global trained_pipeline
    global trained_model
    global target_column
    global task_type
    global X_train
    global X_test
    global y_train
    global y_test
    global model_results
    global evaluation_metrics
    global predictions
    global prediction_probabilities
    global trained_feature_names
    global trained_numeric_features
    global trained_categorical_features

    try:
        if not loaded():
            return (
                "Please load a dataset first.",
                pd.DataFrame(),
                pd.DataFrame(),
                None,
                pd.DataFrame(),
            )

        if task not in [
            "Classification",
            "Regression",
        ]:
            return (
                "Select Classification or Regression.",
                pd.DataFrame(),
                pd.DataFrame(),
                None,
                pd.DataFrame(),
            )

        if not target or target not in processed_df.columns:
            return (
                "Select a valid target.",
                pd.DataFrame(),
                pd.DataFrame(),
                None,
                pd.DataFrame(),
            )

        if task == "Classification":
            if model_name not in CLASSIFICATION_MODELS:
                return (
                    "Selected model is not valid for classification.",
                    pd.DataFrame(),
                    pd.DataFrame(),
                    None,
                    pd.DataFrame(),
                )
        else:
            if model_name not in REGRESSION_MODELS:
                return (
                    "Selected model is not valid for regression.",
                    pd.DataFrame(),
                    pd.DataFrame(),
                    None,
                    pd.DataFrame(),
                )

        X, y, preprocessor, num_cols, cat_cols = prepare_training_data(
            feature_df if feature_df is not None else processed_df,
            target,
            scaling,
            encoding,
        )

        if task == "Classification":
            y = y.astype(str)

        stratify = y if task == "Classification" else None

        X_train_local, X_test_local, y_train_local, y_test_local = train_test_split(
            X,
            y,
            test_size=float(test_size),
            random_state=42,
            stratify=stratify,
        )

        model = get_model(
            task,
            model_name,
        )

        if model is None:
            raise ValueError(
                "Unable to create the selected model."
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
            X_train_local,
            y_train_local,
        )

        pred = pipeline.predict(
            X_test_local
        )

        probabilities = None

        if task == "Classification":
            if hasattr(
                pipeline,
                "predict_proba",
            ):
                try:
                    probabilities = pipeline.predict_proba(
                        X_test_local
                    )
                except Exception:
                    probabilities = None

        metrics_dict = {}

        if task == "Classification":
            metrics_dict["Accuracy"] = accuracy_score(
                y_test_local,
                pred,
            )

            metrics_dict["Precision"] = precision_score(
                y_test_local,
                pred,
                average="weighted",
                zero_division=0,
            )

            metrics_dict["Recall"] = recall_score(
                y_test_local,
                pred,
                average="weighted",
                zero_division=0,
            )

            metrics_dict["F1"] = f1_score(
                y_test_local,
                pred,
                average="weighted",
                zero_division=0,
            )

            if probabilities is not None:
                try:
                    if probabilities.shape[1] == 2:
                        metrics_dict["ROC-AUC"] = roc_auc_score(
                            y_test_local,
                            probabilities[:, 1],
                        )
                    else:
                        metrics_dict["ROC-AUC"] = roc_auc_score(
                            y_test_local,
                            probabilities,
                            multi_class="ovr",
                        )
                except Exception:
                    metrics_dict["ROC-AUC"] = np.nan
            else:
                metrics_dict["ROC-AUC"] = np.nan

        else:
            mse = mean_squared_error(
                y_test_local,
                pred,
            )

            rmse = math.sqrt(mse)

            r2 = r2_score(
                y_test_local,
                pred,
            )

            n = len(y_test_local)

            try:
                p = X_train_local.shape[1]
                adjusted_r2 = (
                    1
                    - (
                        (1 - r2)
                        * (n - 1)
                        / max(n - p - 1, 1)
                    )
                )
            except Exception:
                adjusted_r2 = np.nan

            metrics_dict["MAE"] = mean_absolute_error(
                y_test_local,
                pred,
            )
            metrics_dict["MSE"] = mse
            metrics_dict["RMSE"] = rmse
            metrics_dict["R2"] = r2
            metrics_dict["Adjusted R2"] = adjusted_r2

        # Cross-validation
        folds = int(cv_folds)

        if task == "Classification":
            cv = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=42,
            )

            scoring = {
                "accuracy": "accuracy",
                "f1": "f1_weighted",
            }

        else:
            cv = KFold(
                n_splits=folds,
                shuffle=True,
                random_state=42,
            )

            scoring = {
                "r2": "r2",
                "neg_mae": "neg_mean_absolute_error",
            }

        try:
            cv_result = cross_validate(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
            )

            if task == "Classification":
                cv_score = np.mean(
                    cv_result["test_accuracy"]
                )
                cv_f1 = np.mean(
                    cv_result["test_f1"]
                )
                metrics_dict["CV Accuracy"] = cv_score
                metrics_dict["CV F1"] = cv_f1

            else:
                cv_r2 = np.mean(
                    cv_result["test_r2"]
                )
                cv_mae = -np.mean(
                    cv_result["test_neg_mae"]
                )
                metrics_dict["CV R2"] = cv_r2
                metrics_dict["CV MAE"] = cv_mae

        except Exception as cv_error:
            metrics_dict["CV Error"] = str(cv_error)

        trained_pipeline = pipeline
        trained_model = model
        target_column = target
        task_type = task

        X_train = X_train_local
        X_test = X_test_local
        y_train = y_train_local
        y_test = y_test_local

        predictions = pred
        prediction_probabilities = probabilities

        trained_numeric_features = num_cols
        trained_categorical_features = cat_cols

        evaluation_metrics = pd.DataFrame(
            {
                "Metric": list(metrics_dict.keys()),
                "Value": list(metrics_dict.values()),
            }
        )

        result_row = {
            "Model": model_name,
            "Task": task,
        }

        result_row.update(metrics_dict)

        if model_results.empty:
            model_results = pd.DataFrame(
                [result_row]
            )
        else:
            model_results = pd.concat(
                [
                    model_results,
                    pd.DataFrame([result_row]),
                ],
                ignore_index=True,
            )

        predictions_df = pd.DataFrame(
            {
                "Actual": y_test_local.reset_index(drop=True),
                "Predicted": pd.Series(pred).reset_index(drop=True),
            }
        )

        if probabilities is not None:
            for i in range(probabilities.shape[1]):
                predictions_df[
                    f"Probability_{i + 1}"
                ] = probabilities[:, i]

        status = f"""
        <div class="success-card">
            <b>Model trained successfully.</b><br>
            Task: {task}<br>
            Model: {model_name}<br>
            Training rows: {len(X_train_local):,}<br>
            Testing rows: {len(X_test_local):,}<br>
            Features: {X.shape[1]:,}<br>
            Cross-validation folds: {folds}
        </div>
        """

        return (
            status,
            evaluation_metrics,
            model_results,
            predictions_df,
            feature_importance(),
        )

    except Exception as e:
        return (
            f"Model training failed: {str(e)}",
            pd.DataFrame(),
            model_results,
            None,
            pd.DataFrame(),
        )


# ============================================================
# 22. FEATURE IMPORTANCE
# ============================================================

def get_transformed_feature_names():
    if trained_pipeline is None:
        return []

    try:
        preprocessor = trained_pipeline.named_steps[
            "preprocessor"
        ]

        names = preprocessor.get_feature_names_out()

        return [
            str(x).replace(
                "numeric__",
                "",
            ).replace(
                "categorical__",
                "",
            )
            for x in names
        ]

    except Exception:
        return []


def feature_importance():
    global feature_importance_df

    if trained_pipeline is None:
        return pd.DataFrame()

    try:
        model = trained_pipeline.named_steps["model"]

        names = get_transformed_feature_names()

        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_

        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)

            if coef.ndim == 1:
                values = np.abs(coef)
            else:
                values = np.mean(
                    np.abs(coef),
                    axis=0,
                )

        else:
            return pd.DataFrame()

        n = min(
            len(names),
            len(values),
        )

        result = pd.DataFrame(
            {
                "Feature": names[:n],
                "Importance": values[:n],
            }
        ).sort_values(
            "Importance",
            ascending=False,
        )

        feature_importance_df = result.copy()

        return result

    except Exception:
        return pd.DataFrame()


# ============================================================
# 23. EVALUATION
# ============================================================

def evaluation_analysis():
    try:
        if trained_pipeline is None:
            return (
                "No trained model.",
                None,
                None,
            )

        if task_type == "Classification":
            cm = confusion_matrix(
                y_test,
                predictions,
            )

            fig, ax = plt.subplots(
                figsize=(6, 5)
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
            plt.tight_layout()

            report = classification_report(
                y_test,
                predictions,
                zero_division=0,
            )

            return report, fig, evaluation_metrics

        actual = np.asarray(y_test)
        pred = np.asarray(predictions)

        residuals = actual - pred

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 5),
        )

        axes[0].scatter(
            actual,
            pred,
            alpha=0.7,
        )

        axes[0].set_xlabel("Actual")
        axes[0].set_ylabel("Predicted")
        axes[0].set_title("Actual vs Predicted")

        axes[1].scatter(
            pred,
            residuals,
            alpha=0.7,
        )

        axes[1].axhline(
            0,
            linestyle="--",
        )

        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Residual")
        axes[1].set_title("Residual Plot")

        plt.tight_layout()

        report = (
            f"MAE: {mean_absolute_error(actual, pred):.6f}\n"
            f"MSE: {mean_squared_error(actual, pred):.6f}\n"
            f"RMSE: {math.sqrt(mean_squared_error(actual, pred)):.6f}\n"
            f"R²: {r2_score(actual, pred):.6f}"
        )

        return report, fig, evaluation_metrics

    except Exception as e:
        return (
            f"Evaluation failed: {str(e)}",
            None,
            evaluation_metrics,
        )


# ============================================================
# 24. MODEL COMPARISON
# ============================================================

def compare_models():
    if model_results is None or model_results.empty:
        return (
            pd.DataFrame(),
            "No trained models available.",
        )

    result = model_results.copy()

    if task_type == "Classification":
        if "Accuracy" in result.columns:
            result = result.sort_values(
                "Accuracy",
                ascending=False,
            )
            best = result.iloc[0]["Model"]
            message = f"Best model by Accuracy: {best}"

        else:
            message = "No classification accuracy available."

    else:
        if "R2" in result.columns:
            result = result.sort_values(
                "R2",
                ascending=False,
            )
            best = result.iloc[0]["Model"]
            message = f"Best model by R²: {best}"

        else:
            message = "No regression R² available."

    return result, message


# ============================================================
# 25. HYPERPARAMETER TUNING
# ============================================================

def tune_model(
    target,
    task,
    model_name,
    method,
    cv_folds,
    scaling,
    encoding,
):
    global trained_pipeline
    global trained_model
    global target_column
    global task_type
    global X_train
    global X_test
    global y_train
    global y_test
    global predictions
    global evaluation_metrics

    try:
        if not loaded():
            return (
                "Please load a dataset first.",
                pd.DataFrame(),
                pd.DataFrame(),
            )

        X, y, preprocessor, _, _ = prepare_training_data(
            feature_df if feature_df is not None else processed_df,
            target,
            scaling,
            encoding,
        )

        if task == "Classification":
            y = y.astype(str)

        stratify = y if task == "Classification" else None

        Xtr, Xte, ytr, yte = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )

        model = get_model(
            task,
            model_name,
        )

        if model is None:
            raise ValueError(
                "Invalid model for selected task."
            )

        pipe = Pipeline(
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

        if task == "Classification":
            if model_name == "Logistic Regression":
                params = {
                    "model__C": [0.1, 1, 10],
                }

            elif model_name == "Decision Tree":
                params = {
                    "model__max_depth": [3, 5, 8, None],
                    "model__min_samples_split": [2, 5],
                }

            elif model_name == "Random Forest":
                params = {
                    "model__n_estimators": [100, 150],
                    "model__max_depth": [None, 6, 10],
                }

            elif model_name == "KNN":
                params = {
                    "model__n_neighbors": [3, 5, 7, 9],
                }

            else:
                params = {}

            cv = StratifiedKFold(
                n_splits=int(cv_folds),
                shuffle=True,
                random_state=42,
            )

            scoring = "accuracy"

        else:
            if model_name == "Ridge":
                params = {
                    "model__alpha": [0.01, 0.1, 1, 10],
                }

            elif model_name == "Lasso":
                params = {
                    "model__alpha": [0.001, 0.01, 0.1],
                }

            elif model_name == "Decision Tree":
                params = {
                    "model__max_depth": [3, 5, 8, None],
                }

            elif model_name == "Random Forest":
                params = {
                    "model__n_estimators": [100, 150],
                    "model__max_depth": [None, 6, 10],
                }

            else:
                params = {}

            cv = KFold(
                n_splits=int(cv_folds),
                shuffle=True,
                random_state=42,
            )

            scoring = "r2"

        if not params:
            params = {
                "model__random_state": [42]
            } if "random_state" in model.get_params() else {}

        if not params:
            return (
                "This model does not have a safe predefined tuning grid.",
                pd.DataFrame(),
                pd.DataFrame(),
            )

        if method == "GridSearchCV":
            search = GridSearchCV(
                pipe,
                params,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
            )
        else:
            search = RandomizedSearchCV(
                pipe,
                params,
                n_iter=min(
                    5,
                    max(1, len(list(params.values())[0]))
                ),
                cv=cv,
                scoring=scoring,
                random_state=42,
                n_jobs=1,
            )

        search.fit(
            Xtr,
            ytr,
        )

        pred = search.predict(
            Xte
        )

        trained_pipeline = search.best_estimator_
        trained_model = search.best_estimator_.named_steps["model"]

        target_column = target
        task_type = task

        X_train = Xtr
        X_test = Xte
        y_train = ytr
        y_test = yte

        predictions = pred

        if task == "Classification":
            score = accuracy_score(
                yte,
                pred,
            )

            tuned_metrics = pd.DataFrame(
                {
                    "Metric": [
                        "Best CV Score",
                        "Test Accuracy",
                    ],
                    "Value": [
                        search.best_score_,
                        score,
                    ],
                }
            )

        else:
            score = r2_score(
                yte,
                pred,
            )

            tuned_metrics = pd.DataFrame(
                {
                    "Metric": [
                        "Best CV Score",
                        "Test R2",
                        "Test MAE",
                        "Test RMSE",
                    ],
                    "Value": [
                        search.best_score_,
                        score,
                        mean_absolute_error(
                            yte,
                            pred,
                        ),
                        math.sqrt(
                            mean_squared_error(
                                yte,
                                pred,
                            )
                        ),
                    ],
                }
            )

        evaluation_metrics = tuned_metrics.copy()

        best_params = pd.DataFrame(
            {
                "Parameter": list(
                    search.best_params_.keys()
                ),
                "Value": list(
                    search.best_params_.values()
                ),
            }
        )

        feature_importance()

        message = f"""
        <div class="success-card">
            <b>Hyperparameter tuning completed.</b><br>
            Method: {method}<br>
            Model: {model_name}<br>
            Best CV Score: {search.best_score_:.6f}<br>
            Best Parameters: {search.best_params_}
        </div>
        """

        return (
            message,
            best_params,
            tuned_metrics,
        )

    except Exception as e:
        return (
            f"Tuning failed: {str(e)}",
            pd.DataFrame(),
            pd.DataFrame(),
        )


# ============================================================
# 26. PREDICTION INPUT
# ============================================================

def prediction_template():
    if trained_pipeline is None:
        return pd.DataFrame(
            columns=["Feature", "Value"]
        )

    features = (
        list(X_train.columns)
        if hasattr(X_train, "columns")
        else []
    )

    return pd.DataFrame(
        {
            "Feature": features,
            "Value": ["" for _ in features],
        }
    )


def predict_from_table(input_table):
    try:
        if trained_pipeline is None:
            return (
                "Please train a model first.",
                pd.DataFrame(
                    columns=["Feature", "Value"]
                ),
                None,
            )

        if input_table is None:
            return (
                "Prediction input is empty.",
                prediction_template(),
                None,
            )

        table = pd.DataFrame(
            input_table,
            columns=["Feature", "Value"],
        )

        expected = (
            list(X_train.columns)
            if hasattr(X_train, "columns")
            else []
        )

        if not expected:
            return (
                "The trained model has no input features.",
                table,
                None,
            )

        values = {}

        for feature in expected:
            row = table[
                table["Feature"].astype(str) == str(feature)
            ]

            if row.empty:
                return (
                    f"Prediction input is missing feature: {feature}",
                    table,
                    None,
                )

            value = row.iloc[0]["Value"]

            if pd.isna(value) or str(value).strip() == "":
                return (
                    f"Prediction input is incomplete: {feature}",
                    table,
                    None,
                )

            values[feature] = value

        input_df = pd.DataFrame(
            [values],
            columns=expected,
        )

        for col in trained_numeric_features:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(
                    input_df[col],
                    errors="coerce",
                )

        pred = trained_pipeline.predict(
            input_df
        )

        result = {
            "Prediction": pred[0]
        }

        probability_text = ""

        if task_type == "Classification":
            if hasattr(
                trained_pipeline,
                "predict_proba",
            ):
                try:
                    proba = trained_pipeline.predict_proba(
                        input_df
                    )[0]

                    classes = trained_pipeline.named_steps[
                        "model"
                    ].classes_

                    probability_rows = []

                    for cls, p in zip(classes, proba):
                        probability_rows.append(
                            {
                                "Class": cls,
                                "Probability": float(p),
                                "Percentage": f"{p * 100:.2f}%",
                            }
                        )

                    probability_df = pd.DataFrame(
                        probability_rows
                    )

                    probability_text = (
                        f"Predicted class: {pred[0]}"
                    )

                    return (
                        probability_text,
                        table,
                        probability_df,
                    )

                except Exception:
                    pass

        result_df = pd.DataFrame(
            [result]
        )

        return (
            f"Prediction: {pred[0]}",
            table,
            result_df,
        )

    except Exception as e:
        return (
            f"Prediction failed: {str(e)}",
            input_table,
            None,
        )


# ============================================================
# 27. EXPLAINABILITY
# ============================================================

def explain_model(method):
    try:
        if trained_pipeline is None:
            return pd.DataFrame(), "Train a model first.", None

        if method == "Feature Importance":
            result = feature_importance()

            if result.empty:
                return (
                    pd.DataFrame(),
                    "This model does not expose direct feature importance.",
                    None,
                )

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            top = result.head(20).sort_values(
                "Importance"
            )

            ax.barh(
                top["Feature"],
                top["Importance"],
            )

            ax.set_title(
                "Feature Importance"
            )

            plt.tight_layout()

            return result, "Feature importance calculated.", fig

        if method == "Permutation Importance":
            if X_test is None or y_test is None:
                return (
                    pd.DataFrame(),
                    "Test data is not available.",
                    None,
                )

            result = permutation_importance(
                trained_pipeline,
                X_test,
                y_test,
                n_repeats=3,
                random_state=42,
                n_jobs=1,
            )

            names = X_test.columns

            result_df = pd.DataFrame(
                {
                    "Feature": names,
                    "Importance Mean": result.importances_mean,
                    "Importance Std": result.importances_std,
                }
            ).sort_values(
                "Importance Mean",
                ascending=False,
            )

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            top = result_df.head(20).sort_values(
                "Importance Mean"
            )

            ax.barh(
                top["Feature"],
                top["Importance Mean"],
            )

            ax.set_title(
                "Permutation Importance"
            )

            plt.tight_layout()

            return (
                result_df,
                "Permutation importance calculated.",
                fig,
            )

        if method == "Coefficients":
            model = trained_pipeline.named_steps["model"]

            if not hasattr(model, "coef_"):
                return (
                    pd.DataFrame(),
                    "Selected model does not have coefficients.",
                    None,
                )

            names = get_transformed_feature_names()
            coef = np.asarray(model.coef_)

            if coef.ndim == 1:
                values = coef
            else:
                values = np.mean(
                    np.abs(coef),
                    axis=0,
                )

            n = min(
                len(names),
                len(values),
            )

            result = pd.DataFrame(
                {
                    "Feature": names[:n],
                    "Coefficient": values[:n],
                }
            )

            return (
                result.sort_values(
                    "Coefficient",
                    key=lambda x: x.abs(),
                    ascending=False,
                ),
                "Model coefficients calculated.",
                None,
            )

        if method == "SHAP":
            if not SHAP_AVAILABLE:
                return (
                    pd.DataFrame(),
                    "SHAP is not installed. The dashboard continues normally.",
                    None,
                )

            if X_test is None or X_test.empty:
                return (
                    pd.DataFrame(),
                    "Test data is unavailable.",
                    None,
                )

            try:
                transformed = trained_pipeline.named_steps[
                    "preprocessor"
                ].transform(X_test)

                model = trained_pipeline.named_steps[
                    "model"
                ]

                explainer = shap.Explainer(
                    model,
                    transformed,
                )

                shap_values = explainer(
                    transformed[:100]
                )

                values = np.abs(
                    shap_values.values
                )

                if values.ndim == 3:
                    values = values.mean(axis=2)

                importance = values.mean(axis=0)

                names = get_transformed_feature_names()

                n = min(
                    len(names),
                    len(importance),
                )

                result = pd.DataFrame(
                    {
                        "Feature": names[:n],
                        "Mean Absolute SHAP": importance[:n],
                    }
                ).sort_values(
                    "Mean Absolute SHAP",
                    ascending=False,
                )

                return (
                    result,
                    "SHAP explanation generated.",
                    None,
                )

            except Exception as e:
                return (
                    pd.DataFrame(),
                    f"SHAP could not be calculated: {str(e)}",
                    None,
                )

        return pd.DataFrame(), "Select an explanation method.", None

    except Exception as e:
        return pd.DataFrame(), f"Explainability failed: {str(e)}", None


# ============================================================
# 28. TIME SERIES
# ============================================================

def time_series_analysis(date_column, value_column):
    try:
        if not loaded():
            return None, pd.DataFrame(), "Load a dataset first."

        if date_column not in processed_df.columns:
            return None, pd.DataFrame(), "Select a valid date column."

        if value_column not in processed_df.columns:
            return None, pd.DataFrame(), "Select a valid value column."

        df = processed_df[
            [date_column, value_column]
        ].copy()

        df[date_column] = pd.to_datetime(
            df[date_column],
            errors="coerce",
        )

        df[value_column] = pd.to_numeric(
            df[value_column],
            errors="coerce",
        )

        df = df.dropna()

        if df.empty:
            return None, pd.DataFrame(), "No valid time-series records."

        df = df.sort_values(
            date_column
        )

        fig, ax = plt.subplots(
            figsize=(11, 5)
        )

        ax.plot(
            df[date_column],
            df[value_column],
        )

        ax.set_title(
            f"Time Series: {value_column}"
        )

        ax.set_xlabel(
            date_column
        )

        ax.set_ylabel(
            value_column
        )

        plt.xticks(rotation=45)
        plt.tight_layout()

        summary = pd.DataFrame(
            {
                "Metric": [
                    "Start Date",
                    "End Date",
                    "Observations",
                    "Mean",
                    "Minimum",
                    "Maximum",
                    "Standard Deviation",
                ],
                "Value": [
                    df[date_column].min(),
                    df[date_column].max(),
                    len(df),
                    df[value_column].mean(),
                    df[value_column].min(),
                    df[value_column].max(),
                    df[value_column].std(),
                ],
            }
        )

        return fig, summary, "Time-series analysis completed."

    except Exception as e:
        return None, pd.DataFrame(), f"Time-series error: {str(e)}"


# ============================================================
# 29. EXPORT CENTER
# ============================================================

EXPORT_TYPES = [
    "Original Dataset",
    "Processed Dataset",
    "Currently Viewed Dataset",
    "Statistics",
    "Missing Value Report",
    "Duplicate Report",
    "Outlier Report",
    "Data Quality Report",
    "Model Metrics",
    "Model Comparison",
    "Prediction Result",
    "Feature Importance",
    "Cleaning Change Log",
    "Feature Engineering Log",
    "Complete Analysis Summary",
]


def complete_analysis_summary():
    if not loaded():
        return pd.DataFrame()

    metrics = basic_metrics(processed_df)

    rows = [
        ["File", current_filename],
        ["Rows", metrics["Rows"]],
        ["Columns", metrics["Columns"]],
        ["Numeric Columns", metrics["Numeric"]],
        ["Categorical Columns", metrics["Categorical"]],
        ["Missing Values", metrics["Missing"]],
        ["Duplicate Rows", metrics["Duplicates"]],
        ["Outlier Observations", metrics["Outliers"]],
        ["Target", target_column if target_column else ""],
        ["Task", task_type if task_type else ""],
        ["Cleaning Steps", len(cleaning_history)],
        ["Feature Engineering Steps", len(feature_history)],
        ["Trained Model", type(trained_model).__name__ if trained_model else ""],
    ]

    return pd.DataFrame(
        rows,
        columns=["Metric", "Value"],
    )


def export_data(export_type, file_format):
    try:
        if not loaded():
            return None, "Please load a dataset first."

        excel = file_format == "Excel"

        if export_type == "Original Dataset":
            df = original_df
            name = "original_dataset"

        elif export_type == "Processed Dataset":
            df = processed_df
            name = "processed_dataset"

        elif export_type == "Currently Viewed Dataset":
            df = viewed_df
            name = "currently_viewed_dataset"

            if df is None or df.empty:
                return None, (
                    "No viewed dataset is available. "
                    "Use the Dataset Viewer first."
                )

        elif export_type == "Statistics":
            if last_statistics_df.empty:
                calculate_statistics(
                    numeric_columns(processed_df),
                    50,
                )

            df = last_statistics_df
            name = "statistics"

        elif export_type == "Missing Value Report":
            if last_missing_df.empty:
                missing_report()

            df = last_missing_df
            name = "missing_value_report"

        elif export_type == "Duplicate Report":
            if last_duplicate_df.empty:
                duplicate_report()

            df = last_duplicate_df
            name = "duplicate_report"

        elif export_type == "Outlier Report":
            if last_outlier_df.empty:
                outlier_report()

            df = last_outlier_df
            name = "outlier_report"

        elif export_type == "Data Quality Report":
            if last_quality_df.empty:
                quality_analysis()

            df = last_quality_df
            name = "data_quality_report"

        elif export_type == "Model Metrics":
            df = evaluation_metrics
            name = "model_metrics"

        elif export_type == "Model Comparison":
            df = model_results
            name = "model_comparison"

        elif export_type == "Prediction Result":
            if predictions is None:
                return None, "No prediction result is available."

            df = pd.DataFrame(
                {
                    "Actual": y_test,
                    "Predicted": predictions,
                }
            )

            name = "prediction_result"

        elif export_type == "Feature Importance":
            df = feature_importance_df

            if df.empty:
                df = feature_importance()

            name = "feature_importance"

        elif export_type == "Cleaning Change Log":
            df = pd.DataFrame(
                cleaning_history
            )

            name = "cleaning_change_log"

        elif export_type == "Feature Engineering Log":
            df = pd.DataFrame(
                feature_history
            )

            name = "feature_engineering_log"

        elif export_type == "Complete Analysis Summary":
            df = complete_analysis_summary()
            name = "complete_analysis_summary"

        else:
            return None, "Unknown export type."

        if df is None:
            return None, "No data is available for this export."

        if df.empty:
            return None, "The selected export currently contains no data."

        extension = ".xlsx" if excel else ".csv"

        path = unique_file(
            name + extension
        )

        if excel:
            df.to_excel(
                path,
                index=False,
            )
        else:
            df.to_csv(
                path,
                index=False,
            )

        if not os.path.exists(path):
            return None, "The export file could not be created."

        return (
            path,
            f"{export_type} generated successfully.",
        )

    except Exception as e:
        return None, f"Export failed: {str(e)}"


# ============================================================
# 30. QUICK EXPORTS
# ============================================================

def quick_original():
    if original_df is None:
        return None
    return make_download(
        original_df,
        "original_dataset.csv",
        False,
    )


def quick_processed():
    if processed_df is None:
        return None
    return make_download(
        processed_df,
        "processed_dataset.csv",
        False,
    )


def quick_model():
    if model_results is None or model_results.empty:
        return None
    return make_download(
        model_results,
        "model_results.csv",
        False,
    )


def quick_prediction():
    if predictions is None:
        return None

    df = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": predictions,
        }
    )

    return make_download(
        df,
        "prediction_result.csv",
        False,
    )


# ============================================================
# 31. CSS
# ============================================================

CSS = """
* {
    font-family: "Times New Roman", Times, serif !important;
}

body,
.gradio-container {
    background: #eef4f9 !important;
    color: #14213d !important;
}

.gradio-container {
    max-width: 1500px !important;
    margin: auto !important;
}

#header {
    background: linear-gradient(
        135deg,
        #102a43,
        #1d4ed8
    );
    border-radius: 18px;
    padding: 25px 30px;
    color: white;
    margin-bottom: 18px;
    box-shadow: 0 8px 25px rgba(16,42,67,0.18);
}

#header h1 {
    color: white !important;
    margin: 0;
    font-size: 32px;
}

#header p {
    color: #dbeafe !important;
    font-size: 16px;
}

.section-card {
    background: white;
    border: 1px solid #d7e2ec;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 15px;
    box-shadow: 0 4px 14px rgba(15,23,42,0.05);
}

.kpi {
    background: white;
    border-radius: 13px;
    padding: 15px;
    border: 1px solid #dbe7f0;
    min-height: 95px;
    box-shadow: 0 4px 12px rgba(15,23,42,0.05);
}

.kpi-label {
    font-size: 13px;
    color: #64748b;
}

.kpi-value {
    font-size: 27px;
    font-weight: bold;
    color: #102a43;
}

.success-card {
    background: #ecfdf5;
    color: #166534;
    border: 1px solid #86efac;
    border-radius: 10px;
    padding: 14px;
}

.preview-card {
    background: #eff6ff;
    color: #1e3a8a;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 14px;
}

.upload-success {
    background: linear-gradient(
        135deg,
        #ecfeff,
        #eff6ff
    );
    color: #0f3d5e;
    border: 1px solid #a5f3fc;
    border-radius: 12px;
    padding: 16px;
}

button {
    border-radius: 9px !important;
    font-weight: 600 !important;
}

.primary-btn button {
    background: linear-gradient(
        135deg,
        #1d4ed8,
        #0891b2
    ) !important;
    color: white !important;
    border: none !important;
}

.success-btn button {
    background: #15803d !important;
    color: white !important;
}

.danger-btn button {
    background: #b91c1c !important;
    color: white !important;
}

.secondary-btn button {
    background: #e8f1f8 !important;
    color: #17324d !important;
    border: 1px solid #cbd5e1 !important;
}

.gradio-tabs {
    background: transparent !important;
}

.tab-nav {
    overflow-x: auto !important;
    white-space: nowrap !important;
    display: flex !important;
    scrollbar-width: thin;
    border-bottom: 1px solid #cbd5e1 !important;
}

.tab-nav button {
    min-width: max-content !important;
    color: #24445f !important;
}

textarea,
input,
select,
.gr-input,
.gr-box {
    border-radius: 9px !important;
    border-color: #cbd5e1 !important;
}

.dataframe {
    border-radius: 10px !important;
}

h1, h2, h3, h4 {
    color: #102a43 !important;
}

footer {
    display: none !important;
}
"""


# ============================================================
# 32. GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="AI & ML Analytics Studio",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
    ),
    css=CSS,
) as app:

    gr.HTML(
        """
        <div id="header">
            <h1>AI & ML Analytics Studio</h1>
            <p>
                Data analysis, healthcare-aware quality checks,
                cleaning, visualization, feature engineering,
                machine learning, evaluation and export.
            </p>
        </div>
        """
    )

    # ========================================================
    # UPLOAD
    # ========================================================

    with gr.Group(elem_classes="section-card"):

        gr.Markdown("## Dataset Upload")

        with gr.Row():

            upload = gr.File(
                label="Upload CSV / XLSX / XLS",
                file_types=[
                    ".csv",
                    ".xlsx",
                    ".xls",
                ],
                type="filepath",
            )

            upload_status = gr.HTML(
                value=html_message(
                    "Upload a dataset to begin.",
                    "info",
                )
            )

        with gr.Row():

            load_preview = gr.Dataframe(
                headers=[],
                interactive=False,
                label="Loaded Dataset Preview",
                max_height=300,
            )

            initial_preview = gr.Dataframe(
                headers=[],
                interactive=False,
                label="Initial 50 Rows",
                max_height=300,
            )

    # ========================================================
    # KPI OVERVIEW
    # ========================================================

    with gr.Group(elem_classes="section-card"):

        gr.Markdown("## Dataset Overview")

        overview_status = gr.Markdown(
            "Load a dataset to view its overview."
        )

        with gr.Row():

            kpi_rows = gr.Number(
                label="Rows",
                value=0,
                interactive=False,
            )

            kpi_cols = gr.Number(
                label="Columns",
                value=0,
                interactive=False,
            )

            kpi_num = gr.Number(
                label="Numeric",
                value=0,
                interactive=False,
            )

            kpi_cat = gr.Number(
                label="Categorical",
                value=0,
                interactive=False,
            )

        with gr.Row():

            kpi_missing = gr.Number(
                label="Missing",
                value=0,
                interactive=False,
            )

            kpi_duplicates = gr.Number(
                label="Duplicates",
                value=0,
                interactive=False,
            )

            kpi_outliers = gr.Number(
                label="Outliers",
                value=0,
                interactive=False,
            )

            kpi_target = gr.Textbox(
                label="Suggested Target",
                value="None",
                interactive=False,
            )

        dtype_table = gr.Dataframe(
            label="Column Information",
            interactive=False,
            max_height=300,
        )

        describe_table = gr.Dataframe(
            label="Statistical Description",
            interactive=False,
            max_height=300,
        )

    # ========================================================
    # MAIN TABS
    # ========================================================

    with gr.Tabs():

        # ====================================================
        # VIEWER
        # ====================================================

        with gr.Tab("Viewer"):

            gr.Markdown("## Dataset Viewer")

            with gr.Row():

                viewer_dataset = gr.Dropdown(
                    choices=[
                        "Original",
                        "Processed",
                    ],
                    value="Processed",
                    label="Dataset",
                )

                viewer_type = gr.Dropdown(
                    choices=[
                        "Head",
                        "Tail",
                        "Random",
                        "Entire Dataset",
                        "Custom Range",
                    ],
                    value="Head",
                    label="View",
                )

                viewer_rows = gr.Dropdown(
                    choices=[
                        10,
                        20,
                        50,
                        100,
                        147,
                        500,
                    ],
                    value=50,
                    label="Rows",
                )

            with gr.Row():

                viewer_start = gr.Number(
                    label="Start",
                    value=0,
                    visible=False,
                )

                viewer_end = gr.Number(
                    label="End",
                    value=147,
                    visible=False,
                )

            with gr.Row():

                viewer_search = gr.Textbox(
                    label="Search Dataset",
                    placeholder="Search across all columns...",
                )

                viewer_clear = gr.Button(
                    "Clear Search",
                    elem_classes="secondary-btn",
                )

            viewer_refresh = gr.Button(
                "Refresh View",
                elem_classes="primary-btn",
            )

            viewer_info = gr.HTML(
                "No view generated."
            )

            with gr.Row():

                viewer_displayed = gr.Number(
                    label="Displayed Rows",
                    value=0,
                    interactive=False,
                )

                viewer_total = gr.Number(
                    label="Total Rows",
                    value=0,
                    interactive=False,
                )

                viewer_columns_count = gr.Number(
                    label="Columns",
                    value=0,
                    interactive=False,
                )

            viewer_table = gr.Dataframe(
                headers=[],
                interactive=False,
                label="Dataset",
                max_height=500,
            )

        # ====================================================
        # STATISTICS
        # ====================================================

        with gr.Tab("Statistics"):

            gr.Markdown("## Statistical Analysis")

            with gr.Row():

                stats_columns = gr.Dropdown(
                    choices=[],
                    multiselect=True,
                    label="Numerical Columns",
                )

                percentile = gr.Dropdown(
                    choices=[
                        10,
                        25,
                        50,
                        75,
                        90,
                        95,
                        99,
                    ],
                    value=50,
                    label="Percentile",
                )

            stats_button = gr.Button(
                "Calculate Statistics",
                elem_classes="primary-btn",
            )

            stats_table = gr.Dataframe(
                interactive=False,
                label="Complete Statistics",
                max_height=450,
            )

            stats_status = gr.Markdown(
                "No statistics calculated."
            )

            correlation_button = gr.Button(
                "Calculate Pearson / Spearman / Kendall / Covariance",
                elem_classes="secondary-btn",
            )

            correlation_table = gr.Dataframe(
                interactive=False,
                label="Advanced Correlation Analysis",
                max_height=450,
            )

            correlation_status = gr.Markdown()

        # ====================================================
        # QUALITY
        # ====================================================

        with gr.Tab("Quality"):

            gr.Markdown("## Data Quality Center")

            quality_button = gr.Button(
                "Run Quality Analysis",
                elem_classes="primary-btn",
            )

            with gr.Row():

                quality_score = gr.Number(
                    label="Data Quality Heuristic Score",
                    value=0,
                    interactive=False,
                )

                quality_summary = gr.HTML(
                    "No quality analysis yet."
                )

            quality_table = gr.Dataframe(
                interactive=False,
                label="Column-Level Quality Analysis",
                max_height=450,
            )

            healthcare_warning = gr.HTML(
                "Healthcare-aware validation will appear here."
            )

            missing_button = gr.Button(
                "Generate Missing Value Report",
                elem_classes="secondary-btn",
            )

            missing_table = gr.Dataframe(
                interactive=False,
                label="Missing Value Report",
                max_height=350,
            )

            missing_status = gr.Markdown()

            duplicate_button = gr.Button(
                "Generate Duplicate Report",
                elem_classes="secondary-btn",
            )

            duplicate_table = gr.Dataframe(
                interactive=False,
                label="Duplicate Report",
                max_height=350,
            )

            duplicate_status = gr.Markdown()

            outlier_button = gr.Button(
                "Generate Outlier Report",
                elem_classes="secondary-btn",
            )

            outlier_table = gr.Dataframe(
                interactive=False,
                label="Outlier Report",
                max_height=350,
            )

            outlier_status = gr.Markdown()

        # ====================================================
        # CLEANING
        # ====================================================

        with gr.Tab("Cleaning"):

            gr.Markdown("## Data Cleaning Studio")

            gr.Markdown(
                """
                **Original → Select Operation → Preview → Apply → Processed**
                """
            )

            with gr.Row():

                clean_operation = gr.Dropdown(
                    choices=CLEANING_OPERATIONS,
                    value="Median Imputation",
                    label="Cleaning Operation",
                )

                clean_apply_to = gr.Dropdown(
                    choices=[
                        "Entire Dataset",
                        "Selected Column",
                    ],
                    value="Entire Dataset",
                    label="Apply To",
                )

            clean_help = gr.HTML(
                cleaning_operation_help(
                    "Median Imputation"
                )
            )

            clean_columns = gr.Dropdown(
                choices=[],
                multiselect=True,
                label="Selected Columns",
                visible=False,
            )

            with gr.Row():

                clean_constant = gr.Textbox(
                    label="Constant Value",
                    value="0",
                    visible=False,
                )

                clean_old = gr.Textbox(
                    label="Replace This Value",
                    visible=False,
                )

                clean_new = gr.Textbox(
                    label="Replace With",
                    visible=False,
                )

                clean_threshold = gr.Number(
                    label="Missing Threshold %",
                    value=50,
                    visible=False,
                )

            with gr.Row():

                clean_preview_button = gr.Button(
                    "Preview Changes",
                    elem_classes="secondary-btn",
                )

                clean_apply_button = gr.Button(
                    "Apply Cleaning",
                    elem_classes="success-btn",
                )

                clean_reset_button = gr.Button(
                    "Reset to Original",
                    elem_classes="danger-btn",
                )

            clean_message = gr.HTML(
                "No cleaning operation selected."
            )

            with gr.Row():

                clean_before = gr.Dataframe(
                    interactive=False,
                    label="Before Cleaning",
                    max_height=300,
                )

                clean_after = gr.Dataframe(
                    interactive=False,
                    label="Preview / After Cleaning",
                    max_height=300,
                )

            clean_comparison = gr.Dataframe(
                interactive=False,
                label="Before / After Comparison",
                max_height=300,
            )

            cleaning_history_table = gr.Dataframe(
                headers=[
                    "Step",
                    "Operation",
                    "Column",
                    "Before",
                    "After",
                    "Status",
                ],
                interactive=False,
                label="Cleaning Change Log",
                max_height=300,
            )

        # ====================================================
        # EDA
        # ====================================================

        with gr.Tab("EDA"):

            gr.Markdown("## Exploratory Data Analysis")

            with gr.Row():

                graph_type = gr.Dropdown(
                    choices=EDA_TYPES,
                    value="Histogram",
                    label="Graph Type",
                )

                graph_x = gr.Dropdown(
                    choices=[],
                    label="X",
                )

                graph_y = gr.Dropdown(
                    choices=[],
                    label="Y",
                    visible=False,
                )

                graph_z = gr.Dropdown(
                    choices=[],
                    label="Z",
                    visible=False,
                )

                graph_hue = gr.Dropdown(
                    choices=[],
                    label="Hue",
                    visible=False,
                )

            graph_button = gr.Button(
                "Generate Graph",
                elem_classes="primary-btn",
            )

            graph_status = gr.Markdown()

            graph_output = gr.Plot(
                label="Visualization"
            )

            gr.Markdown("## Time-Series Analysis")

            with gr.Row():

                ts_date = gr.Dropdown(
                    choices=[],
                    label="Date Column",
                )

                ts_value = gr.Dropdown(
                    choices=[],
                    label="Value Column",
                )

            ts_button = gr.Button(
                "Analyze Time Series",
                elem_classes="secondary-btn",
            )

            ts_plot = gr.Plot()

            ts_table = gr.Dataframe(
                interactive=False,
                label="Time-Series Summary",
            )

            ts_status = gr.Markdown()

        # ====================================================
        # TARGET
        # ====================================================

        with gr.Tab("Target"):

            gr.Markdown("## Target Analysis")

            target_column_ui = gr.Dropdown(
                choices=[],
                label="Target Column",
            )

            target_button = gr.Button(
                "Analyze Target",
                elem_classes="primary-btn",
            )

            target_summary = gr.HTML()

            target_table = gr.Dataframe(
                interactive=False,
                label="Target Distribution / Statistics",
            )

            target_task = gr.Textbox(
                label="Suggested Task",
                interactive=False,
            )

            feature_target_button = gr.Button(
                "Feature-Target Analysis",
                elem_classes="secondary-btn",
            )

            feature_target_table = gr.Dataframe(
                interactive=False,
                label="Feature-Target Analysis",
                max_height=400,
            )

            feature_target_status = gr.Markdown()

        # ====================================================
        # FEATURES
        # ====================================================

        with gr.Tab("Features"):

            gr.Markdown("## Feature Engineering Studio")

            with gr.Row():

                feature_operation = gr.Dropdown(
                    choices=FEATURE_OPERATIONS,
                    value="Standardization",
                    label="Feature Engineering Operation",
                )

                feature_columns = gr.Dropdown(
                    choices=[],
                    multiselect=True,
                    label="Columns",
                )

            with gr.Row():

                feature_k = gr.Number(
                    label="K for SelectKBest",
                    value=10,
                    visible=False,
                )

                feature_threshold = gr.Number(
                    label="Threshold",
                    value=0.95,
                    visible=False,
                )

                feature_target = gr.Dropdown(
                    choices=[],
                    label="Target to Preserve",
                )

            with gr.Row():

                feature_apply_button = gr.Button(
                    "Apply Feature Engineering",
                    elem_classes="primary-btn",
                )

                feature_reset_button = gr.Button(
                    "Reset Feature Workspace",
                    elem_classes="secondary-btn",
                )

            feature_status = gr.Markdown()

            feature_preview = gr.Dataframe(
                interactive=False,
                label="Feature Engineering Dataset",
                max_height=400,
            )

            feature_history_table = gr.Dataframe(
                interactive=False,
                label="Feature Engineering History",
                max_height=300,
            )

        # ====================================================
        # TRAINING
        # ====================================================

        with gr.Tab("Training"):

            gr.Markdown("## Machine Learning Training")

            with gr.Row():

                train_task = gr.Dropdown(
                    choices=[
                        "Classification",
                        "Regression",
                    ],
                    value="Classification",
                    label="Task",
                )

                train_target = gr.Dropdown(
                    choices=[],
                    label="Target",
                )

                train_model = gr.Dropdown(
                    choices=CLASSIFICATION_MODELS,
                    value="Random Forest",
                    label="Model",
                )

            with gr.Row():

                test_size = gr.Slider(
                    minimum=0.1,
                    maximum=0.4,
                    value=0.2,
                    step=0.05,
                    label="Test Size",
                )

                scaling = gr.Dropdown(
                    choices=[
                        "None",
                        "Standard",
                        "MinMax",
                        "Robust",
                    ],
                    value="Standard",
                    label="Scaling",
                )

                encoding = gr.Dropdown(
                    choices=[
                        "One-Hot",
                        "Ordinal",
                    ],
                    value="One-Hot",
                    label="Encoding",
                )

                cv_folds = gr.Dropdown(
                    choices=[
                        3,
                        5,
                        10,
                    ],
                    value=5,
                    label="Cross-Validation Folds",
                )

            train_button = gr.Button(
                "Train Model",
                elem_classes="primary-btn",
            )

            train_status = gr.HTML()

            train_metrics = gr.Dataframe(
                interactive=False,
                label="Training Metrics",
            )

            train_prediction_table = gr.Dataframe(
                interactive=False,
                label="Test Predictions",
                max_height=400,
            )

            train_importance_table = gr.Dataframe(
                interactive=False,
                label="Initial Feature Importance",
                max_height=350,
            )

        # ====================================================
        # EVALUATION
        # ====================================================

        with gr.Tab("Evaluation"):

            gr.Markdown("## Model Evaluation")

            evaluation_button = gr.Button(
                "Run Evaluation",
                elem_classes="primary-btn",
            )

            evaluation_report = gr.Textbox(
                label="Evaluation Report",
                lines=10,
                interactive=False,
            )

            evaluation_plot = gr.Plot(
                label="Evaluation Visualization"
            )

            evaluation_table = gr.Dataframe(
                interactive=False,
                label="Evaluation Metrics",
            )

        # ====================================================
        # COMPARISON
        # ====================================================

        with gr.Tab("Comparison"):

            gr.Markdown("## Model Comparison")

            comparison_button = gr.Button(
                "Compare Trained Models",
                elem_classes="primary-btn",
            )

            comparison_status = gr.Markdown()

            comparison_table = gr.Dataframe(
                interactive=False,
                label="Model Leaderboard",
                max_height=400,
            )

        # ====================================================
        # TUNING
        # ====================================================

        with gr.Tab("Tuning"):

            gr.Markdown("## Hyperparameter Tuning")

            with gr.Row():

                tuning_target = gr.Dropdown(
                    choices=[],
                    label="Target",
                )

                tuning_task = gr.Dropdown(
                    choices=[
                        "Classification",
                        "Regression",
                    ],
                    value="Classification",
                    label="Task",
                )

                tuning_model = gr.Dropdown(
                    choices=CLASSIFICATION_MODELS,
                    value="Random Forest",
                    label="Model",
                )

            with gr.Row():

                tuning_method = gr.Dropdown(
                    choices=[
                        "GridSearchCV",
                        "RandomizedSearchCV",
                    ],
                    value="GridSearchCV",
                    label="Search Method",
                )

                tuning_cv = gr.Dropdown(
                    choices=[
                        3,
                        5,
                        10,
                    ],
                    value=3,
                    label="CV Folds",
                )

                tuning_scaling = gr.Dropdown(
                    choices=[
                        "None",
                        "Standard",
                        "MinMax",
                        "Robust",
                    ],
                    value="Standard",
                    label="Scaling",
                )

                tuning_encoding = gr.Dropdown(
                    choices=[
                        "One-Hot",
                        "Ordinal",
                    ],
                    value="One-Hot",
                    label="Encoding",
                )

            tuning_button = gr.Button(
                "Tune Model",
                elem_classes="primary-btn",
            )

            tuning_status = gr.HTML()

            tuning_params = gr.Dataframe(
                interactive=False,
                label="Best Parameters",
            )

            tuning_metrics = gr.Dataframe(
                interactive=False,
                label="Tuned Model Performance",
            )

        # ====================================================
        # PREDICTION
        # ====================================================

        with gr.Tab("Prediction"):

            gr.Markdown("## Prediction Center")

            prediction_setup = gr.Button(
                "Load Trained Model Inputs",
                elem_classes="secondary-btn",
            )

            prediction_input = gr.Dataframe(
                headers=[
                    "Feature",
                    "Value",
                ],
                datatype=[
                    "str",
                    "str",
                ],
                column_count=(2, "fixed"),
                value=pd.DataFrame(
                    columns=[
                        "Feature",
                        "Value",
                    ]
                ),
                interactive=True,
                label="Prediction Input",
                max_height=500,
            )

            prediction_button = gr.Button(
                "Predict",
                elem_classes="primary-btn",
            )

            prediction_status = gr.Markdown()

            probability_table = gr.Dataframe(
                headers=[
                    "Class",
                    "Probability",
                    "Percentage",
                ],
                interactive=False,
                label="Classification Probabilities",
            )

            prediction_result_table = gr.Dataframe(
                interactive=False,
                label="Prediction Result",
            )

        # ====================================================
        # EXPLAINABILITY
        # ====================================================

        with gr.Tab("Explainability"):

            gr.Markdown("## Model Explainability")

            explanation_method = gr.Dropdown(
                choices=[
                    "Feature Importance",
                    "Permutation Importance",
                    "Coefficients",
                    "SHAP",
                ],
                value="Feature Importance",
                label="Explanation Method",
            )

            explain_button = gr.Button(
                "Generate Explanation",
                elem_classes="primary-btn",
            )

            explain_status = gr.Markdown()

            explain_table = gr.Dataframe(
                interactive=False,
                label="Explanation Results",
                max_height=450,
            )

            explain_plot = gr.Plot(
                label="Explanation Visualization"
            )

        # ====================================================
        # EXPORT
        # ====================================================

        with gr.Tab("Export"):

            gr.Markdown("## Export Center")

            with gr.Row():

                export_type = gr.Dropdown(
                    choices=EXPORT_TYPES,
                    value="Processed Dataset",
                    label="Export Type",
                )

                export_format = gr.Dropdown(
                    choices=[
                        "CSV",
                        "Excel",
                    ],
                    value="CSV",
                    label="Format",
                )

            export_button = gr.Button(
                "Generate Download",
                elem_classes="primary-btn",
            )

            export_status = gr.Markdown()

            export_file = gr.File(
                label="Generated Download"
            )

            gr.Markdown("### Quick Downloads")

            with gr.Row():

                quick_original_button = gr.Button(
                    "Download Original",
                    elem_classes="secondary-btn",
                )

                quick_processed_button = gr.Button(
                    "Download Processed",
                    elem_classes="secondary-btn",
                )

                quick_model_button = gr.Button(
                    "Download Model Results",
                    elem_classes="secondary-btn",
                )

                quick_prediction_button = gr.Button(
                    "Download Prediction",
                    elem_classes="secondary-btn",
                )

            with gr.Row():

                quick_original_file = gr.File(
                    label="Original File"
                )

                quick_processed_file = gr.File(
                    label="Processed File"
                )

                quick_model_file = gr.File(
                    label="Model Results"
                )

                quick_prediction_file = gr.File(
                    label="Prediction File"
                )


# ============================================================
# 33. EVENT WIRING
# ============================================================

    # Upload
    upload.change(
        load_dataset,
        inputs=[upload],
        outputs=[
            upload_status,
            load_preview,
            initial_preview,
            overview_status,
            stats_columns,
        ],
    )

    upload.change(
        overview,
        inputs=[],
        outputs=[
            overview_status,
            kpi_rows,
            kpi_cols,
            kpi_num,
            kpi_cat,
            kpi_missing,
            kpi_duplicates,
            kpi_outliers,
            kpi_target,
            dtype_table,
            describe_table,
        ],
    )

    # Dataset choices after upload
    upload.change(
        lambda: gr.update(
            choices=all_columns(processed_df),
            value=all_columns(processed_df)[0]
            if all_columns(processed_df)
            else None,
        ),
        inputs=[],
        outputs=[target_column_ui],
    )

    upload.change(
        lambda: gr.update(
            choices=all_columns(processed_df),
            value=all_columns(processed_df)[0]
            if all_columns(processed_df)
            else None,
        ),
        inputs=[],
        outputs=[train_target],
    )

    upload.change(
        lambda: gr.update(
            choices=all_columns(processed_df),
            value=all_columns(processed_df)[0]
            if all_columns(processed_df)
            else None,
        ),
        inputs=[],
        outputs=[tuning_target],
    )

    upload.change(
        lambda: gr.update(
            choices=all_columns(processed_df),
            value=all_columns(processed_df)[0]
            if all_columns(processed_df)
            else None,
        ),
        inputs=[],
        outputs=[feature_target],
    )

    upload.change(
        lambda: (
            gr.update(
                choices=all_columns(processed_df),
                value=all_columns(processed_df)[0]
                if all_columns(processed_df)
                else None,
            ),
            gr.update(
                choices=all_columns(processed_df),
                value=all_columns(processed_df)[0]
                if all_columns(processed_df)
                else None,
            ),
            gr.update(
                choices=all_columns(processed_df),
                value=None,
            ),
            gr.update(
                choices=all_columns(processed_df),
                value=None,
            ),
            gr.update(
                choices=all_columns(processed_df),
                value=None,
            ),
        ),
        inputs=[],
        outputs=[
            graph_x,
            graph_y,
            graph_z,
            graph_hue,
            ts_date,
        ],
    )

    upload.change(
        lambda: gr.update(
            choices=all_columns(processed_df),
            value=None,
        ),
        inputs=[],
        outputs=[ts_value],
    )

    upload.change(
        lambda: gr.update(
            choices=all_columns(processed_df),
            value=None,
        ),
        inputs=[],
        outputs=[clean_columns],
    )

    upload.change(
        lambda: gr.update(
            choices=all_columns(processed_df),
            value=None,
        ),
        inputs=[],
        outputs=[feature_columns],
    )

    # Viewer
    viewer_type.change(
        update_view_controls,
        inputs=[viewer_type],
        outputs=[
            viewer_rows,
            viewer_start,
            viewer_end,
        ],
    )

    viewer_refresh.click(
        get_viewed_dataset,
        inputs=[
            viewer_dataset,
            viewer_type,
            viewer_rows,
            viewer_start,
            viewer_end,
            viewer_search,
        ],
        outputs=[
            viewer_table,
            viewer_info,
            viewer_displayed,
            viewer_total,
            viewer_columns_count,
        ],
    )

    viewer_clear.click(
        clear_search,
        inputs=[],
        outputs=[
            viewer_search,
            viewer_table,
            viewer_info,
        ],
    )

    # Statistics
    stats_button.click(
        calculate_statistics,
        inputs=[
            stats_columns,
            percentile,
        ],
        outputs=[
            stats_table,
            stats_status,
        ],
    )

    correlation_button.click(
        advanced_correlation,
        inputs=[],
        outputs=[
            correlation_table,
            correlation_status,
        ],
    )

    # Quality
    quality_button.click(
        quality_analysis,
        inputs=[],
        outputs=[
            quality_table,
            quality_score,
            quality_summary,
            healthcare_warning,
        ],
    )

    missing_button.click(
        missing_report,
        inputs=[],
        outputs=[
            missing_table,
            missing_status,
        ],
    )

    duplicate_button.click(
        duplicate_report,
        inputs=[],
        outputs=[
            duplicate_table,
            duplicate_status,
        ],
    )

    outlier_button.click(
        outlier_report,
        inputs=[],
        outputs=[
            outlier_table,
            outlier_status,
        ],
    )

    # Cleaning controls
    clean_operation.change(
        update_cleaning_controls,
        inputs=[
            clean_operation,
            clean_apply_to,
        ],
        outputs=[
            clean_columns,
            clean_constant,
            clean_old,
            clean_new,
            clean_threshold,
            clean_help,
        ],
    )

    clean_apply_to.change(
        update_cleaning_controls,
        inputs=[
            clean_operation,
            clean_apply_to,
        ],
        outputs=[
            clean_columns,
            clean_constant,
            clean_old,
            clean_new,
            clean_threshold,
            clean_help,
        ],
    )

    clean_preview_button.click(
        cleaning_preview,
        inputs=[
            clean_operation,
            clean_apply_to,
            clean_columns,
            clean_constant,
            clean_old,
            clean_new,
            clean_threshold,
        ],
        outputs=[
            clean_before,
            clean_after,
            clean_message,
            clean_comparison,
        ],
    )

    clean_apply_button.click(
        apply_cleaning,
        inputs=[
            clean_operation,
            clean_apply_to,
            clean_columns,
            clean_constant,
            clean_old,
            clean_new,
            clean_threshold,
        ],
        outputs=[
            clean_before,
            clean_after,
            clean_message,
            clean_comparison,
            cleaning_history_table,
        ],
    )

    clean_reset_button.click(
        reset_to_original,
        inputs=[],
        outputs=[
            clean_after,
            clean_message,
            cleaning_history_table,
        ],
    )

    clean_reset_button.click(
        overview,
        inputs=[],
        outputs=[
            overview_status,
            kpi_rows,
            kpi_cols,
            kpi_num,
            kpi_cat,
            kpi_missing,
            kpi_duplicates,
            kpi_outliers,
            kpi_target,
            dtype_table,
            describe_table,
        ],
    )

    # EDA
    graph_type.change(
        update_eda_controls,
        inputs=[graph_type],
        outputs=[
            graph_x,
            graph_y,
            graph_z,
            graph_hue,
        ],
    )

    graph_button.click(
        create_eda_graph,
        inputs=[
            graph_type,
            graph_x,
            graph_y,
            graph_z,
            graph_hue,
        ],
        outputs=[
            graph_output,
            graph_status,
        ],
    )

    ts_button.click(
        time_series_analysis,
        inputs=[
            ts_date,
            ts_value,
        ],
        outputs=[
            ts_plot,
            ts_table,
            ts_status,
        ],
    )

    # Target
    target_button.click(
        target_analysis,
        inputs=[target_column_ui],
        outputs=[
            target_summary,
            target_table,
            target_task,
        ],
    )

    feature_target_button.click(
        feature_target_analysis,
        inputs=[target_column_ui],
        outputs=[
            feature_target_table,
            feature_target_status,
        ],
    )

    # Features
    feature_operation.change(
        update_feature_controls,
        inputs=[feature_operation],
        outputs=[
            feature_k,
            feature_threshold,
        ],
    )

    feature_apply_button.click(
        apply_feature_engineering,
        inputs=[
            feature_operation,
            feature_columns,
            feature_k,
            feature_threshold,
            feature_target,
        ],
        outputs=[
            feature_preview,
            feature_status,
            feature_history_table,
        ],
    )

    feature_reset_button.click(
        reset_features,
        inputs=[],
        outputs=[
            feature_preview,
            feature_status,
            feature_history_table,
        ],
    )

    # Training task/model compatibility
    def update_training_models(task):
        if task == "Classification":
            choices = CLASSIFICATION_MODELS
        else:
            choices = REGRESSION_MODELS

        return gr.update(
            choices=choices,
            value=choices[0] if choices else None,
        )

    train_task.change(
        update_training_models,
        inputs=[train_task],
        outputs=[train_model],
    )

    tuning_task.change(
        update_training_models,
        inputs=[tuning_task],
        outputs=[tuning_model],
    )

    train_button.click(
        train_model,
        inputs=[
            train_target,
            train_task,
            train_model,
            test_size,
            scaling,
            encoding,
            cv_folds,
        ],
        outputs=[
            train_status,
            train_metrics,
            comparison_table,
            train_prediction_table,
            train_importance_table,
        ],
    )

    # Evaluation
    evaluation_button.click(
        evaluation_analysis,
        inputs=[],
        outputs=[
            evaluation_report,
            evaluation_plot,
            evaluation_table,
        ],
    )

    # Comparison
    comparison_button.click(
        compare_models,
        inputs=[],
        outputs=[
            comparison_table,
            comparison_status,
        ],
    )

    # Tuning
    tuning_button.click(
        tune_model,
        inputs=[
            tuning_target,
            tuning_task,
            tuning_model,
            tuning_method,
            tuning_cv,
            tuning_scaling,
            tuning_encoding,
        ],
        outputs=[
            tuning_status,
            tuning_params,
            tuning_metrics,
        ],
    )

    # Prediction
    prediction_setup.click(
        prediction_template,
        inputs=[],
        outputs=[
            prediction_input,
        ],
    )

    prediction_button.click(
        predict_from_table,
        inputs=[
            prediction_input,
        ],
        outputs=[
            prediction_status,
            prediction_input,
            probability_table,
        ],
    )

    # Explainability
    explain_button.click(
        explain_model,
        inputs=[
            explanation_method,
        ],
        outputs=[
            explain_table,
            explain_status,
            explain_plot,
        ],
    )

    # Export
    export_button.click(
        export_data,
        inputs=[
            export_type,
            export_format,
        ],
        outputs=[
            export_file,
            export_status,
        ],
    )

    quick_original_button.click(
        quick_original,
        inputs=[],
        outputs=[quick_original_file],
    )

    quick_processed_button.click(
        quick_processed,
        inputs=[],
        outputs=[quick_processed_file],
    )

    quick_model_button.click(
        quick_model,
        inputs=[],
        outputs=[quick_model_file],
    )

    quick_prediction_button.click(
        quick_prediction,
        inputs=[],
        outputs=[quick_prediction_file],
    )


# ============================================================
# 34. LAUNCH
# ============================================================

print("=" * 70)
print("AI & ML ANALYTICS STUDIO")
print("=" * 70)
print("Installation completed.")
print("Gradio version:", gr.__version__)
print("XGBoost available:", XGBOOST_AVAILABLE)
print("SHAP available:", SHAP_AVAILABLE)
print("=" * 70)

app.launch(
    share=True,
    debug=False,
    show_error=True,
)