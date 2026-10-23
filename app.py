# ============================================================
# AI & ML ANALYTICS STUDIO
# Complete single-cell Google Colab application
# ============================================================
# IMPORTANT:
# - Paste this ENTIRE code into ONE Google Colab cell.
# - Run the cell.
# - Do not add !pip, %pip, Markdown fences, or extra code.
# - This code uses subprocess for installation, so it is valid Python.
# ============================================================

import sys
import subprocess
import os
import re
import uuid
import tempfile
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. INSTALLATION
# ------------------------------------------------------------
REQUIRED_PACKAGES = [
    "gradio==5.50.0",
    "pandas>=2.0",
    "numpy>=1.24",
    "matplotlib>=3.7",
    "seaborn>=0.13",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "openpyxl>=3.1",
    "xlrd>=2.0.1",
]

for package in REQUIRED_PACKAGES:
    try:
        name = package.split(">=")[0].split("==")[0]
        __import__(name.replace("-", "_"))
    except Exception:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", package]
            )
        except Exception as install_error:
            print(f"Package installation warning for {package}: {install_error}")

# ------------------------------------------------------------
# 2. IMPORTS
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

from scipy import stats
from scipy.stats import zscore

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
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


# ------------------------------------------------------------
# 3. GLOBAL STATE
# ------------------------------------------------------------
original_df = None
processed_df = None
viewed_df = None

cleaning_history = []

trained_pipeline = None
trained_model = None
trained_preprocessor = None

target_column = None
task_type = None

X_train = None
X_test = None
y_train = None
y_test = None

predictions = None
prediction_probabilities = None

model_results = pd.DataFrame()
last_metrics = pd.DataFrame()
last_feature_importance = pd.DataFrame()

trained_feature_names = []
trained_input_columns = []

last_evaluation_fig = None
last_roc_fig = None
last_pr_fig = None

last_tuned_model = None
last_tuning_results = None

TEMP_DIR = os.path.join(tempfile.gettempdir(), "ai_ml_analytics_studio")
os.makedirs(TEMP_DIR, exist_ok=True)


# ------------------------------------------------------------
# 4. GENERAL HELPERS
# ------------------------------------------------------------
def empty_df():
    return pd.DataFrame()


def safe_str(value):
    if value is None:
        return ""
    return str(value)


def numeric_columns(df):
    if df is None or df.empty:
        return []
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df):
    if df is None or df.empty:
        return []
    return df.select_dtypes(exclude=np.number).columns.tolist()


def all_columns(df):
    if df is None:
        return []
    return df.columns.astype(str).tolist()


def clean_column_name(value):
    return str(value).strip()


def safe_filename(name):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
    return name[:100]


def make_temp_file(prefix, extension):
    filename = (
        f"{safe_filename(prefix)}_"
        f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex[:8]}.{extension}"
    )
    return os.path.join(TEMP_DIR, filename)


def file_exists(path):
    return path if path and os.path.exists(path) else None


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def count_missing(df):
    if df is None or df.empty:
        return 0
    return int(df.isna().sum().sum())


def count_duplicates(df):
    if df is None or df.empty:
        return 0
    return int(df.duplicated().sum())


def count_outliers(df):
    if df is None or df.empty:
        return 0

    total = 0

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

        total += int(((s < lower) | (s > upper)).sum())

    return total


def dataset_metrics(df):
    if df is None:
        return {
            "Rows": 0,
            "Columns": 0,
            "Missing": 0,
            "Duplicates": 0,
            "Numeric": 0,
            "Categorical": 0,
            "Outliers": 0,
        }

    return {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Missing": count_missing(df),
        "Duplicates": count_duplicates(df),
        "Numeric": len(numeric_columns(df)),
        "Categorical": len(categorical_columns(df)),
        "Outliers": count_outliers(df),
    }


def comparison_table(before, after):
    b = dataset_metrics(before)
    a = dataset_metrics(after)

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


def quality_score(df):
    if df is None or df.empty:
        return 0

    rows = len(df)
    cols = max(len(df.columns), 1)

    missing_rate = count_missing(df) / max(rows * cols, 1)
    duplicate_rate = count_duplicates(df) / max(rows, 1)

    completeness = max(0, 100 - missing_rate * 100)
    uniqueness = max(0, 100 - duplicate_rate * 100)

    invalid_count = 0
    infinite_count = 0

    for col in numeric_columns(df):
        values = pd.to_numeric(df[col], errors="coerce")

        infinite_count += int(np.isinf(values).sum())

        if len(values.dropna()) > 0:
            invalid_count += int((values.dropna() < 0).sum()) if any(
                x in str(col).lower()
                for x in [
                    "age",
                    "bmi",
                    "heart",
                    "rate",
                    "cholesterol",
                    "pressure",
                ]
            ) else 0

    consistency = max(
        0,
        100
        - ((count_duplicates(df) / max(rows, 1)) * 100)
        - ((infinite_count / max(rows * cols, 1)) * 100),
    )

    validity = max(
        0,
        100
        - ((invalid_count / max(rows * cols, 1)) * 100)
        - ((infinite_count / max(rows * cols, 1)) * 100),
    )

    score = round(
        (
            completeness
            + consistency
            + validity
            + uniqueness
        )
        / 4
    )

    return int(max(0, min(100, score)))


# ------------------------------------------------------------
# 5. OVERVIEW
# ------------------------------------------------------------
def overview_text(df):
    if df is None:
        return "Load a CSV, XLSX, or XLS dataset to begin."

    if df.empty:
        return "The uploaded dataset is empty."

    m = dataset_metrics(df)
    score = quality_score(df)

    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    return f"""
# AI & ML Analytics Studio

### Dataset Overview

| Metric | Value |
|---|---:|
| Rows | {m["Rows"]:,} |
| Columns | {m["Columns"]:,} |
| Numeric Columns | {m["Numeric"]:,} |
| Categorical Columns | {m["Categorical"]:,} |
| Missing Values | {m["Missing"]:,} |
| Duplicate Rows | {m["Duplicates"]:,} |
| Estimated IQR Outliers | {m["Outliers"]:,} |
| Memory Usage | {memory_mb:.2f} MB |
| Dashboard Quality Score | {score}/100 |

The quality score is a **dashboard heuristic**, not an official statistical or clinical quality score.

### Dataset Types

- Numeric columns: {m["Numeric"]}
- Categorical/text columns: {m["Categorical"]}
- Total columns: {m["Columns"]}

### Recommended Workflow

Upload → View → Statistics → Quality → Cleaning Preview → Apply Cleaning → EDA → Target → Features → Training → Evaluation → Prediction → Explainability → Export
"""


# ------------------------------------------------------------
# 6. HEALTHCARE VALIDATION
# ------------------------------------------------------------
def healthcare_validation(df):
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["Column", "Detected Pattern", "Suspicious Values", "Recommendation"]
        )

    results = []

    for col in df.columns:
        name = str(col).lower()
        series = pd.to_numeric(df[col], errors="coerce")

        if any(k in name for k in ["age", "patient_age"]):
            mask = (series < 0) | (series > 120)
            n = int(mask.sum())

            if n:
                results.append(
                    {
                        "Column": col,
                        "Detected Pattern": "Possible Age",
                        "Suspicious Values": n,
                        "Recommendation": "Review values below 0 or above 120.",
                    }
                )

        elif "bmi" in name:
            mask = (series <= 0) | (series > 100)
            n = int(mask.sum())

            if n:
                results.append(
                    {
                        "Column": col,
                        "Detected Pattern": "Possible BMI",
                        "Suspicious Values": n,
                        "Recommendation": "Review non-positive or extremely high BMI values.",
                    }
                )

        elif any(k in name for k in ["heart_rate", "heartrate", "pulse"]):
            mask = series <= 0
            n = int(mask.sum())

            if n:
                results.append(
                    {
                        "Column": col,
                        "Detected Pattern": "Possible Heart Rate",
                        "Suspicious Values": n,
                        "Recommendation": "Review non-positive values.",
                    }
                )

        elif "cholesterol" in name:
            mask = series < 0
            n = int(mask.sum())

            if n:
                results.append(
                    {
                        "Column": col,
                        "Detected Pattern": "Possible Cholesterol",
                        "Suspicious Values": n,
                        "Recommendation": "Review negative values.",
                    }
                )

        elif any(
            k in name
            for k in [
                "blood_pressure",
                "bloodpressure",
                "systolic",
                "diastolic",
            ]
        ):
            mask = series <= 0
            n = int(mask.sum())

            if n:
                results.append(
                    {
                        "Column": col,
                        "Detected Pattern": "Possible Blood Pressure",
                        "Suspicious Values": n,
                        "Recommendation": "Review non-positive values.",
                    }
                )

    return pd.DataFrame(
        results,
        columns=[
            "Column",
            "Detected Pattern",
            "Suspicious Values",
            "Recommendation",
        ],
    )


# ------------------------------------------------------------
# 7. DATA QUALITY
# ------------------------------------------------------------
def quality_analysis(df):
    if df is None:
        return (
            "Load a dataset first.",
            pd.DataFrame(),
        )

    rows = []

    for col in df.columns:
        s = df[col]

        numeric_like = False

        if s.dtype == object:
            converted = pd.to_numeric(s, errors="coerce")
            non_null = s.notna().sum()

            if non_null > 0:
                numeric_like = (
                    converted.notna().sum() / non_null >= 0.80
                )

        unique_count = s.nunique(dropna=True)

        high_cardinality = (
            not pd.api.types.is_numeric_dtype(s)
            and unique_count > max(50, len(df) * 0.5)
        )

        empty_column = s.isna().all()
        constant_column = unique_count <= 1

        near_constant = False

        if unique_count > 1:
            frequencies = s.value_counts(normalize=True, dropna=False)

            if len(frequencies) > 0 and frequencies.iloc[0] >= 0.98:
                near_constant = True

        infinite_count = 0

        if pd.api.types.is_numeric_dtype(s):
            infinite_count = int(np.isinf(s).sum())

        rows.append(
            {
                "Column": col,
                "Data Type": str(s.dtype),
                "Missing": int(s.isna().sum()),
                "Unique": int(unique_count),
                "Duplicates": int(s.duplicated().sum()),
                "Empty Column": empty_column,
                "Constant": constant_column,
                "Near Constant": near_constant,
                "Numeric-Looking Text": numeric_like,
                "Infinite": infinite_count,
                "High Cardinality": high_cardinality,
            }
        )

    result = pd.DataFrame(rows)

    healthcare = healthcare_validation(df)

    score = quality_score(df)

    report = f"""
# Data Quality Center

## Dashboard Quality Score: {score}/100

This score is a **dashboard heuristic** intended for educational data-quality assessment. It is not an official statistical, regulatory, or clinical quality score.

### Quality Dimensions

- Completeness: evaluates missingness.
- Consistency: evaluates duplicates and consistency problems.
- Validity: evaluates obvious invalid/infinite values.
- Uniqueness: evaluates duplicate records.

### Automatically Detected Issues

- Empty columns: {(result["Empty Column"]).sum() if not result.empty else 0}
- Constant columns: {(result["Constant"]).sum() if not result.empty else 0}
- Near-constant columns: {(result["Near Constant"]).sum() if not result.empty else 0}
- Numeric-looking text: {(result["Numeric-Looking Text"]).sum() if not result.empty else 0}
- Infinite values: {int(result["Infinite"].sum()) if not result.empty else 0}
- High-cardinality categorical columns: {(result["High Cardinality"]).sum() if not result.empty else 0}
- Duplicate rows: {count_duplicates(df)}
- Missing cells: {count_missing(df)}

### Healthcare-aware Validation

Healthcare checks are warnings only. Records are never automatically deleted.
"""

    if not healthcare.empty:
        report += "\n\n### Healthcare Warnings\nSuspicious values were detected. Review them before changing the data."

    return report, result


# ------------------------------------------------------------
# 8. DATASET VIEWER
# ------------------------------------------------------------
def get_source_df(source):
    if source == "Processed":
        return processed_df if processed_df is not None else original_df

    return original_df


def viewer_mode_update(mode):
    rows_visible = mode in ["Head", "Tail", "Random"]
    custom_visible = mode == "Custom Range"

    return (
        gr.update(visible=rows_visible),
        gr.update(visible=custom_visible),
        gr.update(visible=custom_visible),
    )


def create_view(source, mode, rows, start, end, search):
    global viewed_df

    df = get_source_df(source)

    if df is None:
        viewed_df = None
        return "Load a dataset first.", pd.DataFrame()

    working = df.copy()

    search = safe_str(search).strip()

    if search:
        mask = pd.Series(False, index=working.index)

        for col in working.columns:
            mask = mask | working[col].astype(str).str.contains(
                re.escape(search),
                case=False,
                na=False,
            )

        working = working.loc[mask].copy()

    total = len(working)

    try:
        n = int(rows)
    except Exception:
        n = 50

    if mode == "Head":
        result = working.head(n)
        info = (
            f"Showing first {len(result):,} rows of {total:,}"
        )

    elif mode == "Tail":
        result = working.tail(n)
        info = (
            f"Showing last {len(result):,} rows of {total:,}"
        )

    elif mode == "Random":
        sample_n = min(n, total)

        result = (
            working.sample(
                sample_n,
                random_state=42,
            )
            if sample_n > 0
            else working.head(0)
        )

        info = (
            f"Showing random {len(result):,} rows from {total:,}"
        )

    elif mode == "Custom Range":
        try:
            start_i = max(0, int(start))
        except Exception:
            start_i = 0

        try:
            end_i = min(total, int(end))
        except Exception:
            end_i = min(total, start_i + 50)

        if end_i < start_i:
            end_i = start_i

        result = working.iloc[start_i:end_i].copy()

        info = (
            f"Showing rows {start_i:,}–{max(start_i, end_i - 1):,} "
            f"of {total:,}"
        )

    else:
        safe_max = min(total, 1000)
        result = working.head(safe_max)

        if total > safe_max:
            info = (
                f"Showing first {safe_max:,} of {total:,} rows "
                f"for responsiveness."
            )
        else:
            info = f"Showing entire dataset — {total:,} rows."

    viewed_df = result.copy()

    info += (
        f"\n\nDisplayed rows: {len(result):,} | "
        f"Total rows: {total:,} | "
        f"Columns: {len(working.columns):,}"
    )

    return info, result


def clear_search(source, mode, rows, start, end):
    return "", *create_view(
        source,
        mode,
        rows,
        start,
        end,
        "",
    )


# ------------------------------------------------------------
# 9. STATISTICS
# ------------------------------------------------------------
def statistics_analysis(columns, percentile):
    df = processed_df if processed_df is not None else original_df

    if df is None:
        return pd.DataFrame()

    nums = numeric_columns(df)

    if not columns:
        columns = nums

    columns = [c for c in columns if c in df.columns]

    if not columns:
        return pd.DataFrame()

    try:
        pct = float(percentile) / 100.0
    except Exception:
        pct = 0.90

    rows = []

    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()

        if s.empty:
            continue

        try:
            mode_value = s.mode().iloc[0]
        except Exception:
            mode_value = np.nan

        mad = np.mean(np.abs(s - s.mean()))

        q1 = s.quantile(0.25)
        median = s.quantile(0.50)
        q3 = s.quantile(0.75)

        rows.append(
            {
                "Column": col,
                "Count": len(s),
                "Mean": s.mean(),
                "Median": median,
                "Mode": mode_value,
                "Min": s.min(),
                "Max": s.max(),
                "Range": s.max() - s.min(),
                "Variance": s.var(),
                "Std Dev": s.std(),
                "Q1": q1,
                "Q3": q3,
                "IQR": q3 - q1,
                "Percentile": s.quantile(pct),
                "P10": s.quantile(0.10),
                "P25": q1,
                "P50": median,
                "P75": q3,
                "P90": s.quantile(0.90),
                "P95": s.quantile(0.95),
                "Skewness": s.skew(),
                "Kurtosis": s.kurt(),
                "MAD": mad,
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 10. MISSING / DUPLICATE / OUTLIER REPORTS
# ------------------------------------------------------------
def missing_report(df=None):
    if df is None:
        df = processed_df if processed_df is not None else original_df

    if df is None:
        return pd.DataFrame()

    rows = []

    for col in df.columns:
        missing = int(df[col].isna().sum())

        rows.append(
            {
                "Column": col,
                "Missing": missing,
                "Missing %": round(
                    missing / max(len(df), 1) * 100,
                    2,
                ),
                "Non-Missing": int(df[col].notna().sum()),
                "Data Type": str(df[col].dtype),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "Missing",
        ascending=False,
    )


def duplicate_report(df=None):
    if df is None:
        df = processed_df if processed_df is not None else original_df

    if df is None:
        return pd.DataFrame()

    duplicate_mask = df.duplicated(keep=False)

    return df.loc[duplicate_mask].copy()


def outlier_report(df=None):
    if df is None:
        df = processed_df if processed_df is not None else original_df

    if df is None:
        return pd.DataFrame()

    rows = []

    for col in numeric_columns(df):
        s = pd.to_numeric(df[col], errors="coerce")

        clean = s.dropna()

        if len(clean) < 4:
            continue

        q1 = clean.quantile(0.25)
        q3 = clean.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            iqr_count = 0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            iqr_count = int(
                ((clean < lower) | (clean > upper)).sum()
            )

        z_count = 0

        if clean.std() > 0:
            z_values = np.abs(stats.zscore(clean, nan_policy="omit"))
            z_count = int((z_values > 3).sum())

        rows.append(
            {
                "Column": col,
                "IQR Lower": q1 - 1.5 * iqr,
                "IQR Upper": q3 + 1.5 * iqr,
                "IQR Outliers": iqr_count,
                "Z-score Outliers": z_count,
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 11. CLEANING ENGINE
# ------------------------------------------------------------
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
    "Replace Text Values",
    "Convert Numeric",
    "Convert Category",
    "Convert Date",
    "Replace Negative Numeric Values",
    "Replace Infinite Values",
    "Replace Empty Strings",
    "Replace Common Null Strings",
    "IQR Remove",
    "IQR Cap",
    "Z-score Remove",
    "Z-score Cap",
]


def clean_operation_controls(operation):
    missing_ops = [
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
    ]

    text_ops = [
        "Strip Whitespace",
        "Lowercase Text",
        "Uppercase Text",
        "Title Case Text",
        "Replace Text Values",
    ]

    needs_threshold = operation == "Drop Columns Above Missing %"
    needs_constant = operation == "Constant Imputation"
    needs_replace = operation == "Replace Text Values"
    needs_z = operation in [
        "Z-score Remove",
        "Z-score Cap",
    ]

    selected_column_visible = operation not in [
        "Remove Exact Duplicates",
        "Drop Missing Columns",
        "Drop Columns Above Missing %",
        "Replace Infinite Values",
        "Replace Empty Strings",
        "Replace Common Null Strings",
    ]

    return (
        gr.update(visible=selected_column_visible),
        gr.update(visible=needs_constant),
        gr.update(visible=needs_replace),
        gr.update(visible=needs_threshold),
        gr.update(visible=needs_z),
    )


def selected_clean_columns(scope, selected_column, multi_columns, df):
    if df is None:
        return []

    if scope == "Selected Column":
        if selected_column in df.columns:
            return [selected_column]

        return []

    if multi_columns:
        valid = [c for c in multi_columns if c in df.columns]

        if valid:
            return valid

    return list(df.columns)


def apply_cleaning_operation(
    df,
    operation,
    scope,
    selected_column,
    multi_columns,
    constant_value,
    replace_from,
    replace_to,
    threshold,
    z_threshold,
):
    result = df.copy()

    cols = selected_clean_columns(
        scope,
        selected_column,
        multi_columns,
        result,
    )

    if not cols and operation not in [
        "Remove Exact Duplicates",
        "Drop Missing Columns",
        "Drop Columns Above Missing %",
        "Replace Infinite Values",
        "Replace Empty Strings",
        "Replace Common Null Strings",
    ]:
        return result, "No valid column selected."

    if operation == "Drop Missing Rows":
        if scope == "Entire Dataset":
            result = result.dropna()
        else:
            result = result.dropna(subset=cols)

    elif operation == "Drop Missing Columns":
        if scope == "Entire Dataset":
            result = result.dropna(axis=1)
        else:
            result = result.dropna(
                axis=1,
                subset=cols,
            )

    elif operation == "Drop Columns Above Missing %":
        try:
            limit = float(threshold)
        except Exception:
            limit = 50

        missing_pct = result.isna().mean() * 100

        drop_cols = missing_pct[
            missing_pct > limit
        ].index.tolist()

        result = result.drop(columns=drop_cols)

    elif operation == "Mean Imputation":
        for col in cols:
            values = pd.to_numeric(result[col], errors="coerce")
            replacement = values.mean()

            if pd.notna(replacement):
                result[col] = result[col].fillna(replacement)

    elif operation == "Median Imputation":
        for col in cols:
            values = pd.to_numeric(result[col], errors="coerce")
            replacement = values.median()

            if pd.notna(replacement):
                result[col] = result[col].fillna(replacement)

    elif operation == "Mode Imputation":
        for col in cols:
            mode = result[col].mode(dropna=True)

            if len(mode):
                result[col] = result[col].fillna(mode.iloc[0])

    elif operation == "Constant Imputation":
        result[cols] = result[cols].fillna(
            safe_str(constant_value)
        )

    elif operation == "Forward Fill":
        result[cols] = result[cols].ffill()

    elif operation == "Backward Fill":
        result[cols] = result[cols].bfill()

    elif operation == "Interpolation":
        for col in cols:
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].interpolate()

    elif operation == "Remove Exact Duplicates":
        result = result.drop_duplicates()

    elif operation == "Strip Whitespace":
        for col in cols:
            if result[col].dtype == object:
                result[col] = result[col].astype(str).str.strip()

    elif operation == "Lowercase Text":
        for col in cols:
            if result[col].dtype == object:
                result[col] = result[col].astype(str).str.lower()

    elif operation == "Uppercase Text":
        for col in cols:
            if result[col].dtype == object:
                result[col] = result[col].astype(str).str.upper()

    elif operation == "Title Case Text":
        for col in cols:
            if result[col].dtype == object:
                result[col] = result[col].astype(str).str.title()

    elif operation == "Replace Text Values":
        for col in cols:
            result[col] = result[col].replace(
                safe_str(replace_from),
                safe_str(replace_to),
            )

    elif operation == "Convert Numeric":
        for col in cols:
            result[col] = pd.to_numeric(
                result[col],
                errors="coerce",
            )

    elif operation == "Convert Category":
        for col in cols:
            result[col] = result[col].astype("category")

    elif operation == "Convert Date":
        for col in cols:
            result[col] = pd.to_datetime(
                result[col],
                errors="coerce",
            )

    elif operation == "Replace Negative Numeric Values":
        for col in cols:
            values = pd.to_numeric(
                result[col],
                errors="coerce",
            )

            result.loc[values < 0, col] = np.nan

    elif operation == "Replace Infinite Values":
        numeric_cols = numeric_columns(result)

        result[numeric_cols] = result[numeric_cols].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    elif operation == "Replace Empty Strings":
        for col in cols if cols else result.columns:
            if result[col].dtype == object:
                result[col] = result[col].replace(
                    r"^\s*$",
                    np.nan,
                    regex=True,
                )

    elif operation == "Replace Common Null Strings":
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

        for col in cols if cols else result.columns:
            result[col] = result[col].replace(
                null_values,
                np.nan,
            )

    elif operation in [
        "IQR Remove",
        "IQR Cap",
    ]:
        for col in cols:
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

            if operation == "IQR Remove":
                result = result[
                    (result[col].isna())
                    | (
                        (result[col] >= lower)
                        & (result[col] <= upper)
                    )
                ].copy()

            else:
                result[col] = result[col].clip(
                    lower,
                    upper,
                )

    elif operation in [
        "Z-score Remove",
        "Z-score Cap",
    ]:
        try:
            z_limit = float(z_threshold)
        except Exception:
            z_limit = 3.0

        for col in cols:
            if not pd.api.types.is_numeric_dtype(result[col]):
                continue

            s = result[col]

            mean = s.mean()
            std = s.std()

            if pd.isna(std) or std == 0:
                continue

            lower = mean - z_limit * std
            upper = mean + z_limit * std

            if operation == "Z-score Remove":
                result = result[
                    (result[col].isna())
                    | (
                        (result[col] >= lower)
                        & (result[col] <= upper)
                    )
                ].copy()

            else:
                result[col] = result[col].clip(
                    lower,
                    upper,
                )

    return result.reset_index(drop=True), "Operation completed."


def cleaning_preview(
    operation,
    scope,
    selected_column,
    multi_columns,
    constant_value,
    replace_from,
    replace_to,
    threshold,
    z_threshold,
):
    if processed_df is None:
        return (
            "Load a dataset first.",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    try:
        preview, message = apply_cleaning_operation(
            processed_df,
            operation,
            scope,
            selected_column,
            multi_columns,
            constant_value,
            replace_from,
            replace_to,
            threshold,
            z_threshold,
        )

        comparison = comparison_table(
            processed_df,
            preview,
        )

        before_missing = count_missing(processed_df)
        after_missing = count_missing(preview)

        before_outliers = count_outliers(processed_df)
        after_outliers = count_outliers(preview)

        cols = selected_clean_columns(
            scope,
            selected_column,
            multi_columns,
            processed_df,
        )

        detail = f"""
### Cleaning Preview

**Operation:** {operation}

**Apply To:** {scope}

**Columns:** {", ".join(map(str, cols)) if cols else "Dataset-wide"}

### What will change?

Rows: {len(processed_df):,} → {len(preview):,}

Missing values: {before_missing:,} → {after_missing:,}

Duplicates: {count_duplicates(processed_df):,} → {count_duplicates(preview):,}

Estimated outliers: {before_outliers:,} → {after_outliers:,}

### Important

This is only a **preview**. The processed dataset is not changed until **Apply Change** is clicked.

{message}
"""

        return (
            detail,
            processed_df.head(100),
            preview.head(100),
            comparison,
        )

    except Exception as e:
        return (
            f"Cleaning preview failed: {type(e).__name__}: {str(e)}",
            processed_df.head(100),
            processed_df.head(100),
            pd.DataFrame(),
        )


def apply_cleaning(
    operation,
    scope,
    selected_column,
    multi_columns,
    constant_value,
    replace_from,
    replace_to,
    threshold,
    z_threshold,
):
    global processed_df
    global cleaning_history

    if processed_df is None:
        return (
            "Load a dataset first.",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    before = processed_df.copy()

    try:
        after, message = apply_cleaning_operation(
            before,
            operation,
            scope,
            selected_column,
            multi_columns,
            constant_value,
            replace_from,
            replace_to,
            threshold,
            z_threshold,
        )

        processed_df = after.copy()

        cols = selected_clean_columns(
            scope,
            selected_column,
            multi_columns,
            before,
        )

        step_number = len(cleaning_history) + 1

        cleaning_history.append(
            {
                "Step": step_number,
                "Operation": operation,
                "Column": (
                    ", ".join(map(str, cols))
                    if cols
                    else "All"
                ),
                "Before": f"{len(before):,} rows / {count_missing(before):,} missing",
                "After": f"{len(after):,} rows / {count_missing(after):,} missing",
                "Status": "Applied",
            }
        )

        comparison = comparison_table(
            before,
            after,
        )

        summary = f"""
### Cleaning Applied Successfully

**Step {step_number}: {operation}**

Rows: {len(before):,} → {len(after):,}

Missing values: {count_missing(before):,} → {count_missing(after):,}

Duplicates: {count_duplicates(before):,} → {count_duplicates(after):,}

Outliers: {count_outliers(before):,} → {count_outliers(after):,}

The processed dataset now contains this change.
The original dataset remains untouched.
"""

        return (
            summary,
            after.head(100),
            comparison,
            pd.DataFrame(cleaning_history),
        )

    except Exception as e:
        return (
            f"Cleaning failed: {type(e).__name__}: {str(e)}",
            before.head(100),
            pd.DataFrame(),
            pd.DataFrame(cleaning_history),
        )


def reset_cleaning():
    global processed_df
    global cleaning_history

    if original_df is None:
        return (
            "No original dataset is loaded.",
            pd.DataFrame(),
            pd.DataFrame(),
        )

    processed_df = original_df.copy()
    cleaning_history = []

    return (
        "Processed dataset successfully reset to the original uploaded dataset.",
        processed_df.head(100),
        pd.DataFrame(),
    )


# ------------------------------------------------------------
# 12. EDA
# ------------------------------------------------------------
def eda_control_update(graph):
    x_visible = graph not in [
        "Correlation",
        "Covariance",
        "Missing",
        "Pair Plot",
    ]

    y_visible = graph in [
        "Scatter",
        "Line",
        "Area",
        "Bar",
        "3D Scatter",
    ]

    z_visible = graph == "3D Scatter"

    hue_visible = graph in [
        "Scatter",
        "Box",
        "Violin",
        "Count",
        "Bar",
    ]

    return (
        gr.update(visible=x_visible),
        gr.update(visible=y_visible),
        gr.update(visible=z_visible),
        gr.update(visible=hue_visible),
    )


def generate_eda_plot(
    graph,
    x,
    y,
    z,
    hue,
):
    df = processed_df if processed_df is not None else original_df

    if df is None or df.empty:
        return None, "Load a dataset first."

    try:
        fig = plt.figure(figsize=(10, 6))

        if graph == "Histogram":
            if x not in df.columns:
                return None, "Select a valid X column."

            sns.histplot(
                df[x].dropna(),
                kde=False,
            )
            plt.xlabel(x)
            plt.title(f"Histogram: {x}")

        elif graph == "KDE":
            if x not in df.columns:
                return None, "Select a valid X column."

            values = pd.to_numeric(
                df[x],
                errors="coerce",
            ).dropna()

            if len(values) < 2:
                return None, "KDE requires numerical data."

            sns.kdeplot(values, fill=True)
            plt.title(f"KDE: {x}")

        elif graph == "Box":
            if x not in df.columns:
                return None, "Select a valid X column."

            sns.boxplot(
                x=df[x],
                y=df[y] if y in df.columns else None,
                hue=df[hue] if hue in df.columns else None,
                legend=False if hue in df.columns else None,
            )
            plt.title(f"Box Plot: {x}")

        elif graph == "Violin":
            if x not in df.columns:
                return None, "Select a valid X column."

            sns.violinplot(
                x=df[x],
                y=df[y] if y in df.columns else None,
                hue=df[hue] if hue in df.columns else None,
                legend=False if hue in df.columns else None,
            )
            plt.title(f"Violin Plot: {x}")

        elif graph == "Count":
            if x not in df.columns:
                return None, "Select a valid X column."

            counts = (
                df[x]
                .astype(str)
                .value_counts()
                .head(20)
            )

            sns.barplot(
                x=counts.values,
                y=counts.index,
            )
            plt.title(f"Count Plot: {x}")

        elif graph == "Bar":
            if x not in df.columns or y not in df.columns:
                return None, "Bar chart requires X and Y columns."

            temp = df[[x, y]].dropna()

            if not pd.api.types.is_numeric_dtype(temp[y]):
                return None, "Bar chart Y must be numeric."

            grouped = (
                temp.groupby(x)[y]
                .mean()
                .sort_values(ascending=False)
                .head(20)
            )

            grouped.plot(kind="bar")
            plt.title(f"Bar Chart: {y} by {x}")
            plt.xticks(rotation=45)

        elif graph == "Pie":
            if x not in df.columns:
                return None, "Select a valid X column."

            counts = (
                df[x]
                .astype(str)
                .value_counts()
                .head(10)
            )

            counts.plot(
                kind="pie",
                autopct="%1.1f%%",
            )

            plt.ylabel("")
            plt.title(f"Pie Chart: {x}")

        elif graph == "Scatter":
            if x not in df.columns or y not in df.columns:
                return None, "Scatter plot requires X and Y."

            sns.scatterplot(
                data=df,
                x=x,
                y=y,
                hue=hue if hue in df.columns else None,
            )
            plt.title(f"Scatter: {x} vs {y}")

        elif graph == "Line":
            if x not in df.columns or y not in df.columns:
                return None, "Line chart requires X and Y."

            temp = df[[x, y]].dropna()

            if not pd.api.types.is_numeric_dtype(temp[y]):
                return None, "Y should be numeric."

            temp.plot(
                x=x,
                y=y,
                kind="line",
            )
            plt.title(f"Line Chart: {y} vs {x}")

        elif graph == "Area":
            if x not in df.columns or y not in df.columns:
                return None, "Area chart requires X and Y."

            temp = df[[x, y]].dropna()

            if not pd.api.types.is_numeric_dtype(temp[y]):
                return None, "Y should be numeric."

            temp.plot(
                x=x,
                y=y,
                kind="area",
                alpha=0.5,
            )
            plt.title(f"Area Chart: {y} vs {x}")

        elif graph == "Correlation":
            cols = numeric_columns(df)

            if len(cols) < 2:
                return None, "At least two numerical columns are required."

            corr = df[cols].corr(method="pearson")

            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
            )

            plt.title("Pearson Correlation Heatmap")

        elif graph == "Covariance":
            cols = numeric_columns(df)

            if len(cols) < 2:
                return None, "At least two numerical columns are required."

            covariance = df[cols].cov()

            sns.heatmap(
                covariance,
                annot=True,
                fmt=".2f",
                cmap="viridis",
            )

            plt.title("Covariance Heatmap")

        elif graph == "Pair Plot":
            cols = numeric_columns(df)[:6]

            if len(cols) < 2:
                return None, "At least two numerical columns are required."

            pair = sns.pairplot(
                df[cols].dropna().sample(
                    min(500, len(df[cols].dropna())),
                    random_state=42,
                )
            )

            fig = pair.fig
            fig.suptitle(
                "Pair Plot",
                y=1.02,
            )

        elif graph == "Q-Q":
            if x not in df.columns:
                return None, "Select a numerical X column."

            values = pd.to_numeric(
                df[x],
                errors="coerce",
            ).dropna()

            if len(values) < 3:
                return None, "Q-Q plot requires enough numerical values."

            stats.probplot(
                values,
                dist="norm",
                plot=plt,
            )

            plt.title(f"Q-Q Plot: {x}")

        elif graph == "ECDF":
            if x not in df.columns:
                return None, "Select a numerical X column."

            values = pd.to_numeric(
                df[x],
                errors="coerce",
            ).dropna()

            if values.empty:
                return None, "ECDF requires numerical data."

            values = np.sort(values)

            y_values = np.arange(
                1,
                len(values) + 1,
            ) / len(values)

            plt.step(
                values,
                y_values,
                where="post",
            )

            plt.xlabel(x)
            plt.ylabel("ECDF")
            plt.title(f"ECDF: {x}")

        elif graph == "Missing":
            missing = df.isna().sum()
            missing = missing[missing > 0]

            if missing.empty:
                return None, "No missing values found."

            missing.sort_values(
                ascending=False
            ).plot(kind="bar")

            plt.ylabel("Missing Count")
            plt.title("Missing Values by Column")
            plt.xticks(rotation=45)

        elif graph == "3D Scatter":
            if (
                x not in df.columns
                or y not in df.columns
                or z not in df.columns
            ):
                return None, "3D scatter requires X, Y and Z."

            x_values = pd.to_numeric(
                df[x],
                errors="coerce",
            )

            y_values = pd.to_numeric(
                df[y],
                errors="coerce",
            )

            z_values = pd.to_numeric(
                df[z],
                errors="coerce",
            )

            valid = (
                x_values.notna()
                & y_values.notna()
                & z_values.notna()
            )

            ax = fig.add_subplot(
                111,
                projection="3d",
            )

            if hue in df.columns:
                colors = pd.factorize(
                    df.loc[valid, hue]
                )[0]

                ax.scatter(
                    x_values[valid],
                    y_values[valid],
                    z_values[valid],
                    c=colors,
                )
            else:
                ax.scatter(
                    x_values[valid],
                    y_values[valid],
                    z_values[valid],
                )

            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_zlabel(z)
            ax.set_title("3D Scatter Plot")

        else:
            plt.close(fig)
            return None, "Select a graph type."

        plt.tight_layout()

        return fig, "Graph generated successfully."

    except Exception as e:
        plt.close("all")
        return None, f"Graph generation failed: {type(e).__name__}: {str(e)}"


# ------------------------------------------------------------
# 13. TARGET ANALYSIS
# ------------------------------------------------------------
def target_analysis(target, task):
    df = processed_df if processed_df is not None else original_df

    if df is None:
        return "Load a dataset first.", None

    if target not in df.columns:
        return "Select a valid target.", None

    try:
        fig = plt.figure(figsize=(9, 5))

        if task == "Classification":
            counts = df[target].astype(str).value_counts()

            imbalance = (
                counts.max() / counts.min()
                if len(counts) > 1 and counts.min() > 0
                else np.inf
            )

            percentages = (
                counts / counts.sum() * 100
            ).round(2)

            report = f"""
### Classification Target Analysis

**Target:** {target}

**Number of classes:** {len(counts)}

**Class imbalance ratio:** {imbalance:.2f}

| Class | Count | Percentage |
|---|---:|---:|
"""

            for cls, count in counts.items():
                report += (
                    f"| {cls} | {count:,} | "
                    f"{percentages[cls]:.2f}% |\n"
                )

            counts.plot(kind="bar")
            plt.title(f"Target Class Distribution: {target}")
            plt.ylabel("Count")
            plt.xticks(rotation=45)

        else:
            values = pd.to_numeric(
                df[target],
                errors="coerce",
            ).dropna()

            if values.empty:
                return (
                    "Regression target must contain numerical values.",
                    None,
                )

            skewness = values.skew()
            kurtosis = values.kurt()
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            outliers = int(
                (
                    (values < q1 - 1.5 * iqr)
                    | (values > q3 + 1.5 * iqr)
                ).sum()
            )

            report = f"""
### Regression Target Analysis

**Target:** {target}

**Mean:** {values.mean():.4f}

**Median:** {values.median():.4f}

**Minimum:** {values.min():.4f}

**Maximum:** {values.max():.4f}

**Skewness:** {skewness:.4f}

**Kurtosis:** {kurtosis:.4f}

**IQR outliers:** {outliers:,}
"""

            sns.histplot(
                values,
                kde=True,
            )

            plt.title(f"Regression Target Distribution: {target}")

        plt.tight_layout()

        return report, fig

    except Exception as e:
        return (
            f"Target analysis failed: {type(e).__name__}: {str(e)}",
            None,
        )


# ------------------------------------------------------------
# 14. FEATURE-TARGET ANALYSIS
# ------------------------------------------------------------
def feature_target_analysis(target, feature):
    df = processed_df if processed_df is not None else original_df

    if df is None:
        return "Load a dataset first.", None

    if target not in df.columns or feature not in df.columns:
        return "Select valid target and feature columns.", None

    try:
        fig = plt.figure(figsize=(9, 5))

        if (
            pd.api.types.is_numeric_dtype(df[target])
            and pd.api.types.is_numeric_dtype(df[feature])
        ):
            corr_p = df[[feature, target]].corr(
                method="pearson"
            ).iloc[0, 1]

            corr_s = df[[feature, target]].corr(
                method="spearman"
            ).iloc[0, 1]

            corr_k = df[[feature, target]].corr(
                method="kendall"
            ).iloc[0, 1]

            sns.scatterplot(
                data=df,
                x=feature,
                y=target,
            )

            plt.title(
                f"{feature} vs {target}"
            )

            report = f"""
### Feature-Target Analysis

Feature: **{feature}**

Target: **{target}**

Pearson correlation: **{corr_p:.4f}**

Spearman correlation: **{corr_s:.4f}**

Kendall correlation: **{corr_k:.4f}**
"""

        else:
            sns.boxplot(
                data=df,
                x=feature,
                y=target,
            )

            plt.xticks(rotation=45)

            report = f"""
### Feature-Target Analysis

Feature: **{feature}**

Target: **{target}**

Categorical/numerical relationship displayed using a box plot.
"""

        plt.tight_layout()

        return report, fig

    except Exception as e:
        return (
            f"Feature-target analysis failed: {type(e).__name__}: {str(e)}",
            None,
        )


# ------------------------------------------------------------
# 15. FEATURE ENGINEERING
# ------------------------------------------------------------
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


def feature_engineering(
    operation,
    columns,
    target,
    k,
    variance_threshold,
    correlation_threshold,
):
    global processed_df

    if processed_df is None:
        return "Load a dataset first.", pd.DataFrame()

    df = processed_df.copy()

    columns = [
        c for c in (columns or [])
        if c in df.columns
    ]

    try:
        if operation in [
            "Log Transform",
            "Square Transform",
            "Square Root Transform",
        ]:
            if not columns:
                columns = numeric_columns(df)

            for col in columns:
                values = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

                if operation == "Log Transform":
                    min_value = values.min()

                    if pd.notna(min_value) and min_value <= 0:
                        values = values - min_value + 1

                    df[f"{col}_log"] = np.log1p(values)

                elif operation == "Square Transform":
                    df[f"{col}_square"] = values ** 2

                else:
                    values = values.clip(lower=0)
                    df[f"{col}_sqrt"] = np.sqrt(values)

        elif operation == "Standardization":
            for col in columns:
                values = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

                std = values.std()

                if std and not pd.isna(std):
                    df[col] = (
                        values - values.mean()
                    ) / std

        elif operation == "MinMax Scaling":
            for col in columns:
                values = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

                min_v = values.min()
                max_v = values.max()

                if pd.notna(min_v) and max_v != min_v:
                    df[col] = (
                        values - min_v
                    ) / (max_v - min_v)

        elif operation == "Robust Scaling":
            for col in columns:
                values = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

                median = values.median()
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)

                iqr = q3 - q1

                if pd.notna(iqr) and iqr != 0:
                    df[col] = (
                        values - median
                    ) / iqr

        elif operation == "One-Hot Encoding":
            if not columns:
                columns = categorical_columns(df)

            if columns:
                df = pd.get_dummies(
                    df,
                    columns=columns,
                    dtype=int,
                )

        elif operation == "Ordinal Encoding":
            if not columns:
                return "Select at least one categorical column.", df.head(100)

            for col in columns:
                df[col] = (
                    df[col]
                    .astype("category")
                    .cat.codes
                    .replace(-1, np.nan)
                )

        elif operation == "Frequency Encoding":
            if not columns:
                return "Select at least one column.", df.head(100)

            for col in columns:
                freq = df[col].value_counts(
                    normalize=True
                )

                df[f"{col}_frequency"] = df[col].map(freq)

        elif operation == "Variance Filtering":
            numeric = numeric_columns(df)

            try:
                threshold = float(variance_threshold)
            except Exception:
                threshold = 0.0

            drop_cols = []

            for col in numeric:
                variance = df[col].var()

                if pd.notna(variance) and variance <= threshold:
                    drop_cols.append(col)

            if drop_cols:
                df = df.drop(columns=drop_cols)

        elif operation == "Correlation Filtering":
            numeric = numeric_columns(df)

            try:
                threshold = float(correlation_threshold)
            except Exception:
                threshold = 0.95

            if len(numeric) > 1:
                corr = df[numeric].corr().abs()

                upper = corr.where(
                    np.triu(
                        np.ones(corr.shape),
                        k=1,
                    ).astype(bool)
                )

                drop_cols = [
                    column
                    for column in upper.columns
                    if any(
                        upper[column] > threshold
                    )
                ]

                if drop_cols:
                    df = df.drop(
                        columns=drop_cols
                    )

        elif operation == "SelectKBest":
            if target not in df.columns:
                return "Select a valid target.", df.head(100)

            numeric = [
                c for c in numeric_columns(df)
                if c != target
            ]

            if not numeric:
                return "No numerical features available.", df.head(100)

            X = df[numeric].copy()
            y = df[target].copy()

            valid = y.notna()

            X = X.loc[valid]
            y = y.loc[valid]

            X = X.fillna(X.median())

            try:
                k_value = int(k)
            except Exception:
                k_value = 10

            k_value = min(
                k_value,
                len(numeric),
            )

            if (
                pd.api.types.is_numeric_dtype(y)
                and y.nunique() > 10
            ):
                selector = SelectKBest(
                    score_func=f_regression,
                    k=k_value,
                )
            else:
                y_codes = (
                    y.astype("category")
                    .cat.codes
                )

                selector = SelectKBest(
                    score_func=f_classif,
                    k=k_value,
                )

                y = y_codes

            selector.fit(X, y)

            selected = list(
                np.array(numeric)[
                    selector.get_support()
                ]
            )

            keep = selected + [target]

            df = df[
                [c for c in keep if c in df.columns]
            ].copy()

        elif operation == "PCA":
            numeric = numeric_columns(df)

            if len(numeric) < 2:
                return (
                    "PCA requires at least two numerical columns.",
                    df.head(100),
                )

            X = df[numeric].copy()

            X = X.fillna(
                X.median()
            )

            try:
                k_value = int(k)
            except Exception:
                k_value = 2

            k_value = max(
                1,
                min(k_value, len(numeric)),
            )

            scaler = StandardScaler()

            X_scaled = scaler.fit_transform(X)

            pca = PCA(
                n_components=k_value,
                random_state=42,
            )

            transformed = pca.fit_transform(
                X_scaled
            )

            for i in range(k_value):
                df[f"PCA_{i+1}"] = transformed[:, i]

        processed_df = df.reset_index(drop=True)

        return (
            f"Feature engineering operation '{operation}' applied successfully.",
            processed_df.head(100),
        )

    except Exception as e:
        return (
            f"Feature engineering failed: {type(e).__name__}: {str(e)}",
            processed_df.head(100),
        )


# ------------------------------------------------------------
# 16. MACHINE LEARNING MODELS
# ------------------------------------------------------------
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


def model_choices(task):
    if task == "Classification":
        return gr.update(
            choices=CLASSIFICATION_MODELS,
            value=CLASSIFICATION_MODELS[0],
        )

    return gr.update(
        choices=REGRESSION_MODELS,
        value=REGRESSION_MODELS[0],
    )


def make_model(task, name):
    if task == "Classification":

        if name == "Logistic Regression":
            return LogisticRegression(
                max_iter=1000,
            )

        if name == "KNN":
            return KNeighborsClassifier(
                n_neighbors=5,
            )

        if name == "Naive Bayes":
            return GaussianNB()

        if name == "Decision Tree":
            return DecisionTreeClassifier(
                max_depth=8,
                random_state=42,
            )

        if name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
            )

        if name == "SVM":
            return SVC(
                probability=True,
            )

        if name == "Gradient Boosting":
            return GradientBoostingClassifier(
                random_state=42,
            )

        if name == "XGBoost" and XGBOOST_AVAILABLE:
            return XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=2,
            )

    else:

        if name == "Linear Regression":
            return LinearRegression()

        if name == "Polynomial Regression":
            return Pipeline(
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
            )

        if name == "Ridge":
            return Ridge(alpha=1.0)

        if name == "Lasso":
            return Lasso(alpha=0.01)

        if name == "Decision Tree":
            return DecisionTreeRegressor(
                max_depth=8,
                random_state=42,
            )

        if name == "Random Forest":
            return RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
            )

        if name == "SVR":
            return SVR()

        if name == "Gradient Boosting":
            return GradientBoostingRegressor(
                random_state=42,
            )

        if name == "XGBoost" and XGBOOST_AVAILABLE:
            return XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=2,
            )

    raise ValueError(
        f"Unsupported model: {name}"
    )


def make_preprocessor(X, scaling):
    numeric = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if scaling == "StandardScaler":
        scaler = StandardScaler()
    elif scaling == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaling == "RobustScaler":
        scaler = RobustScaler()
    else:
        scaler = "passthrough"

    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                scaler,
            ),
        ]
    )

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                encoder,
            ),
        ]
    )

    transformers = []

    if numeric:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric,
            )
        )

    if categorical:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ------------------------------------------------------------
# 17. TRAIN MODEL
# ------------------------------------------------------------
def train_model(
    task,
    model_name,
    target,
    test_size,
    scaling,
    folds,
):
    global trained_pipeline
    global trained_model
    global trained_preprocessor
    global target_column
    global task_type
    global X_train
    global X_test
    global y_train
    global y_test
    global last_metrics
    global last_evaluation_fig
    global last_roc_fig
    global last_pr_fig
    global trained_input_columns
    global model_results

    df = processed_df if processed_df is not None else original_df

    if df is None:
        return (
            "Load a dataset first.",
            pd.DataFrame(),
            None,
            None,
            None,
        )

    if target not in df.columns:
        return (
            "Select a valid target column.",
            pd.DataFrame(),
            None,
            None,
            None,
        )

    if len(df) < 10:
        return (
            "At least 10 rows are recommended for model training.",
            pd.DataFrame(),
            None,
            None,
            None,
        )

    try:
        work = df.copy()

        y = work[target]
        X = work.drop(columns=[target])

        valid_target = y.notna()

        X = X.loc[valid_target].copy()
        y = y.loc[valid_target].copy()

        if task == "Classification":
            if y.nunique() < 2:
                return (
                    "Classification requires at least two target classes.",
                    pd.DataFrame(),
                    None,
                    None,
                    None,
                )

            if y.dtype == object or not pd.api.types.is_numeric_dtype(y):
                y = y.astype(str)

        else:
            y = pd.to_numeric(
                y,
                errors="coerce",
            )

            valid = y.notna()

            X = X.loc[valid]
            y = y.loc[valid]

        if len(X) < 10:
            return (
                "Not enough valid rows after target cleaning.",
                pd.DataFrame(),
                None,
                None,
                None,
            )

        try:
            test_ratio = float(test_size)
        except Exception:
            test_ratio = 0.20

        test_ratio = min(
            max(test_ratio, 0.10),
            0.40,
        )

        stratify = (
            y
            if task == "Classification"
            and y.value_counts().min() >= 2
            else None
        )

        X_train_local, X_test_local, y_train_local, y_test_local = (
            train_test_split(
                X,
                y,
                test_size=test_ratio,
                random_state=42,
                stratify=stratify,
            )
        )

        preprocessor = make_preprocessor(
            X_train_local,
            scaling,
        )

        model = make_model(
            task,
            model_name,
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

        y_pred = pipeline.predict(
            X_test_local
        )

        metrics_rows = []

        if task == "Classification":

            accuracy = accuracy_score(
                y_test_local,
                y_pred,
            )

            precision = precision_score(
                y_test_local,
                y_pred,
                average="weighted",
                zero_division=0,
            )

            recall = recall_score(
                y_test_local,
                y_pred,
                average="weighted",
                zero_division=0,
            )

            f1 = f1_score(
                y_test_local,
                y_pred,
                average="weighted",
                zero_division=0,
            )

            try:
                if hasattr(
                    pipeline,
                    "predict_proba",
                ):
                    probabilities = pipeline.predict_proba(
                        X_test_local
                    )

                    if len(
                        np.unique(y_test_local)
                    ) == 2:
                        auc = roc_auc_score(
                            y_test_local,
                            probabilities[:, 1],
                        )
                    else:
                        auc = roc_auc_score(
                            y_test_local,
                            probabilities,
                            multi_class="ovr",
                        )
                else:
                    auc = np.nan
            except Exception:
                auc = np.nan

            metrics_rows = [
                {
                    "Metric": "Accuracy",
                    "Value": accuracy,
                },
                {
                    "Metric": "Precision",
                    "Value": precision,
                },
                {
                    "Metric": "Recall",
                    "Value": recall,
                },
                {
                    "Metric": "F1",
                    "Value": f1,
                },
                {
                    "Metric": "ROC-AUC",
                    "Value": auc,
                },
            ]

            cm = confusion_matrix(
                y_test_local,
                y_pred,
            )

            fig_cm = plt.figure(
                figsize=(7, 6)
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
            )

            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.tight_layout()

            roc_fig = None
            pr_fig = None

            if len(np.unique(y_test_local)) == 2:
                try:
                    probabilities = pipeline.predict_proba(
                        X_test_local
                    )[:, 1]

                    positive_class = (
                        pipeline.classes_[1]
                        if hasattr(
                            pipeline,
                            "classes_",
                        )
                        else sorted(
                            y_test_local.unique()
                        )[1]
                    )

                    y_binary = (
                        y_test_local
                        == positive_class
                    ).astype(int)

                    fpr, tpr, _ = roc_curve(
                        y_binary,
                        probabilities,
                    )

                    roc_fig = plt.figure(
                        figsize=(7, 5)
                    )

                    plt.plot(
                        fpr,
                        tpr,
                        label=f"AUC = {auc:.3f}",
                    )

                    plt.plot(
                        [0, 1],
                        [0, 1],
                        linestyle="--",
                    )

                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("ROC Curve")
                    plt.legend()
                    plt.tight_layout()

                    precision_curve, recall_curve, _ = (
                        precision_recall_curve(
                            y_binary,
                            probabilities,
                        )
                    )

                    pr_fig = plt.figure(
                        figsize=(7, 5)
                    )

                    ap = average_precision_score(
                        y_binary,
                        probabilities,
                    )

                    plt.plot(
                        recall_curve,
                        precision_curve,
                        label=f"AP = {ap:.3f}",
                    )

                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    plt.title(
                        "Precision-Recall Curve"
                    )

                    plt.legend()
                    plt.tight_layout()

                except Exception:
                    pass

        else:
            mae = mean_absolute_error(
                y_test_local,
                y_pred,
            )

            mse = mean_squared_error(
                y_test_local,
                y_pred,
            )

            rmse = np.sqrt(mse)

            r2 = r2_score(
                y_test_local,
                y_pred,
            )

            n = len(y_test_local)

            try:
                p = X_test_local.shape[1]

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

            metrics_rows = [
                {
                    "Metric": "MAE",
                    "Value": mae,
                },
                {
                    "Metric": "MSE",
                    "Value": mse,
                },
                {
                    "Metric": "RMSE",
                    "Value": rmse,
                },
                {
                    "Metric": "R2",
                    "Value": r2,
                },
                {
                    "Metric": "Adjusted R2",
                    "Value": adjusted_r2,
                },
            ]

            fig_cm = plt.figure(
                figsize=(8, 6)
            )

            plt.scatter(
                y_test_local,
                y_pred,
                alpha=0.7,
            )

            min_value = min(
                y_test_local.min(),
                y_pred.min(),
            )

            max_value = max(
                y_test_local.max(),
                y_pred.max(),
            )

            plt.plot(
                [min_value, max_value],
                [min_value, max_value],
                linestyle="--",
            )

            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Actual vs Predicted")
            plt.tight_layout()

            roc_fig = plt.figure(
                figsize=(8, 5)
            )

            residuals = (
                y_test_local.values
                - y_pred
            )

            plt.scatter(
                y_pred,
                residuals,
                alpha=0.7,
            )

            plt.axhline(
                0,
                linestyle="--",
            )

            plt.xlabel("Predicted")
            plt.ylabel("Residual")
            plt.title("Residual Plot")
            plt.tight_layout()

            pr_fig = None

        metrics_df = pd.DataFrame(
            metrics_rows
        )

        # Cross-validation
        try:
            cv_folds = int(folds)
        except Exception:
            cv_folds = 5

        cv_folds = min(
            max(cv_folds, 3),
            10,
        )

        if task == "Classification":
            counts = y_train_local.value_counts()

            if (
                len(counts) > 1
                and counts.min() >= cv_folds
            ):
                cv = StratifiedKFold(
                    n_splits=cv_folds,
                    shuffle=True,
                    random_state=42,
                )
            else:
                cv = None
        else:
            cv = KFold(
                n_splits=min(
                    cv_folds,
                    len(X_train_local),
                ),
                shuffle=True,
                random_state=42,
            )

        cv_score = np.nan

        if cv is not None:
            try:
                scoring = (
                    "accuracy"
                    if task == "Classification"
                    else "r2"
                )

                scores = cross_val_score(
                    pipeline,
                    X_train_local,
                    y_train_local,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=1,
                )

                cv_score = scores.mean()

                metrics_df.loc[
                    len(metrics_df)
                ] = {
                    "Metric": "Cross Validation Score",
                    "Value": cv_score,
                }

            except Exception:
                pass

        # Store global state
        trained_pipeline = pipeline
        trained_model = model
        trained_preprocessor = preprocessor

        target_column = target
        task_type = task

        X_train = X_train_local
        X_test = X_test_local
        y_train = y_train_local
        y_test = y_test_local

        trained_input_columns = X.columns.tolist()

        last_metrics = metrics_df.copy()

        last_evaluation_fig = fig_cm
        last_roc_fig = roc_fig
        last_pr_fig = pr_fig

        model_record = {
            "Model": model_name,
        }

        for _, row in metrics_df.iterrows():
            model_record[row["Metric"]] = row["Value"]

        model_results = pd.concat(
            [
                model_results,
                pd.DataFrame([model_record]),
            ],
            ignore_index=True,
        )

        status = f"""
## Training Complete

Task: **{task}**

Model: **{model_name}**

Target: **{target}**

Training rows: **{len(X_train_local):,}**

Testing rows: **{len(X_test_local):,}**

Cross-validation folds: **{cv_folds}**

The trained pipeline is now available for Evaluation, Comparison, Prediction and Explainability.
"""

        return (
            status,
            metrics_df,
            fig_cm,
            roc_fig,
            pr_fig,
        )

    except Exception as e:
        return (
            f"Model training failed: {type(e).__name__}: {str(e)}",
            pd.DataFrame(),
            None,
            None,
            None,
        )


# ------------------------------------------------------------
# 18. MODEL COMPARISON
# ------------------------------------------------------------
def compare_models(
    task,
    target,
    test_size,
    scaling,
    folds,
):
    df = processed_df if processed_df is not None else original_df

    if df is None:
        return "Load a dataset first.", pd.DataFrame()

    if target not in df.columns:
        return "Select a valid target.", pd.DataFrame()

    models = (
        CLASSIFICATION_MODELS
        if task == "Classification"
        else REGRESSION_MODELS
    )

    results = []

    try:
        work = df.copy()

        y = work[target]
        X = work.drop(columns=[target])

        valid = y.notna()

        X = X.loc[valid]
        y = y.loc[valid]

        if task == "Regression":
            y = pd.to_numeric(
                y,
                errors="coerce",
            )

            valid = y.notna()

            X = X.loc[valid]
            y = y.loc[valid]

        else:
            y = y.astype(str)

        stratify = (
            y
            if task == "Classification"
            and y.value_counts().min() >= 2
            else None
        )

        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=float(test_size),
            random_state=42,
            stratify=stratify,
        )

        for model_name in models:
            try:
                preprocessor = make_preprocessor(
                    X_tr,
                    scaling,
                )

                model = make_model(
                    task,
                    model_name,
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
                    X_tr,
                    y_tr,
                )

                pred = pipeline.predict(
                    X_te
                )

                if task == "Classification":
                    accuracy = accuracy_score(
                        y_te,
                        pred,
                    )

                    precision = precision_score(
                        y_te,
                        pred,
                        average="weighted",
                        zero_division=0,
                    )

                    recall = recall_score(
                        y_te,
                        pred,
                        average="weighted",
                        zero_division=0,
                    )

                    f1 = f1_score(
                        y_te,
                        pred,
                        average="weighted",
                        zero_division=0,
                    )

                    results.append(
                        {
                            "Model": model_name,
                            "Accuracy": accuracy,
                            "Precision": precision,
                            "Recall": recall,
                            "F1": f1,
                        }
                    )

                else:
                    mae = mean_absolute_error(
                        y_te,
                        pred,
                    )

                    mse = mean_squared_error(
                        y_te,
                        pred,
                    )

                    rmse = np.sqrt(mse)

                    r2 = r2_score(
                        y_te,
                        pred,
                    )

                    results.append(
                        {
                            "Model": model_name,
                            "MAE": mae,
                            "RMSE": rmse,
                            "R2": r2,
                        }
                    )

            except Exception as model_error:
                results.append(
                    {
                        "Model": model_name,
                        "Error": str(model_error),
                    }
                )

        result_df = pd.DataFrame(results)

        if result_df.empty:
            return "No models could be trained.", result_df

        if task == "Classification":
            valid_results = result_df[
                "Accuracy"
            ].notna()

            if valid_results.any():
                best = result_df.loc[
                    result_df["Accuracy"].idxmax()
                ]

                status = (
                    f"Model comparison completed. "
                    f"Best accuracy: {best['Model']} "
                    f"({best['Accuracy']:.4f})."
                )
            else:
                status = "Comparison completed with model errors."

        else:
            valid_results = result_df[
                "R2"
            ].notna()

            if valid_results.any():
                best = result_df.loc[
                    result_df["R2"].idxmax()
                ]

                status = (
                    f"Model comparison completed. "
                    f"Best R²: {best['Model']} "
                    f"({best['R2']:.4f})."
                )
            else:
                status = "Comparison completed with model errors."

        return status, result_df

    except Exception as e:
        return (
            f"Model comparison failed: {type(e).__name__}: {str(e)}",
            pd.DataFrame(),
        )


# ------------------------------------------------------------
# 19. HYPERPARAMETER TUNING
# ------------------------------------------------------------
def tune_model(
    task,
    model_name,
    target,
    search_method,
    folds,
):
    global last_tuned_model
    global last_tuning_results

    df = processed_df if processed_df is not None else original_df

    if df is None:
        return (
            "Load a dataset first.",
            pd.DataFrame(),
        )

    if target not in df.columns:
        return (
            "Select a valid target.",
            pd.DataFrame(),
        )

    try:
        y = df[target]
        X = df.drop(columns=[target])

        valid = y.notna()

        X = X.loc[valid]
        y = y.loc[valid]

        if task == "Regression":
            y = pd.to_numeric(
                y,
                errors="coerce",
            )

            valid = y.notna()

            X = X.loc[valid]
            y = y.loc[valid]

        else:
            y = y.astype(str)

        preprocessor = make_preprocessor(
            X,
            "StandardScaler",
        )

        model = make_model(
            task,
            model_name,
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

        if task == "Classification":

            if model_name == "Logistic Regression":
                param_grid = {
                    "model__C": [
                        0.1,
                        1.0,
                        10.0,
                    ]
                }

            elif model_name == "Decision Tree":
                param_grid = {
                    "model__max_depth": [
                        3,
                        5,
                        8,
                        None,
                    ],
                    "model__min_samples_split": [
                        2,
                        5,
                    ],
                }

            elif model_name == "Random Forest":
                param_grid = {
                    "model__n_estimators": [
                        80,
                        120,
                    ],
                    "model__max_depth": [
                        5,
                        10,
                        None,
                    ],
                }

            elif model_name == "KNN":
                param_grid = {
                    "model__n_neighbors": [
                        3,
                        5,
                        7,
                    ]
                }

            else:
                param_grid = {
                    "model__learning_rate": [
                        0.03,
                        0.05,
                        0.1,
                    ],
                    "model__n_estimators": [
                        50,
                        100,
                    ],
                }

            scoring = "accuracy"

        else:

            if model_name == "Ridge":
                param_grid = {
                    "model__alpha": [
                        0.01,
                        0.1,
                        1.0,
                        10.0,
                    ]
                }

            elif model_name == "Lasso":
                param_grid = {
                    "model__alpha": [
                        0.001,
                        0.01,
                        0.1,
                    ]
                }

            elif model_name == "Random Forest":
                param_grid = {
                    "model__n_estimators": [
                        80,
                        120,
                    ],
                    "model__max_depth": [
                        5,
                        10,
                        None,
                    ],
                }

            elif model_name == "Decision Tree":
                param_grid = {
                    "model__max_depth": [
                        3,
                        5,
                        8,
                        None,
                    ]
                }

            else:
                param_grid = {
                    "model__learning_rate": [
                        0.03,
                        0.05,
                        0.1,
                    ],
                    "model__n_estimators": [
                        50,
                        100,
                    ],
                }

            scoring = "r2"

        try:
            cv_folds = int(folds)
        except Exception:
            cv_folds = 5

        cv_folds = min(
            max(cv_folds, 3),
            10,
        )

        if task == "Classification":
            min_class = y.value_counts().min()

            cv_folds = min(
                cv_folds,
                int(min_class),
            )

            if cv_folds < 2:
                return (
                    "Not enough samples per class for tuning.",
                    pd.DataFrame(),
                )

            cv = StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=42,
            )

        else:
            cv = KFold(
                n_splits=min(
                    cv_folds,
                    len(X),
                ),
                shuffle=True,
                random_state=42,
            )

        if search_method == "RandomizedSearchCV":
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=min(
                    6,
                    sum(
                        len(v)
                        for v in param_grid.values()
                    ),
                ),
                scoring=scoring,
                cv=cv,
                random_state=42,
                n_jobs=1,
            )
        else:
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=1,
            )

        search.fit(
            X,
            y,
        )

        last_tuned_model = search.best_estimator_
        last_tuning_results = pd.DataFrame(
            [
                {
                    "Best Score": search.best_score_,
                    "Best Parameters": str(
                        search.best_params_
                    ),
                    "CV Folds": cv_folds,
                    "Search Method": search_method,
                }
            ]
        )

        status = f"""
## Hyperparameter Tuning Complete

Model: **{model_name}**

Task: **{task}**

Search method: **{search_method}**

Best CV score: **{search.best_score_:.4f}**

Best parameters:

`{search.best_params_}`
"""

        return (
            status,
            last_tuning_results,
        )

    except Exception as e:
        return (
            f"Tuning failed: {type(e).__name__}: {str(e)}",
            pd.DataFrame(),
        )


# ------------------------------------------------------------
# 20. PREDICTION
# ------------------------------------------------------------
def build_prediction_table():
    if trained_pipeline is None:
        return pd.DataFrame(
            {
                "Feature": [],
                "Value": [],
            }
        )

    rows = []

    for feature in trained_input_columns:
        default_value = ""

        if (
            X_train is not None
            and feature in X_train.columns
        ):
            series = X_train[feature]

            if pd.api.types.is_numeric_dtype(series):
                value = series.median()

                if pd.notna(value):
                    default_value = str(value)

            else:
                mode = series.dropna().mode()

                if len(mode):
                    default_value = str(
                        mode.iloc[0]
                    )

        rows.append(
            {
                "Feature": feature,
                "Value": default_value,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Feature",
            "Value",
        ],
    )


def prepare_prediction():
    table = build_prediction_table()

    return (
        table,
        "Prediction input prepared. Edit the Value column before predicting."
        if trained_pipeline is not None
        else "Train a model first.",
    )


def predict_from_table(input_table):
    global predictions
    global prediction_probabilities

    if trained_pipeline is None:
        return (
            "Train a model first.",
            pd.DataFrame(),
        )

    try:
        if input_table is None:
            return (
                "Prediction input is incomplete.",
                pd.DataFrame(),
            )

        if not isinstance(
            input_table,
            pd.DataFrame,
        ):
            input_table = pd.DataFrame(
                input_table,
                columns=[
                    "Feature",
                    "Value",
                ],
            )

        if list(input_table.columns) != [
            "Feature",
            "Value",
        ]:
            input_table.columns = [
                "Feature",
                "Value",
            ]

        row = {}

        for feature in trained_input_columns:
            matching = input_table[
                input_table["Feature"].astype(str)
                == str(feature)
            ]

            if matching.empty:
                return (
                    f"Prediction input is missing feature: {feature}",
                    pd.DataFrame(),
                )

            value = matching.iloc[0]["Value"]

            if (
                X_train is not None
                and feature in X_train.columns
                and pd.api.types.is_numeric_dtype(
                    X_train[feature]
                )
            ):
                try:
                    value = float(value)
                except Exception:
                    return (
                        f"Feature '{feature}' requires a numeric value.",
                        pd.DataFrame(),
                    )

            row[feature] = value

        input_df = pd.DataFrame(
            [row],
            columns=trained_input_columns,
        )

        pred = trained_pipeline.predict(
            input_df
        )

        predictions = pred

        result_rows = [
            {
                "Prediction": pred[0],
            }
        ]

        prediction_probabilities = None

        if (
            task_type == "Classification"
            and hasattr(
                trained_pipeline,
                "predict_proba",
            )
        ):
            probabilities = (
                trained_pipeline.predict_proba(
                    input_df
                )[0]
            )

            prediction_probabilities = probabilities

            classes = trained_pipeline.classes_

            for cls, probability in zip(
                classes,
                probabilities,
            ):
                result_rows.append(
                    {
                        "Prediction": (
                            f"Probability - {cls}: "
                            f"{probability:.4f}"
                        )
                    }
                )

        return (
            "Prediction completed successfully.",
            pd.DataFrame(result_rows),
        )

    except Exception as e:
        return (
            f"Prediction failed: {type(e).__name__}: {str(e)}",
            pd.DataFrame(),
        )


# ------------------------------------------------------------
# 21. EXPLAINABILITY
# ------------------------------------------------------------
def get_transformed_feature_names():
    if trained_pipeline is None:
        return []

    try:
        preprocessor = trained_pipeline.named_steps[
            "preprocessor"
        ]

        names = preprocessor.get_feature_names_out()

        return [
            str(x)
            .replace("numeric__", "")
            .replace("categorical__", "")
            for x in names
        ]

    except Exception:
        return []


def explain_model(method):
    global last_feature_importance

    if trained_pipeline is None:
        return (
            "Train a model first.",
            pd.DataFrame(),
            None,
        )

    try:
        model = trained_pipeline.named_steps[
            "model"
        ]

        feature_names = get_transformed_feature_names()

        if method in [
            "Feature Importance",
            "Coefficients",
        ]:
            values = None

            if hasattr(
                model,
                "feature_importances_",
            ):
                values = model.feature_importances_

            elif hasattr(
                model,
                "coef_",
            ):
                coef = model.coef_

                if coef.ndim == 1:
                    values = np.abs(coef)
                else:
                    values = np.mean(
                        np.abs(coef),
                        axis=0,
                    )

            else:
                return (
                    "This model does not expose native feature importance or coefficients.",
                    pd.DataFrame(),
                    None,
                )

            values = np.asarray(values)

            n = min(
                len(feature_names),
                len(values),
            )

            result = pd.DataFrame(
                {
                    "Feature": feature_names[:n],
                    "Importance": values[:n],
                }
            ).sort_values(
                "Importance",
                ascending=False,
            )

        elif method == "Permutation Importance":
            if X_test is None or y_test is None:
                return (
                    "No test data available.",
                    pd.DataFrame(),
                    None,
                )

            scoring = (
                "accuracy"
                if task_type == "Classification"
                else "r2"
            )

            result_perm = permutation_importance(
                trained_pipeline,
                X_test,
                y_test,
                n_repeats=5,
                random_state=42,
                scoring=scoring,
            )

            result = pd.DataFrame(
                {
                    "Feature": X_test.columns,
                    "Importance": result_perm.importances_mean,
                }
            ).sort_values(
                "Importance",
                ascending=False,
            )

        elif method == "SHAP":
            if not SHAP_AVAILABLE:
                return (
                    "SHAP is optional and is not installed. The dashboard continues to work without it.",
                    pd.DataFrame(),
                    None,
                )

            try:
                sample = X_test.head(
                    min(100, len(X_test))
                )

                explainer = shap.Explainer(
                    trained_pipeline.predict,
                    sample,
                )

                shap_values = explainer(
                    sample
                )

                values = np.abs(
                    shap_values.values
                )

                if values.ndim == 3:
                    values = values.mean(
                        axis=(0, 2)
                    )
                elif values.ndim == 2:
                    values = values.mean(
                        axis=0
                    )

                result = pd.DataFrame(
                    {
                        "Feature": sample.columns,
                        "Importance": values,
                    }
                ).sort_values(
                    "Importance",
                    ascending=False,
                )

            except Exception as shap_error:
                return (
                    f"SHAP could not be calculated: {shap_error}",
                    pd.DataFrame(),
                    None,
                )

        else:
            return (
                "Select an explainability method.",
                pd.DataFrame(),
                None,
            )

        result = result.head(30).reset_index(
            drop=True
        )

        last_feature_importance = result.copy()

        fig = plt.figure(
            figsize=(9, 7)
        )

        display_result = result.head(20).sort_values(
            "Importance",
            ascending=True,
        )

        plt.barh(
            display_result["Feature"].astype(str),
            display_result["Importance"],
        )

        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(method)
        plt.tight_layout()

        return (
            f"{method} generated successfully.",
            result,
            fig,
        )

    except Exception as e:
        return (
            f"Explainability failed: {type(e).__name__}: {str(e)}",
            pd.DataFrame(),
            None,
        )


# ------------------------------------------------------------
# 22. FILE LOADING
# ------------------------------------------------------------
def load_dataset(file_path):
    global original_df
    global processed_df
    global viewed_df
    global cleaning_history

    if file_path is None:
        return (
            "Please upload a CSV, XLSX, or XLS file.",
            "No dataset loaded.",
            "No dataset loaded.",
            pd.DataFrame(),
            pd.DataFrame(),
            "No dataset loaded.",
            pd.DataFrame(),
            *[
                gr.update(choices=[], value=None)
                for _ in range(10)
            ],
        )

    try:
        path = (
            file_path
            if isinstance(file_path, str)
            else getattr(file_path, "name", None)
        )

        if not path or not os.path.exists(path):
            return (
                "Uploaded file could not be found.",
                "No dataset loaded.",
                "No dataset loaded.",
                pd.DataFrame(),
                pd.DataFrame(),
                "No dataset loaded.",
                pd.DataFrame(),
                *[
                    gr.update(choices=[], value=None)
                    for _ in range(10)
                ],
            )

        extension = os.path.splitext(
            path
        )[1].lower()

        if extension == ".csv":
            df = pd.read_csv(path)

        elif extension in [".xlsx", ".xls"]:
            df = pd.read_excel(path)

        else:
            return (
                "Unsupported file type. Use CSV, XLSX, or XLS.",
                "No dataset loaded.",
                "No dataset loaded.",
                pd.DataFrame(),
                pd.DataFrame(),
                "No dataset loaded.",
                pd.DataFrame(),
                *[
                    gr.update(choices=[], value=None)
                    for _ in range(10)
                ],
            )

        if df is None:
            raise ValueError(
                "Pandas returned no dataframe."
            )

        df.columns = [
            clean_column_name(c)
            for c in df.columns
        ]

        original_df = df.copy()
        processed_df = df.copy()
        viewed_df = df.head(50).copy()
        cleaning_history = []

        overview = overview_text(
            original_df
        )

        viewer_info, viewer_table = create_view(
            "Original",
            "Head",
            50,
            0,
            50,
            "",
        )

        stats = statistics_analysis(
            numeric_columns(df),
            "90",
        )

        quality_md, quality_table = quality_analysis(
            df
        )

        cols = all_columns(df)
        nums = numeric_columns(df)
        cats = categorical_columns(df)

        update_objects = [
            gr.update(
                choices=cols,
                value=cols[0] if cols else None,
            ),
            gr.update(
                choices=cols,
                value=cols[0] if cols else None,
            ),
            gr.update(
                choices=cols,
                value=cols[: min(3, len(cols))]
                if cols
                else [],
            ),
            gr.update(
                choices=nums,
                value=nums[0] if nums else None,
            ),
            gr.update(
                choices=cols,
                value=cols[1]
                if len(cols) > 1
                else (
                    cols[0]
                    if cols
                    else None
                ),
            ),
            gr.update(
                choices=nums,
                value=nums[0] if nums else None,
            ),
            gr.update(
                choices=cols,
                value=cols[0] if cols else None,
            ),
            gr.update(
                choices=cols,
                value=cols[1]
                if len(cols) > 1
                else (
                    cols[0]
                    if cols
                    else None
                ),
            ),
            gr.update(
                choices=cols,
                value=cols[: min(3, len(cols))]
                if cols
                else [],
            ),
            gr.update(
                choices=cols,
                value=cols[0] if cols else None,
            ),
        ]

        status = f"""
Dataset loaded successfully.

File: **{os.path.basename(path)}**

Rows: **{len(df):,}**

Columns: **{len(df.columns):,}**

The original dataset is protected. Cleaning operates on the processed copy.
"""

        return (
            status,
            overview,
            viewer_info,
            viewer_table,
            stats,
            quality_md,
            quality_table,
            *update_objects,
        )

    except Exception as e:
        return (
            f"Dataset loading failed: {type(e).__name__}: {str(e)}",
            "Dataset could not be loaded.",
            "Dataset could not be loaded.",
            pd.DataFrame(),
            pd.DataFrame(),
            "Quality analysis unavailable.",
            pd.DataFrame(),
            *[
                gr.update(choices=[], value=None)
                for _ in range(10)
            ],
        )


# ------------------------------------------------------------
# 23. REFRESH DATA
# ------------------------------------------------------------
def refresh_dashboard():
    if processed_df is None:
        return (
            "Load a dataset first.",
            "No dataset loaded.",
            "No dataset loaded.",
            pd.DataFrame(),
            pd.DataFrame(),
            "No dataset loaded.",
            pd.DataFrame(),
        )

    overview = overview_text(
        processed_df
    )

    info, view = create_view(
        "Processed",
        "Head",
        50,
        0,
        50,
        "",
    )

    stats_table = statistics_analysis(
        numeric_columns(processed_df),
        "90",
    )

    quality_md, quality_table = quality_analysis(
        processed_df
    )

    return (
        "Dashboard refreshed using the current processed dataset.",
        overview,
        info,
        view,
        stats_table,
        quality_md,
        quality_table,
    )


# ------------------------------------------------------------
# 24. EXPORT SYSTEM
# ------------------------------------------------------------
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
    "Complete Analysis Summary",
]

EXPORT_FORMATS = [
    "CSV",
    "Excel",
]


def dataframe_for_export(export_type):
    if export_type == "Original Dataset":
        return original_df

    if export_type == "Processed Dataset":
        return processed_df

    if export_type == "Currently Viewed Dataset":
        return viewed_df

    if export_type == "Statistics":
        return statistics_analysis(
            numeric_columns(
                processed_df
                if processed_df is not None
                else original_df
            ),
            "90",
        )

    if export_type == "Missing Value Report":
        return missing_report()

    if export_type == "Duplicate Report":
        return duplicate_report()

    if export_type == "Outlier Report":
        return outlier_report()

    if export_type == "Data Quality Report":
        _, table = quality_analysis(
            processed_df
            if processed_df is not None
            else original_df
        )

        return table

    if export_type == "Model Metrics":
        return last_metrics

    if export_type == "Model Comparison":
        return model_results

    if export_type == "Prediction Result":
        if predictions is None:
            return pd.DataFrame()

        return pd.DataFrame(
            {
                "Prediction": predictions
            }
        )

    if export_type == "Feature Importance":
        return last_feature_importance

    if export_type == "Cleaning Change Log":
        return pd.DataFrame(
            cleaning_history
        )

    if export_type == "Complete Analysis Summary":
        df = (
            processed_df
            if processed_df is not None
            else original_df
        )

        if df is None:
            return pd.DataFrame()

        m = dataset_metrics(df)

        return pd.DataFrame(
            [
                {
                    "Metric": "Rows",
                    "Value": m["Rows"],
                },
                {
                    "Metric": "Columns",
                    "Value": m["Columns"],
                },
                {
                    "Metric": "Numeric Columns",
                    "Value": m["Numeric"],
                },
                {
                    "Metric": "Categorical Columns",
                    "Value": m["Categorical"],
                },
                {
                    "Metric": "Missing Values",
                    "Value": m["Missing"],
                },
                {
                    "Metric": "Duplicate Rows",
                    "Value": m["Duplicates"],
                },
                {
                    "Metric": "Outliers",
                    "Value": m["Outliers"],
                },
                {
                    "Metric": "Quality Score",
                    "Value": quality_score(df),
                },
            ]
        )

    return pd.DataFrame()


def generate_export(
    export_type,
    export_format,
):
    data = dataframe_for_export(
        export_type
    )

    if data is None:
        return None, "No data is available for this export."

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        data = pd.DataFrame(data)

    if data.empty:
        return (
            None,
            "The selected export currently contains no data.",
        )

    extension = (
        "csv"
        if export_format == "CSV"
        else "xlsx"
    )

    path = make_temp_file(
        safe_filename(export_type),
        extension,
    )

    try:
        if export_format == "CSV":
            data.to_csv(
                path,
                index=False,
            )

        else:
            data.to_excel(
                path,
                index=False,
                engine="openpyxl",
            )

        if not os.path.exists(path):
            return (
                None,
                "The export file could not be created.",
            )

        return (
            path,
            f"Export created successfully: {os.path.basename(path)}",
        )

    except Exception as e:
        return (
            None,
            f"Export failed: {type(e).__name__}: {str(e)}",
        )


# ------------------------------------------------------------
# 25. QUICK EXPORTS
# ------------------------------------------------------------
def quick_export(kind):
    if kind == "Original":
        data = original_df
        name = "original_dataset"

    elif kind == "Processed":
        data = processed_df
        name = "processed_dataset"

    elif kind == "Model Results":
        data = model_results
        name = "model_results"

    elif kind == "Prediction":
        if predictions is None:
            return None, "No prediction is available."

        data = pd.DataFrame(
            {
                "Prediction": predictions
            }
        )

        name = "prediction_result"

    else:
        return None, "Unknown export."

    if data is None or data.empty:
        return None, "No data is available."

    path = make_temp_file(
        name,
        "csv",
    )

    try:
        data.to_csv(
            path,
            index=False,
        )

        return (
            path,
            f"{name} created successfully.",
        )

    except Exception as e:
        return (
            None,
            f"Export failed: {e}",
        )


# ------------------------------------------------------------
# 26. CSS
# ------------------------------------------------------------
CSS = """
* {
    font-family: "Times New Roman", Times, serif !important;
}

body,
.gradio-container {
    background:
        linear-gradient(
            135deg,
            #eef4fb 0%,
            #f7fbff 45%,
            #eef8f8 100%
        ) !important;
    color: #17233c !important;
}

.gradio-container {
    max-width: 1500px !important;
    margin: auto !important;
}

#studio-header {
    background:
        linear-gradient(
            135deg,
            #102a56,
            #173f78 55%,
            #087f8c
        );
    border-radius: 18px;
    padding: 24px 30px;
    margin-bottom: 16px;
    color: white !important;
    box-shadow: 0 8px 25px rgba(16,42,86,0.18);
}

#studio-header h1,
#studio-header p {
    color: white !important;
}

.kpi-card {
    background: white !important;
    border: 1px solid #d7e2ef !important;
    border-radius: 14px !important;
    padding: 15px !important;
    min-height: 90px !important;
    box-shadow: 0 4px 14px rgba(31,55,88,0.07);
}

.kpi-label {
    color: #61718a !important;
    font-size: 14px !important;
}

.kpi-value {
    color: #102a56 !important;
    font-size: 28px !important;
    font-weight: bold !important;
}

.section-card {
    background: white !important;
    border: 1px solid #d9e4ef !important;
    border-radius: 14px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 3px 12px rgba(30,60,90,0.05);
}

button.primary {
    background:
        linear-gradient(
            135deg,
            #153d78,
            #087f8c
        ) !important;
    color: white !important;
    border: none !important;
}

button.secondary {
    background: #eaf3fb !important;
    color: #17345f !important;
    border: 1px solid #c9d8e8 !important;
}

button.stop {
    background: #a83232 !important;
    color: white !important;
}

input,
textarea,
select {
    border-color: #cbd8e7 !important;
}

.dataframe {
    border-radius: 10px !important;
}

.tabs {
    border-radius: 12px !important;
}

.tab-nav {
    overflow-x: auto !important;
    white-space: nowrap !important;
    scrollbar-width: thin;
}

.tab-nav button {
    color: #18365f !important;
    font-weight: 600 !important;
}

h1, h2, h3, h4 {
    color: #102a56 !important;
}

footer {
    display: none !important;
}

.gradio-container .block {
    border-color: #d7e2ef !important;
}

#status-box {
    background: #edf8f7 !important;
    border-left: 4px solid #087f8c !important;
    border-radius: 8px !important;
}

.warning-box {
    background: #fff7df !important;
    border-left: 4px solid #d99a00 !important;
    padding: 12px !important;
    border-radius: 8px !important;
}
"""


# ------------------------------------------------------------
# 27. GRADIO UI
# ------------------------------------------------------------
with gr.Blocks(
    title="AI & ML Analytics Studio",
    css=CSS,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
    ),
) as app:

    gr.HTML(
        """
        <div id="studio-header">
            <h1>AI & ML ANALYTICS STUDIO</h1>
            <p>
                Healthcare-aware data analytics, EDA,
                machine learning, evaluation and explainability.
            </p>
        </div>
        """
    )

    # --------------------------------------------------------
    # UPLOAD / GLOBAL STATUS
    # --------------------------------------------------------
    with gr.Row():
        file_input = gr.File(
            label="Upload Dataset",
            file_types=[
                ".csv",
                ".xlsx",
                ".xls",
            ],
            type="filepath",
        )

        upload_btn = gr.Button(
            "Upload Dataset",
            variant="primary",
        )

    upload_status = gr.Markdown(
        "Upload a CSV, XLSX, or XLS dataset to begin.",
        elem_id="status-box",
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------
    with gr.Row():
        kpi_rows = gr.Markdown(
            '<div class="kpi-card"><div class="kpi-label">Rows</div><div class="kpi-value">0</div></div>'
        )

        kpi_cols = gr.Markdown(
            '<div class="kpi-card"><div class="kpi-label">Columns</div><div class="kpi-value">0</div></div>'
        )

        kpi_numeric = gr.Markdown(
            '<div class="kpi-card"><div class="kpi-label">Numeric</div><div class="kpi-value">0</div></div>'
        )

        kpi_categorical = gr.Markdown(
            '<div class="kpi-card"><div class="kpi-label">Categorical</div><div class="kpi-value">0</div></div>'
        )

        kpi_missing = gr.Markdown(
            '<div class="kpi-card"><div class="kpi-label">Missing</div><div class="kpi-value">0</div></div>'
        )

        kpi_quality = gr.Markdown(
            '<div class="kpi-card"><div class="kpi-label">Quality</div><div class="kpi-value">0</div></div>'
        )

    refresh_btn = gr.Button(
        "Refresh Dashboard",
        variant="secondary",
    )

    # --------------------------------------------------------
    # MAIN NAVIGATION
    # --------------------------------------------------------
    with gr.Tabs():

        # ====================================================
        # OVERVIEW
        # ====================================================
        with gr.Tab("Overview"):

            overview_output = gr.Markdown(
                "Load a dataset to see the overview."
            )

            gr.Markdown(
                """
### Dataset Safety

The application keeps two separate dataframes:

**Original Dataset**
- Never modified by cleaning operations.

**Processed Dataset**
- Receives cleaning and feature-engineering operations.

Reset always copies the original dataset back into the processed dataset.
"""
            )

        # ====================================================
        # VIEWER
        # ====================================================
        with gr.Tab("Viewer"):

            gr.Markdown(
                "## Dataset Viewer"
            )

            with gr.Row():
                viewer_source = gr.Dropdown(
                    choices=[
                        "Original",
                        "Processed",
                    ],
                    value="Original",
                    label="Dataset",
                )

                viewer_mode = gr.Dropdown(
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
                        "10",
                        "20",
                        "50",
                        "100",
                        "147",
                        "500",
                    ],
                    value="50",
                    label="Rows",
                )

            with gr.Row():
                viewer_start = gr.Number(
                    value=0,
                    label="Start",
                    visible=False,
                )

                viewer_end = gr.Number(
                    value=147,
                    label="End",
                    visible=False,
                )

                viewer_search = gr.Textbox(
                    label="Search Dataset",
                    placeholder="Example: Male, 60, Diabetes",
                )

            with gr.Row():
                viewer_btn = gr.Button(
                    "Refresh View",
                    variant="primary",
                )

                clear_search_btn = gr.Button(
                    "Clear Search",
                    variant="secondary",
                )

            viewer_info = gr.Markdown(
                "No dataset loaded."
            )

            viewer_table = gr.Dataframe(
                headers=[],
                interactive=False,
                wrap=False,
                max_height=450,
            )

        # ====================================================
        # STATISTICS
        # ====================================================
        with gr.Tab("Statistics"):

            gr.Markdown(
                "## Descriptive Statistics"
            )

            with gr.Row():
                stats_columns = gr.Dropdown(
                    label="Numerical Columns",
                    multiselect=True,
                    choices=[],
                )

                percentile_selector = gr.Dropdown(
                    label="Percentile",
                    choices=[
                        "10",
                        "25",
                        "50",
                        "75",
                        "90",
                        "95",
                        "99",
                    ],
                    value="90",
                )

            stats_btn = gr.Button(
                "Calculate Statistics",
                variant="primary",
            )

            stats_table = gr.Dataframe(
                headers=[],
                interactive=False,
                wrap=False,
                max_height=500,
            )

            gr.Markdown(
                """
Included statistics:

Mean, Median, Mode, Minimum, Maximum, Range,
Variance, Standard Deviation, Q1, Q3, IQR,
Percentiles, Five-number summary, Skewness,
Kurtosis and Mean Absolute Deviation.
"""
            )

        # ====================================================
        # QUALITY
        # ====================================================
        with gr.Tab("Quality"):

            gr.Markdown(
                "## Data Quality Center"
            )

            quality_output = gr.Markdown(
                "Load a dataset first."
            )

            quality_table = gr.Dataframe(
                headers=[],
                interactive=False,
                wrap=False,
                max_height=500,
            )

            healthcare_table = gr.Dataframe(
                headers=[
                    "Column",
                    "Detected Pattern",
                    "Suspicious Values",
                    "Recommendation",
                ],
                interactive=False,
                wrap=False,
                max_height=300,
            )

            quality_btn = gr.Button(
                "Run Quality Analysis",
                variant="primary",
            )

        # ====================================================
        # CLEANING
        # ====================================================
        with gr.Tab("Cleaning"):

            gr.Markdown(
                """
# Data Cleaning Studio

ORIGINAL → SELECT OPERATION → PREVIEW → APPLY → PROCESSED

Preview never permanently changes the processed dataset.
"""
            )

            with gr.Row():

                clean_operation = gr.Dropdown(
                    choices=CLEANING_OPERATIONS,
                    value="Median Imputation",
                    label="Cleaning Operation",
                )

                clean_scope = gr.Dropdown(
                    choices=[
                        "Entire Dataset",
                        "Selected Column",
                    ],
                    value="Entire Dataset",
                    label="Apply To",
                )

                clean_column = gr.Dropdown(
                    choices=[],
                    label="Selected Column",
                )

            with gr.Row():

                clean_multi_columns = gr.Dropdown(
                    choices=[],
                    multiselect=True,
                    label="Optional Multiple Columns",
                )

                clean_constant = gr.Textbox(
                    value="0",
                    label="Constant Value",
                    visible=False,
                )

                clean_replace_from = gr.Textbox(
                    label="Replace From",
                    visible=False,
                )

                clean_replace_to = gr.Textbox(
                    label="Replace To",
                    visible=False,
                )

            with gr.Row():

                clean_threshold = gr.Number(
                    value=50,
                    label="Missing Threshold %",
                    visible=False,
                )

                clean_z_threshold = gr.Number(
                    value=3,
                    label="Z-score Threshold",
                    visible=False,
                )

            clean_control_status = gr.Markdown(
                "Select a cleaning operation."
            )

            with gr.Row():
                clean_preview_btn = gr.Button(
                    "Preview Changes",
                    variant="secondary",
                )

                clean_apply_btn = gr.Button(
                    "Apply Change",
                    variant="primary",
                )

                clean_reset_btn = gr.Button(
                    "Reset to Original",
                    variant="stop",
                )

            clean_summary = gr.Markdown(
                "No cleaning operation previewed."
            )

            with gr.Row():

                with gr.Column():
                    gr.Markdown(
                        "### Before Cleaning"
                    )

                    clean_before_table = gr.Dataframe(
                        headers=[],
                        interactive=False,
                        max_height=350,
                    )

                with gr.Column():
                    gr.Markdown(
                        "### After / Preview"
                    )

                    clean_after_table = gr.Dataframe(
                        headers=[],
                        interactive=False,
                        max_height=350,
                    )

            gr.Markdown(
                "### Before / After Comparison"
            )

            clean_comparison = gr.Dataframe(
                headers=[
                    "Metric",
                    "Before",
                    "After",
                    "Change",
                ],
                interactive=False,
            )

            gr.Markdown(
                "### Cleaning Change Log"
            )

            clean_history_table = gr.Dataframe(
                headers=[
                    "Step",
                    "Operation",
                    "Column",
                    "Before",
                    "After",
                    "Status",
                ],
                interactive=False,
                max_height=300,
            )

        # ====================================================
        # EDA
        # ====================================================
        with gr.Tab("EDA"):

            gr.Markdown(
                "## Exploratory Data Analysis"
            )

            with gr.Row():

                eda_graph = gr.Dropdown(
                    choices=[
                        "Histogram",
                        "KDE",
                        "Box",
                        "Violin",
                        "Count",
                        "Bar",
                        "Pie",
                        "Scatter",
                        "Line",
                        "Area",
                        "Correlation",
                        "Covariance",
                        "Pair Plot",
                        "Q-Q",
                        "ECDF",
                        "Missing",
                        "3D Scatter",
                    ],
                    value="Histogram",
                    label="Graph",
                )

                eda_x = gr.Dropdown(
                    choices=[],
                    label="X",
                )

                eda_y = gr.Dropdown(
                    choices=[],
                    label="Y",
                    visible=False,
                )

                eda_z = gr.Dropdown(
                    choices=[],
                    label="Z",
                    visible=False,
                )

                eda_hue = gr.Dropdown(
                    choices=[],
                    label="Hue",
                    visible=False,
                )

            eda_btn = gr.Button(
                "Generate Graph",
                variant="primary",
            )

            eda_status = gr.Markdown(
                "Select a graph."
            )

            eda_plot = gr.Plot()

        # ====================================================
        # TARGET
        # ====================================================
        with gr.Tab("Target"):

            gr.Markdown(
                "## Target Analysis"
            )

            with gr.Row():
                target_task = gr.Dropdown(
                    choices=[
                        "Classification",
                        "Regression",
                    ],
                    value="Classification",
                    label="Task",
                )

                target_column_select = gr.Dropdown(
                    choices=[],
                    label="Target",
                )

                target_feature_select = gr.Dropdown(
                    choices=[],
                    label="Feature",
                )

            with gr.Row():

                target_btn = gr.Button(
                    "Analyze Target",
                    variant="primary",
                )

                feature_target_btn = gr.Button(
                    "Feature-Target Analysis",
                    variant="secondary",
                )

            target_report = gr.Markdown()
            target_plot = gr.Plot()

            feature_target_report = gr.Markdown()
            feature_target_plot = gr.Plot()

        # ====================================================
        # FEATURES
        # ====================================================
        with gr.Tab("Features"):

            gr.Markdown(
                "## Feature Engineering Studio"
            )

            with gr.Row():

                feature_operation = gr.Dropdown(
                    choices=FEATURE_OPERATIONS,
                    value="Log Transform",
                    label="Feature Operation",
                )

                feature_columns = gr.Dropdown(
                    choices=[],
                    multiselect=True,
                    label="Columns",
                )

                feature_target = gr.Dropdown(
                    choices=[],
                    label="Target for SelectKBest",
                )

            with gr.Row():

                feature_k = gr.Number(
                    value=5,
                    label="K Components / Features",
                )

                variance_threshold = gr.Number(
                    value=0.0,
                    label="Variance Threshold",
                )

                correlation_threshold = gr.Number(
                    value=0.95,
                    label="Correlation Threshold",
                )

            feature_apply_btn = gr.Button(
                "Apply Feature Engineering",
                variant="primary",
            )

            feature_status = gr.Markdown()

            feature_preview = gr.Dataframe(
                headers=[],
                interactive=False,
                max_height=400,
            )

        # ====================================================
        # TRAINING
        # ====================================================
        with gr.Tab("Training"):

            gr.Markdown(
                "## Machine Learning Training"
            )

            with gr.Row():

                train_task = gr.Dropdown(
                    choices=[
                        "Classification",
                        "Regression",
                    ],
                    value="Classification",
                    label="Task",
                )

                train_model = gr.Dropdown(
                    choices=CLASSIFICATION_MODELS,
                    value=CLASSIFICATION_MODELS[0],
                    label="Model",
                )

                train_target = gr.Dropdown(
                    choices=[],
                    label="Target",
                )

            with gr.Row():

                train_test_size = gr.Slider(
                    minimum=0.10,
                    maximum=0.40,
                    value=0.20,
                    step=0.05,
                    label="Test Size",
                )

                train_scaling = gr.Dropdown(
                    choices=[
                        "StandardScaler",
                        "MinMaxScaler",
                        "RobustScaler",
                        "No Scaling",
                    ],
                    value="StandardScaler",
                    label="Scaling",
                )

                train_folds = gr.Dropdown(
                    choices=[
                        "3",
                        "5",
                        "10",
                    ],
                    value="5",
                    label="Cross Validation Folds",
                )

            train_btn = gr.Button(
                "Train Model",
                variant="primary",
            )

            train_status = gr.Markdown()

            training_metrics = gr.Dataframe(
                headers=[
                    "Metric",
                    "Value",
                ],
                interactive=False,
            )

            with gr.Row():

                training_main_plot = gr.Plot()

                training_roc_plot = gr.Plot()

                training_pr_plot = gr.Plot()

        # ====================================================
        # EVALUATION
        # ====================================================
        with gr.Tab("Evaluation"):

            gr.Markdown(
                "## Model Evaluation"
            )

            evaluation_refresh_btn = gr.Button(
                "Refresh Evaluation",
                variant="secondary",
            )

            evaluation_metrics = gr.Dataframe(
                headers=[
                    "Metric",
                    "Value",
                ],
                interactive=False,
            )

            with gr.Row():

                evaluation_main_plot = gr.Plot()

                evaluation_roc = gr.Plot()

                evaluation_pr = gr.Plot()

        # ====================================================
        # COMPARISON
        # ====================================================
        with gr.Tab("Comparison"):

            gr.Markdown(
                "## Model Comparison"
            )

            with gr.Row():

                comparison_task = gr.Dropdown(
                    choices=[
                        "Classification",
                        "Regression",
                    ],
                    value="Classification",
                    label="Task",
                )

                comparison_target = gr.Dropdown(
                    choices=[],
                    label="Target",
                )

                comparison_test_size = gr.Slider(
                    minimum=0.10,
                    maximum=0.40,
                    value=0.20,
                    step=0.05,
                    label="Test Size",
                )

                comparison_scaling = gr.Dropdown(
                    choices=[
                        "StandardScaler",
                        "MinMaxScaler",
                        "RobustScaler",
                        "No Scaling",
                    ],
                    value="StandardScaler",
                    label="Scaling",
                )

            comparison_folds = gr.Dropdown(
                choices=[
                    "3",
                    "5",
                    "10",
                ],
                value="5",
                label="Cross Validation Folds",
            )

            comparison_btn = gr.Button(
                "Compare Models",
                variant="primary",
            )

            comparison_status = gr.Markdown()

            comparison_table = gr.Dataframe(
                headers=[],
                interactive=False,
                max_height=450,
            )

        # ====================================================
        # TUNING
        # ====================================================
        with gr.Tab("Tuning"):

            gr.Markdown(
                "## Hyperparameter Tuning"
            )

            with gr.Row():

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
                    value=CLASSIFICATION_MODELS[0],
                    label="Model",
                )

                tuning_target = gr.Dropdown(
                    choices=[],
                    label="Target",
                )

                tuning_method = gr.Dropdown(
                    choices=[
                        "GridSearchCV",
                        "RandomizedSearchCV",
                    ],
                    value="GridSearchCV",
                    label="Search Method",
                )

                tuning_folds = gr.Dropdown(
                    choices=[
                        "3",
                        "5",
                        "10",
                    ],
                    value="5",
                    label="CV Folds",
                )

            tuning_btn = gr.Button(
                "Tune Model",
                variant="primary",
            )

            tuning_status = gr.Markdown()

            tuning_table = gr.Dataframe(
                headers=[],
                interactive=False,
            )

        # ====================================================
        # PREDICTION
        # ====================================================
        with gr.Tab("Prediction"):

            gr.Markdown(
                """
## Prediction

The prediction table always contains exactly two columns:

**Feature | Value**

Feature order is taken directly from the trained model pipeline.
"""
            )

            prediction_prepare_btn = gr.Button(
                "Prepare Prediction Input",
                variant="secondary",
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
                type="pandas",
                column_count=2,
                value=pd.DataFrame(
                    columns=[
                        "Feature",
                        "Value",
                    ]
                ),
                interactive=True,
                max_height=500,
            )

            prediction_btn = gr.Button(
                "Predict",
                variant="primary",
            )

            prediction_status = gr.Markdown()

            prediction_output = gr.Dataframe(
                headers=[
                    "Prediction",
                ],
                interactive=False,
            )

        # ====================================================
        # EXPLAINABILITY
        # ====================================================
        with gr.Tab("Explainability"):

            gr.Markdown(
                "## Model Explainability"
            )

            explain_method = gr.Dropdown(
                choices=[
                    "Feature Importance",
                    "Coefficients",
                    "Permutation Importance",
                    "SHAP",
                ],
                value="Feature Importance",
                label="Explainability Method",
            )

            explain_btn = gr.Button(
                "Generate Explanation",
                variant="primary",
            )

            explain_status = gr.Markdown()

            explain_table = gr.Dataframe(
                headers=[],
                interactive=False,
                max_height=450,
            )

            explain_plot = gr.Plot()

        # ====================================================
        # EXPORT
        # ====================================================
        with gr.Tab("Export"):

            gr.Markdown(
                """
## Export Center

Choose exactly what you want to export. Processed data always reflects the current processed dataset.
Original data remains unchanged.
"""
            )

            with gr.Row():

                export_type = gr.Dropdown(
                    choices=EXPORT_TYPES,
                    value="Processed Dataset",
                    label="Export Type",
                )

                export_format = gr.Dropdown(
                    choices=EXPORT_FORMATS,
                    value="CSV",
                    label="Format",
                )

            export_btn = gr.Button(
                "Generate Download",
                variant="primary",
            )

            export_status = gr.Markdown()

            export_file = gr.File(
                label="Generated File",
                interactive=False,
            )

            gr.Markdown(
                "### Quick Downloads"
            )

            with gr.Row():

                quick_original_btn = gr.Button(
                    "Original Dataset",
                    variant="secondary",
                )

                quick_processed_btn = gr.Button(
                    "Processed Dataset",
                    variant="secondary",
                )

                quick_model_btn = gr.Button(
                    "Model Results",
                    variant="secondary",
                )

                quick_prediction_btn = gr.Button(
                    "Prediction Result",
                    variant="secondary",
                )

            with gr.Row():

                quick_file = gr.File(
                    label="Quick Download",
                    interactive=False,
                )

                quick_status = gr.Markdown()


# ------------------------------------------------------------
# 28. EVENT WIRING
# ------------------------------------------------------------

# Upload
upload_btn.click(
    fn=load_dataset,
    inputs=[
        file_input,
    ],
    outputs=[
        upload_status,
        overview_output,
        viewer_info,
        viewer_table,
        stats_table,
        quality_output,
        quality_table,
        clean_column,
        clean_multi_columns,
        stats_columns,
        eda_x,
        eda_y,
        eda_z,
        eda_hue,
        target_column_select,
        target_feature_select,
        feature_columns,
        feature_target,
        train_target,
        comparison_target,
        tuning_target,
    ],
)

# Viewer mode
viewer_mode.change(
    fn=viewer_mode_update,
    inputs=[
        viewer_mode,
    ],
    outputs=[
        viewer_rows,
        viewer_start,
        viewer_end,
    ],
)

# Viewer refresh
viewer_btn.click(
    fn=create_view,
    inputs=[
        viewer_source,
        viewer_mode,
        viewer_rows,
        viewer_start,
        viewer_end,
        viewer_search,
    ],
    outputs=[
        viewer_info,
        viewer_table,
    ],
)

# Clear search
clear_search_btn.click(
    fn=clear_search,
    inputs=[
        viewer_source,
        viewer_mode,
        viewer_rows,
        viewer_start,
        viewer_end,
    ],
    outputs=[
        viewer_search,
        viewer_info,
        viewer_table,
    ],
)

# Statistics
stats_btn.click(
    fn=statistics_analysis,
    inputs=[
        stats_columns,
        percentile_selector,
    ],
    outputs=[
        stats_table,
    ],
)

# Quality
quality_btn.click(
    fn=lambda: (
        quality_analysis(
            processed_df
            if processed_df is not None
            else original_df
        )[0],
        quality_analysis(
            processed_df
            if processed_df is not None
            else original_df
        )[1],
        healthcare_validation(
            processed_df
            if processed_df is not None
            else original_df
        ),
    ),
    inputs=[],
    outputs=[
        quality_output,
        quality_table,
        healthcare_table,
    ],
)

# Cleaning controls
clean_operation.change(
    fn=clean_operation_controls,
    inputs=[
        clean_operation,
    ],
    outputs=[
        clean_column,
        clean_constant,
        clean_replace_from,
        clean_threshold,
        clean_z_threshold,
    ],
)

# Cleaning preview
clean_preview_btn.click(
    fn=cleaning_preview,
    inputs=[
        clean_operation,
        clean_scope,
        clean_column,
        clean_multi_columns,
        clean_constant,
        clean_replace_from,
        clean_replace_to,
        clean_threshold,
        clean_z_threshold,
    ],
    outputs=[
        clean_summary,
        clean_before_table,
        clean_after_table,
        clean_comparison,
    ],
)

# Cleaning apply
clean_apply_btn.click(
    fn=apply_cleaning,
    inputs=[
        clean_operation,
        clean_scope,
        clean_column,
        clean_multi_columns,
        clean_constant,
        clean_replace_from,
        clean_replace_to,
        clean_threshold,
        clean_z_threshold,
    ],
    outputs=[
        clean_summary,
        clean_after_table,
        clean_comparison,
        clean_history_table,
    ],
)

# Cleaning reset
clean_reset_btn.click(
    fn=reset_cleaning,
    inputs=[],
    outputs=[
        clean_summary,
        clean_after_table,
        clean_comparison,
    ],
)

# EDA controls
eda_graph.change(
    fn=eda_control_update,
    inputs=[
        eda_graph,
    ],
    outputs=[
        eda_x,
        eda_y,
        eda_z,
        eda_hue,
    ],
)

# EDA plot
eda_btn.click(
    fn=generate_eda_plot,
    inputs=[
        eda_graph,
        eda_x,
        eda_y,
        eda_z,
        eda_hue,
    ],
    outputs=[
        eda_plot,
        eda_status,
    ],
)

# Target task/model choices
target_task.change(
    fn=lambda task: gr.update(
        choices=all_columns(
            processed_df
            if processed_df is not None
            else original_df
        ),
        value=(
            all_columns(
                processed_df
                if processed_df is not None
                else original_df
            )[0]
            if all_columns(
                processed_df
                if processed_df is not None
                else original_df
            )
            else None
        ),
    ),
    inputs=[
        target_task,
    ],
    outputs=[
        target_column_select,
    ],
)

target_btn.click(
    fn=target_analysis,
    inputs=[
        target_column_select,
        target_task,
    ],
    outputs=[
        target_report,
        target_plot,
    ],
)

feature_target_btn.click(
    fn=feature_target_analysis,
    inputs=[
        target_column_select,
        target_feature_select,
    ],
    outputs=[
        feature_target_report,
        feature_target_plot,
    ],
)

# Feature engineering
feature_apply_btn.click(
    fn=feature_engineering,
    inputs=[
        feature_operation,
        feature_columns,
        feature_target,
        feature_k,
        variance_threshold,
        correlation_threshold,
    ],
    outputs=[
        feature_status,
        feature_preview,
    ],
)

# Training task changes
train_task.change(
    fn=model_choices,
    inputs=[
        train_task,
    ],
    outputs=[
        train_model,
    ],
)

train_btn.click(
    fn=train_model,
    inputs=[
        train_task,
        train_model,
        train_target,
        train_test_size,
        train_scaling,
        train_folds,
    ],
    outputs=[
        train_status,
        training_metrics,
        training_main_plot,
        training_roc_plot,
        training_pr_plot,
    ],
)

# Evaluation refresh
evaluation_refresh_btn.click(
    fn=lambda: (
        last_metrics
        if isinstance(last_metrics, pd.DataFrame)
        else pd.DataFrame(),
        last_evaluation_fig,
        last_roc_fig,
        last_pr_fig,
    ),
    inputs=[],
    outputs=[
        evaluation_metrics,
        evaluation_main_plot,
        evaluation_roc,
        evaluation_pr,
    ],
)

# Comparison
comparison_task.change(
    fn=lambda task: gr.update(
        choices=all_columns(
            processed_df
            if processed_df is not None
            else original_df
        ),
        value=(
            all_columns(
                processed_df
                if processed_df is not None
                else original_df
            )[0]
            if all_columns(
                processed_df
                if processed_df is not None
                else original_df
            )
            else None
        ),
    ),
    inputs=[
        comparison_task,
    ],
    outputs=[
        comparison_target,
    ],
)

comparison_btn.click(
    fn=compare_models,
    inputs=[
        comparison_task,
        comparison_target,
        comparison_test_size,
        comparison_scaling,
        comparison_folds,
    ],
    outputs=[
        comparison_status,
        comparison_table,
    ],
)

# Tuning
tuning_task.change(
    fn=model_choices,
    inputs=[
        tuning_task,
    ],
    outputs=[
        tuning_model,
    ],
)

tuning_btn.click(
    fn=tune_model,
    inputs=[
        tuning_task,
        tuning_model,
        tuning_target,
        tuning_method,
        tuning_folds,
    ],
    outputs=[
        tuning_status,
        tuning_table,
    ],
)

# Prediction
prediction_prepare_btn.click(
    fn=prepare_prediction,
    inputs=[],
    outputs=[
        prediction_input,
        prediction_status,
    ],
)

prediction_btn.click(
    fn=predict_from_table,
    inputs=[
        prediction_input,
    ],
    outputs=[
        prediction_status,
        prediction_output,
    ],
)

# Explainability
explain_btn.click(
    fn=explain_model,
    inputs=[
        explain_method,
    ],
    outputs=[
        explain_status,
        explain_table,
        explain_plot,
    ],
)

# Export
export_btn.click(
    fn=generate_export,
    inputs=[
        export_type,
        export_format,
    ],
    outputs=[
        export_file,
        export_status,
    ],
)

# Quick exports
quick_original_btn.click(
    fn=lambda: quick_export("Original"),
    inputs=[],
    outputs=[
        quick_file,
        quick_status,
    ],
)

quick_processed_btn.click(
    fn=lambda: quick_export("Processed"),
    inputs=[],
    outputs=[
        quick_file,
        quick_status,
    ],
)

quick_model_btn.click(
    fn=lambda: quick_export("Model Results"),
    inputs=[],
    outputs=[
        quick_file,
        quick_status,
    ],
)

quick_prediction_btn.click(
    fn=lambda: quick_export("Prediction"),
    inputs=[],
    outputs=[
        quick_file,
        quick_status,
    ],
)

# Refresh
refresh_btn.click(
    fn=refresh_dashboard,
    inputs=[],
    outputs=[
        upload_status,
        overview_output,
        viewer_info,
        viewer_table,
        stats_table,
        quality_output,
        quality_table,
    ],
)


# ------------------------------------------------------------
# 29. LAUNCH
# ------------------------------------------------------------
print("AI & ML Analytics Studio is starting...")
print("Upload CSV, XLSX or XLS data after the interface opens.")

app.launch(
    share=True,
    debug=False,
)