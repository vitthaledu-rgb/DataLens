# AI & ML ANALYTICS STUDIO
# Single Google Colab cell / GitHub app.py
# No notebook-only !pip or % commands are used.

import sys
import subprocess
import importlib
import warnings
import os
import re
import uuid
import tempfile
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# INSTALL REQUIRED PACKAGES
# ============================================================

REQUIRED_PACKAGES = {
    "gradio": "gradio>=5.50,<6",
    "pandas": "pandas>=2.0",
    "numpy": "numpy>=1.24",
    "matplotlib": "matplotlib>=3.7",
    "seaborn": "seaborn>=0.12",
    "scipy": "scipy>=1.10",
    "sklearn": "scikit-learn>=1.3",
    "openpyxl": "openpyxl>=3.1",
}

OPTIONAL_PACKAGES = {
    "xlrd": "xlrd>=2.0.1",
}


def install_missing():
    missing = []

    for module, package in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(package)

    if missing:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q"] + missing
            )
        except Exception as exc:
            raise RuntimeError(
                "Required Python packages could not be installed. "
                f"Installer message: {exc}"
            )

    for module, package in OPTIONAL_PACKAGES.items():
        try:
            importlib.import_module(module)
        except Exception:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", package]
                )
            except Exception:
                pass


install_missing()

# ============================================================
# IMPORTS
# ============================================================

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import median_abs_deviation

from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score,
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
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

from sklearn.svm import SVC, SVR

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ============================================================
# GLOBAL FILE LOCATION
# ============================================================

TEMP_DIR = (
    Path(tempfile.gettempdir())
    / "ai_ml_analytics_studio"
)

TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# GENERAL HELPERS
# ============================================================

def new_empty_history():
    return pd.DataFrame(
        columns=[
            "Step",
            "Operation",
            "Column",
            "Before",
            "After",
            "Status",
        ]
    )


def empty_df():
    return pd.DataFrame()


def copy_df(df):
    if isinstance(df, pd.DataFrame):
        return df.copy(deep=True)
    return pd.DataFrame()


def safe_columns(df):
    result = []
    seen = {}

    for raw in df.columns:
        base = str(raw).strip() or "Unnamed"
        count = seen.get(base, 0)
        seen[base] = count + 1

        if count == 0:
            result.append(base)
        else:
            result.append(f"{base}_{count}")

    output = df.copy()
    output.columns = result

    return output


def unique_temp_file(prefix, extension):
    return (
        TEMP_DIR
        / f"{prefix}_{uuid.uuid4().hex[:10]}{extension}"
    )


def numeric_columns(df):
    if df is None or df.empty:
        return []

    return list(
        df.select_dtypes(
            include=np.number
        ).columns
    )


def categorical_columns(df):
    if df is None or df.empty:
        return []

    return list(
        df.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        ).columns
    )


def all_columns(df):
    if isinstance(df, pd.DataFrame):
        return list(df.columns)

    return []


def make_kpi(label, value):
    if isinstance(value, float):
        display = f"{value:.1f}"
    elif isinstance(value, (int, np.integer)):
        display = f"{int(value):,}"
    else:
        display = str(value)

    return (
        '<div class="kpi">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{display}</div>'
        '</div>'
    )

# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(path):

    if not path:
        return (
            pd.DataFrame(),
            "Please upload a CSV, XLSX, or XLS file."
        )

    try:

        file_path = str(path)
        suffix = Path(file_path).suffix.lower()

        if suffix == ".csv":

            df = pd.read_csv(file_path)

        elif suffix == ".xlsx":

            df = pd.read_excel(
                file_path,
                engine="openpyxl"
            )

        elif suffix == ".xls":

            try:
                import xlrd
            except Exception:
                return (
                    pd.DataFrame(),
                    "XLS support requires xlrd. "
                    "Install xlrd>=2.0.1 and reload the app."
                )

            df = pd.read_excel(
                file_path,
                engine="xlrd"
            )

        else:

            return (
                pd.DataFrame(),
                "Unsupported file type. "
                "Use CSV, XLSX, or XLS."
            )

        df = safe_columns(df)

        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        return (
            df,
            f"Loaded {Path(file_path).name}: "
            f"{len(df):,} rows x "
            f"{len(df.columns):,} columns."
        )

    except Exception as exc:

        return (
            pd.DataFrame(),
            f"Could not load the file: "
            f"{type(exc).__name__}: {exc}"
        )

# ============================================================
# HEALTHCARE VALIDATION
# ============================================================

def healthcare_checks(df):

    warnings_list = []
    checked = 0
    valid = 0

    if df is None or df.empty:
        return {
            "score": 100.0,
            "warnings": []
        }

    for column in df.columns:

        name = (
            str(column)
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )

        numeric = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        valid_values = numeric.notna()
        count = int(valid_values.sum())

        if count == 0:
            continue

        checks = []

        if "age" in name:
            checks.append(
                (
                    "age",
                    (numeric < 0) |
                    (numeric > 120)
                )
            )

        if (
            "blood pressure" in name
            or name in {
                "bp",
                "systolic",
                "diastolic"
            }
        ):
            checks.append(
                (
                    "blood pressure",
                    numeric <= 0
                )
            )

        if (
            "heart rate" in name
            or name in {
                "heartrate",
                "pulse"
            }
        ):
            checks.append(
                (
                    "heart rate",
                    numeric <= 0
                )
            )

        if "cholesterol" in name:
            checks.append(
                (
                    "cholesterol",
                    numeric < 0
                )
            )

        if (
            name == "bmi"
            or "body mass index" in name
        ):
            checks.append(
                (
                    "BMI",
                    (numeric <= 0) |
                    (numeric > 100)
                )
            )

        for label, mask in checks:

            bad = int(
                mask.fillna(False).sum()
            )

            checked += count
            valid += count - bad

            if bad:

                warnings_list.append(
                    f"{column}: {bad:,} possible invalid "
                    f"{label} value(s). Warning only; "
                    f"no healthcare records were deleted automatically."
                )

    score = (
        100.0
        if checked == 0
        else 100 * valid / checked
    )

    score = max(
        0.0,
        min(100.0, score)
    )

    return {
        "score": round(score, 1),
        "warnings": warnings_list
    }

# ============================================================
# DATA QUALITY
# ============================================================

def quality_score(df):

    if df is None or df.empty:
        return 0.0

    missing = int(
        df.isna().sum().sum()
    )

    duplicate = int(
        df.duplicated().sum()
    )

    total_cells = max(
        df.size,
        1
    )

    completeness = (
        100 *
        (1 - missing / total_cells)
    )

    uniqueness = (
        100 *
        (1 - duplicate / max(len(df), 1))
    )

    complete_columns = sum(
        df[c].notna().all()
        for c in df.columns
    )

    consistency = (
        100 *
        complete_columns /
        max(len(df.columns), 1)
    )

    health = healthcare_checks(df)["score"]

    score = (
        0.40 * completeness
        + 0.20 * uniqueness
        + 0.20 * consistency
        + 0.20 * health
    )

    return round(
        max(0.0, min(100.0, score)),
        1
    )


def dataset_metrics(df):

    if df is None or df.empty:

        return {
            "rows": 0,
            "columns": 0,
            "numeric": 0,
            "categorical": 0,
            "missing": 0,
            "duplicates": 0,
            "quality": 0.0,
        }

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric": len(
            numeric_columns(df)
        ),
        "categorical": len(
            categorical_columns(df)
        ),
        "missing": int(
            df.isna().sum().sum()
        ),
        "duplicates": int(
            df.duplicated().sum()
        ),
        "quality": quality_score(df),
    }


def overview_table(df):

    if df is None or df.empty:
        return pd.DataFrame()

    rows = []

    for column in df.columns:

        s = df[column]

        mode = (
            s.dropna().mode()
        )

        rows.append(
            [
                column,
                str(s.dtype),
                int(s.notna().sum()),
                int(s.isna().sum()),
                int(s.nunique(dropna=True)),
                (
                    str(mode.iloc[0])
                    if not mode.empty
                    else ""
                ),
            ]
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Column",
            "Type",
            "Non-null",
            "Missing",
            "Unique",
            "Mode",
        ],
    )


def data_quality_report(df):

    if df is None or df.empty:
        return (
            pd.DataFrame(),
            "No dataset loaded."
        )

    rows = []

    numeric_like = []
    empty_columns = []

    for column in df.columns:

        s = df[column]

        non_null = int(
            s.notna().sum()
        )

        missing = int(
            s.isna().sum()
        )

        unique = int(
            s.nunique(dropna=True)
        )

        infinite = 0

        if pd.api.types.is_numeric_dtype(s):

            try:
                infinite = int(
                    np.isinf(
                        s.astype(float)
                    ).sum()
                )
            except Exception:
                infinite = 0

        suspicious = int(
            s.dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                {
                    "",
                    "na",
                    "n/a",
                    "null",
                    "none",
                    "unknown",
                    "?",
                    "-",
                }
            )
            .sum()
        )

        flag = "OK"

        if non_null == 0:

            flag = "Empty"
            empty_columns.append(
                str(column)
            )

        elif unique <= 1:

            flag = "Constant"

        else:

            top_ratio = (
                float(
                    s.value_counts(
                        dropna=True,
                        normalize=True
                    ).iloc[0]
                )
                if non_null
                else 0.0
            )

            if top_ratio >= 0.99:
                flag = "Near-constant"

            if not pd.api.types.is_numeric_dtype(s):

                converted = pd.to_numeric(
                    s,
                    errors="coerce"
                )

                ratio = (
                    converted.notna().sum()
                    / non_null
                )

                if ratio >= 0.90:

                    numeric_like.append(
                        str(column)
                    )

                    flag = (
                        "Numeric-looking text"
                    )

                elif (
                    unique / non_null > 0.90
                ):

                    flag = "High-cardinality"

        rows.append(
            [
                column,
                str(s.dtype),
                len(df),
                non_null,
                missing,
                round(
                    100 * missing /
                    max(len(df), 1),
                    2
                ),
                unique,
                infinite,
                suspicious,
                flag,
            ]
        )

    duplicate_columns = []
    seen_hashes = {}

    for column in df.columns:

        try:

            h = hash(
                tuple(
                    df[column]
                    .astype(str)
                    .tolist()
                )
            )

            if h in seen_hashes:

                duplicate_columns.append(
                    f"{column} duplicates "
                    f"{seen_hashes[h]}"
                )

            else:

                seen_hashes[h] = column

        except Exception:
            pass

    table = pd.DataFrame(
        rows,
        columns=[
            "Column",
            "Type",
            "Rows",
            "Non-null",
            "Missing",
            "Missing %",
            "Unique",
            "Infinite",
            "Suspicious text",
            "Flag",
        ],
    )

    hc = healthcare_checks(df)

    message = (
        f"Data Quality Score: "
        f"{quality_score(df)}/100. "
        f"This is a dashboard heuristic."
    )

    if numeric_like:

        message += (
            "\nNumeric-looking text: "
            + ", ".join(
                numeric_like[:12]
            )
            + "."
        )

    if empty_columns:

        message += (
            "\nEmpty columns: "
            + ", ".join(
                empty_columns[:12]
            )
            + "."
        )

    if duplicate_columns:

        message += (
            "\nDuplicate columns: "
            + "; ".join(
                duplicate_columns[:12]
            )
            + "."
        )

    if hc["warnings"]:

        message += (
            "\n\nHealthcare warnings:\n"
            + "\n".join(
                "- " + item
                for item in hc["warnings"][:15]
            )
        )

    else:

        message += (
            "\nHealthcare checks: "
            "no flagged values."
        )

    return table, message


def missing_report(df):

    if df is None or df.empty:

        return (
            pd.DataFrame(),
            "No dataset loaded."
        )

    missing = df.isna().sum()

    out = pd.DataFrame(
        {
            "Column": missing.index,
            "Missing": missing.values,
            "Missing %": (
                100 *
                missing.values /
                max(len(df), 1)
            ).round(2),
        }
    )

    out = out.sort_values(
        "Missing",
        ascending=False
    )

    return (
        out,
        f"{int(missing.sum()):,} missing cells."
    )


def duplicate_report(df):

    if df is None or df.empty:

        return (
            pd.DataFrame(),
            "No dataset loaded."
        )

    duplicate_count = int(
        df.duplicated().sum()
    )

    out = pd.DataFrame(
        {
            "Metric": [
                "Duplicate rows",
                "Unique rows",
                "Duplicate %",
            ],
            "Value": [
                duplicate_count,
                len(df) - duplicate_count,
                round(
                    100 *
                    duplicate_count /
                    max(len(df), 1),
                    2
                ),
            ],
        }
    )

    return (
        out,
        f"{duplicate_count:,} duplicate rows detected."
    )


def outlier_report(
    df,
    method="IQR"
):

    rows = []

    for column in numeric_columns(df):

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(values) < 3:
            continue

        if method == "Z-score":

            mean = values.mean()
            std = values.std()

            if (
                std == 0
                or pd.isna(std)
            ):

                count = 0

            else:

                count = int(
                    (
                        np.abs(
                            (values - mean) /
                            std
                        ) > 3
                    ).sum()
                )

        else:

            q1 = values.quantile(
                0.25
            )

            q3 = values.quantile(
                0.75
            )

            iqr = q3 - q1

            low = (
                q1 - 1.5 * iqr
            )

            high = (
                q3 + 1.5 * iqr
            )

            count = int(
                (
                    (values < low)
                    |
                    (values > high)
                ).sum()
            )

        rows.append(
            [
                column,
                count,
                round(
                    100 *
                    count /
                    max(len(values), 1),
                    2
                ),
                float(values.min()),
                float(values.max()),
            ]
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Column",
            "Outliers",
            "Outlier %",
            "Minimum",
            "Maximum",
        ],
    )

# ============================================================
# STATISTICS
# ============================================================

def statistics_table(
    df,
    columns=None,
    percentiles=None
):

    if df is None or df.empty:
        return pd.DataFrame()

    columns = (
        columns
        or numeric_columns(df)
    )

    percentiles = (
        percentiles
        or [25, 50, 75]
    )

    rows = []

    for column in columns:

        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if values.empty:
            continue

        mode = values.mode()

        row = {
            "Column": column,
            "Count": len(values),
            "Mean": values.mean(),
            "Median": values.median(),
            "Mode": (
                mode.iloc[0]
                if not mode.empty
                else np.nan
            ),
            "Minimum": values.min(),
            "Maximum": values.max(),
            "Range": (
                values.max()
                - values.min()
            ),
            "Variance": values.var(),
            "Std Dev": values.std(),
            "Q1": values.quantile(0.25),
            "Q2": values.quantile(0.50),
            "Q3": values.quantile(0.75),
            "IQR": (
                values.quantile(0.75)
                - values.quantile(0.25)
            ),
            "Skewness": values.skew(),
            "Kurtosis": values.kurt(),
            "MAD": median_abs_deviation(
                values,
                nan_policy="omit"
            ),
        }

        for percentile in percentiles:

            row[
                f"P{int(percentile)}"
            ] = values.quantile(
                float(percentile) / 100
            )

        rows.append(row)

    return pd.DataFrame(
        rows
    ).round(6)

# ============================================================
# DATASET VIEWER
# ============================================================

def get_source_df(
    original,
    processed,
    dataset_name
):

    if dataset_name == "Original":
        return original

    return processed


def dataset_view(
    original,
    processed,
    dataset_name,
    view_mode,
    row_count,
    start,
    end,
    search,
):

    df = get_source_df(
        original,
        processed,
        dataset_name
    )

    if df is None or df.empty:

        return (
            pd.DataFrame(),
            "Load a dataset first.",
            pd.DataFrame(),
        )

    work = df.copy()

    if search and str(search).strip():

        query = (
            str(search)
            .strip()
            .lower()
        )

        mask = (
            work.astype(str)
            .apply(
                lambda col:
                col.str.lower()
                .str.contains(
                    re.escape(query),
                    na=False
                )
            )
            .any(axis=1)
        )

        work = work.loc[mask]

    total = len(work)

    n = int(
        row_count
        or 50
    )

    if view_mode == "Head":

        result = work.head(n)

        message = (
            f"Showing rows 1–"
            f"{len(result):,} "
            f"of {total:,}"
        )

    elif view_mode == "Tail":

        result = work.tail(n)

        first = max(
            total - len(result) + 1,
            1
        )

        message = (
            f"Showing rows "
            f"{first:,}–{total:,} "
            f"of {total:,}"
        )

    elif view_mode == "Random":

        result = (
            work.sample(
                min(n, total),
                random_state=42
            )
            if total
            else work
        )

        message = (
            f"Showing random "
            f"{len(result):,} "
            f"rows from {total:,}"
        )

    elif view_mode == "Entire Dataset":

        if total > 5000:

            result = work.head(5000)

            message = (
                f"Showing first 5,000 "
                f"of {total:,} rows "
                f"for performance"
            )

        else:

            result = work

            message = (
                f"Showing entire dataset "
                f"— {total:,} rows"
            )

    else:

        start_index = max(
            int(start or 1) - 1,
            0
        )

        end_index = min(
            int(end or start_index + 50),
            total
        )

        result = work.iloc[
            start_index:end_index
        ]

        message = (
            f"Showing rows "
            f"{start_index + 1:,}–"
            f"{start_index + len(result):,} "
            f"of {total:,}"
        )

    message += (
        f" | Displayed {len(result):,}"
        f" | Columns {len(work.columns):,}"
    )

    return (
        result,
        message,
        result.copy()
    )

# ============================================================
# CLEANING ENGINE
# ============================================================

def clean_dataset(
    df,
    operation,
    primary_column,
    selected_columns,
    strategy,
    value,
    threshold,
):

    x = copy_df(df)

    before = dataset_metrics(x)

    selected = list(
        selected_columns or []
    )

    if (
        primary_column
        and primary_column != "(none)"
        and primary_column not in selected
    ):
        selected.insert(
            0,
            primary_column
        )

    try:

        if operation == "Remove duplicates":

            removed = int(
                x.duplicated().sum()
            )

            x = x.drop_duplicates()

            detail = (
                f"Removed {removed:,} "
                f"duplicate row(s)."
            )

        elif operation == "Drop rows with missing values":

            subset = (
                selected
                if selected
                else None
            )

            before_rows = len(x)

            x = x.dropna(
                subset=subset
            )

            detail = (
                f"Removed "
                f"{before_rows - len(x):,} "
                f"row(s) containing "
                f"missing values."
            )

        elif operation == "Drop columns above missing %":

            limit = float(
                threshold
                if threshold is not None
                else 50
            )

            drop_cols = [
                c
                for c in x.columns
                if (
                    x[c].isna().mean()
                    * 100
                ) > limit
            ]

            x = x.drop(
                columns=drop_cols,
                errors="ignore"
            )

            detail = (
                f"Dropped {len(drop_cols)} "
                f"column(s) above "
                f"{limit:g}% missing."
            )

        elif operation == "Missing value treatment":

            target_columns = (
                selected
                or list(x.columns)
            )

            for column in target_columns:

                if column not in x.columns:
                    continue

                if (
                    strategy == "Mean"
                    and pd.api.types.is_numeric_dtype(
                        x[column]
                    )
                ):

                    x[column] = (
                        x[column]
                        .fillna(
                            x[column].mean()
                        )
                    )

                elif (
                    strategy == "Median"
                    and pd.api.types.is_numeric_dtype(
                        x[column]
                    )
                ):

                    x[column] = (
                        x[column]
                        .fillna(
                            x[column].median()
                        )
                    )

                elif strategy == "Mode":

                    mode = (
                        x[column]
                        .mode()
                    )

                    if not mode.empty:

                        x[column] = (
                            x[column]
                            .fillna(
                                mode.iloc[0]
                            )
                        )

                elif strategy == "Constant":

                    x[column] = (
                        x[column]
                        .fillna(
                            value
                            if value != ""
                            else 0
                        )
                    )

                elif strategy == "Forward fill":

                    x[column] = (
                        x[column]
                        .ffill()
                    )

                elif strategy == "Backward fill":

                    x[column] = (
                        x[column]
                        .bfill()
                    )

                elif (
                    strategy == "Interpolation"
                    and pd.api.types.is_numeric_dtype(
                        x[column]
                    )
                ):

                    x[column] = (
                        x[column]
                        .interpolate()
                    )

            detail = (
                f"Applied {strategy} "
                f"to {len(target_columns):,} "
                f"column(s)."
            )

        elif operation == "String cleanup":

            target_columns = (
                selected
                or categorical_columns(x)
            )

            for column in target_columns:

                if column not in x.columns:
                    continue

                series = (
                    x[column]
                    .astype("string")
                )

                if strategy == "Strip":

                    x[column] = (
                        series.str.strip()
                    )

                elif strategy == "Lowercase":

                    x[column] = (
                        series.str.lower()
                    )

                elif strategy == "Uppercase":

                    x[column] = (
                        series.str.upper()
                    )

                elif strategy == "Title case":

                    x[column] = (
                        series.str.title()
                    )

            detail = (
                f"Applied {strategy} "
                f"to {len(target_columns):,} "
                f"column(s)."
            )

        elif operation == "Replace values":

            target_columns = (
                selected
                or list(x.columns)
            )

            raw = str(
                value or ""
            )

            if "=>" not in raw:

                raise ValueError(
                    "Use replacement format: "
                    "old => new"
                )

            old, new = raw.split(
                "=>",
                1
            )

            old = old.strip()
            new = new.strip()

            for column in target_columns:

                if column in x.columns:

                    x[column] = (
                        x[column]
                        .replace(
                            old,
                            new
                        )
                    )

            detail = (
                f"Replaced '{old}' "
                f"with '{new}'."
            )

        elif operation == "Numeric conversion":

            target_columns = (
                selected
                or list(x.columns)
            )

            for column in target_columns:

                if column in x.columns:

                    x[column] = (
                        pd.to_numeric(
                            x[column],
                            errors="coerce"
                        )
                    )

            detail = (
                "Numeric conversion completed."
            )

        elif operation == "Category conversion":

            target_columns = (
                selected
                or categorical_columns(x)
            )

            for column in target_columns:

                if column in x.columns:

                    x[column] = (
                        x[column]
                        .astype("category")
                    )

            detail = (
                "Category conversion completed."
            )

        elif operation == "Date conversion":

            target_columns = (
                selected
                or list(x.columns)
            )

            for column in target_columns:

                if column in x.columns:

                    x[column] = (
                        pd.to_datetime(
                            x[column],
                            errors="coerce"
                        )
                    )

            detail = (
                "Date conversion completed."
            )

        elif operation == "Negative-value handling":

            target_columns = (
                selected
                or numeric_columns(x)
            )

            for column in target_columns:

                if column not in x.columns:
                    continue

                s = pd.to_numeric(
                    x[column],
                    errors="coerce"
                )

                if strategy == "Remove rows":

                    x = x.loc[
                        s.isna()
                        | (s >= 0)
                    ]

                elif strategy == "Replace with 0":

                    x.loc[
                        s < 0,
                        column
                    ] = 0

                elif strategy == "Absolute value":

                    x[column] = s.abs()

            detail = (
                f"Negative values handled "
                f"using {strategy}."
            )

        elif operation == "Infinite-value handling":

            x = x.replace(
                [np.inf, -np.inf],
                np.nan
            )

            if strategy == "Drop rows":

                x = x.dropna(
                    subset=selected
                    or None
                )

            elif strategy == "Replace with 0":

                x = x.fillna(0)

            detail = (
                f"Infinite values handled "
                f"using {strategy}."
            )

        elif operation == "Empty/common-null strings":

            target_columns = (
                selected
                or categorical_columns(x)
            )

            null_tokens = {
                "",
                "na",
                "n/a",
                "null",
                "none",
                "nan",
                "unknown",
                "?",
                "-",
            }

            for column in target_columns:

                if column in x.columns:

                    s = (
                        x[column]
                        .astype("string")
                        .str.strip()
                    )

                    x.loc[
                        s.str.lower()
                        .isin(null_tokens),
                        column
                    ] = np.nan

            detail = (
                "Common null/empty strings "
                "converted to missing values."
            )

        elif operation in {
            "IQR remove",
            "IQR cap",
            "Z-score remove",
            "Z-score cap",
        }:

            target_columns = (
                selected
                or numeric_columns(x)
            )

            removed = 0

            for column in target_columns:

                if column not in x.columns:
                    continue

                s = pd.to_numeric(
                    x[column],
                    errors="coerce"
                )

                values = s.dropna()

                if len(values) < 3:
                    continue

                if operation.startswith("IQR"):

                    q1 = values.quantile(
                        0.25
                    )

                    q3 = values.quantile(
                        0.75
                    )

                    spread = q3 - q1

                    low = (
                        q1 - 1.5 * spread
                    )

                    high = (
                        q3 + 1.5 * spread
                    )

                else:

                    mean = values.mean()
                    std = values.std()

                    if (
                        std == 0
                        or pd.isna(std)
                    ):
                        continue

                    low = (
                        mean - 3 * std
                    )

                    high = (
                        mean + 3 * std
                    )

                mask = (
                    (s < low)
                    | (s > high)
                )

                if operation.endswith(
                    "remove"
                ):

                    removed += int(
                        mask.fillna(
                            False
                        ).sum()
                    )

                    x = x.loc[
                        ~mask.fillna(False)
                    ]

                else:

                    x[column] = (
                        s.clip(
                            low,
                            high
                        )
                    )

            if operation.endswith("remove"):

                detail = (
                    f"{operation}: "
                    f"removed {removed:,} "
                    f"row(s)."
                )

            else:

                detail = (
                    f"{operation}: "
                    f"extreme values capped."
                )

        else:

            detail = (
                "No cleaning operation selected."
            )

        x = x.replace(
            [np.inf, -np.inf],
            np.nan
        )

        after = dataset_metrics(x)

        return (
            x,
            before,
            after,
            detail
        )

    except Exception as exc:

        return (
            df.copy(),
            before,
            before,
            f"Cleaning failed: "
            f"{type(exc).__name__}: {exc}"
        )


def change_summary(
    before,
    after
):

    labels = [
        ("Rows", "rows"),
        ("Columns", "columns"),
        ("Numeric Columns", "numeric"),
        ("Categorical Columns", "categorical"),
        ("Missing", "missing"),
        ("Duplicates", "duplicates"),
        ("Quality Score", "quality"),
    ]

    return pd.DataFrame(
        {
            "Metric": [
                x[0]
                for x in labels
            ],
            "Before": [
                before[x[1]]
                for x in labels
            ],
            "After": [
                after[x[1]]
                for x in labels
            ],
            "Change": [
                after[x[1]]
                - before[x[1]]
                for x in labels
            ],
        }
    )

# ============================================================
# EDA
# ============================================================

def make_chart(
    df,
    chart_type,
    x=None,
    y=None,
    z=None,
    hue=None
):

    if df is None or df.empty:
        raise ValueError(
            "Load a dataset first."
        )

    fig, ax = plt.subplots(
        figsize=(8, 4.8)
    )

    try:

        if chart_type == "Histogram / KDE":

            if not x:
                raise ValueError(
                    "Select a numeric X column."
                )

            sns.histplot(
                df[x].dropna(),
                kde=True,
                ax=ax
            )

            ax.set_title(
                f"Distribution of {x}"
            )

        elif chart_type == "Box Plot":

            if not x:
                raise ValueError(
                    "Select a numeric X column."
                )

            sns.boxplot(
                x=df[x],
                ax=ax
            )

            ax.set_title(
                f"Box Plot: {x}"
            )

        elif chart_type == "Violin Plot":

            if not x:
                raise ValueError(
                    "Select a numeric X column."
                )

            sns.violinplot(
                x=df[x],
                ax=ax
            )

            ax.set_title(
                f"Violin Plot: {x}"
            )

        elif chart_type == "Count Plot":

            if not x:
                raise ValueError(
                    "Select a categorical X column."
                )

            order = (
                df[x]
                .astype(str)
                .value_counts()
                .head(30)
                .index
            )

            sns.countplot(
                y=df[x].astype(str),
                order=order,
                ax=ax
            )

            ax.set_title(
                f"Counts: {x}"
            )

        elif chart_type == "Bar Chart":

            if not x or not y:
                raise ValueError(
                    "Select X and numeric Y."
                )

            grouped = (
                df.groupby(
                    x,
                    dropna=False
                )[y]
                .mean()
                .sort_values(
                    ascending=False
                )
                .head(30)
            )

            grouped.plot(
                kind="bar",
                ax=ax
            )

            ax.set_title(
                f"Mean {y} by {x}"
            )

        elif chart_type == "Pie Chart":

            if not x:
                raise ValueError(
                    "Select a categorical X column."
                )

            (
                df[x]
                .astype(str)
                .value_counts()
                .head(12)
                .plot.pie(
                    autopct="%1.1f%%",
                    ax=ax
                )
            )

            ax.set_ylabel("")

            ax.set_title(
                f"Distribution: {x}"
            )

        elif chart_type == "Scatter Plot":

            if not x or not y:
                raise ValueError(
                    "Select numeric X and Y."
                )

            sns.scatterplot(
                data=df,
                x=x,
                y=y,
                hue=(
                    hue
                    if hue
                    else None
                ),
                ax=ax
            )

            ax.set_title(
                f"{y} vs {x}"
            )

        elif chart_type == "Line Chart":

            if not x or not y:
                raise ValueError(
                    "Select X and Y."
                )

            sns.lineplot(
                data=df,
                x=x,
                y=y,
                hue=(
                    hue
                    if hue
                    else None
                ),
                ax=ax
            )

            ax.set_title(
                f"{y} by {x}"
            )

        elif chart_type == "Area Chart":

            if not x or not y:
                raise ValueError(
                    "Select X and Y."
                )

            temp = (
                df[[x, y]]
                .copy()
                .dropna()
                .head(5000)
            )

            temp[y] = pd.to_numeric(
                temp[y],
                errors="coerce"
            )

            temp = temp.dropna()

            ax.fill_between(
                np.arange(len(temp)),
                temp[y].to_numpy(),
                alpha=0.30
            )

            ax.plot(
                np.arange(len(temp)),
                temp[y].to_numpy()
            )

            ax.set_title(
                f"Area Chart: {y}"
            )

        elif chart_type == "Pearson Correlation":

            columns = numeric_columns(df)

            if len(columns) < 2:
                raise ValueError(
                    "At least two numeric columns are required."
                )

            sns.heatmap(
                df[columns].corr(
                    method="pearson"
                ),
                cmap="coolwarm",
                center=0,
                ax=ax
            )

            ax.set_title(
                "Pearson Correlation Heatmap"
            )

        elif chart_type == "Covariance Heatmap":

            columns = numeric_columns(df)

            if len(columns) < 2:
                raise ValueError(
                    "At least two numeric columns are required."
                )

            sns.heatmap(
                df[columns].cov(),
                cmap="vlag",
                center=0,
                ax=ax
            )

            ax.set_title(
                "Covariance Heatmap"
            )

        elif chart_type == "Pair Plot":

            columns = numeric_columns(df)[:6]

            if len(columns) < 2:
                raise ValueError(
                    "At least two numeric columns are required."
                )

            sample = (
                df[columns]
                .dropna()
            )

            sample = sample.sample(
                min(500, len(sample)),
                random_state=42
            )

            plt.close(fig)

            pair = sns.pairplot(
                sample
            )

            return pair.fig

        elif chart_type == "Q-Q Plot":

            if not x:
                raise ValueError(
                    "Select a numeric X column."
                )

            values = pd.to_numeric(
                df[x],
                errors="coerce"
            ).dropna()

            stats.probplot(
                values,
                dist="norm",
                plot=ax
            )

            ax.set_title(
                f"Q-Q Plot: {x}"
            )

        elif chart_type == "ECDF":

            if not x:
                raise ValueError(
                    "Select a numeric X column."
                )

            values = np.sort(
                pd.to_numeric(
                    df[x],
                    errors="coerce"
                )
                .dropna()
                .to_numpy()
            )

            if len(values) == 0:
                raise ValueError(
                    "No numeric values available."
                )

            y_values = (
                np.arange(
                    1,
                    len(values) + 1
                )
                / len(values)
            )

            ax.step(
                values,
                y_values,
                where="post"
            )

            ax.set_xlabel(x)
            ax.set_ylabel("ECDF")

            ax.set_title(
                f"ECDF: {x}"
            )

        elif chart_type == "Missing Values":

            missing = (
                df.isna()
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            missing = (
                missing[
                    missing > 0
                ]
                .head(40)
            )

            if missing.empty:
                raise ValueError(
                    "No missing values found."
                )

            missing.plot(
                kind="bar",
                ax=ax
            )

            ax.set_title(
                "Missing Values by Column"
            )

            ax.set_ylabel(
                "Missing Cells"
            )

        elif chart_type == "3D Scatter":

            if not x or not y or not z:
                raise ValueError(
                    "Select X, Y and Z."
                )

            plt.close(fig)

            fig = plt.figure(
                figsize=(8, 5)
            )

            ax3 = fig.add_subplot(
                111,
                projection="3d"
            )

            ax3.scatter(
                pd.to_numeric(
                    df[x],
                    errors="coerce"
                ),
                pd.to_numeric(
                    df[y],
                    errors="coerce"
                ),
                pd.to_numeric(
                    df[z],
                    errors="coerce"
                ),
                s=15,
                alpha=0.65
            )

            ax3.set_xlabel(x)
            ax3.set_ylabel(y)
            ax3.set_zlabel(z)

            ax3.set_title(
                "3D Scatter"
            )

        elif chart_type == "Time Series":

            if not x or not y:
                raise ValueError(
                    "Select date/time X and numeric Y."
                )

            temp = df[
                [x, y]
            ].copy()

            temp[x] = pd.to_datetime(
                temp[x],
                errors="coerce"
            )

            temp[y] = pd.to_numeric(
                temp[y],
                errors="coerce"
            )

            temp = (
                temp
                .dropna()
                .sort_values(x)
            )

            if temp.empty:
                raise ValueError(
                    "X could not be interpreted as dates."
                )

            ax.plot(
                temp[x],
                temp[y]
            )

            ax.set_title(
                f"Time Series: {y}"
            )

            fig.autofmt_xdate()

        elif chart_type in {
            "Spearman Correlation",
            "Kendall Correlation",
        }:

            columns = numeric_columns(df)

            if len(columns) < 2:
                raise ValueError(
                    "At least two numeric columns are required."
                )

            method = (
                "spearman"
                if chart_type.startswith(
                    "Spearman"
                )
                else "kendall"
            )

            corr = df[
                columns
            ].corr(
                method=method
            )

            sns.heatmap(
                corr,
                cmap="coolwarm",
                center=0,
                ax=ax
            )

            ax.set_title(
                f"{method.title()} "
                f"Correlation Heatmap"
            )

        elif chart_type == "Categorical Association":

            cats = categorical_columns(df)

            if len(cats) < 2:
                raise ValueError(
                    "At least two categorical columns are required."
                )

            table = pd.crosstab(
                df[cats[0]],
                df[cats[1]]
            )

            table = table.iloc[
                :20,
                :20
            ]

            sns.heatmap(
                table,
                cmap="Blues",
                ax=ax
            )

            ax.set_title(
                f"Categorical Association: "
                f"{cats[0]} vs {cats[1]}"
            )

        else:

            raise ValueError(
                "Select a chart type."
            )

        fig.tight_layout()

        return fig

    except Exception:

        plt.close("all")

        raise

# ============================================================
# TARGET ANALYSIS
# ============================================================

def analyze_target(
    df,
    target,
    task
):

    if df is None or df.empty:
        return (
            pd.DataFrame(),
            "Load a dataset first."
        )

    if target not in df.columns:
        return (
            pd.DataFrame(),
            "Select a valid target."
        )

    series = df[target]

    if task == "Classification":

        counts = (
            series
            .astype(str)
            .value_counts(
                dropna=False
            )
        )

        percentages = (
            100 *
            counts /
            max(counts.sum(), 1)
        ).round(2)

        table = pd.DataFrame(
            {
                "Class": counts.index,
                "Count": counts.values,
                "Percent": percentages.values,
            }
        )

        ratio = (
            counts.max()
            / max(
                counts.min(),
                1
            )
        )

        numeric_features = [
            c
            for c in numeric_columns(df)
            if c != target
        ]

        association_lines = []

        if numeric_features:

            work = df[
                numeric_features
            ].copy()

            encoded_target = pd.Series(
                pd.factorize(
                    series.astype(str)
                )[0],
                index=df.index
            )

            for column in numeric_features:

                try:

                    value = pd.to_numeric(
                        work[column],
                        errors="coerce"
                    ).corr(
                        encoded_target
                    )

                    if pd.notna(value):

                        association_lines.append(
                            (
                                column,
                                abs(float(value))
                            )
                        )

                except Exception:
                    pass

        association_lines.sort(
            key=lambda item: item[1],
            reverse=True
        )

        text = (
            f"Classification target: "
            f"{len(table):,} class(es). "
            f"Imbalance ratio: "
            f"{ratio:.2f}."
        )

        if association_lines:

            text += (
                "\nTop numeric "
                "feature-target associations: "
                + ", ".join(
                    f"{name} ({score:.3f})"
                    for name, score
                    in association_lines[:8]
                )
            )

        return table, text

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if values.empty:

        return (
            pd.DataFrame(),
            "Regression target must contain numeric values."
        )

    temp = pd.DataFrame(
        {
            target: values
        }
    )

    outliers = outlier_report(
        temp
    )

    count = (
        int(
            outliers.iloc[0][
                "Outliers"
            ]
        )
        if not outliers.empty
        else 0
    )

    table = pd.DataFrame(
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
                "Outliers",
            ],
            "Value": [
                len(values),
                values.mean(),
                values.median(),
                values.std(),
                values.min(),
                values.max(),
                values.skew(),
                values.kurt(),
                count,
            ],
        }
    ).round(6)

    associations = []

    for column in numeric_columns(df):

        if column == target:
            continue

        try:

            feature = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            target_values = pd.to_numeric(
                df[target],
                errors="coerce"
            )

            correlation = feature.corr(
                target_values,
                method="pearson"
            )

            if pd.notna(
                correlation
            ):

                associations.append(
                    (
                        column,
                        abs(float(correlation))
                    )
                )

        except Exception:
            pass

    associations.sort(
        key=lambda item: item[1],
        reverse=True
    )

    message = (
        "Regression target distribution summary."
    )

    if associations:

        message += (
            "\nTop numeric "
            "feature-target associations: "
            + ", ".join(
                f"{name} ({score:.3f})"
                for name, score
                in associations[:8]
            )
        )

    return table, message

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(
    df,
    operation,
    columns,
    k,
    threshold
):

    if df is None or df.empty:
        return (
            df,
            "Load a dataset first."
        )

    x = df.copy()

    columns = [
        c
        for c in (columns or [])
        if c in x.columns
    ]

    try:

        if operation == "Log transform":

            for c in columns:

                values = pd.to_numeric(
                    x[c],
                    errors="coerce"
                )

                x[c] = np.log1p(
                    values.clip(
                        lower=0
                    )
                )

        elif operation == "Square":

            for c in columns:

                x[c] = (
                    pd.to_numeric(
                        x[c],
                        errors="coerce"
                    ) ** 2
                )

        elif operation == "Square root":

            for c in columns:

                values = pd.to_numeric(
                    x[c],
                    errors="coerce"
                )

                x[c] = np.sqrt(
                    values.clip(
                        lower=0
                    )
                )

        elif operation == "StandardScaler":

            cols = (
                columns
                or numeric_columns(x)
            )

            if cols:

                x[cols] = (
                    StandardScaler()
                    .fit_transform(
                        x[cols].fillna(
                            x[cols].median()
                        )
                    )
                )

        elif operation == "MinMaxScaler":

            cols = (
                columns
                or numeric_columns(x)
            )

            if cols:

                x[cols] = (
                    MinMaxScaler()
                    .fit_transform(
                        x[cols].fillna(
                            x[cols].median()
                        )
                    )
                )

        elif operation == "RobustScaler":

            cols = (
                columns
                or numeric_columns(x)
            )

            if cols:

                x[cols] = (
                    RobustScaler()
                    .fit_transform(
                        x[cols].fillna(
                            x[cols].median()
                        )
                    )
                )

        elif operation == "One-hot encoding":

            if columns:

                x = pd.get_dummies(
                    x,
                    columns=columns,
                    dummy_na=True
                )

        elif operation == "Ordinal encoding":

            if columns:

                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                )

                x[columns] = (
                    encoder.fit_transform(
                        x[columns].astype(str)
                    )
                )

        elif operation == "Frequency encoding":

            for c in columns:

                frequencies = (
                    x[c]
                    .value_counts(
                        normalize=True
                    )
                )

                x[c] = (
                    x[c]
                    .map(frequencies)
                )

        elif operation == "Variance filter":

            nums = numeric_columns(x)

            limit = float(
                threshold
                if threshold is not None
                else 0
            )

            keep = [
                c
                for c in nums
                if (
                    x[c]
                    .fillna(
                        x[c].median()
                    )
                    .var()
                    > limit
                )
            ]

            cats = categorical_columns(x)

            x = x[
                keep + cats
            ]

        elif operation == "Correlation filter":

            nums = numeric_columns(x)

            limit = float(
                threshold
                if threshold is not None
                else 0.90
            )

            if len(nums) > 1:

                corr = (
                    x[nums]
                    .corr()
                    .abs()
                )

                drop = set()

                for i, first in enumerate(nums):

                    for second in nums[
                        i + 1:
                    ]:

                        if (
                            corr.loc[
                                first,
                                second
                            ] > limit
                        ):

                            drop.add(
                                second
                            )

                x = x.drop(
                    columns=list(drop),
                    errors="ignore"
                )

        elif operation == "SelectKBest":

            nums = numeric_columns(x)

            keep_n = min(
                int(k or 10),
                len(nums)
            )

            if (
                keep_n
                and len(nums) > keep_n
            ):

                variances = (
                    x[nums]
                    .fillna(
                        x[nums].median()
                    )
                    .var()
                    .sort_values(
                        ascending=False
                    )
                )

                keep = list(
                    variances.head(
                        keep_n
                    ).index
                )

                x = x[
                    keep
                    + [
                        c
                        for c in x.columns
                        if c not in nums
                    ]
                ]

        elif operation == "PCA":

            nums = (
                columns
                or numeric_columns(x)
            )

            nums = [
                c
                for c in nums
                if c in x.columns
            ]

            if len(nums) < 2:

                raise ValueError(
                    "PCA requires at least "
                    "two numeric columns."
                )

            matrix = (
                x[nums]
                .apply(
                    pd.to_numeric,
                    errors="coerce"
                )
                .fillna(0)
            )

            components = min(
                int(k or 2),
                len(nums),
                max(
                    1,
                    len(matrix)
                )
            )

            scaled = (
                StandardScaler()
                .fit_transform(matrix)
            )

            transformed = (
                PCA(
                    n_components=components
                )
                .fit_transform(
                    scaled
                )
            )

            for i in range(
                components
            ):

                x[
                    f"PCA_{i + 1}"
                ] = transformed[:, i]

        return (
            x,
            f"Feature engineering applied: "
            f"{operation}. The original uploaded "
            f"dataset remains unchanged."
        )

    except Exception as exc:

        return (
            df.copy(),
            f"Feature engineering failed: "
            f"{type(exc).__name__}: {exc}"
        )

# ============================================================
# MACHINE LEARNING
# ============================================================

def model_choices(task):

    if task == "Classification":

        choices = [
            "Logistic Regression",
            "KNN",
            "Naive Bayes",
            "Decision Tree",
            "Random Forest",
            "SVM",
            "Gradient Boosting",
        ]

    else:

        choices = [
            "Linear Regression",
            "Polynomial Regression",
            "Ridge",
            "Lasso",
            "Decision Tree",
            "Random Forest",
            "SVR",
            "Gradient Boosting",
        ]

    if XGB_AVAILABLE:
        choices.append(
            "XGBoost"
        )

    return choices


def build_preprocessor(
    X,
    scaling="StandardScaler",
    encoding="One-hot"
):

    numeric = list(
        X.select_dtypes(
            include=np.number
        ).columns
    )

    categorical = [
        c
        for c in X.columns
        if c not in numeric
    ]

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]

    scaler_map = {
        "StandardScaler":
            StandardScaler(),
        "MinMaxScaler":
            MinMaxScaler(),
        "RobustScaler":
            RobustScaler(),
    }

    scaler = scaler_map.get(
        scaling
    )

    if scaler is not None:

        numeric_steps.append(
            (
                "scaler",
                scaler
            )
        )

    if encoding == "Ordinal":

        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1
                    )
                ),
            ]
        )

    else:

        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                ),
            ]
        )

    transformers = []

    if numeric:

        transformers.append(
            (
                "numeric",
                Pipeline(
                    numeric_steps
                ),
                numeric
            )
        )

    if categorical:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical
            )
        )

    if not transformers:

        raise ValueError(
            "No usable feature columns remain."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )


def make_model(
    task,
    name
):

    if task == "Classification":

        models = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=1200
                ),

            "KNN":
                KNeighborsClassifier(),

            "Naive Bayes":
                GaussianNB(),

            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42
                ),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=150,
                    random_state=42,
                    n_jobs=-1
                ),

            "SVM":
                SVC(
                    probability=True,
                    random_state=42
                ),

            "Gradient Boosting":
                GradientBoostingClassifier(
                    random_state=42
                ),
        }

        if XGB_AVAILABLE:

            models[
                "XGBoost"
            ] = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
                n_jobs=2
            )

    else:

        models = {

            "Linear Regression":
                LinearRegression(),

            "Polynomial Regression":
                Pipeline(
                    [
                        (
                            "polynomial",
                            PolynomialFeatures(
                                degree=2,
                                include_bias=False
                            )
                        ),
                        (
                            "model",
                            Ridge(
                                alpha=1.0
                            )
                        ),
                    ]
                ),

            "Ridge":
                Ridge(
                    alpha=1.0
                ),

            "Lasso":
                Lasso(
                    alpha=0.001,
                    max_iter=5000
                ),

            "Decision Tree":
                DecisionTreeRegressor(
                    random_state=42
                ),

            "Random Forest":
                RandomForestRegressor(
                    n_estimators=150,
                    random_state=42,
                    n_jobs=-1
                ),

            "SVR":
                SVR(),

            "Gradient Boosting":
                GradientBoostingRegressor(
                    random_state=42
                ),
        }

        if XGB_AVAILABLE:

            models[
                "XGBoost"
            ] = XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=2
            )

    return models.get(
        name
    )


def train_pipeline(
    df,
    task,
    target,
    model_name,
    test_size,
    scaling,
    encoding,
    random_state,
):

    if df is None or df.empty:

        return (
            None,
            pd.DataFrame(),
            pd.Series(
                dtype=float
            ),
            pd.Series(
                dtype=float
            ),
            pd.DataFrame(),
            [],
            "Load a dataset first."
        )

    if target not in df.columns:

        return (
            None,
            pd.DataFrame(),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.DataFrame(),
            [],
            "Select a valid target."
        )

    data = (
        df
        .dropna(
            subset=[target]
        )
        .copy()
    )

    if task == "Classification":

        if (
            data[target]
            .nunique(
                dropna=True
            ) < 2
        ):

            return (
                None,
                pd.DataFrame(),
                pd.Series(dtype=float),
                pd.Series(dtype=float),
                pd.DataFrame(),
                [],
                "Classification target "
                "needs at least two classes."
            )

        y = data[target]

    else:

        y = pd.to_numeric(
            data[target],
            errors="coerce"
        )

        data = data.loc[
            y.notna()
        ].copy()

        y = y.loc[
            data.index
        ]

    if len(data) < 12:

        return (
            None,
            pd.DataFrame(),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.DataFrame(),
            [],
            "At least 12 usable rows "
            "are recommended for training."
        )

    X = data.drop(
        columns=[target]
    )

    if X.shape[1] == 0:

        return (
            None,
            pd.DataFrame(),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.DataFrame(),
            [],
            "No feature columns remain."
        )

    try:

        model = make_model(
            task,
            model_name
        )

        if model is None:

            return (
                None,
                pd.DataFrame(),
                pd.Series(dtype=float),
                pd.Series(dtype=float),
                pd.DataFrame(),
                [],
                "Selected model is not compatible "
                "with the selected task."
            )

        preprocessor = build_preprocessor(
            X,
            scaling=scaling,
            encoding=encoding
        )

        pipeline = Pipeline(
            [
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

        stratify = None

        if task == "Classification":

            counts = y.value_counts()

            if (
                len(counts) > 1
                and counts.min() >= 2
            ):

                stratify = y

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = train_test_split(
            X,
            y,
            test_size=float(
                test_size
            ),
            random_state=int(
                random_state
            ),
            stratify=stratify
        )

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        if task == "Classification":

            metrics = {

                "Accuracy":
                    accuracy_score(
                        y_test,
                        predictions
                    ),

                "Precision":
                    precision_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    ),

                "Recall":
                    recall_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    ),

                "F1":
                    f1_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    ),
            }

            try:

                probabilities = (
                    pipeline
                    .predict_proba(
                        X_test
                    )
                )

                if (
                    probabilities.shape[1]
                    == 2
                ):

                    auc = roc_auc_score(
                        y_test,
                        probabilities[:, 1]
                    )

                else:

                    auc = roc_auc_score(
                        y_test,
                        probabilities,
                        multi_class="ovr"
                    )

                metrics[
                    "ROC-AUC"
                ] = auc

            except Exception:

                metrics[
                    "ROC-AUC"
                ] = np.nan

        else:

            mse = mean_squared_error(
                y_test,
                predictions
            )

            r2 = r2_score(
                y_test,
                predictions
            )

            n = len(y_test)
            p = X_test.shape[1]

            metrics = {

                "MAE":
                    mean_absolute_error(
                        y_test,
                        predictions
                    ),

                "MSE":
                    mse,

                "RMSE":
                    np.sqrt(mse),

                "R²":
                    r2,

                "Adjusted R²":
                    (
                        1
                        - (1 - r2)
                        * (n - 1)
                        / max(
                            n - p - 1,
                            1
                        )
                    ),
            }

        metrics_df = pd.DataFrame(
            {
                "Metric":
                    list(metrics.keys()),
                "Value": [
                    (
                        round(
                            float(v),
                            6
                        )
                        if pd.notna(v)
                        else np.nan
                    )
                    for v in metrics.values()
                ],
            }
        )

        features = list(
            X.columns
        )

        message = (
            f"Training complete. "
            f"Train rows: {len(X_train):,}. "
            f"Test rows: {len(X_test):,}. "
            f"Features: {len(features):,}."
        )

        return (
            pipeline,
            X_test,
            y_test,
            predictions,
            metrics_df,
            features,
            message
        )

    except Exception as exc:

        return (
            None,
            pd.DataFrame(),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.DataFrame(),
            [],
            f"Model training failed: "
            f"{type(exc).__name__}: {exc}"
        )

# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_pipeline(
    pipeline,
    X_test,
    y_test,
    predictions,
    task
):

    if pipeline is None:

        return (
            pd.DataFrame(),
            "Train a model first.",
            None
        )

    try:

        if task == "Classification":

            table = pd.DataFrame(
                {
                    "Actual":
                        np.asarray(y_test),
                    "Predicted":
                        np.asarray(predictions),
                }
            )

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            fig = plt.figure(
                figsize=(11, 8)
            )

            grid = fig.add_gridspec(
                2,
                2
            )

            ax1 = fig.add_subplot(
                grid[0, 0]
            )

            cm = confusion_matrix(
                y_test,
                predictions
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax1
            )

            ax1.set_xlabel(
                "Predicted"
            )

            ax1.set_ylabel(
                "Actual"
            )

            ax1.set_title(
                "Confusion Matrix"
            )

            ax2 = fig.add_subplot(
                grid[0, 1]
            )

            ax3 = fig.add_subplot(
                grid[1, 0]
            )

            ax4 = fig.add_subplot(
                grid[1, 1]
            )

            ax4.axis(
                "off"
            )

            curves_available = False

            try:

                probabilities = (
                    pipeline
                    .predict_proba(
                        X_test
                    )
                )

                classes = list(
                    pipeline.classes_
                )

                if len(classes) == 2:

                    positive = (
                        probabilities[:, 1]
                    )

                    binary_actual = (
                        np.asarray(y_test)
                        == classes[1]
                    ).astype(int)

                    fpr, tpr, _ = (
                        roc_curve(
                            binary_actual,
                            positive
                        )
                    )

                    auc = roc_auc_score(
                        binary_actual,
                        positive
                    )

                    ax2.plot(
                        fpr,
                        tpr,
                        label=(
                            f"ROC-AUC = "
                            f"{auc:.3f}"
                        )
                    )

                    ax2.plot(
                        [0, 1],
                        [0, 1],
                        "--"
                    )

                    ax2.set_xlabel(
                        "False Positive Rate"
                    )

                    ax2.set_ylabel(
                        "True Positive Rate"
                    )

                    ax2.set_title(
                        "ROC Curve"
                    )

                    ax2.legend()

                    precision_curve, recall_curve, _ = (
                        precision_recall_curve(
                            binary_actual,
                            positive
                        )
                    )

                    ax3.plot(
                        recall_curve,
                        precision_curve
                    )

                    ax3.set_xlabel(
                        "Recall"
                    )

                    ax3.set_ylabel(
                        "Precision"
                    )

                    ax3.set_title(
                        "Precision-Recall Curve"
                    )

                    curves_available = True

                else:

                    ax2.text(
                        0.5,
                        0.5,
                        "ROC curve is shown "
                        "for binary classification.",
                        ha="center",
                        va="center"
                    )

                    ax2.set_title(
                        "ROC Curve"
                    )

                    ax3.text(
                        0.5,
                        0.5,
                        "Precision-Recall curve "
                        "is shown for binary classification.",
                        ha="center",
                        va="center"
                    )

                    ax3.set_title(
                        "Precision-Recall Curve"
                    )

            except Exception:

                ax2.text(
                    0.5,
                    0.5,
                    "Probability curves "
                    "unavailable for this model.",
                    ha="center",
                    va="center"
                )

                ax2.set_title(
                    "ROC Curve"
                )

                ax3.text(
                    0.5,
                    0.5,
                    "Probability curves "
                    "unavailable for this model.",
                    ha="center",
                    va="center"
                )

                ax3.set_title(
                    "Precision-Recall Curve"
                )

            ax4.text(
                0.02,
                0.95,
                f"Accuracy: {accuracy:.4f}\n"
                f"Precision: {precision:.4f}\n"
                f"Recall: {recall:.4f}\n"
                f"F1: {f1:.4f}\n\n"
                f"ROC/PR curves available: "
                f"{'Yes' if curves_available else 'No'}",
                va="top",
                fontsize=12
            )

            fig.suptitle(
                "Classification Evaluation",
                fontsize=16,
                fontweight="bold"
            )

            fig.tight_layout()

            message = (
                f"Accuracy: {accuracy:.4f} | "
                f"Precision: {precision:.4f} | "
                f"Recall: {recall:.4f} | "
                f"F1: {f1:.4f}"
            )

            return (
                table,
                message,
                fig
            )

        mse = mean_squared_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        residuals = (
            np.asarray(y_test)
            - np.asarray(predictions)
        )

        fig = plt.figure(
            figsize=(10, 5)
        )

        grid = fig.add_gridspec(
            1,
            2
        )

        ax1 = fig.add_subplot(
            grid[0, 0]
        )

        ax1.scatter(
            y_test,
            predictions,
            alpha=0.7
        )

        low = min(
            float(np.min(y_test)),
            float(np.min(predictions))
        )

        high = max(
            float(np.max(y_test)),
            float(np.max(predictions))
        )

        ax1.plot(
            [low, high],
            [low, high],
            "--"
        )

        ax1.set_xlabel(
            "Actual"
        )

        ax1.set_ylabel(
            "Predicted"
        )

        ax1.set_title(
            "Actual vs Predicted"
        )

        ax2 = fig.add_subplot(
            grid[0, 1]
        )

        ax2.scatter(
            predictions,
            residuals,
            alpha=0.7
        )

        ax2.axhline(
            0,
            linestyle="--"
        )

        ax2.set_xlabel(
            "Predicted"
        )

        ax2.set_ylabel(
            "Residual"
        )

        ax2.set_title(
            "Residual Plot"
        )

        fig.tight_layout()

        table = pd.DataFrame(
            {
                "Actual":
                    np.asarray(y_test),
                "Predicted":
                    np.asarray(predictions),
                "Residual":
                    residuals,
            }
        )

        message = (
            f"MAE: "
            f"{mean_absolute_error(y_test, predictions):.4f} | "
            f"MSE: {mse:.4f} | "
            f"RMSE: {np.sqrt(mse):.4f} | "
            f"R²: {r2:.4f}"
        )

        return (
            table,
            message,
            fig
        )

    except Exception as exc:

        return (
            pd.DataFrame(),
            f"Evaluation failed: "
            f"{type(exc).__name__}: {exc}",
            None
        )

# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_models(
    df,
    task,
    target,
    selected_models,
    scaling,
    encoding,
    test_size
):

    if df is None or df.empty:

        return (
            pd.DataFrame(),
            "Load a dataset first."
        )

    if target not in df.columns:

        return (
            pd.DataFrame(),
            "Select a valid target."
        )

    models = (
        selected_models
        or model_choices(task)
    )

    data = (
        df
        .dropna(
            subset=[target]
        )
        .copy()
    )

    if task == "Regression":

        y = pd.to_numeric(
            data[target],
            errors="coerce"
        )

        mask = y.notna()

        data = data.loc[
            mask
        ]

        y = y.loc[
            data.index
        ]

    else:

        y = data[target]

    X = data.drop(
        columns=[target]
    )

    if len(X) < 15:

        return (
            pd.DataFrame(),
            "Not enough rows for model comparison."
        )

    stratify = None

    if task == "Classification":

        counts = y.value_counts()

        if (
            len(counts) > 1
            and counts.min() >= 2
        ):

            stratify = y

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=float(
            test_size
        ),
        random_state=42,
        stratify=stratify
    )

    rows = []

    for name in models:

        try:

            model = make_model(
                task,
                name
            )

            if model is None:
                continue

            pipe = Pipeline(
                [
                    (
                        "preprocessor",
                        build_preprocessor(
                            X_train,
                            scaling,
                            encoding
                        )
                    ),
                    (
                        "model",
                        model
                    ),
                ]
            )

            pipe.fit(
                X_train,
                y_train
            )

            pred = pipe.predict(
                X_test
            )

            if task == "Classification":

                try:

                    probabilities = (
                        pipe.predict_proba(
                            X_test
                        )
                    )

                    if (
                        probabilities.shape[1]
                        == 2
                    ):

                        auc = roc_auc_score(
                            y_test,
                            probabilities[:, 1]
                        )

                    else:

                        auc = roc_auc_score(
                            y_test,
                            probabilities,
                            multi_class="ovr"
                        )

                except Exception:

                    auc = np.nan

                rows.append(
                    [
                        name,
                        accuracy_score(
                            y_test,
                            pred
                        ),
                        precision_score(
                            y_test,
                            pred,
                            average="weighted",
                            zero_division=0
                        ),
                        recall_score(
                            y_test,
                            pred,
                            average="weighted",
                            zero_division=0
                        ),
                        f1_score(
                            y_test,
                            pred,
                            average="weighted",
                            zero_division=0
                        ),
                        auc,
                    ]
                )

            else:

                mse = mean_squared_error(
                    y_test,
                    pred
                )

                r2 = r2_score(
                    y_test,
                    pred
                )

                n = len(y_test)
                p = X_test.shape[1]

                rows.append(
                    [
                        name,
                        mean_absolute_error(
                            y_test,
                            pred
                        ),
                        mse,
                        np.sqrt(mse),
                        r2,
                        (
                            1
                            - (1 - r2)
                            * (n - 1)
                            / max(
                                n - p - 1,
                                1
                            )
                        ),
                    ]
                )

        except Exception:
            continue

    if task == "Classification":

        columns = [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
        ]

        result = pd.DataFrame(
            rows,
            columns=columns
        )

        if not result.empty:

            result = (
                result
                .sort_values(
                    "F1",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )

    else:

        columns = [
            "Model",
            "MAE",
            "MSE",
            "RMSE",
            "R²",
            "Adjusted R²",
        ]

        result = pd.DataFrame(
            rows,
            columns=columns
        )

        if not result.empty:

            result = (
                result
                .sort_values(
                    "R²",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )

    if result.empty:

        return (
            result,
            "No compatible models completed successfully."
        )

    best = result.iloc[0][
        "Model"
    ]

    return (
        result.round(6),
        f"Comparison complete. "
        f"Best model by primary metric: {best}."
    )

# ============================================================
# CROSS VALIDATION
# ============================================================

def cross_validate(
    df,
    task,
    target,
    model_name,
    scaling,
    encoding,
    folds
):

    if df is None or df.empty:

        return (
            pd.DataFrame(),
            "Load a dataset first."
        )

    if target not in df.columns:

        return (
            pd.DataFrame(),
            "Select a valid target."
        )

    data = (
        df
        .dropna(
            subset=[target]
        )
    )

    if task == "Regression":

        y = pd.to_numeric(
            data[target],
            errors="coerce"
        )

        mask = y.notna()

        data = data.loc[
            mask
        ]

        y = y.loc[
            data.index
        ]

    else:

        y = data[target]

    X = data.drop(
        columns=[target]
    )

    try:

        model = make_model(
            task,
            model_name
        )

        if model is None:

            return (
                pd.DataFrame(),
                "Model is unavailable."
            )

        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(
                        X,
                        scaling,
                        encoding
                    )
                ),
                (
                    "model",
                    model
                ),
            ]
        )

        if task == "Classification":

            minimum_class = int(
                y.value_counts().min()
            )

            folds = min(
                int(folds),
                minimum_class
            )

            if folds < 2:

                return (
                    pd.DataFrame(),
                    "Not enough examples per class "
                    "for StratifiedKFold."
                )

            cv = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=42
            )

            scoring = "f1_weighted"

        else:

            folds = min(
                int(folds),
                len(X)
            )

            if folds < 2:

                return (
                    pd.DataFrame(),
                    "Not enough rows for KFold."
                )

            cv = KFold(
                n_splits=folds,
                shuffle=True,
                random_state=42
            )

            scoring = "r2"

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring
        )

        table = pd.DataFrame(
            {
                "Fold":
                    range(
                        1,
                        len(scores) + 1
                    ),
                "Score":
                    scores,
            }
        )

        return (
            table.round(6),
            f"Mean CV score: "
            f"{scores.mean():.4f} "
            f"± {scores.std():.4f}"
        )

    except Exception as exc:

        return (
            pd.DataFrame(),
            f"Cross-validation failed: "
            f"{type(exc).__name__}: {exc}"
        )

# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

def tune_model(
    df,
    task,
    target,
    model_name,
    method,
    scaling,
    encoding,
    folds
):

    if df is None or df.empty:

        return (
            pd.DataFrame(),
            "Load a dataset first."
        )

    if target not in df.columns:

        return (
            pd.DataFrame(),
            "Select a valid target."
        )

    data = (
        df
        .dropna(
            subset=[target]
        )
        .copy()
    )

    if task == "Regression":

        y = pd.to_numeric(
            data[target],
            errors="coerce"
        )

        mask = y.notna()

        data = data.loc[
            mask
        ]

        y = y.loc[
            data.index
        ]

    else:

        y = data[target]

    X = data.drop(
        columns=[target]
    )

    model = make_model(
        task,
        model_name
    )

    if model is None:

        return (
            pd.DataFrame(),
            "Model is unavailable."
        )

    parameter_grids = {

        "Random Forest": {
            "model__n_estimators":
                [80, 150],
            "model__max_depth":
                [None, 6, 12],
        },

        "Decision Tree": {
            "model__max_depth":
                [None, 4, 8],
            "model__min_samples_split":
                [2, 5],
        },

        "Logistic Regression": {
            "model__C":
                [0.1, 1, 10],
        },

        "KNN": {
            "model__n_neighbors":
                [3, 5, 7],
        },

        "Ridge": {
            "model__alpha":
                [0.1, 1, 10],
        },

        "Lasso": {
            "model__alpha":
                [0.0001, 0.001, 0.01],
        },

        "Gradient Boosting": {
            "model__n_estimators":
                [80, 150],
            "model__learning_rate":
                [0.03, 0.10],
            "model__max_depth":
                [2, 3],
        },
    }

    grid = parameter_grids.get(
        model_name
    )

    if not grid:

        return (
            pd.DataFrame(
                {
                    "Parameter":
                        ["Search"],
                    "Value": [
                        "No safe tuning grid "
                        "defined for this model."
                    ],
                }
            ),
            "No tuning grid is defined "
            "for this model."
        )

    try:

        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(
                        X,
                        scaling,
                        encoding
                    )
                ),
                (
                    "model",
                    model
                ),
            ]
        )

        if task == "Classification":

            minimum_class = int(
                y.value_counts().min()
            )

            cv_folds = min(
                int(folds),
                minimum_class
            )

            if cv_folds < 2:

                return (
                    pd.DataFrame(),
                    "Not enough examples "
                    "per class for tuning."
                )

            scoring = "f1_weighted"

        else:

            cv_folds = min(
                int(folds),
                len(X)
            )

            if cv_folds < 2:

                return (
                    pd.DataFrame(),
                    "Not enough rows for tuning."
                )

            scoring = "r2"

        if method == "RandomizedSearchCV":

            search = RandomizedSearchCV(
                pipeline,
                grid,
                n_iter=min(
                    6,
                    max(
                        1,
                        int(
                            np.prod(
                                [
                                    len(v)
                                    for v
                                    in grid.values()
                                ]
                            )
                        )
                    )
                ),
                cv=cv_folds,
                scoring=scoring,
                n_jobs=1,
                random_state=42
            )

        else:

            search = GridSearchCV(
                pipeline,
                grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=1
            )

        search.fit(
            X,
            y
        )

        table = pd.DataFrame(
            {
                "Parameter": [
                    "Best parameters",
                    "Best CV score",
                ],
                "Value": [
                    str(
                        search.best_params_
                    ),
                    round(
                        float(
                            search.best_score_
                        ),
                        6
                    ),
                ],
            }
        )

        return (
            table,
            "Hyperparameter tuning completed."
        )

    except Exception as exc:

        return (
            pd.DataFrame(),
            f"Tuning failed: "
            f"{type(exc).__name__}: {exc}"
        )

# ============================================================
# EXPLAINABILITY
# ============================================================

def native_feature_importance(
    pipeline
):

    if pipeline is None:

        return (
            pd.DataFrame(),
            "Train a model first."
        )

    try:

        model = pipeline.named_steps[
            "model"
        ]

        preprocessor = (
            pipeline.named_steps[
                "preprocessor"
            ]
        )

        names = list(
            preprocessor
            .get_feature_names_out()
        )

        if hasattr(
            model,
            "feature_importances_"
        ):

            values = np.asarray(
                model.feature_importances_
            )

        elif hasattr(
            model,
            "coef_"
        ):

            coefficients = np.asarray(
                model.coef_
            )

            values = (
                np.mean(
                    np.abs(
                        coefficients
                    ),
                    axis=0
                )
                if coefficients.ndim > 1
                else np.abs(
                    coefficients
                )
            )

        else:

            return (
                pd.DataFrame(),
                "The selected model does not "
                "expose native feature importance."
            )

        result = pd.DataFrame(
            {
                "Feature":
                    names[:len(values)],
                "Importance":
                    values[:len(names)],
            }
        )

        result = (
            result
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(50)
        )

        return (
            result,
            "Feature importance generated."
        )

    except Exception as exc:

        return (
            pd.DataFrame(),
            f"Feature importance unavailable: {exc}"
        )


def permutation_importance_report(
    pipeline,
    X_test,
    y_test,
    task
):

    if pipeline is None:

        return (
            pd.DataFrame(),
            "Train a model first."
        )

    try:

        from sklearn.inspection import (
            permutation_importance
        )

        scoring = (
            "f1_weighted"
            if task == "Classification"
            else "r2"
        )

        result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=3,
            random_state=42,
            n_jobs=1,
            scoring=scoring
        )

        table = pd.DataFrame(
            {
                "Feature":
                    list(X_test.columns),
                "Importance":
                    result.importances_mean,
            }
        )

        table = (
            table
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(50)
        )

        return (
            table,
            "Permutation importance generated."
        )

    except Exception as exc:

        return (
            pd.DataFrame(),
            f"Permutation importance failed: {exc}"
        )


def shap_report(
    pipeline,
    X_test
):

    if not SHAP_AVAILABLE:

        return (
            pd.DataFrame(),
            "SHAP is unavailable. "
            "The dashboard continues normally."
        )

    if pipeline is None:

        return (
            pd.DataFrame(),
            "Train a model first."
        )

    try:

        model = pipeline.named_steps[
            "model"
        ]

        preprocessor = (
            pipeline.named_steps[
                "preprocessor"
            ]
        )

        transformed = (
            preprocessor.transform(
                X_test
            )
        )

        explainer = shap.Explainer(
            model,
            transformed
        )

        shap_values = explainer(
            transformed
        )

        names = list(
            preprocessor
            .get_feature_names_out()
        )

        values = np.abs(
            shap_values.values
        )

        if values.ndim > 2:

            values = values.mean(
                axis=-1
            )

        mean_values = (
            values.mean(axis=0)
        )

        table = pd.DataFrame(
            {
                "Feature":
                    names[:len(mean_values)],
                "Mean |SHAP|":
                    mean_values[:len(names)],
            }
        )

        table = (
            table
            .sort_values(
                "Mean |SHAP|",
                ascending=False
            )
            .head(50)
        )

        return (
            table,
            "SHAP analysis generated."
        )

    except Exception as exc:

        return (
            pd.DataFrame(),
            f"SHAP could not be generated: {exc}"
        )

# ============================================================
# PREDICTION
# ============================================================

def prediction_from_table(
    pipeline,
    original_df,
    task,
    input_table
):

    empty_result = pd.DataFrame(
        columns=[
            "Feature",
            "Value"
        ]
    )

    if pipeline is None:

        return (
            empty_result,
            "Train a model first.",
            pd.DataFrame()
        )

    if (
        original_df is None
        or original_df.empty
    ):

        return (
            empty_result,
            "Load a dataset first.",
            pd.DataFrame()
        )

    if input_table is None:

        return (
            empty_result,
            "Enter prediction values.",
            pd.DataFrame()
        )

    try:

        table = pd.DataFrame(
            input_table
        )

        if table.shape[1] < 2:

            return (
                empty_result,
                "Prediction input needs "
                "Feature and Value columns.",
                pd.DataFrame()
            )

        table = table.iloc[
            :,
            :2
        ].copy()

        table.columns = [
            "Feature",
            "Value"
        ]

        table = table.dropna(
            subset=["Feature"]
        )

        features = list(
            table["Feature"]
            .astype(str)
        )

        if not features:

            return (
                empty_result,
                "Enter at least one feature.",
                pd.DataFrame()
            )

        expected = list(
            pipeline
            .named_steps[
                "preprocessor"
            ]
            .feature_names_in_
        )

        missing_features = [
            feature
            for feature in expected
            if feature not in features
        ]

        if missing_features:

            return (
                table,
                "Prediction input is incomplete. "
                "Missing: "
                + ", ".join(
                    missing_features[:15]
                ),
                pd.DataFrame()
            )

        row = {}

        for feature in expected:

            raw = table.loc[
                table["Feature"]
                .astype(str)
                == str(feature),
                "Value"
            ]

            if raw.empty:

                return (
                    empty_result,
                    f"Missing value for {feature}.",
                    pd.DataFrame()
                )

            value = raw.iloc[0]

            if pd.api.types.is_numeric_dtype(
                original_df[feature]
            ):

                converted = pd.to_numeric(
                    pd.Series(
                        [value]
                    ),
                    errors="coerce"
                ).iloc[0]

                if pd.isna(converted):

                    return (
                        table,
                        f"{feature} requires "
                        "a numeric value.",
                        pd.DataFrame()
                    )

                row[feature] = float(
                    converted
                )

            else:

                row[feature] = str(
                    value
                )

        X = pd.DataFrame(
            [row],
            columns=expected
        )

        prediction = pipeline.predict(
            X
        )[0]

        result = pd.DataFrame(
            {
                "Feature": [
                    "Prediction"
                ],
                "Value": [
                    str(prediction)
                ],
            }
        )

        probabilities = pd.DataFrame()

        if (
            task == "Classification"
            and hasattr(
                pipeline,
                "predict_proba"
            )
        ):

            try:

                probability = (
                    pipeline
                    .predict_proba(X)[0]
                )

                classes = list(
                    pipeline.classes_
                )

                probabilities = (
                    pd.DataFrame(
                        {
                            "Class":
                                classes,
                            "Probability":
                                probability,
                        }
                    )
                    .sort_values(
                        "Probability",
                        ascending=False
                    )
                )

            except Exception:

                probabilities = (
                    pd.DataFrame()
                )

        return (
            result,
            f"Prediction generated: "
            f"{prediction}",
            probabilities
        )

    except Exception as exc:

        return (
            empty_result,
            f"Prediction failed: "
            f"{type(exc).__name__}: {exc}",
            pd.DataFrame()
        )

# ============================================================
# EXPORT
# ============================================================

def export_data(
    export_type,
    export_format,
    original,
    processed,
    viewed,
    statistics,
    quality,
    missing,
    duplicates,
    outliers,
    metrics,
    comparison,
    prediction,
    importance,
    history,
):

    datasets = {

        "Original Dataset":
            original,

        "Processed Dataset":
            processed,

        "Currently Viewed Dataset":
            viewed,

        "Statistics":
            statistics,

        "Missing Value Report":
            missing,

        "Duplicate Report":
            duplicates,

        "Outlier Report":
            outliers,

        "Data Quality Report":
            quality,

        "Model Metrics":
            metrics,

        "Model Comparison":
            comparison,

        "Prediction Result":
            prediction,

        "Feature Importance":
            importance,

        "Cleaning Change Log":
            history,
    }

    if (
        export_type
        != "Complete Analysis Summary"
    ):

        data = datasets.get(
            export_type,
            pd.DataFrame()
        )

        if (
            not isinstance(
                data,
                pd.DataFrame
            )
            or data.empty
        ):

            return (
                None,
                f"No exportable data is available "
                f"for {export_type}."
            )

        extension = (
            ".xlsx"
            if export_format == "Excel"
            else ".csv"
        )

        output = unique_temp_file(
            re.sub(
                r"[^A-Za-z0-9]+",
                "_",
                export_type.lower()
            ),
            extension
        )

        try:

            if extension == ".csv":

                data.to_csv(
                    output,
                    index=False
                )

            else:

                with pd.ExcelWriter(
                    output,
                    engine="openpyxl"
                ) as writer:

                    data.to_excel(
                        writer,
                        index=False,
                        sheet_name="Export"
                    )

            if output.exists():

                return (
                    str(output),
                    f"Download ready: "
                    f"{output.name}"
                )

            return (
                None,
                "The export file could not be created."
            )

        except Exception as exc:

            return (
                None,
                f"Export failed: "
                f"{type(exc).__name__}: {exc}"
            )

    try:

        if export_format == "Excel":

            output = unique_temp_file(
                "complete_analysis_summary",
                ".xlsx"
            )

            with pd.ExcelWriter(
                output,
                engine="openpyxl"
            ) as writer:

                for name, data in datasets.items():

                    if (
                        isinstance(
                            data,
                            pd.DataFrame
                        )
                        and not data.empty
                    ):

                        sheet = re.sub(
                            r"[^A-Za-z0-9]+",
                            "_",
                            name
                        )[:31] or "Sheet"

                        data.to_excel(
                            writer,
                            index=False,
                            sheet_name=sheet
                        )

        else:

            output = unique_temp_file(
                "complete_analysis_summary",
                ".csv"
            )

            pieces = []

            for name, data in datasets.items():

                if (
                    isinstance(
                        data,
                        pd.DataFrame
                    )
                    and not data.empty
                ):

                    pieces.append(
                        f"--- {name} ---\n"
                        + data.to_csv(
                            index=False
                        )
                    )

            output.write_text(
                "\n".join(pieces),
                encoding="utf-8"
            )

        if output.exists():

            return (
                str(output),
                f"Download ready: "
                f"{output.name}"
            )

        return (
            None,
            "The summary file could not be created."
        )

    except Exception as exc:

        return (
            None,
            f"Summary export failed: "
            f"{type(exc).__name__}: {exc}"
        )

# ============================================================
# CSS
# ============================================================

CSS = r"""
* {
    font-family: "Times New Roman", Times, serif !important;
}

body,
.gradio-container {
    background: #edf3f7 !important;
    color: #18263a !important;
}

.gradio-container {
    max-width: 1500px !important;
    margin: auto !important;
}

.app-header {
    background: linear-gradient(
        135deg,
        #102b52 0%,
        #176b88 60%,
        #1395a5 100%
    );
    color: white;
    padding: 24px 28px;
    border-radius: 16px;
    margin-bottom: 14px;
    box-shadow: 0 10px 28px rgba(
        16,
        43,
        82,
        .18
    );
}

.app-header h1 {
    margin: 0;
    font-size: 32px;
    letter-spacing: .2px;
}

.app-header p {
    margin: 6px 0 0;
    font-size: 15px;
    opacity: .95;
}

.kpi {
    background: linear-gradient(
        145deg,
        #ffffff,
        #e5f2f7
    );
    border: 1px solid #c8d9e3;
    border-radius: 12px;
    padding: 11px 8px;
    text-align: center;
    min-height: 74px;
    box-shadow: 0 3px 12px rgba(
        34,
        69,
        95,
        .06
    );
}

.kpi-label {
    color: #61758a;
    font-size: 12px;
}

.kpi-value {
    color: #123e66;
    font-size: 24px;
    font-weight: bold;
    margin-top: 3px;
}

.section-title {
    font-size: 21px;
    color: #123e66;
    font-weight: bold;
    border-bottom: 2px solid #d4e2eb;
    padding-bottom: 7px;
    margin-bottom: 12px;
}

.gr-panel,
.panel {
    border: 1px solid #ccdbe4 !important;
    border-radius: 12px !important;
    background: #ffffff !important;
}

.gr-button {
    border-radius: 9px !important;
}

button.primary {
    background: linear-gradient(
        135deg,
        #176b88,
        #1395a5
    ) !important;
    color: white !important;
    border: none !important;
}

button.stop {
    background: #b43d3d !important;
    color: white !important;
}

.gr-input,
.gr-dropdown,
.gr-textbox,
textarea {
    border-radius: 8px !important;
}

.tab-nav {
    overflow-x: auto !important;
    white-space: nowrap !important;
}

table {
    font-size: 13px !important;
}

footer {
    display: none !important;
}
"""

# ============================================================
# GRADIO STATES
# ============================================================

original_state = gr.State(
    pd.DataFrame()
)

processed_state = gr.State(
    pd.DataFrame()
)

viewed_state = gr.State(
    pd.DataFrame()
)

history_state = gr.State(
    new_empty_history()
)

pipeline_state = gr.State(
    None
)

X_test_state = gr.State(
    pd.DataFrame()
)

y_test_state = gr.State(
    pd.Series(dtype=float)
)

prediction_state = gr.State(
    pd.Series(dtype=float)
)

metrics_state = gr.State(
    pd.DataFrame()
)

# ============================================================
# CALLBACKS
# ============================================================

def load_callback(path):

    df, message = load_dataset(
        path
    )

    if df.empty:

        zero = make_kpi(
            "Rows",
            0
        )

        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            new_empty_history(),
            pd.DataFrame(),

            gr.update(
                choices=[],
                value=None
            ),

            gr.update(
                choices=[],
                value=[]
            ),

            gr.update(
                choices=[],
                value=None
            ),

            gr.update(
                choices=[],
                value=None
            ),

            gr.update(
                choices=[],
                value=None
            ),

            gr.update(
                choices=[],
                value=None
            ),

            gr.update(
                choices=[],
                value=None
            ),

            gr.update(
                choices=[],
                value=[]
            ),

            pd.DataFrame(
                columns=[
                    "Feature",
                    "Value"
                ]
            ),

            gr.update(
                choices=model_choices(
                    "Classification"
                ),
                value="Random Forest"
            ),

            zero,

            make_kpi(
                "Columns",
                0
            ),

            make_kpi(
                "Numeric",
                0
            ),

            make_kpi(
                "Categorical",
                0
            ),

            make_kpi(
                "Missing",
                0
            ),

            make_kpi(
                "Duplicates",
                0
            ),

            make_kpi(
                "Quality",
                0
            ),

            message,

            "No healthcare checks available.",

            new_empty_history()
        )

    metrics = dataset_metrics(
        df
    )

    quality, quality_message = (
        data_quality_report(df)
    )

    health = healthcare_checks(
        df
    )

    columns = all_columns(
        df
    )

    nums = numeric_columns(
        df
    )

    cats = categorical_columns(
        df
    )

    healthcare_message = (
        quality_message
    )

    if health["warnings"]:

        healthcare_message += (
            "\n\n"
            + "\n".join(
                "- " + item
                for item in health[
                    "warnings"
                ]
            )
        )

    return (
        df.copy(),
        df.copy(),
        df.head(100).copy(),
        new_empty_history(),

        overview_table(df),

        gr.update(
            choices=[
                "(none)"
            ] + columns,
            value=(
                columns[0]
                if columns
                else "(none)"
            )
        ),

        gr.update(
            choices=columns,
            value=columns[:5]
        ),

        gr.update(
            choices=nums,
            value=(
                nums[0]
                if nums
                else None
            )
        ),

        gr.update(
            choices=nums,
            value=(
                nums[1]
                if len(nums) > 1
                else None
            )
        ),

        gr.update(
            choices=nums,
            value=(
                nums[2]
                if len(nums) > 2
                else None
            )
        ),

        gr.update(
            choices=[""] + cats,
            value=""
        ),

        gr.update(
            choices=columns,
            value=(
                columns[-1]
                if columns
                else None
            )
        ),

        gr.update(
            choices=nums,
            value=nums[:5]
        ),

        pd.DataFrame(
            {
                "Feature":
                    columns,
                "Value":
                    ["" for _ in columns],
            }
        ),

        gr.update(
            choices=model_choices(
                "Classification"
            ),
            value="Random Forest"
        ),

        make_kpi(
            "Rows",
            metrics["rows"]
        ),

        make_kpi(
            "Columns",
            metrics["columns"]
        ),

        make_kpi(
            "Numeric",
            metrics["numeric"]
        ),

        make_kpi(
            "Categorical",
            metrics["categorical"]
        ),

        make_kpi(
            "Missing",
            metrics["missing"]
        ),

        make_kpi(
            "Duplicates",
            metrics["duplicates"]
        ),

        make_kpi(
            "Quality",
            metrics["quality"]
        ),

        message,

        healthcare_message,

        new_empty_history()
    )


def viewer_callback(
    original,
    processed,
    dataset_name,
    view_mode,
    row_count,
    start,
    end,
    search,
):

    return dataset_view(
        original,
        processed,
        dataset_name,
        view_mode,
        row_count,
        start,
        end,
        search
    )


def clear_search_callback(
    original,
    processed,
    dataset_name,
    view_mode,
    row_count,
    start,
    end,
):

    table, message, viewed = (
        dataset_view(
            original,
            processed,
            dataset_name,
            view_mode,
            row_count,
            start,
            end,
            ""
        )
    )

    return (
        "",
        table,
        message,
        viewed
    )


def clean_preview_callback(
    processed,
    operation,
    primary,
    columns,
    strategy,
    value,
    threshold,
):

    if (
        processed is None
        or processed.empty
    ):

        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            "Load a dataset first."
        )

    new_df, before, after, detail = (
        clean_dataset(
            processed,
            operation,
            primary,
            columns,
            strategy,
            value,
            threshold
        )
    )

    return (
        new_df.head(100),
        change_summary(
            before,
            after
        ),
        pd.DataFrame(
            [
                {
                    "Status":
                        "PREVIEW ONLY",
                    "Operation":
                        operation,
                    "Detail":
                        detail,
                    "Rows before":
                        before["rows"],
                    "Rows after":
                        after["rows"],
                    "Missing before":
                        before["missing"],
                    "Missing after":
                        after["missing"],
                }
            ]
        ),
        "Preview created. The processed dataset was NOT changed."
    )


def clean_apply_callback(
    processed,
    history,
    operation,
    primary,
    columns,
    strategy,
    value,
    threshold,
):

    if (
        processed is None
        or processed.empty
    ):

        return (
            processed,
            history,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            "Load a dataset first.",
            make_kpi("Rows", 0),
            make_kpi("Missing", 0),
            make_kpi("Duplicates", 0),
            make_kpi("Quality", 0)
        )

    new_df, before, after, detail = (
        clean_dataset(
            processed,
            operation,
            primary,
            columns,
            strategy,
            value,
            threshold
        )
    )

    hist = (
        history.copy()
        if isinstance(
            history,
            pd.DataFrame
        )
        else new_empty_history()
    )

    selected = list(
        columns or []
    )

    if (
        primary
        and primary != "(none)"
        and primary not in selected
    ):

        selected.insert(
            0,
            primary
        )

    column_text = (
        ", ".join(selected)
        if selected
        else "Applicable columns"
    )

    step = len(hist) + 1

    entry = pd.DataFrame(
        [
            [
                step,
                operation,
                column_text,
                before["rows"],
                after["rows"],
                "Applied",
            ]
        ],
        columns=hist.columns
    )

    hist = pd.concat(
        [
            hist,
            entry
        ],
        ignore_index=True
    )

    metrics = dataset_metrics(
        new_df
    )

    message = (
        f"Cleaning applied successfully. "
        f"{detail} Processed dataset: "
        f"{len(new_df):,} rows x "
        f"{len(new_df.columns):,} columns."
    )

    return (
        new_df,
        hist,
        new_df.head(100),
        pd.DataFrame([before]),
        pd.DataFrame([after]),
        message,
        make_kpi(
            "Rows",
            metrics["rows"]
        ),
        make_kpi(
            "Missing",
            metrics["missing"]
        ),
        make_kpi(
            "Duplicates",
            metrics["duplicates"]
        ),
        make_kpi(
            "Quality",
            metrics["quality"]
        )
    )


def reset_callback(
    original
):

    if (
        original is None
        or original.empty
    ):

        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            new_empty_history(),
            "No original dataset is available.",
            make_kpi("Rows", 0),
            make_kpi("Missing", 0),
            make_kpi("Duplicates", 0),
            make_kpi("Quality", 0)
        )

    metrics = dataset_metrics(
        original
    )

    return (
        original.copy(),
        original.head(100).copy(),
        overview_table(original),
        new_empty_history(),
        "Reset complete. The processed dataset has been restored to the original uploaded dataset.",
        make_kpi(
            "Rows",
            metrics["rows"]
        ),
        make_kpi(
            "Missing",
            metrics["missing"]
        ),
        make_kpi(
            "Duplicates",
            metrics["duplicates"]
        ),
        make_kpi(
            "Quality",
            metrics["quality"]
        )
    )


def chart_callback(
    processed,
    chart_type,
    x,
    y,
    z,
    hue,
):

    try:

        return (
            make_chart(
                processed,
                chart_type,
                x,
                y,
                z,
                hue
            ),
            "Chart generated successfully."
        )

    except Exception as exc:

        return (
            None,
            f"Chart unavailable: "
            f"{type(exc).__name__}: {exc}"
        )


def quality_callback(
    processed
):

    quality, message = (
        data_quality_report(
            processed
        )
    )

    missing, _ = missing_report(
        processed
    )

    duplicates, _ = (
        duplicate_report(
            processed
        )
    )

    outliers = outlier_report(
        processed
    )

    return (
        quality,
        missing,
        duplicates,
        outliers,
        message
    )


def statistics_callback(
    processed,
    columns,
    percentiles
):

    try:

        return statistics_table(
            processed,
            columns,
            percentiles
        )

    except Exception:

        return pd.DataFrame()


def task_change_callback(
    task
):

    choices = model_choices(
        task
    )

    return (
        gr.update(
            choices=choices,
            value=(
                choices[0]
                if choices
                else None
            )
        ),
        gr.update(
            choices=choices,
            value=choices[:]
        )
    )


def target_callback(
    processed,
    target,
    task
):

    return analyze_target(
        processed,
        target,
        task
    )


def feature_callback(
    processed,
    operation,
    columns,
    k,
    threshold
):

    new_df, message = (
        engineer_features(
            processed,
            operation,
            columns,
            k,
            threshold
        )
    )

    return (
        new_df,
        new_df.head(100),
        message
    )


def cleaning_control_callback(
    operation
):

    strategy_choices = [
        "Mean",
        "Median",
        "Mode",
        "Constant",
        "Forward fill",
        "Backward fill",
        "Interpolation",
    ]

    value_visible = (
        operation
        in {
            "Replace values",
            "Missing value treatment",
            "String cleanup",
        }
    )

    threshold_visible = (
        operation
        == "Drop columns above missing %"
    )

    if operation == "Negative-value handling":

        strategy_choices = [
            "Remove rows",
            "Replace with 0",
            "Absolute value",
        ]

    elif operation == "Infinite-value handling":

        strategy_choices = [
            "Drop rows",
            "Replace with 0",
        ]

    elif operation == "String cleanup":

        strategy_choices = [
            "Strip",
            "Lowercase",
            "Uppercase",
            "Title case",
        ]

    elif operation in {
        "IQR remove",
        "IQR cap",
        "Z-score remove",
        "Z-score cap",
    }:

        strategy_choices = [
            "Not required"
        ]

    elif operation in {
        "Remove duplicates",
        "Drop rows with missing values",
        "Drop columns above missing %",
        "Replace values",
        "Numeric conversion",
        "Category conversion",
        "Date conversion",
        "Empty/common-null strings",
    }:

        strategy_choices = [
            "Not required"
        ]

    return (
        gr.update(
            choices=strategy_choices,
            value=strategy_choices[0]
        ),
        gr.update(
            visible=value_visible
        ),
        gr.update(
            visible=threshold_visible
        )
    )


def train_callback(
    processed,
    task,
    target,
    model_name,
    test_size,
    scaling,
    encoding,
    random_state,
):

    (
        pipeline,
        X_test,
        y_test,
        predictions,
        metrics,
        features,
        message
    ) = train_pipeline(
        processed,
        task,
        target,
        model_name,
        test_size,
        scaling,
        encoding,
        random_state
    )

    prediction_template = pd.DataFrame(
        {
            "Feature":
                features,
            "Value":
                ["" for _ in features],
        }
    )

    return (
        pipeline,
        X_test,
        y_test,
        predictions,
        metrics,
        prediction_template,
        message
    )


def evaluation_callback(
    pipeline,
    X_test,
    y_test,
    predictions,
    task
):

    return evaluate_pipeline(
        pipeline,
        X_test,
        y_test,
        predictions,
        task
    )


def comparison_callback(
    processed,
    task,
    target,
    models,
    scaling,
    encoding,
    test_size
):

    return compare_models(
        processed,
        task,
        target,
        models,
        scaling,
        encoding,
        test_size
    )


def cv_callback(
    processed,
    task,
    target,
    model,
    scaling,
    encoding,
    folds
):

    return cross_validate(
        processed,
        task,
        target,
        model,
        scaling,
        encoding,
        folds
    )


def tuning_callback(
    processed,
    task,
    target,
    model,
    method,
    scaling,
    encoding,
    folds
):

    return tune_model(
        processed,
        task,
        target,
        model,
        method,
        scaling,
        encoding,
        folds
    )


def importance_callback(
    pipeline
):

    return native_feature_importance(
        pipeline
    )


def permutation_callback(
    pipeline,
    X_test,
    y_test,
    task
):

    return permutation_importance_report(
        pipeline,
        X_test,
        y_test,
        task
    )


def shap_callback(
    pipeline,
    X_test
):

    return shap_report(
        pipeline,
        X_test
    )


def prediction_callback(
    pipeline,
    original,
    task,
    input_table
):

    return prediction_from_table(
        pipeline,
        original,
        task,
        input_table
    )

# ============================================================
# BUILD DASHBOARD
# ============================================================

with gr.Blocks(
    title="AI & ML ANALYTICS STUDIO",
    css=CSS,
    theme=gr.themes.Soft()
) as dashboard:

    gr.HTML(
        '<div class="app-header">'
        '<h1>AI & ML ANALYTICS STUDIO</h1>'
        '<p>Data analysis, quality, cleaning, visualization, feature engineering, machine learning, evaluation and export.</p>'
        '</div>'
    )

    status_message = gr.Markdown(
        "Upload a CSV, XLSX, or XLS dataset to begin."
    )

    with gr.Row():

        kpi_rows = gr.HTML(
            make_kpi("Rows", 0)
        )

        kpi_columns = gr.HTML(
            make_kpi("Columns", 0)
        )

        kpi_numeric = gr.HTML(
            make_kpi("Numeric", 0)
        )

        kpi_categorical = gr.HTML(
            make_kpi("Categorical", 0)
        )

        kpi_missing = gr.HTML(
            make_kpi("Missing", 0)
        )

        kpi_duplicates = gr.HTML(
            make_kpi("Duplicates", 0)
        )

        kpi_quality = gr.HTML(
            make_kpi("Quality", 0)
        )

    # ========================================================
    # OVERVIEW
    # ========================================================

    with gr.Tab("Overview"):

        gr.Markdown(
            "## Dataset Overview",
            elem_classes="section-title"
        )

        upload = gr.File(
            label="Upload Dataset",
            file_types=[
                ".csv",
                ".xlsx",
                ".xls"
            ],
            type="filepath"
        )

        load_button = gr.Button(
            "Load Dataset",
            variant="primary"
        )

        overview_table_component = (
            gr.Dataframe(
                interactive=False
            )
        )

        healthcare_message = gr.Markdown()

    # ========================================================
    # VIEWER
    # ========================================================

    with gr.Tab("Viewer"):

        gr.Markdown(
            "## Dataset Viewer",
            elem_classes="section-title"
        )

        with gr.Row():

            viewer_dataset = gr.Radio(
                [
                    "Original",
                    "Processed"
                ],
                value="Processed",
                label="Dataset"
            )

            viewer_mode = gr.Dropdown(
                [
                    "Head",
                    "Tail",
                    "Random",
                    "Entire Dataset",
                    "Custom Range",
                ],
                value="Head",
                label="View"
            )

            viewer_rows = gr.Dropdown(
                [
                    "10",
                    "20",
                    "50",
                    "100",
                    "147",
                    "500",
                ],
                value="50",
                label="Rows"
            )

        with gr.Row():

            viewer_start = gr.Number(
                value=1,
                minimum=1,
                step=1,
                label="Custom Start"
            )

            viewer_end = gr.Number(
                value=50,
                minimum=1,
                step=1,
                label="Custom End"
            )

        with gr.Row():

            search_box = gr.Textbox(
                label="Search Dataset",
                placeholder="Search across all columns"
            )

            search_button = gr.Button(
                "Search / View",
                variant="primary"
            )

            clear_search_button = gr.Button(
                "Clear Search"
            )

        viewer_info = gr.Markdown()

        viewer_output = gr.Dataframe(
            interactive=False
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    with gr.Tab("Statistics"):

        gr.Markdown(
            "## Statistics",
            elem_classes="section-title"
        )

        statistics_columns = gr.Dropdown(
            multiselect=True,
            label="Numerical Columns"
        )

        percentile_selector = gr.CheckboxGroup(
            choices=[
                10,
                25,
                50,
                75,
                90,
                95,
                99,
            ],
            value=[
                25,
                50,
                75
            ],
            label="Percentiles"
        )

        statistics_button = gr.Button(
            "Generate Statistics",
            variant="primary"
        )

        statistics_output = gr.Dataframe(
            interactive=False
        )

    # ========================================================
    # QUALITY
    # ========================================================

    with gr.Tab("Quality"):

        gr.Markdown(
            "## Data Quality",
            elem_classes="section-title"
        )

        quality_button = gr.Button(
            "Analyze Quality",
            variant="primary"
        )

        quality_output = gr.Dataframe(
            interactive=False
        )

        missing_output = gr.Dataframe(
            interactive=False
        )

        duplicate_output = gr.Dataframe(
            interactive=False
        )

        outlier_output = gr.Dataframe(
            interactive=False
        )

        quality_message = gr.Markdown()

    # ========================================================
    # CLEANING
    # ========================================================

    with gr.Tab("Cleaning"):

        gr.Markdown(
            "## Data Cleaning Studio",
            elem_classes="section-title"
        )

        gr.Markdown(
            "ORIGINAL → SELECT OPERATION → PREVIEW → APPLY → PROCESSED"
        )

        cleaning_operation = gr.Dropdown(
            [
                "Remove duplicates",
                "Drop rows with missing values",
                "Drop columns above missing %",
                "Missing value treatment",
                "String cleanup",
                "Replace values",
                "Numeric conversion",
                "Category conversion",
                "Date conversion",
                "Negative-value handling",
                "Infinite-value handling",
                "Empty/common-null strings",
                "IQR remove",
                "IQR cap",
                "Z-score remove",
                "Z-score cap",
            ],
            value="Remove duplicates",
            label="Cleaning Operation"
        )

        with gr.Row():

            cleaning_primary = gr.Dropdown(
                choices=[
                    "(none)"
                ],
                value="(none)",
                label="Primary Column"
            )

            cleaning_columns = gr.Dropdown(
                multiselect=True,
                label="Additional Columns"
            )

        with gr.Row():

            cleaning_strategy = gr.Dropdown(
                [
                    "Mean",
                    "Median",
                    "Mode",
                    "Constant",
                    "Forward fill",
                    "Backward fill",
                    "Interpolation",
                ],
                value="Median",
                label="Strategy"
            )

            cleaning_value = gr.Textbox(
                label="Value / Replacement",
                placeholder="For replacement use: old => new"
            )

            cleaning_threshold = gr.Number(
                value=50,
                minimum=0,
                maximum=100,
                label="Missing Threshold %"
            )

        with gr.Row():

            preview_clean_button = gr.Button(
                "Preview Change"
            )

            apply_clean_button = gr.Button(
                "Apply Change",
                variant="primary"
            )

            reset_button = gr.Button(
                "Reset to Original",
                variant="stop"
            )

        cleaning_preview_table = gr.Dataframe(
            interactive=False
        )

        cleaning_change_summary = gr.Dataframe(
            interactive=False
        )

        cleaning_status_table = gr.Dataframe(
            interactive=False
        )

        cleaning_message = gr.Markdown()

        gr.Markdown(
            "### Before / After Comparison",
            elem_classes="section-title"
        )

        with gr.Row():

            cleaning_before = gr.Dataframe(
                interactive=False
            )

            cleaning_after = gr.Dataframe(
                interactive=False
            )

        gr.Markdown(
            "### Cleaning History",
            elem_classes="section-title"
        )

        cleaning_history_output = (
            gr.Dataframe(
                interactive=False
            )
        )

    # ========================================================
    # EDA
    # ========================================================

    with gr.Tab("EDA"):

        gr.Markdown(
            "## Exploratory Data Analysis",
            elem_classes="section-title"
        )

        chart_type = gr.Dropdown(
            [
                "Histogram / KDE",
                "Box Plot",
                "Violin Plot",
                "Count Plot",
                "Bar Chart",
                "Pie Chart",
                "Scatter Plot",
                "Line Chart",
                "Area Chart",
                "Pearson Correlation",
                "Covariance Heatmap",
                "Pair Plot",
                "Q-Q Plot",
                "ECDF",
                "Missing Values",
                "3D Scatter",
                "Time Series",
                "Spearman Correlation",
                "Kendall Correlation",
                "Categorical Association",
            ],
            value="Histogram / KDE",
            label="Chart Type"
        )

        with gr.Row():

            eda_x = gr.Dropdown(
                label="X"
            )

            eda_y = gr.Dropdown(
                label="Y"
            )

            eda_z = gr.Dropdown(
                label="Z"
            )

            eda_hue = gr.Dropdown(
                label="Hue",
                choices=[""]
            )

        chart_button = gr.Button(
            "Generate Chart",
            variant="primary"
        )

        chart_output = gr.Plot()

        chart_message = gr.Markdown()

    # ========================================================
    # TARGET
    # ========================================================

    with gr.Tab("Target"):

        gr.Markdown(
            "## Target Analysis",
            elem_classes="section-title"
        )

        with gr.Row():

            task_type = gr.Radio(
                [
                    "Classification",
                    "Regression",
                ],
                value="Classification",
                label="Task"
            )

            target_column = gr.Dropdown(
                label="Target Column"
            )

        target_button = gr.Button(
            "Analyze Target",
            variant="primary"
        )

        target_output = gr.Dataframe(
            interactive=False
        )

        target_message = gr.Markdown()

    # ========================================================
    # FEATURES
    # ========================================================

    with gr.Tab("Features"):

        gr.Markdown(
            "## Feature Engineering",
            elem_classes="section-title"
        )

        feature_operation = gr.Dropdown(
            [
                "Log transform",
                "Square",
                "Square root",
                "StandardScaler",
                "MinMaxScaler",
                "RobustScaler",
                "One-hot encoding",
                "Ordinal encoding",
                "Frequency encoding",
                "Variance filter",
                "Correlation filter",
                "SelectKBest",
                "PCA",
            ],
            value="Log transform",
            label="Feature Operation"
        )

        feature_columns = gr.Dropdown(
            multiselect=True,
            label="Columns"
        )

        with gr.Row():

            feature_k = gr.Number(
                value=10,
                minimum=1,
                step=1,
                label="K / Components"
            )

            feature_threshold = gr.Number(
                value=0.90,
                minimum=0,
                maximum=1,
                step=0.05,
                label="Threshold"
            )

        feature_button = gr.Button(
            "Apply Feature Engineering",
            variant="primary"
        )

        feature_output = gr.Dataframe(
            interactive=False
        )

        feature_message = gr.Markdown()

    # ========================================================
    # TRAINING
    # ========================================================

    with gr.Tab("Training"):

        gr.Markdown(
            "## Model Training",
            elem_classes="section-title"
        )

        with gr.Row():

            training_model = gr.Dropdown(
                choices=model_choices(
                    "Classification"
                ),
                value="Random Forest",
                label="Model"
            )

            training_scaling = gr.Dropdown(
                [
                    "StandardScaler",
                    "MinMaxScaler",
                    "RobustScaler",
                    "None",
                ],
                value="StandardScaler",
                label="Scaling"
            )

            training_encoding = gr.Dropdown(
                [
                    "One-hot",
                    "Ordinal"
                ],
                value="One-hot",
                label="Encoding"
            )

        with gr.Row():

            training_test_size = gr.Slider(
                0.15,
                0.40,
                value=0.20,
                step=0.05,
                label="Test Size"
            )

            training_random_state = gr.Number(
                value=42,
                minimum=0,
                step=1,
                label="Random State"
            )

        train_button = gr.Button(
            "Train Model",
            variant="primary"
        )

        training_metrics = gr.Dataframe(
            interactive=False
        )

        training_message = gr.Markdown()

    # ========================================================
    # EVALUATION
    # ========================================================

    with gr.Tab("Evaluation"):

        gr.Markdown(
            "## Model Evaluation",
            elem_classes="section-title"
        )

        evaluate_button = gr.Button(
            "Evaluate Model",
            variant="primary"
        )

        evaluation_table = gr.Dataframe(
            interactive=False
        )

        evaluation_message = gr.Markdown()

        evaluation_plot = gr.Plot()

    # ========================================================
    # COMPARISON
    # ========================================================

    with gr.Tab("Comparison"):

        gr.Markdown(
            "## Model Comparison and Cross-validation",
            elem_classes="section-title"
        )

        comparison_models = gr.Dropdown(
            choices=model_choices(
                "Classification"
            ),
            value=model_choices(
                "Classification"
            ),
            multiselect=True,
            label="Models"
        )

        comparison_folds = gr.Dropdown(
            [
                3,
                5,
                10
            ],
            value=5,
            label="Cross-validation folds"
        )

        with gr.Row():

            compare_button = gr.Button(
                "Compare Models",
                variant="primary"
            )

            cross_validation_button = gr.Button(
                "Cross-Validate Selected Model"
            )

        comparison_output = gr.Dataframe(
            interactive=False
        )

        comparison_message = gr.Markdown()

    # ========================================================
    # TUNING
    # ========================================================

    with gr.Tab("Tuning"):

        gr.Markdown(
            "## Hyperparameter Tuning",
            elem_classes="section-title"
        )

        tuning_method = gr.Radio(
            [
                "GridSearchCV",
                "RandomizedSearchCV",
            ],
            value="GridSearchCV",
            label="Method"
        )

        tuning_button = gr.Button(
            "Tune Model",
            variant="primary"
        )

        tuning_output = gr.Dataframe(
            interactive=False
        )

        tuning_message = gr.Markdown()

    # ========================================================
    # PREDICTION
    # ========================================================

    with gr.Tab("Prediction"):

        gr.Markdown(
            "## Prediction",
            elem_classes="section-title"
        )

        gr.Markdown(
            "After training, the exact training feature list "
            "is shown below. Keep one row per feature and "
            "enter values in the Value column."
        )

        prediction_input = gr.Dataframe(
            headers=[
                "Feature",
                "Value"
            ],
            datatype=[
                "str",
                "str"
            ],
            value=[],
            row_count=(
                1,
                "dynamic"
            ),
            column_count=(
                2,
                "fixed"
            ),
            interactive=True
        )

        predict_button = gr.Button(
            "Predict",
            variant="primary"
        )

        prediction_result = gr.Dataframe(
            headers=[
                "Feature",
                "Value"
            ],
            datatype=[
                "str",
                "str"
            ],
            row_count=1,
            column_count=(
                2,
                "fixed"
            ),
            interactive=False
        )

        prediction_message = gr.Markdown()

        probability_output = gr.Dataframe(
            interactive=False
        )

    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    with gr.Tab("Explainability"):

        gr.Markdown(
            "## Explainability",
            elem_classes="section-title"
        )

        with gr.Row():

            native_importance_button = gr.Button(
                "Feature Importance",
                variant="primary"
            )

            permutation_button = gr.Button(
                "Permutation Importance"
            )

            shap_button = gr.Button(
                "SHAP Analysis"
            )

        importance_output = gr.Dataframe(
            interactive=False
        )

        explainability_message = gr.Markdown()

    # ========================================================
    # EXPORT
    # ========================================================

    with gr.Tab("Export"):

        gr.Markdown(
            "## Export Center",
            elem_classes="section-title"
        )

        export_type = gr.Dropdown(
            [
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
            ],
            value="Processed Dataset",
            label="Export Type"
        )

        export_format = gr.Radio(
            [
                "CSV",
                "Excel"
            ],
            value="CSV",
            label="Format"
        )

        export_button = gr.Button(
            "Generate Download",
            variant="primary"
        )

        export_download = gr.File(
            label="Download"
        )

        export_message = gr.Markdown()

    gr.Markdown(
        "AI & ML Analytics Studio | "
        "Original data remains protected. "
        "Healthcare checks are warnings only."
    )

    # ========================================================
    # LOAD EVENTS
    # ========================================================

    load_outputs = [
        original_state,
        processed_state,
        viewed_state,
        history_state,
        overview_table_component,
        cleaning_primary,
        cleaning_columns,
        eda_x,
        eda_y,
        eda_z,
        eda_hue,
        target_column,
        feature_columns,
        prediction_input,
        training_model,
        kpi_rows,
        kpi_columns,
        kpi_numeric,
        kpi_categorical,
        kpi_missing,
        kpi_duplicates,
        kpi_quality,
        status_message,
        healthcare_message,
        cleaning_history_output,
    ]

    load_button.click(
        load_callback,
        inputs=[
            upload
        ],
        outputs=load_outputs
    )

    upload.change(
        load_callback,
        inputs=[
            upload
        ],
        outputs=load_outputs
    )

    # ========================================================
    # VIEWER EVENTS
    # ========================================================

    viewer_outputs = [
        viewer_output,
        viewer_info,
        viewed_state,
    ]

    search_button.click(
        viewer_callback,
        inputs=[
            original_state,
            processed_state,
            viewer_dataset,
            viewer_mode,
            viewer_rows,
            viewer_start,
            viewer_end,
            search_box,
        ],
        outputs=viewer_outputs
    )

    viewer_dataset.change(
        viewer_callback,
        inputs=[
            original_state,
            processed_state,
            viewer_dataset,
            viewer_mode,
            viewer_rows,
            viewer_start,
            viewer_end,
            search_box,
        ],
        outputs=viewer_outputs
    )

    viewer_mode.change(
        viewer_callback,
        inputs=[
            original_state,
            processed_state,
            viewer_dataset,
            viewer_mode,
            viewer_rows,
            viewer_start,
            viewer_end,
            search_box,
        ],
        outputs=viewer_outputs
    )

    clear_search_button.click(
        clear_search_callback,
        inputs=[
            original_state,
            processed_state,
            viewer_dataset,
            viewer_mode,
            viewer_rows,
            viewer_start,
            viewer_end,
        ],
        outputs=[
            search_box,
            viewer_output,
            viewer_info,
            viewed_state,
        ]
    )

    # ========================================================
    # STATISTICS EVENTS
    # ========================================================

    statistics_button.click(
        statistics_callback,
        inputs=[
            processed_state,
            statistics_columns,
            percentile_selector,
        ],
        outputs=[
            statistics_output
        ]
    )

    # ========================================================
    # QUALITY EVENTS
    # ========================================================

    quality_button.click(
        quality_callback,
        inputs=[
            processed_state
        ],
        outputs=[
            quality_output,
            missing_output,
            duplicate_output,
            outlier_output,
            quality_message,
        ]
    )

    # ========================================================
    # CLEANING EVENTS
    # ========================================================

    cleaning_operation.change(
        cleaning_control_callback,
        inputs=[
            cleaning_operation
        ],
        outputs=[
            cleaning_strategy,
            cleaning_value,
            cleaning_threshold,
        ]
    )

    preview_clean_button.click(
        clean_preview_callback,
        inputs=[
            processed_state,
            cleaning_operation,
            cleaning_primary,
            cleaning_columns,
            cleaning_strategy,
            cleaning_value,
            cleaning_threshold,
        ],
        outputs=[
            cleaning_preview_table,
            cleaning_change_summary,
            cleaning_status_table,
            cleaning_message,
        ]
    )

    apply_clean_button.click(
        clean_apply_callback,
        inputs=[
            processed_state,
            history_state,
            cleaning_operation,
            cleaning_primary,
            cleaning_columns,
            cleaning_strategy,
            cleaning_value,
            cleaning_threshold,
        ],
        outputs=[
            processed_state,
            history_state,
            viewer_output,
            cleaning_before,
            cleaning_after,
            cleaning_message,
            kpi_rows,
            kpi_missing,
            kpi_duplicates,
            kpi_quality,
        ]
    )

    reset_button.click(
        reset_callback,
        inputs=[
            original_state
        ],
        outputs=[
            processed_state,
            viewer_output,
            overview_table_component,
            history_state,
            cleaning_message,
            kpi_rows,
            kpi_missing,
            kpi_duplicates,
            kpi_quality,
        ]
    )

    # ========================================================
    # EDA EVENTS
    # ========================================================

    chart_button.click(
        chart_callback,
        inputs=[
            processed_state,
            chart_type,
            eda_x,
            eda_y,
            eda_z,
            eda_hue,
        ],
        outputs=[
            chart_output,
            chart_message,
        ]
    )

    # ========================================================
    # TASK EVENTS
    # ========================================================

    task_type.change(
        task_change_callback,
        inputs=[
            task_type
        ],
        outputs=[
            training_model,
            comparison_models,
        ]
    )

    # ========================================================
    # TARGET EVENTS
    # ========================================================

    target_button.click(
        target_callback,
        inputs=[
            processed_state,
            target_column,
            task_type,
        ],
        outputs=[
            target_output,
            target_message,
        ]
    )

    # ========================================================
    # FEATURE EVENTS
    # ========================================================

    feature_button.click(
        feature_callback,
        inputs=[
            processed_state,
            feature_operation,
            feature_columns,
            feature_k,
            feature_threshold,
        ],
        outputs=[
            processed_state,
            feature_output,
            feature_message,
        ]
    )

    # ========================================================
    # TRAINING EVENTS
    # ========================================================

    train_button.click(
        train_callback,
        inputs=[
            processed_state,
            task_type,
            target_column,
            training_model,
            training_test_size,
            training_scaling,
            training_encoding,
            training_random_state,
        ],
        outputs=[
            pipeline_state,
            X_test_state,
            y_test_state,
            prediction_state,
            metrics_state,
            prediction_input,
            training_message,
        ]
    )

    # ========================================================
    # EVALUATION EVENTS
    # ========================================================

    evaluate_button.click(
        evaluation_callback,
        inputs=[
            pipeline_state,
            X_test_state,
            y_test_state,
            prediction_state,
            task_type,
        ],
        outputs=[
            evaluation_table,
            evaluation_message,
            evaluation_plot,
        ]
    )

    # ========================================================
    # COMPARISON EVENTS
    # ========================================================

    compare_button.click(
        comparison_callback,
        inputs=[
            processed_state,
            task_type,
            target_column,
            comparison_models,
            training_scaling,
            training_encoding,
            training_test_size,
        ],
        outputs=[
            comparison_output,
            comparison_message,
        ]
    )

    cross_validation_button.click(
        cv_callback,
        inputs=[
            processed_state,
            task_type,
            target_column,
            training_model,
            training_scaling,
            training_encoding,
            comparison_folds,
        ],
        outputs=[
            comparison_output,
            comparison_message,
        ]
    )

    # ========================================================
    # TUNING EVENTS
    # ========================================================

    tuning_button.click(
        tuning_callback,
        inputs=[
            processed_state,
            task_type,
            target_column,
            training_model,
            tuning_method,
            training_scaling,
            training_encoding,
            comparison_folds,
        ],
        outputs=[
            tuning_output,
            tuning_message,
        ]
    )

    # ========================================================
    # EXPLAINABILITY EVENTS
    # ========================================================

    native_importance_button.click(
        importance_callback,
        inputs=[
            pipeline_state
        ],
        outputs=[
            importance_output,
            explainability_message,
        ]
    )

    permutation_button.click(
        permutation_callback,
        inputs=[
            pipeline_state,
            X_test_state,
            y_test_state,
            task_type,
        ],
        outputs=[
            importance_output,
            explainability_message,
        ]
    )

    shap_button.click(
        shap_callback,
        inputs=[
            pipeline_state,
            X_test_state,
        ],
        outputs=[
            importance_output,
            explainability_message,
        ]
    )

    # ========================================================
    # PREDICTION EVENTS
    # ========================================================

    predict_button.click(
        prediction_callback,
        inputs=[
            pipeline_state,
            original_state,
            task_type,
            prediction_input,
        ],
        outputs=[
            prediction_result,
            prediction_message,
            probability_output,
        ]
    )

    # ========================================================
    # EXPORT EVENTS
    # ========================================================

    export_button.click(
        export_data,
        inputs=[
            export_type,
            export_format,
            original_state,
            processed_state,
            viewed_state,
            statistics_output,
            quality_output,
            missing_output,
            duplicate_output,
            outlier_output,
            metrics_state,
            comparison_output,
            prediction_result,
            importance_output,
            history_state,
        ],
        outputs=[
            export_download,
            export_message,
        ]
    )

# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    # For Colab, the default creates a public Gradio URL.
    # For another environment:
    # GRADIO_SHARE=false python app.py

    share_value = (
        os.getenv(
            "GRADIO_SHARE",
            "true"
        )
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
        }
    )

    dashboard.launch(
        share=share_value,
        show_error=True
    )