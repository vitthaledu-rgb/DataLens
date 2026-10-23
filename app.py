# ============================================================
# AI & ML ANALYTICS STUDIO
# COMPLETE SINGLE-CELL GOOGLE COLAB APPLICATION
# ============================================================

!pip -q install "gradio==5.50.0" pandas numpy matplotlib seaborn scipy scikit-learn openpyxl xlrd

import os
import re
import tempfile
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import gradio as gr

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV
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
    PolynomialFeatures
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
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.inspection import permutation_importance

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)

from sklearn.svm import SVC, SVR


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# ============================================================
# OPTIONAL LIBRARIES
# ============================================================

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
# APPLICATION STATE
# ============================================================

APP = {
    "original_df": None,
    "processed_df": None,
    "viewed_df": None,
    "filename": "",

    "cleaning_history": [],

    "trained_pipeline": None,
    "trained_model": None,
    "preprocessor": None,

    "feature_names": [],
    "raw_feature_names": [],

    "target_column": None,
    "task_type": None,

    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,

    "predictions": None,
    "probabilities": None,

    "model_results": pd.DataFrame(),
    "evaluation_metrics": pd.DataFrame(),
    "feature_importance": pd.DataFrame(),
    "prediction_df": pd.DataFrame(),

    "viewed_description": "",

    "last_graph": None,
    "last_statistics": pd.DataFrame(),
    "last_quality": pd.DataFrame(),
    "last_missing": pd.DataFrame(),
    "last_outliers": pd.DataFrame(),
    "last_duplicate": pd.DataFrame(),
    "last_tuning": pd.DataFrame(),
    "last_target": pd.DataFrame(),
    "last_feature_result": pd.DataFrame(),
}


# ============================================================
# BASIC HELPERS
# ============================================================

def clear_state():
    for key in list(APP.keys()):

        if key in [
            "model_results",
            "evaluation_metrics",
            "feature_importance",
            "prediction_df",
            "last_statistics",
            "last_quality",
            "last_missing",
            "last_outliers",
            "last_duplicate",
            "last_tuning",
            "last_target",
            "last_feature_result"
        ]:
            APP[key] = pd.DataFrame()

        elif key == "cleaning_history":
            APP[key] = []

        elif key in [
            "original_df",
            "processed_df",
            "viewed_df",
            "trained_pipeline",
            "trained_model",
            "preprocessor",
            "X_train",
            "X_test",
            "y_train",
            "y_test",
            "predictions",
            "probabilities"
        ]:
            APP[key] = None

        elif key in [
            "filename",
            "target_column",
            "task_type",
            "viewed_description"
        ]:
            APP[key] = ""

        elif key in [
            "feature_names",
            "raw_feature_names"
        ]:
            APP[key] = []

        elif key == "last_graph":
            APP[key] = None


def safe_df(df):
    if isinstance(df, pd.DataFrame):
        return df.copy()
    return pd.DataFrame()


def numeric_cols(df):
    if df is None:
        return []
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_cols(df):
    if df is None:
        return []
    return df.select_dtypes(exclude=np.number).columns.tolist()


def total_missing(df):
    if df is None or df.empty:
        return 0
    return int(df.isna().sum().sum())


def duplicate_count(df):
    if df is None or df.empty:
        return 0
    return int(df.duplicated().sum())


def count_mixed_type_columns(df):
    if df is None or df.empty:
        return 0

    count = 0

    for column in df.columns:
        series = df[column].dropna()

        if series.empty:
            continue

        type_count = series.map(
            lambda x: type(x).__name__
        ).nunique()

        if type_count > 1:
            count += 1

    return count


def count_invalid_values(df):
    if df is None or df.empty:
        return 0

    invalid = 0

    numeric = df.select_dtypes(include=np.number)

    if not numeric.empty:
        invalid += int(
            np.isinf(numeric).sum().sum()
        )

    for column in df.columns:

        name = str(column).lower()

        series = df[column]

        numeric_series = pd.to_numeric(
            series,
            errors="coerce"
        )

        if "age" in name:

            invalid += int(
                ((numeric_series < 0) |
                 (numeric_series > 120)).sum()
            )

        elif (
            "blood pressure" in name
            or "blood_pressure" in name
        ):

            invalid += int(
                (numeric_series <= 0).sum()
            )

        elif (
            "heart rate" in name
            or "heartrate" in name
            or "pulse" in name
        ):

            invalid += int(
                (numeric_series <= 0).sum()
            )

        elif "cholesterol" in name:

            invalid += int(
                (numeric_series < 0).sum()
            )

        elif "bmi" in name:

            invalid += int(
                (
                    (numeric_series <= 0)
                    |
                    (numeric_series > 100)
                ).sum()
            )

    return invalid


def outlier_count_total(df):

    if df is None or df.empty:
        return 0

    total = 0

    for column in numeric_cols(df):

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        total += int(
            (
                (series < lower)
                |
                (series > upper)
            ).sum()
        )

    return total


def quality_score(df):

    if df is None or df.empty:
        return 0

    rows, columns = df.shape

    completeness = max(
        0,
        100 -
        (
            total_missing(df)
            /
            max(1, rows * columns)
        ) * 100
    )

    uniqueness = max(
        0,
        100 -
        (
            duplicate_count(df)
            /
            max(1, rows)
        ) * 100
    )

    invalid = count_invalid_values(df)

    validity = max(
        0,
        100 -
        (
            invalid
            /
            max(1, rows * columns)
        ) * 100
    )

    consistency = max(
        0,
        100 -
        (
            count_mixed_type_columns(df)
            /
            max(1, columns)
        ) * 100
    )

    return int(
        round(
            (
                completeness
                + uniqueness
                + validity
                + consistency
            ) / 4
        )
    )


def fmt_num(value):

    try:

        if pd.isna(value):
            return "NA"

        return (
            f"{float(value):,.4f}"
            .rstrip("0")
            .rstrip(".")
        )

    except Exception:

        return str(value)


def file_name_safe(text):

    text = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(text)
    )

    return text[:80] or "dataset"


def ensure_df():

    if (
        APP["processed_df"] is None
        or APP["processed_df"].empty
    ):
        return None, "Load a dataset first."

    return APP["processed_df"], ""


def write_df(df, stem, fmt="CSV"):

    if df is None:
        return None

    df = safe_df(df)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    stem = file_name_safe(stem)

    if fmt == "Excel":

        path = os.path.join(
            tempfile.gettempdir(),
            f"{stem}_{timestamp}.xlsx"
        )

        df.to_excel(
            path,
            index=False
        )

    else:

        path = os.path.join(
            tempfile.gettempdir(),
            f"{stem}_{timestamp}.csv"
        )

        df.to_csv(
            path,
            index=False
        )

    if os.path.exists(path):
        return path

    return None


def metric_snapshot(df):

    return {
        "Rows": int(len(df)),
        "Columns": int(df.shape[1]),
        "Missing Values": total_missing(df),
        "Duplicates": duplicate_count(df),
        "Numeric Columns": len(numeric_cols(df)),
        "Categorical Columns": len(categorical_cols(df)),
        "Outliers": outlier_count_total(df)
    }


def comparison_df(before, after):

    before_metrics = metric_snapshot(before)
    after_metrics = metric_snapshot(after)

    rows = []

    for key in before_metrics:

        rows.append([
            key,
            before_metrics[key],
            after_metrics[key],
            after_metrics[key] -
            before_metrics[key]
        ])

    return pd.DataFrame(
        rows,
        columns=[
            "Metric",
            "Before",
            "After",
            "Change"
        ]
    )


def summary_html(df):

    if df is None or df.empty:

        return """
        <div class="notice">
            No dataset loaded.
        </div>
        """

    score = quality_score(df)

    target = (
        APP.get("target_column")
        or "Not selected"
    )

    return f"""
    <div class="kpi-grid">

        <div class="kpi k1">
            <span>Rows</span>
            <b>{len(df):,}</b>
        </div>

        <div class="kpi k2">
            <span>Columns</span>
            <b>{df.shape[1]:,}</b>
        </div>

        <div class="kpi k3">
            <span>Numeric</span>
            <b>{len(numeric_cols(df))}</b>
        </div>

        <div class="kpi k4">
            <span>Categorical</span>
            <b>{len(categorical_cols(df))}</b>
        </div>

        <div class="kpi k5">
            <span>Missing</span>
            <b>{total_missing(df):,}</b>
        </div>

        <div class="kpi k6">
            <span>Duplicates</span>
            <b>{duplicate_count(df):,}</b>
        </div>

        <div class="kpi k7">
            <span>Quality</span>
            <b>{score}/100</b>
        </div>

        <div class="kpi k8">
            <span>Target</span>
            <b>{str(target)[:18]}</b>
        </div>

    </div>
    """


# ============================================================
# DATASET REPORTS
# ============================================================

def dataset_info(df):

    if df is None or df.empty:
        return pd.DataFrame()

    rows = []

    for column in df.columns:

        series = df[column]

        rows.append({
            "Column": column,
            "Data Type": str(series.dtype),
            "Non-Null": int(series.notna().sum()),
            "Missing": int(series.isna().sum()),
            "Missing %": round(
                series.isna().mean() * 100,
                2
            ),
            "Unique": int(
                series.nunique(
                    dropna=True
                )
            ),
            "Sample": ", ".join(
                map(
                    str,
                    series.dropna()
                    .head(3)
                    .tolist()
                )
            )
        })

    return pd.DataFrame(rows)


def missing_report(df):

    if df is None:
        return pd.DataFrame()

    report = pd.DataFrame({
        "Column": df.columns,
        "Missing": df.isna().sum().values,
        "Missing %": (
            df.isna().mean().values * 100
        ).round(2)
    })

    return report.sort_values(
        ["Missing", "Column"],
        ascending=[False, True]
    ).reset_index(drop=True)


def duplicate_report(df):

    if df is None:
        return pd.DataFrame()

    mask = df.duplicated(
        keep=False
    )

    if not mask.any():

        return pd.DataFrame({
            "Message": [
                "No duplicate rows found."
            ]
        })

    duplicated = df.loc[mask].copy()

    duplicated.insert(
        0,
        "Duplicate Group",
        duplicated.groupby(
            list(df.columns),
            dropna=False
        ).ngroup() + 1
    )

    return duplicated


def unique_report(df):

    if df is None:
        return pd.DataFrame()

    return pd.DataFrame({
        "Column": df.columns,
        "Unique Values": [
            df[column].nunique(
                dropna=True
            )
            for column in df.columns
        ],
        "Unique %": [
            round(
                df[column].nunique(
                    dropna=True
                )
                /
                max(1, len(df))
                * 100,
                2
            )
            for column in df.columns
        ]
    })


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(file_obj):

    try:

        if file_obj is None:

            return (
                "Please upload a CSV, XLSX, or XLS file.",
                "<div class='notice'>No dataset loaded.</div>",
                pd.DataFrame(),
                pd.DataFrame(),
                [],
                [],
                [],
                [],
                [],
                [],
                "No dataset"
            )

        if isinstance(file_obj, str):

            path = file_obj

        else:

            path = getattr(
                file_obj,
                "name",
                str(file_obj)
            )

        extension = os.path.splitext(
            path
        )[1].lower()

        if extension == ".csv":

            df = pd.read_csv(path)

        elif extension == ".xlsx":

            df = pd.read_excel(
                path,
                engine="openpyxl"
            )

        elif extension == ".xls":

            df = pd.read_excel(
                path,
                engine="xlrd"
            )

        else:

            raise ValueError(
                "Unsupported file. Use CSV, XLSX, or XLS."
            )

        if (
            df.empty
            or df.shape[1] == 0
        ):

            raise ValueError(
                "The uploaded dataset is empty."
            )

        clear_state()

        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        APP["original_df"] = df.copy()
        APP["processed_df"] = df.copy()
        APP["filename"] = os.path.basename(path)

        info = dataset_info(df)

        missing = missing_report(df)
        duplicates = duplicate_report(df)
        statistics = statistics_report(df)

        APP["last_missing"] = missing
        APP["last_duplicate"] = duplicates
        APP["last_statistics"] = statistics

        columns = df.columns.tolist()
        numerical = numeric_cols(df)

        message = (
            f"Loaded: {APP['filename']} | "
            f"{len(df):,} rows | "
            f"{df.shape[1]:,} columns"
        )

        return (
            message,
            summary_html(df),
            df.head(50),
            info,
            numerical,
            numerical,
            columns,
            columns,
            columns,
            columns,
            "Showing first 50 rows"
        )

    except Exception as error:

        return (
            f"Could not load dataset: {error}",
            "<div class='error'>Dataset could not be loaded.</div>",
            pd.DataFrame(),
            pd.DataFrame(),
            [],
            [],
            [],
            [],
            [],
            [],
            "No dataset"
        )


# ============================================================
# DATASET VIEWER
# ============================================================

def update_view_controls(view):

    if view in [
        "Head",
        "Tail",
        "Random"
    ]:

        return (
            gr.update(
                visible=True
            ),
            gr.update(
                visible=False
            ),
            "Rows"
        )

    if view == "Custom Range":

        return (
            gr.update(
                visible=False
            ),
            gr.update(
                visible=True
            ),
            "Custom range"
        )

    return (
        gr.update(
            visible=False
        ),
        gr.update(
            visible=False
        ),
        "Entire dataset"
    )


def make_view(
    dataset_choice,
    view,
    n_rows,
    start,
    end,
    search
):

    try:

        if dataset_choice == "Original":

            df = APP["original_df"]

        else:

            df = APP["processed_df"]

        if df is None:

            return (
                pd.DataFrame(),
                "Load a dataset first."
            )

        work = df.copy()

        if search and str(search).strip():

            term = str(
                search
            ).strip().lower()

            mask = (
                work.astype(str)
                .apply(
                    lambda column:
                    column.str.lower()
                    .str.contains(
                        re.escape(term),
                        na=False
                    )
                )
                .any(axis=1)
            )

            work = work.loc[mask]

        total = len(work)

        n = int(
            n_rows or 50
        )

        if view == "Head":

            shown = work.head(n)

            description = (
                f"Showing first "
                f"{len(shown):,} rows "
                f"of {total:,}"
            )

        elif view == "Tail":

            shown = work.tail(n)

            description = (
                f"Showing last "
                f"{len(shown):,} rows "
                f"of {total:,}"
            )

        elif view == "Random":

            sample_size = min(
                n,
                total
            )

            if sample_size:

                shown = work.sample(
                    sample_size,
                    random_state=42
                )

            else:

                shown = work.head(0)

            description = (
                f"Showing random "
                f"{len(shown):,} rows "
                f"from {total:,}"
            )

        elif view == "Entire Dataset":

            shown = work.head(5000)

            description = (
                f"Showing entire dataset — "
                f"{total:,} rows"
            )

            if total > 5000:

                description += (
                    " (display capped at "
                    "5,000 for responsiveness)"
                )

        else:

            start_value = max(
                0,
                int(start or 0)
            )

            end_value = min(
                total,
                int(end or 147)
            )

            if end_value < start_value:

                end_value = start_value

            shown = work.iloc[
                start_value:end_value
            ]

            description = (
                f"Showing rows "
                f"{start_value + 1:,}–"
                f"{end_value:,} "
                f"of {total:,}"
            )

        APP["viewed_df"] = shown.copy()
        APP["viewed_description"] = description

        final_description = (
            f"{description} | "
            f"Displayed: {len(shown):,} | "
            f"Columns: {shown.shape[1]}"
        )

        return (
            shown,
            final_description
        )

    except Exception as error:

        return (
            pd.DataFrame(),
            f"Viewer error: {error}"
        )


def clear_search(
    dataset_choice,
    view,
    n_rows,
    start,
    end
):

    return make_view(
        dataset_choice,
        view,
        n_rows,
        start,
        end,
        ""
    )


# ============================================================
# STATISTICS
# ============================================================

def statistics_report(
    df,
    selected=None,
    percentile=50
):

    if df is None or df.empty:
        return pd.DataFrame()

    columns = (
        selected
        if selected
        else numeric_cols(df)
    )

    columns = [
        column
        for column in columns
        if (
            column in df.columns
            and
            pd.api.types.is_numeric_dtype(
                df[column]
            )
        )
    ]

    if not columns:
        return pd.DataFrame()

    percentile = float(
        percentile or 50
    )

    rows = []

    for column in columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if series.empty:
            continue

        mode = series.mode()

        q1 = series.quantile(.25)
        median = series.quantile(.50)
        q3 = series.quantile(.75)

        rows.append({

            "Column": column,

            "Count": len(series),

            "Mean": series.mean(),

            "Median": median,

            "Mode": (
                mode.iloc[0]
                if not mode.empty
                else np.nan
            ),

            "Min": series.min(),

            "Max": series.max(),

            "Range":
                series.max() -
                series.min(),

            "Variance":
                series.var(),

            "Std Dev":
                series.std(),

            "Q1": q1,

            "Q3": q3,

            "IQR":
                q3 - q1,

            f"P{int(percentile)}":
                series.quantile(
                    percentile / 100
                ),

            "P10":
                series.quantile(.10),

            "P90":
                series.quantile(.90),

            "Skewness":
                series.skew(),

            "Kurtosis":
                series.kurt(),

            "MAD":
                np.median(
                    np.abs(
                        series -
                        series.median()
                    )
                ),

            "Five Number":
                (
                    f"{series.min():.4g} | "
                    f"{q1:.4g} | "
                    f"{median:.4g} | "
                    f"{q3:.4g} | "
                    f"{series.max():.4g}"
                )
        })

    return pd.DataFrame(
        rows
    ).round(5)


def run_statistics(
    columns,
    percentile
):

    df, error = ensure_df()

    if error:

        return (
            pd.DataFrame(),
            error
        )

    result = statistics_report(
        df,
        columns,
        percentile
    )

    APP["last_statistics"] = result

    if result.empty:

        return (
            result,
            "Select at least one numerical column."
        )

    return (
        result,
        f"Statistics calculated for "
        f"{len(result)} column(s)."
    )


# ============================================================
# HEALTHCARE VALIDATION
# ============================================================

def healthcare_validation(df):

    rows = []

    for column in df.columns:

        name = str(
            column
        ).lower()

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        mask = pd.Series(
            False,
            index=df.index
        )

        rule = None

        if "age" in name:

            mask = (
                (series < 0)
                |
                (series > 120)
            )

            rule = "Age outside 0–120"

        elif (
            "blood pressure" in name
            or
            "blood_pressure" in name
        ):

            mask = series <= 0

            rule = "Blood pressure <= 0"

        elif (
            "heart rate" in name
            or
            "heartrate" in name
            or
            "pulse" in name
        ):

            mask = series <= 0

            rule = "Heart rate <= 0"

        elif "cholesterol" in name:

            mask = series < 0

            rule = "Cholesterol < 0"

        elif "bmi" in name:

            mask = (
                (series <= 0)
                |
                (series > 100)
            )

            rule = "BMI <= 0 or > 100"

        if rule and mask.any():

            rows.append([
                column,
                rule,
                int(mask.sum()),
                (
                    "Warning only; records are "
                    "not deleted automatically."
                )
            ])

    if not rows:

        return pd.DataFrame({
            "Status": [
                "No healthcare-specific "
                "suspicious values detected."
            ]
        })

    return pd.DataFrame(
        rows,
        columns=[
            "Column",
            "Rule",
            "Suspicious Values",
            "Action"
        ]
    )


# ============================================================
# DATA QUALITY CENTER
# ============================================================

def quality_analysis():

    df, error = ensure_df()

    if error:

        return (
            pd.DataFrame(),
            "<div class='notice'>Load a dataset first.</div>",
            pd.DataFrame(),
            pd.DataFrame()
        )

    rows = []

    for column in df.columns:

        series = df[column]

        issues = []

        if series.isna().any():
            issues.append("missing")

        if series.nunique(
            dropna=True
        ) <= 1:
            issues.append("constant")

        if series.nunique(
            dropna=True
        ) <= max(
            1,
            int(len(df) * .01)
        ):
            issues.append("near-constant")

        if series.notna().sum() == 0:
            issues.append("empty")

        if series.dtype == "object":

            converted = pd.to_numeric(
                series,
                errors="coerce"
            )

            if (
                converted.notna().sum() > 0
                and
                converted.notna().mean() >= .8
            ):

                issues.append(
                    "numeric-looking text"
                )

        if pd.api.types.is_numeric_dtype(
            series
        ):

            if np.isinf(series).any():
                issues.append("infinite")

            if len(series.dropna()) >= 4:

                q1 = series.quantile(.25)
                q3 = series.quantile(.75)

                iqr = q3 - q1

                if iqr != 0:

                    outliers = int(
                        (
                            (
                                series <
                                q1 - 1.5 * iqr
                            )
                            |
                            (
                                series >
                                q3 + 1.5 * iqr
                            )
                        ).sum()
                    )

                else:

                    outliers = 0

            else:

                outliers = 0

        else:

            outliers = 0

        high_cardinality = (
            series.nunique(
                dropna=True
            )
            >
            max(
                50,
                int(len(df) * .5)
            )
            and
            not pd.api.types.is_numeric_dtype(
                series
            )
        )

        if high_cardinality:

            issues.append(
                "high-cardinality"
            )

        rows.append([
            column,
            str(series.dtype),
            int(series.isna().sum()),
            int(
                series.nunique(
                    dropna=True
                )
            ),
            outliers,
            (
                ", ".join(issues)
                if issues
                else "No major issue"
            )
        ])

    report = pd.DataFrame(
        rows,
        columns=[
            "Column",
            "Type",
            "Missing",
            "Unique",
            "Outliers",
            "Detected Issues"
        ]
    )

    APP["last_quality"] = report

    score = quality_score(df)

    healthcare = healthcare_validation(
        df
    )

    duplicate_columns = []

    seen = {}

    for index, column in enumerate(
        df.columns
    ):

        key = str(
            column
        ).strip().lower()

        if key in seen:

            duplicate_columns.append(
                str(column)
            )

        else:

            seen[key] = index

    html = f"""
    <div class="quality-box">

        <div class="quality-number">
            {score}/100
        </div>

        <div>
            <b>Data Quality Score</b>
            <br>
            Dashboard heuristic based on
            completeness, consistency,
            validity and uniqueness.
            <br>
            Duplicate column names detected:
            {len(duplicate_columns)}
        </div>

    </div>
    """

    return (
        report,
        html,
        healthcare,
        unique_report(df)
    )


# ============================================================
# CLEANING OPERATIONS
# ============================================================

def apply_clean_operation(
    df,
    operation,
    apply_to,
    column,
    constant,
    replace_from,
    replace_to,
    threshold,
    z_threshold,
    dry_run=False
):

    work = df.copy()

    if apply_to == "Entire Dataset":

        columns = work.columns.tolist()

    else:

        columns = (
            [column]
            if column in work.columns
            else []
        )

    if operation == "Drop rows":

        before = len(work)

        work = (
            work
            .dropna()
            .reset_index(drop=True)
        )

        return (
            work,
            f"Dropped "
            f"{before - len(work):,} "
            f"rows containing missing values."
        )

    if operation == "Drop columns":

        targets = (
            columns
            or work.columns.tolist()
        )

        work = work.drop(
            columns=targets,
            errors="ignore"
        )

        return (
            work,
            f"Dropped {len(targets)} column(s)."
        )

    if operation == "Drop columns above selected %":

        percentage = float(
            threshold or 50
        )

        missing_percentage = (
            work.isna().mean() * 100
        )

        drop_columns = (
            missing_percentage[
                missing_percentage >= percentage
            ]
            .index
            .tolist()
        )

        work = work.drop(
            columns=drop_columns,
            errors="ignore"
        )

        return (
            work,
            f"Dropped "
            f"{len(drop_columns)} column(s) "
            f"with missingness >= "
            f"{percentage:g}%."
        )

    if operation == "Mean imputation":

        targets = (
            columns
            or numeric_cols(work)
        )

        for col in targets:

            if (
                col in work.columns
                and
                pd.api.types.is_numeric_dtype(
                    work[col]
                )
            ):

                work[col] = (
                    work[col]
                    .fillna(
                        work[col].mean()
                    )
                )

        return (
            work,
            f"Mean imputation applied to "
            f"{len(targets)} column(s)."
        )

    if operation == "Median imputation":

        targets = (
            columns
            or numeric_cols(work)
        )

        for col in targets:

            if (
                col in work.columns
                and
                pd.api.types.is_numeric_dtype(
                    work[col]
                )
            ):

                work[col] = (
                    work[col]
                    .fillna(
                        work[col].median()
                    )
                )

        return (
            work,
            f"Median imputation applied to "
            f"{len(targets)} column(s)."
        )

    if operation == "Mode imputation":

        targets = (
            columns
            or
            work.columns.tolist()
        )

        for col in targets:

            if col in work.columns:

                mode = work[col].mode(
                    dropna=True
                )

                if not mode.empty:

                    work[col] = (
                        work[col]
                        .fillna(
                            mode.iloc[0]
                        )
                    )

        return (
            work,
            f"Mode imputation applied to "
            f"{len(targets)} column(s)."
        )

    if operation == "Constant value":

        targets = (
            columns
            or
            work.columns.tolist()
        )

        value = (
            constant
            if constant != ""
            else "0"
        )

        for col in targets:

            if col not in work.columns:
                continue

            try:

                if pd.api.types.is_numeric_dtype(
                    work[col]
                ):

                    work[col] = (
                        work[col]
                        .fillna(
                            float(value)
                        )
                    )

                else:

                    work[col] = (
                        work[col]
                        .fillna(
                            str(value)
                        )
                    )

            except Exception:

                work[col] = (
                    work[col]
                    .fillna(
                        str(value)
                    )
                )

        return (
            work,
            f"Missing values replaced "
            f"with {value!r}."
        )

    if operation == "Forward fill":

        targets = (
            columns
            or
            work.columns.tolist()
        )

        for col in targets:

            if col in work.columns:

                work[col] = (
                    work[col]
                    .ffill()
                )

        return (
            work,
            f"Forward fill applied to "
            f"{len(targets)} column(s)."
        )

    if operation == "Backward fill":

        targets = (
            columns
            or
            work.columns.tolist()
        )

        for col in targets:

            if col in work.columns:

                work[col] = (
                    work[col]
                    .bfill()
                )

        return (
            work,
            f"Backward fill applied to "
            f"{len(targets)} column(s)."
        )

    if operation == "Interpolation":

        targets = (
            columns
            or
            numeric_cols(work)
        )

        for col in targets:

            if (
                col in work.columns
                and
                pd.api.types.is_numeric_dtype(
                    work[col]
                )
            ):

                work[col] = (
                    work[col]
                    .interpolate(
                        limit_direction="both"
                    )
                )

        return (
            work,
            f"Interpolation applied to "
            f"{len(targets)} numeric column(s)."
        )

    if operation == "Remove exact duplicates":

        before = len(work)

        work = (
            work
            .drop_duplicates()
            .reset_index(drop=True)
        )

        return (
            work,
            f"Removed "
            f"{before - len(work):,} "
            f"duplicate row(s)."
        )

    if operation in [
        "Strip whitespace",
        "Lowercase",
        "Uppercase",
        "Title case"
    ]:

        targets = (
            columns
            or
            categorical_cols(work)
        )

        for col in targets:

            if col not in work.columns:
                continue

            work[col] = (
                work[col]
                .astype("string")
                .str.strip()
            )

            if operation == "Lowercase":

                work[col] = (
                    work[col]
                    .str.lower()
                )

            elif operation == "Uppercase":

                work[col] = (
                    work[col]
                    .str.upper()
                )

            elif operation == "Title case":

                work[col] = (
                    work[col]
                    .str.title()
                )

        return (
            work,
            f"{operation} applied to "
            f"{len(targets)} column(s)."
        )

    if operation == "Replace values":

        targets = (
            columns
            or
            work.columns.tolist()
        )

        for col in targets:

            if col in work.columns:

                work[col] = (
                    work[col]
                    .replace(
                        str(replace_from),
                        replace_to
                    )
                )

                work[col] = (
                    work[col]
                    .replace(
                        replace_from,
                        replace_to
                    )
                )

        return (
            work,
            f"Replaced "
            f"{replace_from!r} "
            f"with {replace_to!r}."
        )

    if operation == "Convert numeric":

        targets = (
            columns
            or
            work.columns.tolist()
        )

        for col in targets:

            if col in work.columns:

                work[col] = pd.to_numeric(
                    work[col],
                    errors="coerce"
                )

        return (
            work,
            f"Converted "
            f"{len(targets)} column(s) "
            f"to numeric where possible."
        )

    if operation == "Convert category":

        targets = (
            columns
            or
            categorical_cols(work)
        )

        for col in targets:

            if col in work.columns:

                work[col] = (
                    work[col]
                    .astype("category")
                )

        return (
            work,
            f"Converted "
            f"{len(targets)} column(s) "
            f"to category."
        )

    if operation == "Convert date":

        targets = (
            columns
            or
            work.columns.tolist()
        )

        for col in targets:

            if col in work.columns:

                work[col] = pd.to_datetime(
                    work[col],
                    errors="coerce"
                )

        return (
            work,
            f"Converted "
            f"{len(targets)} column(s) "
            f"to date/time."
        )

    if operation == "Replace negative numeric values":

        targets = (
            columns
            or
            numeric_cols(work)
        )

        count = 0

        for col in targets:

            if (
                col in work.columns
                and
                pd.api.types.is_numeric_dtype(
                    work[col]
                )
            ):

                mask = work[col] < 0

                count += int(
                    mask.sum()
                )

                work.loc[
                    mask,
                    col
                ] = np.nan

        return (
            work,
            f"Replaced "
            f"{count:,} negative numeric "
            f"value(s) with missing values."
        )

    if operation == "Replace infinite values":

        targets = (
            columns
            or
            numeric_cols(work)
        )

        count = 0

        for col in targets:

            if (
                col in work.columns
                and
                pd.api.types.is_numeric_dtype(
                    work[col]
                )
            ):

                count += int(
                    np.isinf(
                        work[col]
                    ).sum()
                )

                work[col] = (
                    work[col]
                    .replace(
                        [np.inf, -np.inf],
                        np.nan
                    )
                )

        return (
            work,
            f"Replaced "
            f"{count:,} infinite value(s) "
            f"with missing values."
        )

    if operation == "Replace empty strings":

        targets = (
            columns
            or
            work.columns.tolist()
        )

        count = 0

        for col in targets:

            if col in work.columns:

                mask = (
                    work[col]
                    .astype(str)
                    .str.strip()
                    .eq("")
                )

                count += int(
                    mask.sum()
                )

                work.loc[
                    mask,
                    col
                ] = np.nan

        return (
            work,
            f"Replaced "
            f"{count:,} empty string(s) "
            f"with missing values."
        )

    if operation == "Replace common null strings":

        targets = (
            columns
            or
            work.columns.tolist()
        )

        null_values = {
            "na",
            "n/a",
            "null",
            "none",
            "nan",
            "missing",
            "unknown",
            "-",
            "?"
        }

        count = 0

        for col in targets:

            if col in work.columns:

                mask = (
                    work[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(null_values)
                )

                count += int(
                    mask.sum()
                )

                work.loc[
                    mask,
                    col
                ] = np.nan

        return (
            work,
            f"Replaced "
            f"{count:,} common null marker(s) "
            f"with missing values."
        )

    if operation in [
        "IQR remove",
        "IQR cap"
    ]:

        targets = (
            columns
            or
            numeric_cols(work)
        )

        count = 0

        keep = pd.Series(
            True,
            index=work.index
        )

        for col in targets:

            if (
                col not in work.columns
                or
                not pd.api.types.is_numeric_dtype(
                    work[col]
                )
            ):
                continue

            series = work[col]

            q1 = series.quantile(.25)
            q3 = series.quantile(.75)

            iqr = q3 - q1

            if (
                iqr == 0
                or
                pd.isna(iqr)
            ):
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            mask = (
                (series < lower)
                |
                (series > upper)
            )

            count += int(
                mask.sum()
            )

            if operation == "IQR remove":

                keep &= (
                    ~mask.fillna(False)
                )

            else:

                work[col] = series.clip(
                    lower,
                    upper
                )

        if operation == "IQR remove":

            work = (
                work.loc[keep]
                .reset_index(drop=True)
            )

        return (
            work,
            f"{operation} affected "
            f"{count:,} outlier value(s)."
        )

    if operation in [
        "Z-score remove",
        "Z-score cap"
    ]:

        targets = (
            columns
            or
            numeric_cols(work)
        )

        threshold_value = float(
            z_threshold or 3
        )

        count = 0

        keep = pd.Series(
            True,
            index=work.index
        )

        for col in targets:

            if (
                col not in work.columns
                or
                not pd.api.types.is_numeric_dtype(
                    work[col]
                )
            ):
                continue

            series = work[col]

            standard_deviation = (
                series.std()
            )

            if (
                standard_deviation == 0
                or
                pd.isna(
                    standard_deviation
                )
            ):
                continue

            zscore = (
                (
                    series -
                    series.mean()
                )
                /
                standard_deviation
            ).abs()

            mask = (
                zscore >
                threshold_value
            )

            count += int(
                mask.sum()
            )

            if operation == "Z-score remove":

                keep &= (
                    ~mask.fillna(False)
                )

            else:

                mean = series.mean()

                work[col] = series.clip(
                    mean -
                    threshold_value *
                    standard_deviation,

                    mean +
                    threshold_value *
                    standard_deviation
                )

        if operation == "Z-score remove":

            work = (
                work.loc[keep]
                .reset_index(drop=True)
            )

        return (
            work,
            f"{operation} affected "
            f"{count:,} value(s) using "
            f"z={threshold_value:g}."
        )

    return (
        work,
        "No operation selected."
    )


# ============================================================
# CLEANING PREVIEW
# ============================================================

def cleaning_before_preview(
    operation,
    apply_to,
    column,
    constant,
    replace_from,
    replace_to,
    threshold,
    z_threshold
):

    try:

        df = APP["processed_df"]

        if df is None:

            return (
                pd.DataFrame(),
                pd.DataFrame(),
                "Load a dataset first.",
                pd.DataFrame()
            )

        before = df.copy()

        after, detail = apply_clean_operation(
            before,
            operation,
            apply_to,
            column,
            constant,
            replace_from,
            replace_to,
            threshold,
            z_threshold,
            dry_run=True
        )

        comparison = comparison_df(
            before,
            after
        )

        return (
            before.head(50),
            after.head(50),
            detail,
            comparison
        )

    except Exception as error:

        return (
            pd.DataFrame(),
            pd.DataFrame(),
            f"Preview error: {error}",
            pd.DataFrame()
        )


def apply_cleaning(
    operation,
    apply_to,
    column,
    constant,
    replace_from,
    replace_to,
    threshold,
    z_threshold
):

    try:

        if APP["processed_df"] is None:

            return (
                pd.DataFrame(),
                "<div class='notice'>Load a dataset first.</div>",
                pd.DataFrame(),
                pd.DataFrame(
                    APP["cleaning_history"]
                ),
                "No change."
            )

        before = (
            APP["processed_df"]
            .copy()
        )

        after, detail = apply_clean_operation(
            before,
            operation,
            apply_to,
            column,
            constant,
            replace_from,
            replace_to,
            threshold,
            z_threshold
        )

        APP["processed_df"] = (
            after.copy()
        )

        comparison = comparison_df(
            before,
            after
        )

        step = (
            len(
                APP["cleaning_history"]
            ) + 1
        )

        APP["cleaning_history"].append({

            "Step": step,

            "Operation":
                operation,

            "Column":
                (
                    column
                    if apply_to ==
                    "Selected Column"
                    else
                    "Entire Dataset"
                ),

            "Before":
                (
                    f"{len(before):,} rows / "
                    f"{total_missing(before):,} missing"
                ),

            "After":
                (
                    f"{len(after):,} rows / "
                    f"{total_missing(after):,} missing"
                ),

            "Status":
                "Applied"
        })

        history = pd.DataFrame(
            APP["cleaning_history"]
        )

        status = (
            f"Applied: {detail}"
        )

        return (
            after.head(50),
            summary_html(after),
            comparison,
            history,
            status
        )

    except Exception as error:

        return (
            pd.DataFrame(),
            "<div class='error'>Cleaning failed.</div>",
            pd.DataFrame(),
            pd.DataFrame(
                APP["cleaning_history"]
            ),
            f"Cleaning failed: {error}"
        )


def reset_cleaning():

    try:

        if APP["original_df"] is None:

            return (
                pd.DataFrame(),
                "<div class='notice'>Nothing to reset.</div>",
                pd.DataFrame(),
                pd.DataFrame(),
                "No original dataset loaded."
            )

        APP["processed_df"] = (
            APP["original_df"].copy()
        )

        APP["cleaning_history"] = []

        return (
            APP["processed_df"].head(50),
            summary_html(
                APP["processed_df"]
            ),
            pd.DataFrame(),
            pd.DataFrame(),
            "Processed dataset restored to the original dataset."
        )

    except Exception as error:

        return (
            pd.DataFrame(),
            "<div class='error'>Reset failed.</div>",
            pd.DataFrame(),
            pd.DataFrame(),
            f"Reset failed: {error}"
        )


# ============================================================
# CLEANING UI STATE
# ============================================================

def clean_ui_state(
    operation,
    apply_to
):

    df = APP["processed_df"]

    columns = (
        df.columns.tolist()
        if df is not None
        else []
    )

    numerical = (
        numeric_cols(df)
        if df is not None
        else []
    )

    needs_column = operation not in [
        "Drop rows",
        "Drop columns",
        "Drop columns above selected %",
        "Remove exact duplicates",
        "Replace negative numeric values",
        "Replace infinite values",
        "Replace empty strings",
        "Replace common null strings"
    ]

    if operation in [
        "Mean imputation",
        "Median imputation",
        "IQR remove",
        "IQR cap",
        "Z-score remove",
        "Z-score cap",
        "Convert numeric"
    ]:

        choices = numerical

    else:

        choices = columns

    return (
        gr.update(
            choices=choices,
            visible=(
                apply_to ==
                "Selected Column"
                or
                needs_column
            )
        ),

        gr.update(
            visible=(
                operation ==
                "Constant value"
            )
        ),

        gr.update(
            visible=(
                operation ==
                "Replace values"
            )
        ),

        gr.update(
            visible=(
                operation ==
                "Replace values"
            )
        ),

        gr.update(
            visible=(
                operation ==
                "Drop columns above selected %"
            )
        ),

        gr.update(
            visible=(
                "Z-score"
                in operation
            )
        )
    )


# ============================================================
# OUTLIERS
# ============================================================

def outlier_report(
    method="IQR",
    z=3
):

    df, error = ensure_df()

    if error:

        return (
            pd.DataFrame(),
            error
        )

    rows = []

    for column in numeric_cols(df):

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if method == "IQR":

            q1 = series.quantile(.25)
            q3 = series.quantile(.75)

            iqr = q3 - q1

            if (
                iqr == 0
                or
                pd.isna(iqr)
            ):

                count = 0
                lower = np.nan
                upper = np.nan

            else:

                lower = (
                    q1 -
                    1.5 * iqr
                )

                upper = (
                    q3 +
                    1.5 * iqr
                )

                count = int(
                    (
                        (series < lower)
                        |
                        (series > upper)
                    ).sum()
                )

        else:

            standard_deviation = (
                series.std()
            )

            if (
                standard_deviation == 0
                or
                pd.isna(
                    standard_deviation
                )
            ):

                count = 0
                lower = np.nan
                upper = np.nan

            else:

                threshold = float(z)

                lower = (
                    series.mean()
                    -
                    threshold *
                    standard_deviation
                )

                upper = (
                    series.mean()
                    +
                    threshold *
                    standard_deviation
                )

                count = int(
                    (
                        (
                            (
                                series -
                                series.mean()
                            )
                            /
                            standard_deviation
                        ).abs()
                        >
                        threshold
                    ).sum()
                )

        rows.append([
            column,
            count,
            lower,
            upper
        ])

    result = pd.DataFrame(
        rows,
        columns=[
            "Column",
            "Outliers",
            "Lower Bound",
            "Upper Bound"
        ]
    )

    result = result.sort_values(
        "Outliers",
        ascending=False
    )

    APP["last_outliers"] = result

    total = (
        int(result["Outliers"].sum())
        if not result.empty
        else 0
    )

    return (
        result,
        f"{total:,} total outlier observations using {method}."
    )


# ============================================================
# EDA CONTROLS
# ============================================================

def graph_controls(graph):

    x_visible = graph not in [
        "Correlation",
        "Covariance",
        "Missing"
    ]

    y_visible = graph in [
        "Scatter",
        "Line",
        "Area",
        "Bar",
        "Count",
        "Time Series",
        "3D Scatter"
    ]

    z_visible = (
        graph ==
        "3D Scatter"
    )

    hue_visible = graph in [
        "Scatter",
        "Line",
        "Area",
        "Bar",
        "Count",
        "Box",
        "Violin",
        "KDE",
        "Time Series"
    ]

    return (
        gr.update(
            visible=x_visible
        ),

        gr.update(
            visible=y_visible
        ),

        gr.update(
            visible=z_visible
        ),

        gr.update(
            visible=hue_visible
        )
    )


# ============================================================
# EDA PLOTTING
# ============================================================

def plot_eda(
    graph,
    x,
    y,
    z,
    hue
):

    try:

        df, error = ensure_df()

        if error:

            return (
                None,
                error
            )

        figure = None

        if graph == "Histogram":

            if x not in df.columns:

                raise ValueError(
                    "Select a column."
                )

            figure, axis = plt.subplots(
                figsize=(8, 5)
            )

            sns.histplot(
                df[x].dropna(),
                kde=False,
                ax=axis
            )

            axis.set_title(
                f"Histogram: {x}"
            )

        elif graph == "KDE":

            if (
                x not in df.columns
                or
                not pd.api.types.is_numeric_dtype(
                    df[x]
                )
            ):

                raise ValueError(
                    "KDE requires a numerical X column."
                )

            figure, axis = plt.subplots(
                figsize=(8, 5)
            )

            sns.kdeplot(
                df[x].dropna(),
                fill=True,
                ax=axis
            )

            axis.set_title(
                f"KDE: {x}"
            )

        elif graph == "Box":

            if x not in df.columns:

                raise ValueError(
                    "Select X."
                )

            figure, axis = plt.subplots(
                figsize=(8, 5)
            )

            sns.boxplot(
                data=df,
                y=x,
                hue=(
                    hue
                    if hue in df.columns
                    else None
                ),
                ax=axis
            )

            axis.set_title(
                f"Box Plot: {x}"
            )

        elif graph == "Violin":

            if x not in df.columns:

                raise ValueError(
                    "Select X."
                )

            figure, axis = plt.subplots(
                figsize=(8, 5)
            )

            sns.violinplot(
                data=df,
                y=x,
                hue=(
                    hue
                    if hue in df.columns
                    else None
                ),
                ax=axis
            )

            axis.set_title(
                f"Violin Plot: {x}"
            )

        elif graph == "Count":

            if x not in df.columns:

                raise ValueError(
                    "Select a categorical column."
                )

            figure, axis = plt.subplots(
                figsize=(8, 5)
            )

            order = (
                df[x]
                .value_counts()
                .head(20)
                .index
            )

            sns.countplot(
                data=df,
                x=x,
                order=order,
                ax=axis
            )

            axis.tick_params(
                axis="x",
                rotation=45
            )

            axis.set_title(
                f"Count Plot: {x}"
            )

        elif graph == "Bar":

            if (
                x not in df.columns
                or
                y not in df.columns
            ):

                raise ValueError(
                    "Bar chart requires X and Y."
                )

            grouped = (
                df
                .groupby(
                    x,
                    dropna=False
                )[y]
                .mean()
                .sort_values(
                    ascending=False
                )
                .head(20)
            )

            figure, axis = plt.subplots(
                figsize=(8, 5)
            )

            grouped.plot(
                kind="bar",
                ax=axis
            )

            axis.set_title(
                f"Bar Chart: mean {y} by {x}"
            )

            axis.tick_params(
                axis="x",
                rotation=45
            )

        elif graph == "Pie":

            if x not in df.columns:

                raise ValueError(
                    "Select a categorical column."
                )

            values = (
                df[x]
                .value_counts()
                .head(10)
            )

            figure, axis = plt.subplots(
                figsize=(7, 7)
            )

            axis.pie(
                values.values,
                labels=values.index.astype(str),
                autopct="%1.1f%%"
            )

            axis.set_title(
                f"Pie Chart: {x}"
            )

        elif graph == "Scatter":

            if (
                x not in df.columns
                or
                y not in df.columns
            ):

                raise ValueError(
                    "Scatter requires X and Y."
                )

            figure, axis = plt.subplots(
                figsize=(8, 5)
            )

            sns.scatterplot(
                data=df,
                x=x,
                y=y,
                hue=(
                    hue
                    if hue in df.columns
                    else None
                ),
                ax=axis
            )

            axis.set_title(
                f"Scatter: {x} vs {y}"
            )

        elif graph == "Time Series":

            if (
                x not in df.columns
                or
                y not in df.columns
            ):

                raise ValueError(
                    "Time Series requires a date/time X column and numerical Y column."
                )

            data = df[
                [x, y]
            ].copy()

            data[x] = pd.to_datetime(
                data[x],
                errors="coerce"
            )

            data[y] = pd.to_numeric(
                data[y],
                errors="coerce"
            )

            data = (
                data
                .dropna()
                .sort_values(x)
            )

            if data.empty:

                raise ValueError(
                    "No valid date/numerical rows were found."
                )

            figure, axis = plt.subplots(
                figsize=(8, 5)
            )

            axis.plot(
                data[x],
                data[y]
            )

            axis.set_title(
                f"Time Series: {y}"
            )

            axis.set_xlabel(x)
            axis.set_ylabel(y)

            figure.autofmt_xdate()

        elif graph in [
            "Line",
            "Area"
        ]:

            if (
                x not in df.columns
                or
                y not in df.columns
            ):

                raise ValueError(
                    f"{graph} requires X and Y."
                )

            data = (
                df[[x, y]]
                .dropna()
                .sort_values(x)
            )

            figure, axis = plt.subplots(
                figsize=(8, 5)
            )

            if graph == "Line":

                axis.plot(
                    data[x],
                    data[y]
                )

            else:

                axis.fill_between(
                    np.arange(
                        len(data)
                    ),
                    data[y].to_numpy(),
                    alpha=.35
                )

                axis.set_xlabel(
                    f"{x} (ordered)"
                )

            axis.set_ylabel(y)

            axis.set_title(
                f"{graph}: {y} by {x}"
            )

            if len(data) > 30:

                axis.tick_params(
                    axis="x",
                    rotation=45
                )

        elif graph == "Correlation":

            columns = numeric_cols(df)

            if len(columns) < 2:

                raise ValueError(
                    "Need at least two numerical columns."
                )

            figure, axis = plt.subplots(
                figsize=(9, 7)
            )

            sns.heatmap(
                df[columns].corr(),
                annot=False,
                cmap="coolwarm",
                center=0,
                ax=axis
            )

            axis.set_title(
                "Pearson Correlation Heatmap"
            )

        elif graph == "Covariance":

            columns = numeric_cols(df)

            if len(columns) < 2:

                raise ValueError(
                    "Need at least two numerical columns."
                )

            figure, axis = plt.subplots(
                figsize=(9, 7)
            )

            sns.heatmap(
                df[columns].cov(),
                annot=False,
                cmap="viridis",
                ax=axis
            )

            axis.set_title(
                "Covariance Heatmap"
            )

        elif graph == "Pair Plot":

            columns = numeric_cols(df)[:6]

            if len(columns) < 2:

                raise ValueError(
                    "Need at least two numerical columns."
                )

            pair = sns.pairplot(
                df[columns]
                .dropna()
                .head(500)
            )

            pair.fig.suptitle(
                "Pair Plot",
                y=1.02
            )

            figure = pair.fig

        elif graph == "Q-Q":

            if (
                x not in df.columns
                or
                not pd.api.types.is_numeric_dtype(
                    df[x]
                )
            ):

                raise ValueError(
                    "Q-Q plot requires numerical X."
                )

            figure, axis = plt.subplots(
                figsize=(7, 5)
            )

            stats.probplot(
                df[x].dropna(),
                dist="norm",
                plot=axis
            )

            axis.set_title(
                f"Q-Q Plot: {x}"
            )

        elif graph == "ECDF":

            if (
                x not in df.columns
                or
                not pd.api.types.is_numeric_dtype(
                    df[x]
                )
            ):

                raise ValueError(
                    "ECDF requires numerical X."
                )

            values = np.sort(
                df[x]
                .dropna()
                .to_numpy()
            )

            y_values = (
                np.arange(
                    1,
                    len(values) + 1
                )
                /
                len(values)
            )

            figure, axis = plt.subplots(
                figsize=(8, 5)
            )

            axis.step(
                values,
                y_values,
                where="post"
            )

            axis.set_title(
                f"ECDF: {x}"
            )

            axis.set_ylabel(
                "ECDF"
            )

        elif graph == "Missing":

            missing = (
                df.isna()
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            missing = (
                missing[missing > 0]
                .head(30)
            )

            if missing.empty:

                raise ValueError(
                    "No missing values found."
                )

            figure, axis = plt.subplots(
                figsize=(9, 5)
            )

            missing.plot(
                kind="bar",
                ax=axis
            )

            axis.set_title(
                "Missing Values by Column"
            )

            axis.tick_params(
                axis="x",
                rotation=45
            )

        elif graph == "3D Scatter":

            if any(
                column not in df.columns
                for column in [
                    x,
                    y,
                    z
                ]
            ):

                raise ValueError(
                    "3D scatter requires X, Y and Z."
                )

            if not all(
                pd.api.types.is_numeric_dtype(
                    df[column]
                )
                for column in [
                    x,
                    y,
                    z
                ]
            ):

                raise ValueError(
                    "3D scatter requires numerical columns."
                )

            figure = plt.figure(
                figsize=(8, 6)
            )

            axis = figure.add_subplot(
                111,
                projection="3d"
            )

            data = (
                df[
                    [x, y, z]
                ]
                .dropna()
                .head(1000)
            )

            axis.scatter(
                data[x],
                data[y],
                data[z],
                s=18
            )

            axis.set_xlabel(x)
            axis.set_ylabel(y)
            axis.set_zlabel(z)
            axis.set_title(
                "3D Scatter"
            )

        else:

            raise ValueError(
                "Unknown graph."
            )

        figure.tight_layout()

        APP["last_graph"] = figure

        return (
            figure,
            f"{graph} generated successfully."
        )

    except Exception as error:

        return (
            None,
            f"Graph error: {error}"
        )


# ============================================================
# TARGET ANALYSIS
# ============================================================

def target_analysis(
    target,
    task
):

    try:

        df, error = ensure_df()

        if error:

            return (
                pd.DataFrame(),
                None,
                error
            )

        if target not in df.columns:

            return (
                pd.DataFrame(),
                None,
                "Select a valid target."
            )

        series = df[target]

        rows = []

        if task == "Classification":

            value_counts = (
                series
                .astype(str)
                .value_counts(
                    dropna=False
                )
            )

            total = max(
                1,
                len(series)
            )

            for cls, count in (
                value_counts
                .head(30)
                .items()
            ):

                rows.append([
                    str(cls),
                    int(count),
                    round(
                        count / total * 100,
                        2
                    )
                ])

            table = pd.DataFrame(
                rows,
                columns=[
                    "Class",
                    "Count",
                    "Percentage"
                ]
            )

            figure, axis = plt.subplots(
                figsize=(8, 5)
            )

            value_counts.head(20).plot(
                kind="bar",
                ax=axis
            )

            axis.set_title(
                f"Target Class Distribution: {target}"
            )

            axis.tick_params(
                axis="x",
                rotation=45
            )

            numeric_association = []

            target_codes = pd.factorize(
                series.astype(str)
            )[0]

            for feature in numeric_cols(df):

                if feature == target:
                    continue

                try:

                    values = pd.to_numeric(
                        df[feature],
                        errors="coerce"
                    )

                    association = abs(
                        pd.Series(values).corr(
                            pd.Series(target_codes),
                            method="spearman"
                        )
                    )

                    numeric_association.append([
                        feature,
                        association
                    ])

                except Exception:
                    pass

            if numeric_association:

                association_df = (
                    pd.DataFrame(
                        numeric_association,
                        columns=[
                            "Feature",
                            "Target Association"
                        ]
                    )
                    .sort_values(
                        "Target Association",
                        ascending=False
                    )
                    .head(20)
                )

        else:

            if not pd.api.types.is_numeric_dtype(
                series
            ):

                return (
                    pd.DataFrame(),
                    None,
                    "Regression target must be numerical."
                )

            values = series.dropna()

            q1 = values.quantile(.25)
            q3 = values.quantile(.75)
            iqr = q3 - q1

            if iqr != 0:

                outlier_count = int(
                    (
                        (
                            values <
                            q1 - 1.5 * iqr
                        )
                        |
                        (
                            values >
                            q3 + 1.5 * iqr
                        )
                    ).sum()
                )

            else:

                outlier_count = 0

            table = pd.DataFrame([
                {
                    "Target": target,
                    "Count": len(values),
                    "Mean": values.mean(),
                    "Median": values.median(),
                    "Std Dev": values.std(),
                    "Skewness": values.skew(),
                    "Kurtosis": values.kurt(),
                    "Min": values.min(),
                    "Max": values.max(),
                    "IQR": iqr,
                    "Outliers": outlier_count
                }
            ]).round(5)

            figure, axis = plt.subplots(
                figsize=(8, 5)
            )

            sns.histplot(
                values,
                kde=True,
                ax=axis
            )

            axis.set_title(
                f"Regression Target Distribution: {target}"
            )

        APP["last_target"] = table

        return (
            table,
            figure,
            f"Target analysis complete for {target}."
        )

    except Exception as error:

        return (
            pd.DataFrame(),
            None,
            f"Target analysis failed: {error}"
        )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def feature_engineering(
    operation,
    columns,
    k
):

    try:

        df, error = ensure_df()

        if error:

            return (
                pd.DataFrame(),
                error
            )

        output = df.copy()

        selected = [
            column
            for column in (columns or [])
            if column in output.columns
        ]

        if operation == "Log":

            for column in selected:

                if pd.api.types.is_numeric_dtype(
                    output[column]
                ):

                    output[
                        column + "_log"
                    ] = np.log1p(
                        np.clip(
                            output[column],
                            0,
                            None
                        )
                    )

        elif operation == "Square":

            for column in selected:

                if pd.api.types.is_numeric_dtype(
                    output[column]
                ):

                    output[
                        column + "_square"
                    ] = (
                        output[column] ** 2
                    )

        elif operation == "Square root":

            for column in selected:

                if pd.api.types.is_numeric_dtype(
                    output[column]
                ):

                    output[
                        column + "_sqrt"
                    ] = np.sqrt(
                        np.clip(
                            output[column],
                            0,
                            None
                        )
                    )

        elif operation == "Standardization":

            for column in selected:

                if pd.api.types.is_numeric_dtype(
                    output[column]
                ):

                    series = output[column]
                    standard_deviation = series.std()

                    if standard_deviation == 0:

                        output[
                            column + "_standard"
                        ] = 0

                    else:

                        output[
                            column + "_standard"
                        ] = (
                            series -
                            series.mean()
                        ) / standard_deviation

        elif operation == "MinMax":

            for column in selected:

                if pd.api.types.is_numeric_dtype(
                    output[column]
                ):

                    series = output[column]

                    minimum = series.min()
                    maximum = series.max()

                    if maximum == minimum:

                        output[
                            column + "_minmax"
                        ] = 0

                    else:

                        output[
                            column + "_minmax"
                        ] = (
                            series -
                            minimum
                        ) / (
                            maximum -
                            minimum
                        )

        elif operation == "Robust scaling":

            for column in selected:

                if pd.api.types.is_numeric_dtype(
                    output[column]
                ):

                    series = output[column]

                    median = series.median()

                    iqr = (
                        series.quantile(.75)
                        -
                        series.quantile(.25)
                    )

                    if iqr == 0:

                        output[
                            column + "_robust"
                        ] = 0

                    else:

                        output[
                            column + "_robust"
                        ] = (
                            series -
                            median
                        ) / iqr

        elif operation == "One-hot encoding":

            if selected:

                encoded = pd.get_dummies(
                    output[selected],
                    dummy_na=True,
                    prefix=selected,
                    dtype=int
                )

                output = pd.concat(
                    [
                        output.drop(
                            columns=selected
                        ),
                        encoded
                    ],
                    axis=1
                )

        elif operation == "Ordinal encoding":

            for column in selected:

                if column in output.columns:

                    output[column] = (
                        pd.factorize(
                            output[column]
                            .astype(str)
                        )[0]
                    )

        elif operation == "Frequency encoding":

            for column in selected:

                if column in output.columns:

                    frequency = (
                        output[column]
                        .value_counts(
                            normalize=True
                        )
                    )

                    output[
                        column + "_frequency"
                    ] = output[column].map(
                        frequency
                    )

        elif operation == "Variance filtering":

            numerical = (
                output
                .select_dtypes(
                    include=np.number
                )
            )

            low_variance = (
                numerical
                .var()
                .sort_values()
            )

            threshold = float(
                k or .01
            )

            drop_columns = (
                low_variance[
                    low_variance <= threshold
                ]
                .index
                .tolist()
            )

            output = output.drop(
                columns=drop_columns,
                errors="ignore"
            )

        elif operation == "Correlation filtering":

            numerical = (
                output
                .select_dtypes(
                    include=np.number
                )
            )

            correlation = (
                numerical
                .corr()
                .abs()
            )

            upper = correlation.where(
                np.triu(
                    np.ones(
                        correlation.shape
                    ),
                    k=1
                ).astype(bool)
            )

            threshold = float(
                k or .9
            )

            drop_columns = [
                column
                for column in upper.columns
                if any(
                    upper[column] >
                    threshold
                )
            ]

            output = output.drop(
                columns=drop_columns,
                errors="ignore"
            )

        elif operation == "SelectKBest":

            numerical = (
                output
                .select_dtypes(
                    include=np.number
                )
            )

            k_value = int(
                k or 5
            )

            if len(numerical.columns) > k_value:

                keep = (
                    numerical
                    .var()
                    .sort_values(
                        ascending=False
                    )
                    .head(k_value)
                    .index
                    .tolist()
                )

                drop_columns = [
                    column
                    for column in numerical.columns
                    if column not in keep
                ]

                output = output.drop(
                    columns=drop_columns,
                    errors="ignore"
                )

        elif operation == "PCA":

            numerical_columns = (
                output
                .select_dtypes(
                    include=np.number
                )
                .columns
                .tolist()
            )

            if len(numerical_columns) < 2:

                raise ValueError(
                    "PCA needs at least two numerical columns."
                )

            data = (
                output[
                    numerical_columns
                ]
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
            )

            data = data.fillna(
                data.median()
            )

            components = max(
                1,
                min(
                    int(k or 2),
                    len(numerical_columns),
                    len(data)
                )
            )

            pca = PCA(
                n_components=components,
                random_state=42
            )

            transformed = pca.fit_transform(
                StandardScaler()
                .fit_transform(data)
            )

            output = output.drop(
                columns=numerical_columns
            )

            for index in range(
                components
            ):

                output[
                    f"PCA_{index + 1}"
                ] = transformed[:, index]

        else:

            raise ValueError(
                "Select a feature-engineering operation."
            )

        APP["last_feature_result"] = (
            output.copy()
        )

        return (
            output.head(100),
            (
                "Feature engineering complete. "
                f"Result: {output.shape[0]:,} rows × "
                f"{output.shape[1]:,} columns."
            )
        )

    except Exception as error:

        return (
            pd.DataFrame(),
            f"Feature engineering failed: {error}"
        )


# ============================================================
# MACHINE LEARNING PREPROCESSOR
# ============================================================

def build_preprocessor(
    X,
    scaling="Standard",
    encoding="One-Hot"
):

    numerical = (
        X.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    categorical = (
        X.select_dtypes(
            exclude=np.number
        )
        .columns
        .tolist()
    )

    transformers = []

    if numerical:

        scaler = "passthrough"

        if scaling == "Standard":

            scaler = StandardScaler()

        elif scaling == "MinMax":

            scaler = MinMaxScaler()

        elif scaling == "Robust":

            scaler = RobustScaler()

        transformers.append(
            (
                "num",

                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "scaler",
                        scaler
                    )
                ]),

                numerical
            )
        )

    if categorical:

        if encoding == "Ordinal":

            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )

        else:

            try:

                encoder = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )

            except TypeError:

                encoder = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse=False
                )

        transformers.append(
            (
                "cat",

                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "encoder",
                        encoder
                    )
                ]),

                categorical
            )
        )

    if not transformers:

        raise ValueError(
            "No usable feature columns."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# ============================================================
# MODEL FACTORY
# ============================================================

def make_model(
    task,
    name
):

    if task == "Classification":

        models = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000
                ),

            "KNN":
                KNeighborsClassifier(),

            "Naive Bayes":
                GaussianNB(),

            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42,
                    max_depth=8
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
                )
        }

        if XGB_AVAILABLE:

            models["XGBoost"] = (
                XGBClassifier(
                    n_estimators=120,
                    max_depth=4,
                    learning_rate=.08,
                    subsample=.85,
                    colsample_bytree=.85,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1
                )
            )

    else:

        models = {

            "Linear Regression":
                LinearRegression(),

            "Polynomial Regression":
                Pipeline([
                    (
                        "poly",
                        PolynomialFeatures(
                            degree=2,
                            include_bias=False
                        )
                    ),
                    (
                        "model",
                        LinearRegression()
                    )
                ]),

            "Ridge":
                Ridge(
                    alpha=1.0
                ),

            "Lasso":
                Lasso(
                    alpha=.01
                ),

            "Decision Tree":
                DecisionTreeRegressor(
                    random_state=42,
                    max_depth=8
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
                )
        }

        if XGB_AVAILABLE:

            models["XGBoost"] = (
                XGBRegressor(
                    n_estimators=120,
                    max_depth=4,
                    learning_rate=.08,
                    subsample=.85,
                    colsample_bytree=.85,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1
                )
            )

    if name not in models:

        raise ValueError(
            f"Model {name} is not compatible with {task}."
        )

    return models[name]


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(
    task,
    target,
    model_name,
    test_size,
    scaling,
    encoding,
    cv_folds
):

    try:

        df, error = ensure_df()

        if error:

            return (
                None,
                None,
                None,
                None,
                error
            )

        if target not in df.columns:

            return (
                None,
                None,
                None,
                None,
                "Select a valid target."
            )

        if (
            task == "Regression"
            and
            not pd.api.types.is_numeric_dtype(
                df[target]
            )
        ):

            return (
                None,
                None,
                None,
                None,
                "Regression target must be numerical."
            )

        if len(df) < 20:

            return (
                None,
                None,
                None,
                None,
                "At least 20 rows are recommended."
            )

        data = (
            df.copy()
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

        X = data.drop(
            columns=[target]
        )

        y = data[target]

        if task == "Classification":

            mask = y.notna()

            X = X.loc[mask]
            y = y.loc[mask]

            if y.nunique() < 2:

                return (
                    None,
                    None,
                    None,
                    None,
                    "Classification needs at least two classes."
                )

            if y.nunique() > 50:

                return (
                    None,
                    None,
                    None,
                    None,
                    "Target has too many classes for this classifier."
                )

        else:

            mask = y.notna()

            X = X.loc[mask]

            y = pd.to_numeric(
                y.loc[mask],
                errors="coerce"
            )

            mask = y.notna()

            X = X.loc[mask]
            y = y.loc[mask]

        if task == "Classification":

            if y.value_counts().min() >= 2:

                stratify = y

            else:

                stratify = None

        else:

            stratify = None

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=float(test_size),
                random_state=42,
                stratify=stratify
            )
        )

        preprocessor = build_preprocessor(
            X_train,
            scaling,
            encoding
        )

        model = make_model(
            task,
            model_name
        )

        pipeline = Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ])

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        probabilities = None

        if (
            task == "Classification"
            and
            hasattr(
                pipeline,
                "predict_proba"
            )
        ):

            try:

                probabilities = (
                    pipeline
                    .predict_proba(
                        X_test
                    )
                )

            except Exception:

                probabilities = None

        APP.update({

            "trained_pipeline":
                pipeline,

            "trained_model":
                model,

            "preprocessor":
                preprocessor,

            "target_column":
                target,

            "task_type":
                task,

            "X_train":
                X_train,

            "X_test":
                X_test,

            "y_train":
                y_train,

            "y_test":
                y_test,

            "predictions":
                predictions,

            "probabilities":
                probabilities,

            "raw_feature_names":
                X.columns.tolist()
        })

        metrics = evaluate_current()

        comparison = (
            run_model_comparison(
                task,
                target,
                test_size,
                scaling,
                encoding,
                cv_folds,
                model_name
            )
        )

        importance = (
            feature_importance_table()
        )

        APP["model_results"] = (
            comparison
            if isinstance(
                comparison,
                pd.DataFrame
            )
            else pd.DataFrame()
        )

        APP["evaluation_metrics"] = (
            metrics
            if isinstance(
                metrics,
                pd.DataFrame
            )
            else pd.DataFrame()
        )

        APP["feature_importance"] = (
            importance
            if isinstance(
                importance,
                pd.DataFrame
            )
            else pd.DataFrame()
        )

        prediction_template = (
            prediction_table()
        )

        status = (
            f"Model trained successfully: "
            f"{model_name} | "
            f"{task} | "
            f"Test rows: {len(y_test):,}"
        )

        return (
            metrics,
            importance,
            prediction_template,
            comparison,
            status
        )

    except Exception as error:

        return (
            None,
            None,
            None,
            None,
            f"Model training failed: {error}"
        )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_current():

    task = APP["task_type"]

    y_true = APP["y_test"]
    predictions = APP["predictions"]

    if task == "Classification":

        average = (
            "binary"
            if len(
                pd.Series(y_true).unique()
            ) == 2
            else
            "weighted"
        )

        rows = [

            [
                "Accuracy",
                accuracy_score(
                    y_true,
                    predictions
                )
            ],

            [
                "Precision",
                precision_score(
                    y_true,
                    predictions,
                    average=average,
                    zero_division=0
                )
            ],

            [
                "Recall",
                recall_score(
                    y_true,
                    predictions,
                    average=average,
                    zero_division=0
                )
            ],

            [
                "F1",
                f1_score(
                    y_true,
                    predictions,
                    average=average,
                    zero_division=0
                )
            ]
        ]

        if APP["probabilities"] is not None:

            try:

                if len(
                    np.unique(y_true)
                ) == 2:

                    probability = (
                        APP["probabilities"][:, 1]
                    )

                    rows.append([
                        "ROC-AUC",
                        roc_auc_score(
                            y_true,
                            probability
                        )
                    ])

                else:

                    rows.append([
                        "ROC-AUC",
                        roc_auc_score(
                            y_true,
                            APP["probabilities"],
                            multi_class="ovr",
                            average="weighted"
                        )
                    ])

            except Exception:
                pass

        return pd.DataFrame(
            rows,
            columns=[
                "Metric",
                "Value"
            ]
        ).round(5)

    mse = mean_squared_error(
        y_true,
        predictions
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    n = len(y_true)

    p = max(
        1,
        APP["X_test"].shape[1]
    )

    adjusted_r2 = (
        1 -
        (
            1 - r2
        )
        *
        (
            n - 1
        )
        /
        max(
            1,
            n - p - 1
        )
    )

    return pd.DataFrame([
        [
            "MAE",
            mean_absolute_error(
                y_true,
                predictions
            )
        ],

        [
            "MSE",
            mse
        ],

        [
            "RMSE",
            np.sqrt(mse)
        ],

        [
            "R2",
            r2
        ],

        [
            "Adjusted R2",
            adjusted_r2
        ]

    ], columns=[
        "Metric",
        "Value"
    ]).round(5)


# ============================================================
# EVALUATION PLOTS
# ============================================================

def evaluation_plot(kind):

    try:

        if APP["predictions"] is None:

            return (
                None,
                "Train a model first."
            )

        y_true = APP["y_test"]
        predictions = APP["predictions"]

        figure, axis = plt.subplots(
            figsize=(8, 5)
        )

        if APP["task_type"] == "Classification":

            matrix = confusion_matrix(
                y_true,
                predictions
            )

            if kind == "Confusion Matrix":

                sns.heatmap(
                    matrix,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    ax=axis
                )

                axis.set_xlabel(
                    "Predicted"
                )

                axis.set_ylabel(
                    "Actual"
                )

                axis.set_title(
                    "Confusion Matrix"
                )

            elif kind == "ROC":

                if APP["probabilities"] is None:

                    raise ValueError(
                        "Probability output is unavailable for this model."
                    )

                if len(
                    np.unique(y_true)
                ) != 2:

                    raise ValueError(
                        "ROC plot currently uses binary classification."
                    )

                false_positive, true_positive, _ = (
                    roc_curve(
                        y_true,
                        APP["probabilities"][:, 1]
                    )
                )

                axis.plot(
                    false_positive,
                    true_positive
                )

                axis.plot(
                    [0, 1],
                    [0, 1],
                    "--"
                )

                axis.set_xlabel(
                    "False Positive Rate"
                )

                axis.set_ylabel(
                    "True Positive Rate"
                )

                axis.set_title(
                    "ROC Curve"
                )

            else:

                if APP["probabilities"] is None:

                    raise ValueError(
                        "Probability output is unavailable for this model."
                    )

                if len(
                    np.unique(y_true)
                ) != 2:

                    raise ValueError(
                        "Precision-Recall plot currently uses binary classification."
                    )

                precision, recall, _ = (
                    precision_recall_curve(
                        y_true,
                        APP["probabilities"][:, 1]
                    )
                )

                axis.plot(
                    recall,
                    precision
                )

                axis.set_xlabel(
                    "Recall"
                )

                axis.set_ylabel(
                    "Precision"
                )

                axis.set_title(
                    "Precision-Recall Curve"
                )

        else:

            if kind == "Actual vs Predicted":

                axis.scatter(
                    y_true,
                    predictions
                )

                lower = min(
                    np.min(y_true),
                    np.min(predictions)
                )

                upper = max(
                    np.max(y_true),
                    np.max(predictions)
                )

                axis.plot(
                    [lower, upper],
                    [lower, upper],
                    "--"
                )

                axis.set_xlabel(
                    "Actual"
                )

                axis.set_ylabel(
                    "Predicted"
                )

                axis.set_title(
                    "Actual vs Predicted"
                )

            else:

                residuals = (
                    y_true -
                    predictions
                )

                sns.scatterplot(
                    x=predictions,
                    y=residuals,
                    ax=axis
                )

                axis.axhline(
                    0,
                    ls="--"
                )

                axis.set_xlabel(
                    "Predicted"
                )

                axis.set_ylabel(
                    "Residual"
                )

                axis.set_title(
                    "Residual Plot"
                )

        figure.tight_layout()

        return (
            figure,
            f"{kind} generated."
        )

    except Exception as error:

        return (
            None,
            f"Evaluation plot failed: {error}"
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

def run_model_comparison(
    task,
    target,
    test_size,
    scaling,
    encoding,
    cv_folds,
    selected_model
):

    try:

        df = (
            APP["processed_df"]
            .copy()
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

        X = df.drop(
            columns=[target]
        )

        y = df[target]

        if task == "Regression":

            y = pd.to_numeric(
                y,
                errors="coerce"
            )

        mask = y.notna()

        X = X.loc[mask]
        y = y.loc[mask]

        if len(X) < 20:

            return pd.DataFrame()

        if task == "Classification":

            model_names = [
                "Logistic Regression",
                "KNN",
                "Naive Bayes",
                "Decision Tree",
                "Random Forest",
                "SVM",
                "Gradient Boosting"
            ]

        else:

            model_names = [
                "Linear Regression",
                "Ridge",
                "Lasso",
                "Decision Tree",
                "Random Forest",
                "SVR",
                "Gradient Boosting"
            ]

        if XGB_AVAILABLE:

            model_names.append(
                "XGBoost"
            )

        if task == "Classification":

            stratify = (
                y
                if y.value_counts().min() >= 2
                else None
            )

        else:

            stratify = None

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=float(test_size),
                random_state=42,
                stratify=stratify
            )
        )

        rows = []

        for name in model_names:

            try:

                preprocessor = build_preprocessor(
                    X_train,
                    scaling,
                    encoding
                )

                pipeline = Pipeline([
                    (
                        "preprocessor",
                        preprocessor
                    ),
                    (
                        "model",
                        make_model(
                            task,
                            name
                        )
                    )
                ])

                pipeline.fit(
                    X_train,
                    y_train
                )

                prediction = pipeline.predict(
                    X_test
                )

                if task == "Classification":

                    score = accuracy_score(
                        y_test,
                        prediction
                    )

                    rows.append([
                        name,
                        score,
                        "Accuracy"
                    ])

                else:

                    rmse = np.sqrt(
                        mean_squared_error(
                            y_test,
                            prediction
                        )
                    )

                    r2 = r2_score(
                        y_test,
                        prediction
                    )

                    rows.append([
                        name,
                        rmse,
                        r2,
                        "RMSE"
                    ])

            except Exception:

                if task == "Classification":

                    rows.append([
                        name,
                        np.nan,
                        "Failed"
                    ])

                else:

                    rows.append([
                        name,
                        np.nan,
                        np.nan,
                        "Failed"
                    ])

        if task == "Classification":

            result = pd.DataFrame(
                rows,
                columns=[
                    "Model",
                    "Score",
                    "Metric"
                ]
            )

            result["Score"] = (
                result["Score"]
                .round(5)
            )

            result = (
                result
                .sort_values(
                    "Score",
                    ascending=False
                )
                .reset_index(drop=True)
            )

        else:

            result = pd.DataFrame(
                rows,
                columns=[
                    "Model",
                    "RMSE",
                    "R2",
                    "Metric"
                ]
            )

            result = (
                result
                .sort_values(
                    "RMSE",
                    ascending=True
                )
                .reset_index(drop=True)
            )

            result["RMSE"] = (
                result["RMSE"]
                .round(5)
            )

            result["R2"] = (
                result["R2"]
                .round(5)
            )

        return result

    except Exception:

        return pd.DataFrame()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def feature_importance_table():

    try:

        pipeline = APP[
            "trained_pipeline"
        ]

        if pipeline is None:

            return pd.DataFrame()

        model = pipeline.named_steps[
            "model"
        ]

        preprocessor = (
            pipeline
            .named_steps[
                "preprocessor"
            ]
        )

        try:

            names = (
                preprocessor
                .get_feature_names_out()
            )

        except Exception:

            names = [
                f"Feature {index + 1}"
                for index in range(
                    len(
                        getattr(
                            model,
                            "feature_importances_",
                            []
                        )
                    )
                )
            ]

        if hasattr(
            model,
            "feature_importances_"
        ):

            values = (
                model.feature_importances_
            )

        elif hasattr(
            model,
            "coef_"
        ):

            coefficients = np.asarray(
                model.coef_
            )

            if coefficients.ndim > 1:

                values = np.mean(
                    np.abs(
                        coefficients
                    ),
                    axis=0
                )

            else:

                values = np.abs(
                    coefficients
                )

        else:

            return permutation_feature_importance()

        result = pd.DataFrame({

            "Feature":
                names,

            "Importance":
                values
        })

        return (
            result
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(100)
            .round(6)
        )

    except Exception:

        return permutation_feature_importance()


def permutation_feature_importance():

    try:

        pipeline = APP[
            "trained_pipeline"
        ]

        X_test = APP[
            "X_test"
        ]

        y_test = APP[
            "y_test"
        ]

        if pipeline is None:

            return pd.DataFrame()

        result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=5,
            random_state=42,
            n_jobs=-1
        )

        names = (
            X_test.columns.tolist()
        )

        return (
            pd.DataFrame({
                "Feature": names,
                "Importance":
                    result.importances_mean
            })
            .sort_values(
                "Importance",
                ascending=False
            )
            .round(6)
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# CROSS VALIDATION
# ============================================================

def run_cross_validation(
    folds
):

    try:

        if (
            APP["processed_df"] is None
            or
            APP["target_column"] is None
        ):

            return (
                pd.DataFrame(),
                "Train a model first."
            )

        df = (
            APP["processed_df"]
            .copy()
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

        target = APP[
            "target_column"
        ]

        task = APP[
            "task_type"
        ]

        X = df.drop(
            columns=[target]
        )

        y = df[target]

        mask = y.notna()

        X = X.loc[mask]
        y = y.loc[mask]

        if APP["trained_model"] is not None:

            trained_class = (
                APP["trained_model"]
                .__class__.__name__
            )

        else:

            trained_class = ""

        if task == "Classification":

            if "Forest" in trained_class:

                model_name = (
                    "Random Forest"
                )

            elif (
                "Logistic"
                in trained_class
            ):

                model_name = (
                    "Logistic Regression"
                )

            else:

                model_name = (
                    "Random Forest"
                )

        else:

            if "Forest" in trained_class:

                model_name = (
                    "Random Forest"
                )

            else:

                model_name = (
                    "Linear Regression"
                )

        preprocessor = build_preprocessor(
            X,
            "Standard",
            "One-Hot"
        )

        pipeline = Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                make_model(
                    task,
                    model_name
                )
            )
        ])

        if task == "Classification":

            cv = StratifiedKFold(
                n_splits=int(folds),
                shuffle=True,
                random_state=42
            )

            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring="accuracy"
            )

        else:

            cv = KFold(
                n_splits=int(folds),
                shuffle=True,
                random_state=42
            )

            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring="r2"
            )

        result = pd.DataFrame({
            "Fold":
                range(
                    1,
                    len(scores) + 1
                ),

            "Score":
                scores
        })

        return (
            result.round(5),
            (
                f"Cross-validation mean: "
                f"{scores.mean():.4f} ± "
                f"{scores.std():.4f}"
            )
        )

    except Exception as error:

        return (
            pd.DataFrame(),
            f"Cross-validation failed: {error}"
        )


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

def tune_model(
    search_type,
    task,
    target,
    model_name,
    folds
):

    try:

        df, error = ensure_df()

        if error:

            return (
                pd.DataFrame(),
                "",
                "Load dataset first."
            )

        if target not in df.columns:

            return (
                pd.DataFrame(),
                "",
                "Select a valid target."
            )

        X = (
            df.drop(
                columns=[target]
            )
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

        y = df[target]

        if task == "Regression":

            y = pd.to_numeric(
                y,
                errors="coerce"
            )

        mask = y.notna()

        X = X.loc[mask]
        y = y.loc[mask]

        preprocessor = build_preprocessor(
            X,
            "Standard",
            "One-Hot"
        )

        model = make_model(
            task,
            model_name
        )

        pipeline = Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ])

        if task == "Classification":

            scoring = "accuracy"

            if model_name == "Random Forest":

                parameter_grid = {
                    "model__n_estimators":
                        [80, 120],

                    "model__max_depth":
                        [None, 6, 10]
                }

            elif model_name == "Logistic Regression":

                parameter_grid = {
                    "model__C":
                        [0.1, 1, 10]
                }

            else:

                parameter_grid = {}

        else:

            scoring = (
                "neg_root_mean_squared_error"
            )

            if model_name == "Random Forest":

                parameter_grid = {
                    "model__n_estimators":
                        [80, 120],

                    "model__max_depth":
                        [None, 6, 10]
                }

            elif model_name == "Ridge":

                parameter_grid = {
                    "model__alpha":
                        [0.1, 1, 10]
                }

            elif model_name == "Lasso":

                parameter_grid = {
                    "model__alpha":
                        [0.001, 0.01, 0.1]
                }

            else:

                parameter_grid = {}

        if not parameter_grid:

            return (
                pd.DataFrame(),
                "",
                (
                    "No safe tuning grid is defined "
                    "for this model. Use Random Forest, "
                    "Logistic Regression, Ridge or Lasso."
                )
            )

        fold_count = int(
            folds
        )

        if search_type == "GridSearchCV":

            search = GridSearchCV(
                pipeline,
                parameter_grid,
                cv=fold_count,
                scoring=scoring,
                n_jobs=-1
            )

        else:

            search = RandomizedSearchCV(
                pipeline,
                parameter_grid,
                n_iter=min(
                    4,
                    sum(
                        len(values)
                        for values in
                        parameter_grid.values()
                    ) * 2
                ),
                cv=fold_count,
                scoring=scoring,
                random_state=42,
                n_jobs=-1
            )

        search.fit(
            X,
            y
        )

        best_pipeline = (
            search.best_estimator_
        )

        APP["trained_pipeline"] = (
            best_pipeline
        )

        APP["trained_model"] = (
            best_pipeline
            .named_steps["model"]
        )

        APP["target_column"] = target
        APP["task_type"] = task

        rows = []

        for parameters, score in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"]
        ):

            rows.append([
                str(parameters),
                float(score)
            ])

        result = pd.DataFrame(
            rows,
            columns=[
                "Parameters",
                "CV Score"
            ]
        )

        result = (
            result
            .sort_values(
                "CV Score",
                ascending=False
            )
        )

        APP["last_tuning"] = (
            result
        )

        return (
            result,
            str(
                search.best_params_
            ),
            (
                f"Best CV score: "
                f"{search.best_score_:.5f}"
            )
        )

    except Exception as error:

        return (
            pd.DataFrame(),
            "",
            f"Tuning failed: {error}"
        )


# ============================================================
# PREDICTION
# ============================================================

def prediction_table():

    pipeline = APP[
        "trained_pipeline"
    ]

    if (
        pipeline is None
        or
        not APP["raw_feature_names"]
    ):

        return pd.DataFrame(
            columns=[
                "Feature",
                "Value"
            ]
        )

    values = []

    X_train = APP[
        "X_train"
    ]

    for column in APP[
        "raw_feature_names"
    ]:

        default = ""

        if column in X_train.columns:

            series = X_train[column]

            if pd.api.types.is_numeric_dtype(
                series
            ):

                default = fmt_num(
                    series.median()
                )

            else:

                mode = (
                    series
                    .mode()
                )

                if not mode.empty:

                    default = str(
                        mode.iloc[0]
                    )

        values.append([
            column,
            default
        ])

    result = pd.DataFrame(
        values,
        columns=[
            "Feature",
            "Value"
        ]
    )

    APP["prediction_df"] = (
        result.copy()
    )

    return result


def predict_from_table(
    table
):

    try:

        if APP["trained_pipeline"] is None:

            return (
                pd.DataFrame(
                    columns=[
                        "Result",
                        "Value"
                    ]
                ),
                "Train a model first."
            )

        if table is None:

            return (
                pd.DataFrame(
                    columns=[
                        "Result",
                        "Value"
                    ]
                ),
                "Prediction input is incomplete."
            )

        prediction_input = pd.DataFrame(
            table,
            columns=[
                "Feature",
                "Value"
            ]
        )

        if prediction_input.empty:

            return (
                pd.DataFrame(
                    columns=[
                        "Result",
                        "Value"
                    ]
                ),
                "Prediction input is empty."
            )

        data = {}

        X_train = APP[
            "X_train"
        ]

        for _, row in (
            prediction_input
            .iterrows()
        ):

            feature = str(
                row["Feature"]
            )

            if feature not in (
                APP["raw_feature_names"]
            ):

                continue

            value = row["Value"]

            if pd.api.types.is_numeric_dtype(
                X_train[feature]
            ):

                try:

                    value = float(
                        value
                    )

                except Exception:

                    value = np.nan

            data[feature] = value

        for feature in (
            APP["raw_feature_names"]
        ):

            if feature not in data:

                data[feature] = np.nan

        input_df = pd.DataFrame(
            [
                data
            ],
            columns=APP[
                "raw_feature_names"
            ]
        )

        prediction = (
            APP["trained_pipeline"]
            .predict(
                input_df
            )
        )

        rows = [
            [
                "Prediction",
                str(
                    prediction[0]
                )
            ]
        ]

        if APP["task_type"] == "Classification":

            try:

                probabilities = (
                    APP["trained_pipeline"]
                    .predict_proba(
                        input_df
                    )[0]
                )

                classes = (
                    APP["trained_pipeline"]
                    .classes_
                )

                for cls, probability in zip(
                    classes,
                    probabilities
                ):

                    rows.append([
                        f"Probability: {cls}",
                        f"{probability * 100:.2f}%"
                    ])

            except Exception:

                pass

        result = pd.DataFrame(
            rows,
            columns=[
                "Result",
                "Value"
            ]
        )

        APP["prediction_df"] = (
            result.copy()
        )

        return (
            result,
            "Prediction completed successfully."
        )

    except Exception as error:

        return (
            pd.DataFrame(
                columns=[
                    "Result",
                    "Value"
                ]
            ),
            f"Prediction failed: {error}"
        )


# ============================================================
# EXPLAINABILITY
# ============================================================

def explain_model(
    method
):

    try:

        if APP["trained_pipeline"] is None:

            return (
                pd.DataFrame(),
                "Train a model first."
            )

        if method == "Feature Importance":

            result = (
                feature_importance_table()
            )

            APP["feature_importance"] = (
                result
            )

            return (
                result,
                "Feature importance extracted from the trained model."
            )

        if method == "Permutation Importance":

            result = (
                permutation_feature_importance()
            )

            APP["feature_importance"] = (
                result
            )

            return (
                result,
                "Permutation importance calculated on the test set."
            )

        if method == "Coefficients":

            model = (
                APP["trained_pipeline"]
                .named_steps["model"]
            )

            if not hasattr(
                model,
                "coef_"
            ):

                return (
                    pd.DataFrame(),
                    "Selected model has no coefficients."
                )

            preprocessor = (
                APP["trained_pipeline"]
                .named_steps[
                    "preprocessor"
                ]
            )

            names = (
                preprocessor
                .get_feature_names_out()
            )

            coefficients = np.asarray(
                model.coef_
            )

            if coefficients.ndim > 1:

                values = np.mean(
                    np.abs(
                        coefficients
                    ),
                    axis=0
                )

            else:

                values = np.abs(
                    coefficients
                )

            result = (
                pd.DataFrame({
                    "Feature":
                        names,

                    "Coefficient Magnitude":
                        values
                })
                .sort_values(
                    "Coefficient Magnitude",
                    ascending=False
                )
                .head(100)
            )

            return (
                result.round(6),
                "Coefficient magnitude calculated."
            )

        if method == "SHAP":

            if not SHAP_AVAILABLE:

                return (
                    pd.DataFrame(),
                    "SHAP is unavailable; the dashboard continues without it."
                )

            return (
                shap_explain(),
                "SHAP explanation generated."
            )

        return (
            pd.DataFrame(),
            "Select an explanation method."
        )

    except Exception as error:

        return (
            pd.DataFrame(),
            f"Explainability failed: {error}"
        )


def shap_explain():

    try:

        model = APP[
            "trained_model"
        ]

        X_test = APP[
            "X_test"
        ]

        preprocessor = (
            APP["trained_pipeline"]
            .named_steps[
                "preprocessor"
            ]
        )

        transformed = (
            preprocessor
            .transform(
                X_test.head(100)
            )
        )

        names = (
            preprocessor
            .get_feature_names_out()
        )

        if hasattr(
            model,
            "feature_importances_"
        ):

            explainer = (
                shap.TreeExplainer(
                    model
                )
            )

            shap_values = (
                explainer.shap_values(
                    transformed
                )
            )

            array = np.asarray(
                shap_values
            )

            if array.ndim == 3:

                array = np.mean(
                    np.abs(array),
                    axis=0
                )

            else:

                array = np.abs(
                    array
                )

            values = np.mean(
                array,
                axis=0
            )

        else:

            return (
                permutation_feature_importance()
            )

        return (
            pd.DataFrame({
                "Feature": names,
                "SHAP Mean Abs": values
            })
            .sort_values(
                "SHAP Mean Abs",
                ascending=False
            )
            .head(100)
            .round(6)
        )

    except Exception as error:

        raise ValueError(
            "SHAP could not be generated "
            f"for this model: {error}"
        )


# ============================================================
# EXPORT CENTER
# ============================================================

def export_file(
    export_type,
    fmt
):

    try:

        data = None

        stem = (
            export_type
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if export_type == "Original Dataset":

            data = APP[
                "original_df"
            ]

        elif export_type == "Processed Dataset":

            data = APP[
                "processed_df"
            ]

        elif export_type == "Currently Viewed Dataset":

            data = APP[
                "viewed_df"
            ]

        elif export_type == "Statistics":

            if (
                not APP[
                    "last_statistics"
                ].empty
            ):

                data = APP[
                    "last_statistics"
                ]

            else:

                data = statistics_report(
                    APP["processed_df"]
                )

        elif export_type == "Missing Value Report":

            if (
                not APP[
                    "last_missing"
                ].empty
            ):

                data = APP[
                    "last_missing"
                ]

            else:

                data = missing_report(
                    APP["processed_df"]
                )

        elif export_type == "Duplicate Report":

            if (
                not APP[
                    "last_duplicate"
                ].empty
            ):

                data = APP[
                    "last_duplicate"
                ]

            else:

                data = duplicate_report(
                    APP["processed_df"]
                )

        elif export_type == "Outlier Report":

            if (
                not APP[
                    "last_outliers"
                ].empty
            ):

                data = APP[
                    "last_outliers"
                ]

            else:

                data = outlier_report()[0]

        elif export_type == "Data Quality Report":

            if (
                not APP[
                    "last_quality"
                ].empty
            ):

                data = APP[
                    "last_quality"
                ]

            else:

                data = quality_analysis()[0]

        elif export_type == "Model Metrics":

            data = APP[
                "evaluation_metrics"
            ]

        elif export_type == "Model Comparison":

            data = APP[
                "model_results"
            ]

        elif export_type == "Prediction Result":

            data = APP[
                "prediction_df"
            ]

        elif export_type == "Feature Importance":

            data = APP[
                "feature_importance"
            ]

        elif export_type == "Cleaning Change Log":

            data = pd.DataFrame(
                APP[
                    "cleaning_history"
                ]
            )

        elif export_type == "Complete Analysis Summary":

            processed = APP[
                "processed_df"
            ]

            original = APP[
                "original_df"
            ]

            data = pd.DataFrame([

                [
                    "File",
                    APP["filename"]
                ],

                [
                    "Rows Original",
                    (
                        len(original)
                        if original is not None
                        else 0
                    )
                ],

                [
                    "Rows Processed",
                    (
                        len(processed)
                        if processed is not None
                        else 0
                    )
                ],

                [
                    "Columns Processed",
                    (
                        processed.shape[1]
                        if processed is not None
                        else 0
                    )
                ],

                [
                    "Quality Score",
                    (
                        quality_score(
                            processed
                        )
                        if processed is not None
                        else 0
                    )
                ],

                [
                    "Missing Values",
                    (
                        total_missing(
                            processed
                        )
                        if processed is not None
                        else 0
                    )
                ],

                [
                    "Duplicates",
                    (
                        duplicate_count(
                            processed
                        )
                        if processed is not None
                        else 0
                    )
                ],

                [
                    "Target",
                    APP[
                        "target_column"
                    ]
                    or ""
                ],

                [
                    "Task",
                    APP[
                        "task_type"
                    ]
                    or ""
                ]

            ], columns=[
                "Metric",
                "Value"
            ])

        if data is None:

            return (
                None,
                "No data is available for this export."
            )

        if isinstance(
            data,
            pd.Series
        ):

            data = data.to_frame()

        if data.empty:

            return (
                None,
                "No data is available for this export."
            )

        path = write_df(
            data,
            stem,
            fmt
        )

        if path is None:

            return (
                None,
                "Could not create the download file."
            )

        return (
            path,
            f"Download ready: {os.path.basename(path)}"
        )

    except Exception as error:

        return (
            None,
            f"Export failed: {error}"
        )


# ============================================================
# RESET DASHBOARD
# ============================================================

def reset_all():

    clear_state()

    return (
        "Dashboard reset. Upload a dataset to begin.",

        "<div class='notice'>No dataset loaded.</div>",

        pd.DataFrame(),

        pd.DataFrame(),

        [],

        [],

        [],

        [],

        [],

        [],

        [],

        [],

        "No dataset loaded."
    )


# ============================================================
# PROFESSIONAL CSS
# ============================================================

CSS = """

@import url(
'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap'
);

:root {

    --navy: #102a43;
    --blue: #2563eb;
    --cyan: #0891b2;
    --teal: #0f766e;
    --green: #16a34a;
    --amber: #d97706;
    --red: #dc2626;

    --background: #eef5fb;
    --card: #ffffff;
    --border: #d8e3ef;

    --text: #1e293b;
    --muted: #64748b;
}

body,
.gradio-container {

    background:
        linear-gradient(
            135deg,
            #eef5fb 0%,
            #f8fbff 55%,
            #eef7ff 100%
        ) !important;

    color:
        var(--text) !important;

    font-family:
        'DM Sans',
        Arial,
        sans-serif !important;
}

.gradio-container {

    max-width:
        1500px !important;

    margin:
        auto !important;
}

.header {

    padding:
        24px 28px !important;

    border-radius:
        20px !important;

    background:
        linear-gradient(
            135deg,
            #102a43,
            #1d4ed8
        ) !important;

    color:
        white !important;

    box-shadow:
        0 10px 30px
        rgba(
            16,
            42,
            67,
            .18
        );

    margin-bottom:
        16px !important;
}

.header h1 {

    margin:
        0;

    font-size:
        30px !important;

    color:
        white !important;
}

.header p {

    margin:
        5px 0 0;

    color:
        #dbeafe !important;
}

.kpi-grid {

    display:
        grid;

    grid-template-columns:
        repeat(
            8,
            minmax(
                100px,
                1fr
            )
        );

    gap:
        10px;

    margin:
        4px 0 10px;
}

.kpi {

    background:
        white;

    border:
        1px solid
        var(--border);

    border-radius:
        13px;

    padding:
        12px 13px;

    box-shadow:
        0 4px 12px
        rgba(
            15,
            42,
            67,
            .05
        );
}

.kpi span {

    font-size:
        12px;

    color:
        var(--muted);

    display:
        block;
}

.kpi b {

    font-size:
        22px;

    color:
        var(--navy);

    display:
        block;

    margin-top:
        3px;

    overflow:
        hidden;

    text-overflow:
        ellipsis;

    white-space:
        nowrap;
}

.k1 {
    border-top:
        4px solid #2563eb;
}

.k2 {
    border-top:
        4px solid #7c3aed;
}

.k3 {
    border-top:
        4px solid #0891b2;
}

.k4 {
    border-top:
        4px solid #0f766e;
}

.k5 {
    border-top:
        4px solid #d97706;
}

.k6 {
    border-top:
        4px solid #dc2626;
}

.k7 {
    border-top:
        4px solid #16a34a;
}

.k8 {
    border-top:
        4px solid #334155;
}

.quality-box {

    display:
        flex;

    align-items:
        center;

    gap:
        18px;

    padding:
        16px;

    border-radius:
        14px;

    background:
        linear-gradient(
            135deg,
            #eff6ff,
            #ecfeff
        );

    border:
        1px solid
        #cfe1f5;

    margin-bottom:
        10px;
}

.quality-number {

    font-size:
        34px;

    font-weight:
        700;

    color:
        #1d4ed8;
}

.notice {

    padding:
        12px 14px;

    border-radius:
        10px;

    background:
        #eff6ff;

    border:
        1px solid
        #bfdbfe;

    color:
        #1e40af;
}

.error {

    padding:
        12px 14px;

    border-radius:
        10px;

    background:
        #fef2f2;

    border:
        1px solid
        #fecaca;

    color:
        #991b1b;
}

button.primary {

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #0891b2
        ) !important;

    color:
        white !important;

    border:
        0 !important;
}

button.secondary {

    background:
        #eaf2ff !important;

    color:
        #1e40af !important;
}

button.danger {

    background:
        #fee2e2 !important;

    color:
        #991b1b !important;
}

textarea,
input,
select {

    border-radius:
        10px !important;

    border-color:
        #cbd5e1 !important;
}

.gradio-tab-nav {

    overflow-x:
        auto !important;

    white-space:
        nowrap !important;
}

.gradio-tab-nav button {

    font-weight:
        600 !important;
}

.dataframe {

    border-radius:
        12px !important;

    overflow:
        hidden !important;
}

footer {

    display:
        none !important;
}

@media(max-width:1000px) {

    .kpi-grid {

        grid-template-columns:
            repeat(
                4,
                minmax(
                    100px,
                    1fr
                )
            );
    }
}

@media(max-width:650px) {

    .kpi-grid {

        grid-template-columns:
            repeat(
                2,
                minmax(
                    100px,
                    1fr
                )
            );
    }
}

"""


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="AI & ML Analytics Studio",
    css=CSS,
    theme=gr.themes.Soft()
) as app:

    gr.HTML("""
    <div class="header">

        <h1>
            AI & ML Analytics Studio
        </h1>

        <p>
            Data analysis, cleaning, visualization,
            healthcare-aware validation and machine learning.
        </p>

    </div>
    """)

    status = gr.Markdown(
        "Upload a CSV, XLSX, or XLS dataset to begin."
    )


    # ========================================================
    # OVERVIEW
    # ========================================================

    with gr.Tab("Overview"):

        with gr.Row():

            upload = gr.File(
                label="Upload Dataset",
                file_types=[
                    ".csv",
                    ".xlsx",
                    ".xls"
                ],
                type="filepath"
            )

            reset_all_btn = gr.Button(
                "Reset Dashboard",
                variant="stop"
            )

        overview = gr.HTML(
            "<div class='notice'>"
            "No dataset loaded."
            "</div>"
        )

        with gr.Row():

            original_preview = gr.Dataframe(
                label="Original Dataset Preview",
                interactive=False
            )

            overview_info = gr.Dataframe(
                label="Column Information",
                interactive=False
            )


    # ========================================================
    # VIEWER
    # ========================================================

    with gr.Tab("Viewer"):

        with gr.Row():

            viewer_dataset = gr.Dropdown(
                [
                    "Original",
                    "Processed"
                ],
                value="Processed",
                label="Dataset"
            )

            viewer_type = gr.Dropdown(
                [
                    "Head",
                    "Tail",
                    "Random",
                    "Entire Dataset",
                    "Custom Range"
                ],
                value="Head",
                label="View"
            )

            viewer_rows = gr.Dropdown(
                [
                    10,
                    20,
                    50,
                    100,
                    147,
                    500
                ],
                value=50,
                label="Rows"
            )

        with gr.Row(
            visible=False
        ) as custom_range_row:

            viewer_start = gr.Number(
                value=0,
                precision=0,
                label="Start"
            )

            viewer_end = gr.Number(
                value=147,
                precision=0,
                label="End"
            )

        with gr.Row():

            viewer_search = gr.Textbox(
                label="Search Dataset",
                placeholder=(
                    "Search any value across all columns"
                )
            )

            clear_search_btn = gr.Button(
                "Clear Search",
                variant="secondary"
            )

        viewer_info = gr.Markdown(
            "No dataset loaded."
        )

        viewer_table = gr.Dataframe(
            label="Dataset Viewer",
            interactive=False
        )


    # ========================================================
    # STATISTICS
    # ========================================================

    with gr.Tab("Statistics"):

        with gr.Row():

            stat_cols = gr.Dropdown(
                multiselect=True,
                label="Numerical Columns"
            )

            percentile = gr.Slider(
                1,
                99,
                value=50,
                step=1,
                label="Percentile"
            )

            stat_btn = gr.Button(
                "Calculate Statistics",
                variant="primary"
            )

        stat_status = gr.Markdown()

        stat_table = gr.Dataframe(
            label="Statistical Summary",
            interactive=False
        )


    # ========================================================
    # QUALITY
    # ========================================================

    with gr.Tab("Quality"):

        quality_btn = gr.Button(
            "Refresh Data Quality",
            variant="primary"
        )

        quality_score_html = gr.HTML()

        with gr.Row():

            quality_table = gr.Dataframe(
                label="Data Quality Findings",
                interactive=False
            )

            healthcare_table = gr.Dataframe(
                label="Healthcare-Aware Validation Warnings",
                interactive=False
            )

        unique_table = gr.Dataframe(
            label="Unique Value Analysis",
            interactive=False
        )


    # ========================================================
    # CLEANING
    # ========================================================

    with gr.Tab("Cleaning"):

        gr.Markdown(
            "## Data Cleaning Studio"
        )

        gr.Markdown(
            "Select one operation, preview it, then apply it. "
            "The original dataset is never overwritten."
        )

        with gr.Row():

            clean_operation = gr.Dropdown(
                [
                    "Drop rows",
                    "Drop columns",
                    "Drop columns above selected %",
                    "Mean imputation",
                    "Median imputation",
                    "Mode imputation",
                    "Constant value",
                    "Forward fill",
                    "Backward fill",
                    "Interpolation",
                    "Remove exact duplicates",
                    "Strip whitespace",
                    "Lowercase",
                    "Uppercase",
                    "Title case",
                    "Replace values",
                    "Convert numeric",
                    "Convert category",
                    "Convert date",
                    "Replace negative numeric values",
                    "Replace infinite values",
                    "Replace empty strings",
                    "Replace common null strings",
                    "IQR remove",
                    "IQR cap",
                    "Z-score remove",
                    "Z-score cap"
                ],
                label="Cleaning Operation",
                value="Remove exact duplicates"
            )

            clean_apply_to = gr.Dropdown(
                [
                    "Entire Dataset",
                    "Selected Column"
                ],
                value="Entire Dataset",
                label="Apply To"
            )

            clean_column = gr.Dropdown(
                label="Column",
                visible=False
            )

        with gr.Row():

            clean_constant = gr.Textbox(
                label="Constant Value",
                visible=False
            )

            clean_replace_from = gr.Textbox(
                label="Replace From",
                visible=False
            )

            clean_replace_to = gr.Textbox(
                label="Replace To",
                visible=False
            )

            clean_threshold = gr.Number(
                value=50,
                label="Missing % Threshold",
                visible=False
            )

            clean_z = gr.Number(
                value=3,
                label="Z-score Threshold",
                visible=False
            )

        with gr.Row():

            preview_clean_btn = gr.Button(
                "Preview Change",
                variant="secondary"
            )

            apply_clean_btn = gr.Button(
                "Apply Cleaning",
                variant="primary"
            )

            reset_clean_btn = gr.Button(
                "Reset to Original",
                variant="stop"
            )

        clean_status = gr.Markdown()

        with gr.Row():

            clean_before = gr.Dataframe(
                label="Before Cleaning",
                interactive=False
            )

            clean_after = gr.Dataframe(
                label="Preview / After Cleaning",
                interactive=False
            )

        clean_what = gr.Markdown()

        clean_comparison = gr.Dataframe(
            label="Before / After Comparison",
            interactive=False
        )

        clean_history = gr.Dataframe(
            label="Cleaning Change Log",
            interactive=False
        )


    # ========================================================
    # EDA
    # ========================================================

    with gr.Tab("EDA"):

        with gr.Row():

            graph_type = gr.Dropdown(
                [
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
                    "Time Series",
                    "Missing",
                    "3D Scatter"
                ],
                value="Histogram",
                label="Graph"
            )

            graph_x = gr.Dropdown(
                label="X"
            )

            graph_y = gr.Dropdown(
                label="Y",
                visible=False
            )

            graph_z = gr.Dropdown(
                label="Z",
                visible=False
            )

            graph_hue = gr.Dropdown(
                label="Hue",
                visible=False,
                allow_custom_value=True
            )

            graph_btn = gr.Button(
                "Generate Graph",
                variant="primary"
            )

        graph_plot = gr.Plot(
            label="Visualization"
        )

        graph_status = gr.Markdown()


    # ========================================================
    # TARGET
    # ========================================================

    with gr.Tab("Target"):

        with gr.Row():

            target_col = gr.Dropdown(
                label="Target Column"
            )

            target_task = gr.Radio(
                [
                    "Classification",
                    "Regression"
                ],
                value="Classification",
                label="Task"
            )

            target_btn = gr.Button(
                "Analyze Target",
                variant="primary"
            )

        with gr.Row():

            target_table = gr.Dataframe(
                label="Target Analysis",
                interactive=False
            )

            target_plot = gr.Plot(
                label="Target Distribution"
            )

        target_status = gr.Markdown()


    # ========================================================
    # FEATURES
    # ========================================================

    with gr.Tab("Features"):

        with gr.Row():

            feat_operation = gr.Dropdown(
                [
                    "Log",
                    "Square",
                    "Square root",
                    "Standardization",
                    "MinMax",
                    "Robust scaling",
                    "One-hot encoding",
                    "Ordinal encoding",
                    "Frequency encoding",
                    "Variance filtering",
                    "Correlation filtering",
                    "SelectKBest",
                    "PCA"
                ],
                value="Log",
                label="Feature Engineering"
            )

            feat_cols = gr.Dropdown(
                multiselect=True,
                label="Columns"
            )

            feat_k = gr.Number(
                value=5,
                precision=0,
                label="K / Threshold"
            )

            feat_btn = gr.Button(
                "Apply Feature Engineering",
                variant="primary"
            )

        feat_status = gr.Markdown()

        feat_table = gr.Dataframe(
            label="Feature-Engineered Preview",
            interactive=False
        )


    # ========================================================
    # TRAINING
    # ========================================================

    with gr.Tab("Training"):

        with gr.Row():

            train_task = gr.Radio(
                [
                    "Classification",
                    "Regression"
                ],
                value="Classification",
                label="Task"
            )

            train_target = gr.Dropdown(
                label="Target"
            )

            train_model_name = gr.Dropdown(
                (
                    [
                        "Logistic Regression",
                        "KNN",
                        "Naive Bayes",
                        "Decision Tree",
                        "Random Forest",
                        "SVM",
                        "Gradient Boosting"
                    ]
                    +
                    (
                        ["XGBoost"]
                        if XGB_AVAILABLE
                        else []
                    )
                ),
                value="Random Forest",
                label="Model"
            )

        with gr.Row():

            train_test = gr.Slider(
                .1,
                .4,
                value=.2,
                step=.05,
                label="Test Size"
            )

            train_scaling = gr.Dropdown(
                [
                    "Standard",
                    "MinMax",
                    "Robust",
                    "None"
                ],
                value="Standard",
                label="Scaling"
            )

            train_encoding = gr.Dropdown(
                [
                    "One-Hot",
                    "Ordinal"
                ],
                value="One-Hot",
                label="Encoding"
            )

            train_cv = gr.Dropdown(
                [
                    3,
                    5,
                    10
                ],
                value=5,
                label="CV Folds"
            )

        train_btn = gr.Button(
            "Train Model",
            variant="primary"
        )

        train_status = gr.Markdown()

        train_metrics = gr.Dataframe(
            label="Initial Evaluation Metrics",
            interactive=False
        )

        train_importance = gr.Dataframe(
            label="Feature Importance",
            interactive=False
        )

        train_prediction_template = gr.Dataframe(
            label="Prediction Input — exactly two columns",
            headers=[
                "Feature",
                "Value"
            ],
            datatype=[
                "str",
                "str"
            ],
            column_count=(2, 2),
            interactive=True
        )


    # ========================================================
    # EVALUATION
    # ========================================================

    with gr.Tab("Evaluation"):

        with gr.Row():

            eval_kind = gr.Dropdown(
                [
                    "Confusion Matrix",
                    "ROC",
                    "Precision-Recall",
                    "Actual vs Predicted",
                    "Residuals"
                ],
                value="Confusion Matrix",
                label="Evaluation Plot"
            )

            eval_btn = gr.Button(
                "Generate Evaluation",
                variant="primary"
            )

        eval_plot = gr.Plot()

        eval_status = gr.Markdown()

        eval_metrics = gr.Dataframe(
            label="Evaluation Metrics",
            interactive=False
        )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    with gr.Tab("Comparison"):

        with gr.Row():

            compare_btn = gr.Button(
                "Compare Models",
                variant="primary"
            )

            cv_folds = gr.Dropdown(
                [
                    3,
                    5,
                    10
                ],
                value=5,
                label="Cross-Validation Folds"
            )

            cv_btn = gr.Button(
                "Run Cross-Validation",
                variant="secondary"
            )

        compare_table = gr.Dataframe(
            label="Model Leaderboard",
            interactive=False
        )

        compare_status = gr.Markdown()

        cv_table = gr.Dataframe(
            label="Cross-Validation Results",
            interactive=False
        )

        cv_status = gr.Markdown()


    # ========================================================
    # TUNING
    # ========================================================

    with gr.Tab("Tuning"):

        with gr.Row():

            tune_search = gr.Dropdown(
                [
                    "GridSearchCV",
                    "RandomizedSearchCV"
                ],
                value="GridSearchCV",
                label="Search"
            )

            tune_task = gr.Radio(
                [
                    "Classification",
                    "Regression"
                ],
                value="Classification",
                label="Task"
            )

            tune_target = gr.Dropdown(
                label="Target"
            )

            tune_model_choice = gr.Dropdown(
                label="Model"
            )

            tune_folds = gr.Dropdown(
                [
                    3,
                    5,
                    10
                ],
                value=3,
                label="CV Folds"
            )

            tune_btn = gr.Button(
                "Tune Model",
                variant="primary"
            )

        tune_table = gr.Dataframe(
            label="Tuning Results",
            interactive=False
        )

        tune_best = gr.Textbox(
            label="Best Parameters"
        )

        tune_status = gr.Markdown()


    # ========================================================
    # PREDICTION
    # ========================================================

    with gr.Tab("Prediction"):

        gr.Markdown(
            "After training, edit the Value column only. "
            "The Feature column is fixed to the training feature order."
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
            column_count=(2, 2),
            label="Prediction Input",
            interactive=True
        )

        predict_btn = gr.Button(
            "Predict",
            variant="primary"
        )

        prediction_output = gr.Dataframe(
            headers=[
                "Result",
                "Value"
            ],
            datatype=[
                "str",
                "str"
            ],
            column_count=(2, 2),
            label="Prediction Result",
            interactive=False
        )

        prediction_status = gr.Markdown()


    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    with gr.Tab("Explainability"):

        with gr.Row():

            explain_method = gr.Dropdown(
                [
                    "Feature Importance",
                    "Permutation Importance",
                    "Coefficients",
                    "SHAP"
                ],
                value="Feature Importance",
                label="Method"
            )

            explain_btn = gr.Button(
                "Explain Model",
                variant="primary"
            )

        explain_table = gr.Dataframe(
            label="Explainability Results",
            interactive=False
        )

        explain_status = gr.Markdown()


    # ========================================================
    # EXPORT
    # ========================================================

    with gr.Tab("Export"):

        with gr.Row():

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
                    "Complete Analysis Summary"
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

            export_btn = gr.Button(
                "Generate Download",
                variant="primary"
            )

        export_file_out = gr.File(
            label="Generated File",
            interactive=False
        )

        export_status = gr.Markdown()


    # ========================================================
    # EVENT WIRING
    # ========================================================

    upload_outputs = [
        status,
        overview,
        original_preview,
        overview_info,
        stat_cols,
        feat_cols,
        graph_x,
        graph_y,
        target_col,
        train_target,
        tune_target
    ]


    upload.upload(
        load_dataset,
        inputs=[upload],
        outputs=upload_outputs
    )


    def refresh_choices():

        if APP["processed_df"] is not None:

            columns = (
                APP["processed_df"]
                .columns
                .tolist()
            )

            numerical = (
                numeric_cols(
                    APP["processed_df"]
                )
            )

        else:

            columns = []
            numerical = []

        return (

            gr.update(
                choices=columns,
                value=(
                    columns[0]
                    if columns
                    else None
                )
            ),

            gr.update(
                choices=columns,
                value=None
            ),

            gr.update(
                choices=columns,
                value=(
                    columns[0]
                    if columns
                    else None
                )
            ),

            gr.update(
                choices=columns,
                value=(
                    columns[0]
                    if columns
                    else None
                )
            ),

            gr.update(
                choices=columns,
                value=(
                    columns[0]
                    if columns
                    else None
                )
            ),

            gr.update(
                choices=numerical,
                value=(
                    numerical[:1]
                )
            ),

            gr.update(
                choices=columns,
                value=columns[:1]
            )
        )


    upload.upload(
        refresh_choices,
        inputs=[],
        outputs=[
            clean_column,
            graph_hue,
            target_col,
            train_target,
            tune_target,
            stat_cols,
            feat_cols
        ]
    )


    # ========================================================
    # VIEWER EVENTS
    # ========================================================

    viewer_type.change(
        update_view_controls,
        inputs=[
            viewer_type
        ],
        outputs=[
            viewer_rows,
            custom_range_row,
            viewer_info
        ]
    )


    for component in [
        viewer_dataset,
        viewer_type,
        viewer_rows,
        viewer_start,
        viewer_end,
        viewer_search
    ]:

        component.change(
            make_view,
            inputs=[
                viewer_dataset,
                viewer_type,
                viewer_rows,
                viewer_start,
                viewer_end,
                viewer_search
            ],
            outputs=[
                viewer_table,
                viewer_info
            ]
        )


    clear_search_btn.click(
        clear_search,
        inputs=[
            viewer_dataset,
            viewer_type,
            viewer_rows,
            viewer_start,
            viewer_end
        ],
        outputs=[
            viewer_table,
            viewer_info
        ]
    )


    # ========================================================
    # STATISTICS EVENTS
    # ========================================================

    stat_btn.click(
        run_statistics,
        inputs=[
            stat_cols,
            percentile
        ],
        outputs=[
            stat_table,
            stat_status
        ]
    )


    # ========================================================
    # QUALITY EVENTS
    # ========================================================

    quality_btn.click(
        quality_analysis,
        inputs=[],
        outputs=[
            quality_table,
            quality_score_html,
            healthcare_table,
            unique_table
        ]
    )


    # ========================================================
    # CLEANING EVENTS
    # ========================================================

    clean_operation.change(
        clean_ui_state,
        inputs=[
            clean_operation,
            clean_apply_to
        ],
        outputs=[
            clean_column,
            clean_constant,
            clean_replace_from,
            clean_replace_to,
            clean_threshold,
            clean_z
        ]
    )


    clean_apply_to.change(
        clean_ui_state,
        inputs=[
            clean_operation,
            clean_apply_to
        ],
        outputs=[
            clean_column,
            clean_constant,
            clean_replace_from,
            clean_replace_to,
            clean_threshold,
            clean_z
        ]
    )


    preview_clean_btn.click(
        cleaning_before_preview,
        inputs=[
            clean_operation,
            clean_apply_to,
            clean_column,
            clean_constant,
            clean_replace_from,
            clean_replace_to,
            clean_threshold,
            clean_z
        ],
        outputs=[
            clean_before,
            clean_after,
            clean_what,
            clean_comparison
        ]
    )


    apply_clean_btn.click(
        apply_cleaning,
        inputs=[
            clean_operation,
            clean_apply_to,
            clean_column,
            clean_constant,
            clean_replace_from,
            clean_replace_to,
            clean_threshold,
            clean_z
        ],
        outputs=[
            clean_after,
            overview,
            clean_comparison,
            clean_history,
            clean_status
        ]
    )


    reset_clean_btn.click(
        reset_cleaning,
        inputs=[],
        outputs=[
            clean_after,
            overview,
            clean_comparison,
            clean_history,
            clean_status
        ]
    )


    # ========================================================
    # EDA EVENTS
    # ========================================================

    graph_type.change(
        graph_controls,
        inputs=[
            graph_type
        ],
        outputs=[
            graph_x,
            graph_y,
            graph_z,
            graph_hue
        ]
    )


    graph_btn.click(
        plot_eda,
        inputs=[
            graph_type,
            graph_x,
            graph_y,
            graph_z,
            graph_hue
        ],
        outputs=[
            graph_plot,
            graph_status
        ]
    )


    # ========================================================
    # TARGET EVENTS
    # ========================================================

    target_btn.click(
        target_analysis,
        inputs=[
            target_col,
            target_task
        ],
        outputs=[
            target_table,
            target_plot,
            target_status
        ]
    )


    # ========================================================
    # FEATURE EVENTS
    # ========================================================

    feat_btn.click(
        feature_engineering,
        inputs=[
            feat_operation,
            feat_cols,
            feat_k
        ],
        outputs=[
            feat_table,
            feat_status
        ]
    )


    # ========================================================
    # TRAINING MODEL OPTIONS
    # ========================================================

    def train_model_choices(
        task
    ):

        if task == "Classification":

            choices = [
                "Logistic Regression",
                "KNN",
                "Naive Bayes",
                "Decision Tree",
                "Random Forest",
                "SVM",
                "Gradient Boosting"
            ]

            if XGB_AVAILABLE:

                choices.append(
                    "XGBoost"
                )

            return gr.update(
                choices=choices,
                value="Random Forest"
            )

        choices = [
            "Linear Regression",
            "Polynomial Regression",
            "Ridge",
            "Lasso",
            "Decision Tree",
            "Random Forest",
            "SVR",
            "Gradient Boosting"
        ]

        if XGB_AVAILABLE:

            choices.append(
                "XGBoost"
            )

        return gr.update(
            choices=choices,
            value="Random Forest"
        )


    train_task.change(
        train_model_choices,
        inputs=[
            train_task
        ],
        outputs=[
            train_model_name
        ]
    )


    train_btn.click(
        train_model,
        inputs=[
            train_task,
            train_target,
            train_model_name,
            train_test,
            train_scaling,
            train_encoding,
            train_cv
        ],
        outputs=[
            train_metrics,
            train_importance,
            train_prediction_template,
            compare_table,
            train_status
        ]
    )


    # ========================================================
    # EVALUATION EVENTS
    # ========================================================

    eval_btn.click(
        evaluation_plot,
        inputs=[
            eval_kind
        ],
        outputs=[
            eval_plot,
            eval_status
        ]
    )


    eval_btn.click(
        lambda:
            APP["evaluation_metrics"],
        inputs=[],
        outputs=[
            eval_metrics
        ]
    )


    # ========================================================
    # COMPARISON EVENTS
    # ========================================================

    def compare_current_models():

        if APP["trained_model"] is None:

            return (
                pd.DataFrame(),
                "Train a model first."
            )

        result = run_model_comparison(
            APP["task_type"],
            APP["target_column"],
            .2,
            "Standard",
            "One-Hot",
            5,
            ""
        )

        APP["model_results"] = (
            result.copy()
            if isinstance(
                result,
                pd.DataFrame
            )
            else pd.DataFrame()
        )

        return (
            result,
            "Model comparison complete."
        )


    compare_btn.click(
        compare_current_models,
        inputs=[],
        outputs=[
            compare_table,
            compare_status
        ]
    )


    cv_btn.click(
        run_cross_validation,
        inputs=[
            cv_folds
        ],
        outputs=[
            cv_table,
            cv_status
        ]
    )


    # ========================================================
    # TUNING MODEL OPTIONS
    # ========================================================

    def tuning_model_choices(
        task
    ):

        if task == "Classification":

            choices = [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest"
            ]

            if XGB_AVAILABLE:

                choices.append(
                    "XGBoost"
                )

        else:

            choices = [
                "Ridge",
                "Lasso",
                "Decision Tree",
                "Random Forest"
            ]

            if XGB_AVAILABLE:

                choices.append(
                    "XGBoost"
                )

        return gr.update(
            choices=choices,
            value=choices[0]
        )


    tune_task.change(
        tuning_model_choices,
        inputs=[
            tune_task
        ],
        outputs=[
            tune_model_choice
        ]
    )


    tune_btn.click(
        tune_model,
        inputs=[
            tune_search,
            tune_task,
            tune_target,
            tune_model_choice,
            tune_folds
        ],
        outputs=[
            tune_table,
            tune_best,
            tune_status
        ]
    )


    # ========================================================
    # PREDICTION EVENTS
    # ========================================================

    predict_btn.click(
        predict_from_table,
        inputs=[
            prediction_input
        ],
        outputs=[
            prediction_output,
            prediction_status
        ]
    )


    # ========================================================
    # EXPLAINABILITY EVENTS
    # ========================================================

    explain_btn.click(
        explain_model,
        inputs=[
            explain_method
        ],
        outputs=[
            explain_table,
            explain_status
        ]
    )


    # ========================================================
    # EXPORT EVENTS
    # ========================================================

    export_btn.click(
        export_file,
        inputs=[
            export_type,
            export_format
        ],
        outputs=[
            export_file_out,
            export_status
        ]
    )


    # ========================================================
    # COMPLETE DASHBOARD RESET
    # ========================================================

    reset_all_btn.click(
        reset_all,
        inputs=[],
        outputs=[
            status,
            overview,
            original_preview,
            overview_info,
            viewer_table,
            stat_table,
            quality_table,
            healthcare_table,
            unique_table,
            clean_before,
            clean_after,
            clean_comparison,
            clean_history
        ]
    )


# ============================================================
# LAUNCH
# ============================================================

print(
    "AI & ML Analytics Studio is ready."
)

app.launch(
    share=True,
    debug=False
)