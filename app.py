# ============================================================
# AI AND ML ANALYTICS DASHBOARD
# Complete single-file Gradio application
# ============================================================
#
# Required packages:
# gradio>=5,<6
# pandas
# numpy
# matplotlib
# seaborn
# scikit-learn
# openpyxl
# xlrd
# scipy
#
# Save this entire code as:
# app.py
#
# Do NOT add !pip install, %pip, ```python, or notebook commands.
# ============================================================

import os
import tempfile
import warnings
from io import BytesIO

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew, kurtosis, zscore

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

warnings.filterwarnings("ignore")

# ============================================================
# GLOBAL STYLE
# ============================================================

sns.set_theme(style="whitegrid")

CSS = """
* {
    font-family: "Times New Roman", Times, serif !important;
}

body {
    background: #f4f6f8 !important;
}

.gradio-container {
    max-width: 1500px !important;
    margin: auto !important;
    background: #f4f6f8 !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: "Times New Roman", Times, serif !important;
    color: #172033 !important;
}

p, label, span, div, button, input, textarea, select {
    font-family: "Times New Roman", Times, serif !important;
}

button {
    border-radius: 6px !important;
}

.primary-button {
    background: #1f4e79 !important;
    color: white !important;
}

.danger-button {
    background: #a61b1b !important;
    color: white !important;
}

.success-button {
    background: #276749 !important;
    color: white !important;
}

.section-box {
    border: 1px solid #d8dee8 !important;
    border-radius: 8px !important;
    padding: 18px !important;
    background: white !important;
}

.status-box {
    border-radius: 6px !important;
    padding: 10px !important;
}

.gradio-container .tabs {
    border-bottom: 1px solid #cfd6df !important;
}

.gradio-container .tab-nav button {
    font-size: 16px !important;
    padding: 10px 14px !important;
}

.dataframe {
    font-size: 13px !important;
}
"""

# ============================================================
# HELPER FUNCTIONS
# ============================================================


def safe_copy(df):
    if df is None:
        return None
    return df.copy(deep=True)


def get_numeric_columns(df):
    if df is None:
        return []
    return df.select_dtypes(include=np.number).columns.tolist()


def get_categorical_columns(df):
    if df is None:
        return []
    return df.select_dtypes(exclude=np.number).columns.tolist()


def dataframe_message(df, max_rows=15):
    if df is None:
        return pd.DataFrame()
    return df.head(max_rows)


def clean_column_names(df):
    df = df.copy()
    new_columns = []

    for col in df.columns:
        col = str(col).strip()
        col = " ".join(col.split())
        new_columns.append(col)

    df.columns = new_columns
    return df


def format_number(value):
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return str(value)


def make_download_file(df, fmt):
    if df is None or df.empty:
        return None

    suffix = ".csv" if fmt == "CSV" else ".xlsx"

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    temp.close()

    if fmt == "CSV":
        df.to_csv(temp.name, index=False)
    else:
        df.to_excel(temp.name, index=False)

    return temp.name


def empty_plot():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(
        0.5,
        0.5,
        "No data available",
        ha="center",
        va="center",
        fontsize=14,
    )
    ax.axis("off")
    return fig


# ============================================================
# FILE UPLOAD
# ============================================================


def load_dataset(file):
    if file is None:
        return (
            None,
            None,
            "",
            "No dataset selected.",
            pd.DataFrame(),
            "",
        )

    try:
        filepath = file if isinstance(file, str) else file.name

        extension = os.path.splitext(filepath)[1].lower()

        if extension == ".csv":
            df = pd.read_csv(filepath)

        elif extension in [".xlsx", ".xls"]:
            df = pd.read_excel(filepath)

        else:
            return (
                None,
                None,
                "",
                "Unsupported file format. Please upload CSV, XLSX, or XLS.",
                pd.DataFrame(),
                "",
            )

        df = clean_column_names(df)

        original = df.copy(deep=True)
        current = df.copy(deep=True)

        filename = os.path.basename(filepath)

        summary = (
            f"Dataset loaded successfully.\n\n"
            f"File: {filename}\n"
            f"Rows: {len(df):,}\n"
            f"Columns: {len(df.columns):,}\n"
            f"Missing cells: {int(df.isna().sum().sum()):,}\n"
            f"Duplicate rows: {int(df.duplicated().sum()):,}"
        )

        return (
            original,
            current,
            filename,
            summary,
            dataframe_message(current, 15),
            "Dataset loaded successfully.",
        )

    except Exception as e:
        return (
            None,
            None,
            "",
            f"Unable to load dataset.\n\nError: {str(e)}",
            pd.DataFrame(),
            "Upload failed.",
        )


# ============================================================
# RESET DATASET
# ============================================================


def reset_dataset(original_df):
    if original_df is None:
        return (
            None,
            pd.DataFrame(),
            "There is no original dataset to reset.",
        )

    restored = original_df.copy(deep=True)

    return (
        restored,
        dataframe_message(restored, 15),
        "Dataset successfully reset to the original uploaded data.",
    )


# ============================================================
# OVERVIEW
# ============================================================


def overview_analysis(df):
    if df is None:
        return (
            "No dataset loaded.",
            pd.DataFrame(),
            pd.DataFrame(),
        )

    numeric = get_numeric_columns(df)
    categorical = get_categorical_columns(df)

    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    summary = f"""
DATASET OVERVIEW

Rows                  : {len(df):,}
Columns               : {len(df.columns):,}
Numeric columns       : {len(numeric):,}
Categorical columns   : {len(categorical):,}
Missing cells         : {missing:,}
Duplicate rows        : {duplicates:,}
Memory usage          : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
"""

    column_rows = []

    for col in df.columns:
        column_rows.append(
            {
                "Column": col,
                "Data Type": str(df[col].dtype),
                "Non-Null": int(df[col].notna().sum()),
                "Missing": int(df[col].isna().sum()),
                "Missing %": round(df[col].isna().mean() * 100, 2),
                "Unique": int(df[col].nunique(dropna=True)),
            }
        )

    return (
        summary,
        pd.DataFrame(column_rows),
        dataframe_message(df, 20),
    )


# ============================================================
# STATISTICS
# ============================================================


def statistics_analysis(df):
    if df is None:
        return pd.DataFrame(), pd.DataFrame(), "No dataset loaded."

    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            "No numeric columns are available for statistical analysis.",
        )

    stats = numeric.describe().T.reset_index()
    stats = stats.rename(columns={"index": "Column"})

    extra_rows = []

    for col in numeric.columns:
        series = numeric[col].dropna()

        if len(series) > 2:
            try:
                sk = float(skew(series))
                ku = float(kurtosis(series))
            except Exception:
                sk = np.nan
                ku = np.nan
        else:
            sk = np.nan
            ku = np.nan

        extra_rows.append(
            {
                "Column": col,
                "Skewness": sk,
                "Kurtosis": ku,
                "Zero Count": int((series == 0).sum()),
                "Negative Count": int((series < 0).sum()),
            }
        )

    extra = pd.DataFrame(extra_rows)

    return (
        stats,
        extra,
        "Statistical analysis completed.",
    )


# ============================================================
# VALUE COUNTS
# ============================================================


def value_counts_analysis(df, column):
    if df is None:
        return pd.DataFrame()

    if not column or column not in df.columns:
        return pd.DataFrame()

    counts = (
        df[column]
        .value_counts(dropna=False)
        .reset_index()
    )

    counts.columns = ["Value", "Count"]

    counts["Percentage"] = (
        counts["Count"] / len(df) * 100
    ).round(2)

    return counts


# ============================================================
# DATA QUALITY
# ============================================================


def data_quality_analysis(df):
    if df is None:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            "No dataset loaded.",
        )

    missing = []

    for col in df.columns:
        count = int(df[col].isna().sum())

        missing.append(
            {
                "Column": col,
                "Missing Count": count,
                "Missing %": round(
                    count / len(df) * 100,
                    2
                ),
                "Present Count": int(df[col].notna().sum()),
            }
        )

    duplicates = pd.DataFrame(
        [
            {
                "Total Rows": len(df),
                "Duplicate Rows": int(df.duplicated().sum()),
                "Unique Rows": int(df.drop_duplicates().shape[0]),
            }
        ]
    )

    quality = []

    for col in df.columns:
        series = df[col]

        quality.append(
            {
                "Column": col,
                "Type": str(series.dtype),
                "Unique Values": int(series.nunique(dropna=True)),
                "Missing": int(series.isna().sum()),
                "Missing %": round(series.isna().mean() * 100, 2),
                "Constant": bool(series.nunique(dropna=False) <= 1),
                "High Cardinality": bool(
                    series.nunique(dropna=True) > max(50, len(df) * 0.8)
                ),
            }
        )

    return (
        pd.DataFrame(missing),
        duplicates,
        pd.DataFrame(quality),
        "Data quality analysis completed.",
    )


# ============================================================
# EDA PLOTS
# ============================================================


def histogram_plot(df, column):
    if df is None or not column or column not in df.columns:
        return empty_plot()

    if not pd.api.types.is_numeric_dtype(df[column]):
        return empty_plot()

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        df[column].dropna(),
        kde=True,
        ax=ax,
    )

    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    plt.tight_layout()
    return fig


def box_plot(df, column):
    if df is None or not column or column not in df.columns:
        return empty_plot()

    if not pd.api.types.is_numeric_dtype(df[column]):
        return empty_plot()

    fig, ax = plt.subplots(figsize=(9, 4))

    sns.boxplot(
        x=df[column],
        ax=ax,
    )

    ax.set_title(f"Box Plot of {column}")
    ax.set_xlabel(column)

    plt.tight_layout()
    return fig


def bar_plot(df, column):
    if df is None or not column or column not in df.columns:
        return empty_plot()

    counts = df[column].astype(str).value_counts().head(20)

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=counts.values,
        y=counts.index,
        ax=ax,
    )

    ax.set_title(f"Top Values of {column}")
    ax.set_xlabel("Count")
    ax.set_ylabel(column)

    plt.tight_layout()
    return fig


def correlation_plot(df):
    if df is None:
        return empty_plot()

    numeric = df.select_dtypes(include=np.number)

    if numeric.shape[1] < 2:
        return empty_plot()

    corr = numeric.corr()

    fig, ax = plt.subplots(
        figsize=(
            max(8, len(corr.columns) * 0.7),
            max(6, len(corr.columns) * 0.6),
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

    ax.set_title("Correlation Heatmap")

    plt.tight_layout()
    return fig


def scatter_plot(df, x_column, y_column):
    if (
        df is None
        or not x_column
        or not y_column
        or x_column not in df.columns
        or y_column not in df.columns
    ):
        return empty_plot()

    if not pd.api.types.is_numeric_dtype(df[x_column]):
        return empty_plot()

    if not pd.api.types.is_numeric_dtype(df[y_column]):
        return empty_plot()

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.scatterplot(
        data=df,
        x=x_column,
        y=y_column,
        ax=ax,
    )

    ax.set_title(f"{y_column} vs {x_column}")

    plt.tight_layout()
    return fig


# ============================================================
# OUTLIER ANALYSIS
# ============================================================


def outlier_report(df):
    if df is None:
        return pd.DataFrame(), "No dataset loaded."

    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:
        return (
            pd.DataFrame(),
            "No numeric columns available.",
        )

    rows = []

    for col in numeric.columns:
        series = numeric[col].dropna()

        if len(series) < 4:
            rows.append(
                {
                    "Column": col,
                    "IQR Outliers": 0,
                    "IQR %": 0,
                    "Z Score Outliers": 0,
                    "Minimum": series.min(),
                    "Maximum": series.max(),
                }
            )
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        iqr_count = int(
            ((series < lower) | (series > upper)).sum()
        )

        try:
            z_count = int(
                (np.abs(zscore(series)) > 3).sum()
            )
        except Exception:
            z_count = 0

        rows.append(
            {
                "Column": col,
                "IQR Outliers": iqr_count,
                "IQR %": round(iqr_count / len(series) * 100, 2),
                "Z Score Outliers": z_count,
                "Minimum": series.min(),
                "Maximum": series.max(),
            }
        )

    return (
        pd.DataFrame(rows),
        "Outlier analysis completed.",
    )


# ============================================================
# TARGET ANALYSIS
# ============================================================


def target_analysis(df, target):
    if df is None or not target or target not in df.columns:
        return pd.DataFrame(), empty_plot(), "Select a target column."

    counts = (
        df[target]
        .value_counts(dropna=False)
        .reset_index()
    )

    counts.columns = ["Value", "Count"]

    if pd.api.types.is_numeric_dtype(df[target]):
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(df[target].dropna(), kde=True, ax=ax)
        ax.set_title(f"Target Distribution: {target}")
        plt.tight_layout()
    else:
        top = df[target].astype(str).value_counts().head(20)

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(
            x=top.values,
            y=top.index,
            ax=ax,
        )
        ax.set_title(f"Target Distribution: {target}")
        plt.tight_layout()

    return (
        counts,
        fig,
        f"Target analysis completed for {target}.",
    )


# ============================================================
# FEATURE ANALYSIS
# ============================================================


def feature_analysis(df):
    if df is None:
        return pd.DataFrame(), "No dataset loaded."

    rows = []

    for col in df.columns:
        series = df[col]

        row = {
            "Feature": col,
            "Type": str(series.dtype),
            "Unique": int(series.nunique(dropna=True)),
            "Missing": int(series.isna().sum()),
            "Missing %": round(series.isna().mean() * 100, 2),
        }

        if pd.api.types.is_numeric_dtype(series):
            values = series.dropna()

            if len(values) > 1:
                row["Mean"] = values.mean()
                row["Median"] = values.median()
                row["Std"] = values.std()
                row["Min"] = values.min()
                row["Max"] = values.max()
            else:
                row["Mean"] = np.nan
                row["Median"] = np.nan
                row["Std"] = np.nan
                row["Min"] = np.nan
                row["Max"] = np.nan
        else:
            row["Mean"] = ""
            row["Median"] = ""
            row["Std"] = ""
            row["Min"] = ""
            row["Max"] = ""

        rows.append(row)

    return (
        pd.DataFrame(rows),
        "Feature analysis completed.",
    )


# ============================================================
# CLEANING ENGINE
# ============================================================


def apply_cleaning(
    current_df,
    duplicate_treatment,
    numeric_missing_treatment,
    categorical_missing_treatment,
    constant_value,
    numeric_outlier_treatment,
    outlier_column,
    outlier_threshold,
    convert_numeric,
    convert_datetime,
    trim_text,
    standardize_text,
):
    if current_df is None:
        return (
            None,
            pd.DataFrame(),
            "No dataset is loaded.",
            "",
        )

    try:
        df = current_df.copy(deep=True)

        before_rows = len(df)
        before_columns = len(df.columns)
        before_missing = int(df.isna().sum().sum())
        before_duplicates = int(df.duplicated().sum())

        operations = []

        # ----------------------------------------------------
        # REMOVE DUPLICATES
        # ----------------------------------------------------

        if duplicate_treatment == "Remove duplicates":
            old_rows = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            removed = old_rows - len(df)

            operations.append(
                f"Duplicate rows removed: {removed:,}"
            )

        # ----------------------------------------------------
        # TRIM TEXT
        # ----------------------------------------------------

        if trim_text:
            text_columns = df.select_dtypes(
                include=["object", "string"]
            ).columns

            for col in text_columns:
                df[col] = df[col].apply(
                    lambda x: x.strip()
                    if isinstance(x, str)
                    else x
                )

            operations.append(
                "Leading and trailing spaces removed from text columns."
            )

        # ----------------------------------------------------
        # STANDARDIZE TEXT
        # ----------------------------------------------------

        if standardize_text:
            text_columns = df.select_dtypes(
                include=["object", "string"]
            ).columns

            for col in text_columns:
                df[col] = df[col].apply(
                    lambda x: x.lower()
                    if isinstance(x, str)
                    else x
                )

            operations.append(
                "Text values converted to lowercase."
            )

        # ----------------------------------------------------
        # NUMERIC CONVERSION
        # ----------------------------------------------------

        if convert_numeric:
            converted = 0

            for col in df.columns:
                if df[col].dtype == "object":
                    converted_series = pd.to_numeric(
                        df[col],
                        errors="coerce",
                    )

                    original_non_null = df[col].notna().sum()

                    if (
                        converted_series.notna().sum()
                        >= original_non_null * 0.8
                    ):
                        df[col] = converted_series
                        converted += 1

            operations.append(
                f"Automatic numeric conversion attempted on {converted} column(s)."
            )

        # ----------------------------------------------------
        # DATETIME CONVERSION
        # ----------------------------------------------------

        if convert_datetime:
            converted = 0

            for col in df.columns:
                if df[col].dtype == "object":
                    converted_series = pd.to_datetime(
                        df[col],
                        errors="coerce",
                    )

                    original_non_null = df[col].notna().sum()

                    if (
                        original_non_null > 0
                        and converted_series.notna().sum()
                        >= original_non_null * 0.8
                    ):
                        df[col] = converted_series
                        converted += 1

            operations.append(
                f"Automatic date/time conversion attempted on {converted} column(s)."
            )

        # ----------------------------------------------------
        # NUMERIC MISSING VALUES
        # ----------------------------------------------------

        numeric_columns = get_numeric_columns(df)

        if numeric_missing_treatment != "Do nothing":
            for col in numeric_columns:
                if df[col].isna().any():

                    if numeric_missing_treatment == "Mean":
                        value = df[col].mean()
                        df[col] = df[col].fillna(value)

                    elif numeric_missing_treatment == "Median":
                        value = df[col].median()
                        df[col] = df[col].fillna(value)

                    elif numeric_missing_treatment == "Zero":
                        df[col] = df[col].fillna(0)

                    elif numeric_missing_treatment == "Forward fill":
                        df[col] = df[col].ffill()

                    elif numeric_missing_treatment == "Backward fill":
                        df[col] = df[col].bfill()

                    elif numeric_missing_treatment == "Constant":
                        df[col] = df[col].fillna(
                            float(constant_value)
                        )

            operations.append(
                f"Numeric missing-value treatment: {numeric_missing_treatment}."
            )

        # ----------------------------------------------------
        # CATEGORICAL MISSING VALUES
        # ----------------------------------------------------

        categorical_columns = get_categorical_columns(df)

        if categorical_missing_treatment != "Do nothing":

            for col in categorical_columns:
                if df[col].isna().any():

                    if categorical_missing_treatment == "Mode":
                        mode = df[col].mode(dropna=True)

                        if len(mode) > 0:
                            df[col] = df[col].fillna(
                                mode.iloc[0]
                            )

                    elif categorical_missing_treatment == "Unknown":
                        df[col] = df[col].fillna("Unknown")

                    elif categorical_missing_treatment == "Forward fill":
                        df[col] = df[col].ffill()

                    elif categorical_missing_treatment == "Backward fill":
                        df[col] = df[col].bfill()

                    elif categorical_missing_treatment == "Constant":
                        df[col] = df[col].fillna(
                            str(constant_value)
                        )

            operations.append(
                f"Categorical missing-value treatment: {categorical_missing_treatment}."
            )

        # ----------------------------------------------------
        # OUTLIER TREATMENT
        # ----------------------------------------------------

        numeric_columns = get_numeric_columns(df)

        if (
            numeric_outlier_treatment != "Do nothing"
            and len(numeric_columns) > 0
        ):

            if (
                outlier_column
                and outlier_column != "All numeric columns"
                and outlier_column in numeric_columns
            ):
                columns_to_process = [outlier_column]
            else:
                columns_to_process = numeric_columns

            threshold = float(outlier_threshold)

            rows_before = len(df)

            if numeric_outlier_treatment == "Remove IQR outliers":

                keep_mask = pd.Series(
                    True,
                    index=df.index,
                )

                for col in columns_to_process:
                    series = df[col]

                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1

                    lower = q1 - threshold * iqr
                    upper = q3 + threshold * iqr

                    keep_mask &= (
                        (series >= lower)
                        | series.isna()
                    ) & (
                        (series <= upper)
                        | series.isna()
                    )

                df = df.loc[keep_mask].reset_index(drop=True)

            elif numeric_outlier_treatment == "Cap IQR outliers":

                for col in columns_to_process:
                    series = df[col]

                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1

                    lower = q1 - threshold * iqr
                    upper = q3 + threshold * iqr

                    df[col] = series.clip(
                        lower=lower,
                        upper=upper,
                    )

            elif numeric_outlier_treatment == "Remove Z-score outliers":

                keep_mask = pd.Series(
                    True,
                    index=df.index,
                )

                for col in columns_to_process:
                    series = df[col]

                    mean = series.mean()
                    std = series.std()

                    if pd.notna(std) and std != 0:
                        z = (
                            (series - mean)
                            / std
                        ).abs()

                        keep_mask &= (
                            (z <= threshold)
                            | series.isna()
                        )

                df = df.loc[keep_mask].reset_index(drop=True)

            rows_changed = rows_before - len(df)

            operations.append(
                f"Outlier treatment: {numeric_outlier_treatment}. "
                f"Rows affected: {rows_changed:,}."
            )

        # ----------------------------------------------------
        # FINAL CLEANUP
        # ----------------------------------------------------

        df = df.reset_index(drop=True)

        after_rows = len(df)
        after_columns = len(df.columns)
        after_missing = int(df.isna().sum().sum())
        after_duplicates = int(df.duplicated().sum())

        summary = (
            "CLEANING COMPLETED\n\n"
            f"Rows before       : {before_rows:,}\n"
            f"Rows after        : {after_rows:,}\n"
            f"Rows removed      : {before_rows - after_rows:,}\n\n"
            f"Columns before    : {before_columns:,}\n"
            f"Columns after     : {after_columns:,}\n\n"
            f"Missing before    : {before_missing:,}\n"
            f"Missing after     : {after_missing:,}\n\n"
            f"Duplicates before : {before_duplicates:,}\n"
            f"Duplicates after  : {after_duplicates:,}\n"
        )

        if operations:
            summary += "\nOPERATIONS APPLIED\n\n"
            summary += "\n".join(
                f"{i}. {item}"
                for i, item in enumerate(
                    operations,
                    start=1,
                )
            )
        else:
            summary += "\nNo cleaning operation was selected."

        audit = (
            f"Cleaning operation completed. "
            f"Rows changed from {before_rows:,} to {after_rows:,}. "
            f"Missing cells changed from {before_missing:,} "
            f"to {after_missing:,}."
        )

        return (
            df,
            dataframe_message(df, 20),
            summary,
            audit,
        )

    except Exception as e:
        return (
            current_df,
            dataframe_message(current_df, 20),
            f"Cleaning could not be completed.\n\nError: {str(e)}",
            "Cleaning failed. Original current data was preserved.",
        )


# ============================================================
# DELETE SELECTED COLUMNS
# ============================================================


def remove_columns(df, columns):
    if df is None:
        return None, pd.DataFrame(), "No dataset loaded."

    if not columns:
        return (
            df,
            dataframe_message(df),
            "No columns selected.",
        )

    try:
        result = df.drop(
            columns=[
                col
                for col in columns
                if col in df.columns
            ]
        ).copy()

        return (
            result,
            dataframe_message(result),
            f"Removed {len(columns)} column(s).",
        )

    except Exception as e:
        return (
            df,
            dataframe_message(df),
            f"Unable to remove columns: {str(e)}",
        )


# ============================================================
# TRAINING HELPERS
# ============================================================


def prepare_ml_data(df, target):
    work = df.copy()

    if target not in work.columns:
        raise ValueError("Target column was not found.")

    work = work.dropna(subset=[target])

    X = work.drop(columns=[target])
    y = work[target]

    # Remove datetime columns
    datetime_cols = X.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    if datetime_cols:
        X = X.drop(columns=datetime_cols)

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    return X, y, numeric_features, categorical_features


def build_preprocessor(
    numeric_features,
    categorical_features,
    scaling,
):
    numeric_steps = [
        (
            "imputer",
            SimpleImputer(strategy="median"),
        )
    ]

    if scaling == "StandardScaler":
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# TRAIN MODEL
# ============================================================


def train_model(
    df,
    problem_type,
    target,
    model_name,
    test_size,
    scaling,
    random_state,
):
    if df is None:
        return (
            "No dataset loaded.",
            pd.DataFrame(),
            empty_plot(),
            None,
        )

    if not target or target not in df.columns:
        return (
            "Please select a valid target column.",
            pd.DataFrame(),
            empty_plot(),
            None,
        )

    try:
        X, y, numeric_features, categorical_features = (
            prepare_ml_data(
                df,
                target,
            )
        )

        if len(X) < 10:
            raise ValueError(
                "At least 10 usable rows are required for training."
            )

        if problem_type == "Classification":

            if y.nunique() < 2:
                raise ValueError(
                    "Classification target must contain at least two classes."
                )

            if model_name == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=200,
                    random_state=int(random_state),
                    class_weight="balanced",
                )

            elif model_name == "Logistic Regression":
                model = LogisticRegression(
                    max_iter=2000,
                )

            elif model_name == "Gradient Boosting":
                model = GradientBoostingClassifier(
                    random_state=int(random_state),
                )

            else:
                model = KNeighborsClassifier(
                    n_neighbors=5,
                )

        else:

            if model_name == "Random Forest":
                model = RandomForestRegressor(
                    n_estimators=200,
                    random_state=int(random_state),
                )

            elif model_name == "Linear Regression":
                model = LinearRegression()

            elif model_name == "Gradient Boosting":
                model = GradientBoostingRegressor(
                    random_state=int(random_state),
                )

            else:
                model = KNeighborsRegressor(
                    n_neighbors=5,
                )

        if problem_type == "Classification":
            stratify_value = y

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=float(test_size),
                    random_state=int(random_state),
                    stratify=stratify_value,
                )
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=float(test_size),
                    random_state=int(random_state),
                )

        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=float(test_size),
                random_state=int(random_state),
            )

        preprocessor = build_preprocessor(
            numeric_features,
            categorical_features,
            scaling,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        if problem_type == "Classification":

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

            metrics = pd.DataFrame(
                {
                    "Metric": [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1 Score",
                    ],
                    "Value": [
                        accuracy,
                        precision,
                        recall,
                        f1,
                    ],
                }
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

            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            plt.tight_layout()

            report = classification_report(
                y_test,
                predictions,
                zero_division=0,
            )

            result_text = (
                "CLASSIFICATION MODEL RESULTS\n\n"
                f"Model: {model_name}\n"
                f"Training rows: {len(X_train):,}\n"
                f"Testing rows: {len(X_test):,}\n\n"
                f"Accuracy: {accuracy:.4f}\n"
                f"Precision: {precision:.4f}\n"
                f"Recall: {recall:.4f}\n"
                f"F1 Score: {f1:.4f}\n\n"
                f"Classification Report:\n{report}"
            )

        else:

            mae = mean_absolute_error(
                y_test,
                predictions,
            )

            mse = mean_squared_error(
                y_test,
                predictions,
            )

            rmse = np.sqrt(mse)

            r2 = r2_score(
                y_test,
                predictions,
            )

            metrics = pd.DataFrame(
                {
                    "Metric": [
                        "MAE",
                        "MSE",
                        "RMSE",
                        "R2 Score",
                    ],
                    "Value": [
                        mae,
                        mse,
                        rmse,
                        r2,
                    ],
                }
            )

            fig, ax = plt.subplots(
                figsize=(8, 6)
            )

            sns.scatterplot(
                x=y_test,
                y=predictions,
                ax=ax,
            )

            minimum = min(
                np.min(y_test),
                np.min(predictions),
            )

            maximum = max(
                np.max(y_test),
                np.max(predictions),
            )

            ax.plot(
                [minimum, maximum],
                [minimum, maximum],
                linestyle="--",
            )

            ax.set_title(
                "Actual vs Predicted"
            )

            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")

            plt.tight_layout()

            result_text = (
                "REGRESSION MODEL RESULTS\n\n"
                f"Model: {model_name}\n"
                f"Training rows: {len(X_train):,}\n"
                f"Testing rows: {len(X_test):,}\n\n"
                f"MAE: {mae:.4f}\n"
                f"MSE: {mse:.4f}\n"
                f"RMSE: {rmse:.4f}\n"
                f"R2 Score: {r2:.4f}"
            )

        return (
            result_text,
            metrics,
            fig,
            pipeline,
        )

    except Exception as e:
        return (
            f"MODEL TRAINING FAILED\n\n{str(e)}",
            pd.DataFrame(),
            empty_plot(),
            None,
        )


# ============================================================
# MODEL COMPARISON
# ============================================================


def compare_models(
    df,
    problem_type,
    target,
    test_size,
    scaling,
    random_state,
):
    if df is None:
        return pd.DataFrame(), "No dataset loaded."

    if not target or target not in df.columns:
        return pd.DataFrame(), "Select a target."

    try:
        X, y, numeric_features, categorical_features = (
            prepare_ml_data(
                df,
                target,
            )
        )

        if problem_type == "Classification":
            models = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=150,
                    random_state=int(random_state),
                    class_weight="balanced",
                ),
                "Logistic Regression": LogisticRegression(
                    max_iter=2000
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    random_state=int(random_state)
                ),
                "KNN": KNeighborsClassifier(
                    n_neighbors=5
                ),
            }
        else:
            models = {
                "Random Forest": RandomForestRegressor(
                    n_estimators=150,
                    random_state=int(random_state),
                ),
                "Linear Regression": LinearRegression(),
                "Gradient Boosting": GradientBoostingRegressor(
                    random_state=int(random_state)
                ),
                "KNN": KNeighborsRegressor(
                    n_neighbors=5
                ),
            }

        rows = []

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=float(test_size),
                random_state=int(random_state),
            )
        except Exception:
            return pd.DataFrame(), "Unable to split the dataset."

        for name, model in models.items():

            try:
                preprocessor = build_preprocessor(
                    numeric_features,
                    categorical_features,
                    scaling,
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

                if problem_type == "Classification":
                    score = accuracy_score(
                        y_test,
                        predictions,
                    )

                    rows.append(
                        {
                            "Model": name,
                            "Accuracy": round(score, 4),
                        }
                    )

                else:
                    score = r2_score(
                        y_test,
                        predictions,
                    )

                    rows.append(
                        {
                            "Model": name,
                            "R2 Score": round(score, 4),
                        }
                    )

            except Exception as model_error:

                rows.append(
                    {
                        "Model": name,
                        "Score": np.nan,
                        "Error": str(model_error),
                    }
                )

        result = pd.DataFrame(rows)

        return (
            result,
            "Model comparison completed.",
        )

    except Exception as e:
        return (
            pd.DataFrame(),
            f"Model comparison failed: {str(e)}",
        )


# ============================================================
# MODEL EVALUATION
# ============================================================


def evaluate_model(
    df,
    problem_type,
    target,
    model_name,
    scaling,
    random_state,
):
    if df is None:
        return pd.DataFrame(), "No dataset loaded."

    if not target or target not in df.columns:
        return pd.DataFrame(), "Select a target."

    try:
        X, y, numeric_features, categorical_features = (
            prepare_ml_data(
                df,
                target,
            )
        )

        preprocessor = build_preprocessor(
            numeric_features,
            categorical_features,
            scaling,
        )

        if problem_type == "Classification":

            if model_name == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=150,
                    random_state=int(random_state),
                )
            elif model_name == "Gradient Boosting":
                model = GradientBoostingClassifier(
                    random_state=int(random_state)
                )
            elif model_name == "KNN":
                model = KNeighborsClassifier(
                    n_neighbors=5
                )
            else:
                model = LogisticRegression(
                    max_iter=2000
                )

            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            scoring = "accuracy"

        else:

            if model_name == "Random Forest":
                model = RandomForestRegressor(
                    n_estimators=150,
                    random_state=int(random_state),
                )
            elif model_name == "Gradient Boosting":
                model = GradientBoostingRegressor(
                    random_state=int(random_state)
                )
            elif model_name == "KNN":
                model = KNeighborsRegressor(
                    n_neighbors=5
                )
            else:
                model = LinearRegression()

            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            scoring = "r2"

        cv_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=5,
            scoring=scoring,
        )

        result = pd.DataFrame(
            {
                "Fold": list(
                    range(1, len(cv_scores) + 1)
                ),
                "Score": cv_scores,
            }
        )

        summary = (
            f"MODEL EVALUATION\n\n"
            f"Model: {model_name}\n"
            f"Cross-validation: 5-fold\n"
            f"Mean score: {cv_scores.mean():.4f}\n"
            f"Standard deviation: {cv_scores.std():.4f}\n"
            f"Minimum: {cv_scores.min():.4f}\n"
            f"Maximum: {cv_scores.max():.4f}"
        )

        return result, summary

    except Exception as e:
        return (
            pd.DataFrame(),
            f"Evaluation failed: {str(e)}",
        )


# ============================================================
# FEATURE IMPORTANCE / EXPLAINABILITY
# ============================================================


def explain_model(
    trained_pipeline,
    target,
):
    if trained_pipeline is None:
        return (
            pd.DataFrame(),
            "Train a model first.",
            empty_plot(),
        )

    try:
        model = trained_pipeline.named_steps["model"]
        preprocessor = trained_pipeline.named_steps[
            "preprocessor"
        ]

        feature_names = (
            preprocessor.get_feature_names_out()
        )

        importance = None

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_

        elif hasattr(model, "coef_"):
            coef = model.coef_

            if coef.ndim == 1:
                importance = np.abs(coef)
            else:
                importance = np.mean(
                    np.abs(coef),
                    axis=0,
                )

        if importance is None:
            return (
                pd.DataFrame(),
                "This model does not expose feature importance.",
                empty_plot(),
            )

        result = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importance,
            }
        ).sort_values(
            "Importance",
            ascending=False,
        )

        result = result.head(30)

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        sns.barplot(
            data=result,
            x="Importance",
            y="Feature",
            ax=ax,
        )

        ax.set_title(
            f"Feature Importance for {target}"
        )

        plt.tight_layout()

        return (
            result,
            "Feature importance generated successfully.",
            fig,
        )

    except Exception as e:
        return (
            pd.DataFrame(),
            f"Explainability could not be generated: {str(e)}",
            empty_plot(),
        )


# ============================================================
# PREDICTION
# ============================================================


def predict_from_uploaded_data(
    trained_pipeline,
    prediction_file,
):
    if trained_pipeline is None:
        return (
            pd.DataFrame(),
            "Train a model before prediction.",
            None,
        )

    if prediction_file is None:
        return (
            pd.DataFrame(),
            "Upload a prediction dataset.",
            None,
        )

    try:
        filepath = (
            prediction_file
            if isinstance(prediction_file, str)
            else prediction_file.name
        )

        extension = os.path.splitext(filepath)[1].lower()

        if extension == ".csv":
            new_df = pd.read_csv(filepath)

        elif extension in [".xlsx", ".xls"]:
            new_df = pd.read_excel(filepath)

        else:
            raise ValueError(
                "Only CSV, XLSX, and XLS are supported."
            )

        new_df = clean_column_names(new_df)

        predictions = trained_pipeline.predict(
            new_df
        )

        result = new_df.copy()
        result["Prediction"] = predictions

        file_path = make_download_file(
            result,
            "CSV",
        )

        return (
            result.head(100),
            f"Prediction completed for {len(result):,} rows.",
            file_path,
        )

    except Exception as e:
        return (
            pd.DataFrame(),
            f"Prediction failed: {str(e)}",
            None,
        )


# ============================================================
# DOWNLOAD CLEANED DATA
# ============================================================


def download_cleaned_data(df, file_format):
    if df is None:
        return (
            None,
            "No cleaned dataset is available.",
        )

    try:
        path = make_download_file(
            df,
            file_format,
        )

        return (
            path,
            f"Cleaned dataset prepared in {file_format} format.",
        )

    except Exception as e:
        return (
            None,
            f"Download preparation failed: {str(e)}",
        )


# ============================================================
# COLUMN CHOICE UPDATE
# ============================================================


def update_column_choices(df):
    if df is None:
        return (
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(choices=[]),
        )

    columns = df.columns.tolist()
    numeric = get_numeric_columns(df)

    outlier_choices = [
        "All numeric columns"
    ] + numeric

    return (
        gr.update(choices=columns, value=columns[0] if columns else None),
        gr.update(choices=columns, value=columns[0] if columns else None),
        gr.update(choices=columns, value=columns[0] if columns else None),
        gr.update(choices=columns, value=columns[0] if columns else None),
        gr.update(
            choices=outlier_choices,
            value=outlier_choices[0]
            if outlier_choices
            else None,
        ),
        gr.update(choices=columns),
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

with gr.Blocks(
    title="AI and ML Analytics Dashboard",
    css=CSS,
    theme=gr.themes.Soft(),
) as dashboard:

    # --------------------------------------------------------
    # SESSION STATE
    # --------------------------------------------------------

    original_state = gr.State(None)
    current_state = gr.State(None)
    filename_state = gr.State("")
    trained_model_state = gr.State(None)

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    gr.Markdown(
        """
        # AI and ML Analytics Dashboard

        **Data Analysis, Data Quality, Visualization, Cleaning, Feature Analysis, Outlier Detection, Machine Learning and Prediction**
        """
    )

    status_box = gr.Markdown(
        "Upload a CSV, XLSX, or XLS dataset to begin."
    )

    # ========================================================
    # DATASET
    # ========================================================

    with gr.Row():

        with gr.Column(scale=2):

            gr.Markdown("## Dataset")

            upload = gr.File(
                label="Upload Dataset",
                file_types=[
                    ".csv",
                    ".xlsx",
                    ".xls",
                ],
                type="filepath",
            )

            loaded_file = gr.Textbox(
                label="Loaded File",
                interactive=False,
            )

            reset_button = gr.Button(
                "Reset to Original",
                variant="secondary",
            )

        with gr.Column(scale=5):

            dataset_preview = gr.Dataframe(
                label="Current Dataset Preview",
                interactive=False,
                wrap=True,
            )

    # ========================================================
    # MAIN TABS
    # ========================================================

    with gr.Tabs():

        # ====================================================
        # OVERVIEW
        # ====================================================

        with gr.Tab("Overview"):

            gr.Markdown("## Dataset Overview")

            overview_button = gr.Button(
                "Refresh Overview",
                variant="primary",
            )

            overview_text = gr.Textbox(
                label="Summary",
                lines=10,
                interactive=False,
            )

            overview_columns = gr.Dataframe(
                label="Column Information",
                interactive=False,
            )

            overview_preview = gr.Dataframe(
                label="Dataset Preview",
                interactive=False,
            )

        # ====================================================
        # STATISTICS
        # ====================================================

        with gr.Tab("Statistics"):

            gr.Markdown("## Statistical Analysis")

            statistics_button = gr.Button(
                "Calculate Statistics",
                variant="primary",
            )

            statistics_table = gr.Dataframe(
                label="Descriptive Statistics",
                interactive=False,
            )

            advanced_statistics_table = gr.Dataframe(
                label="Distribution Statistics",
                interactive=False,
            )

        # ====================================================
        # DATA QUALITY
        # ====================================================

        with gr.Tab("Data Quality"):

            gr.Markdown("## Data Quality Analysis")

            quality_button = gr.Button(
                "Analyze Data Quality",
                variant="primary",
            )

            missing_table = gr.Dataframe(
                label="Missing Value Report",
                interactive=False,
            )

            duplicate_table = gr.Dataframe(
                label="Duplicate Analysis",
                interactive=False,
            )

            quality_table = gr.Dataframe(
                label="Column Quality Report",
                interactive=False,
            )

            gr.Markdown("### Value Counts")

            quality_column = gr.Dropdown(
                label="Select Column",
                choices=[],
            )

            value_count_button = gr.Button(
                "Calculate Value Counts",
            )

            value_count_table = gr.Dataframe(
                label="Value Counts",
                interactive=False,
            )

        # ====================================================
        # EDA
        # ====================================================

        with gr.Tab("EDA"):

            gr.Markdown("## Exploratory Data Analysis")

            with gr.Row():

                histogram_column = gr.Dropdown(
                    label="Histogram Column",
                    choices=[],
                )

                histogram_button = gr.Button(
                    "Create Histogram",
                )

            histogram_output = gr.Plot(
                label="Histogram"
            )

            with gr.Row():

                box_column = gr.Dropdown(
                    label="Box Plot Column",
                    choices=[],
                )

                box_button = gr.Button(
                    "Create Box Plot",
                )

            box_output = gr.Plot(
                label="Box Plot"
            )

            with gr.Row():

                bar_column = gr.Dropdown(
                    label="Bar Chart Column",
                    choices=[],
                )

                bar_button = gr.Button(
                    "Create Bar Chart",
                )

            bar_output = gr.Plot(
                label="Bar Chart"
            )

            correlation_button = gr.Button(
                "Create Correlation Heatmap",
                variant="primary",
            )

            correlation_output = gr.Plot(
                label="Correlation Heatmap"
            )

            with gr.Row():

                scatter_x = gr.Dropdown(
                    label="X Variable",
                    choices=[],
                )

                scatter_y = gr.Dropdown(
                    label="Y Variable",
                    choices=[],
                )

                scatter_button = gr.Button(
                    "Create Scatter Plot",
                )

            scatter_output = gr.Plot(
                label="Scatter Plot"
            )

        # ====================================================
        # CLEANING
        # ====================================================

        with gr.Tab("Cleaning"):

            gr.Markdown("## Data Cleaning")

            gr.Markdown(
                """
                Select the operations you want and then click **Apply Cleaning to Current Data**.
                Every new cleaning operation is applied to the already cleaned dataset.
                Reset always returns to the original uploaded dataset.
                """
            )

            with gr.Row():

                with gr.Column():

                    duplicate_treatment = gr.Dropdown(
                        label="Duplicate Treatment",
                        choices=[
                            "Do nothing",
                            "Remove duplicates",
                        ],
                        value="Do nothing",
                    )

                    numeric_missing = gr.Dropdown(
                        label="Numeric Missing Values",
                        choices=[
                            "Do nothing",
                            "Mean",
                            "Median",
                            "Zero",
                            "Forward fill",
                            "Backward fill",
                            "Constant",
                        ],
                        value="Do nothing",
                    )

                    constant_value = gr.Number(
                        label="Constant Value",
                        value=0,
                    )

                    numeric_outlier = gr.Dropdown(
                        label="Outlier Treatment",
                        choices=[
                            "Do nothing",
                            "Remove IQR outliers",
                            "Cap IQR outliers",
                            "Remove Z-score outliers",
                        ],
                        value="Do nothing",
                    )

                    outlier_column = gr.Dropdown(
                        label="Outlier Column",
                        choices=[
                            "All numeric columns"
                        ],
                        value="All numeric columns",
                    )

                    outlier_threshold = gr.Number(
                        label="Outlier Threshold",
                        value=1.5,
                    )

                with gr.Column():

                    categorical_missing = gr.Dropdown(
                        label="Categorical Missing Values",
                        choices=[
                            "Do nothing",
                            "Mode",
                            "Unknown",
                            "Forward fill",
                            "Backward fill",
                            "Constant",
                        ],
                        value="Do nothing",
                    )

                    convert_numeric = gr.Checkbox(
                        label="Automatically Convert Numeric Columns",
                        value=False,
                    )

                    convert_datetime = gr.Checkbox(
                        label="Automatically Convert Date/Time Columns",
                        value=False,
                    )

                    trim_text = gr.Checkbox(
                        label="Trim Text Spaces",
                        value=False,
                    )

                    standardize_text = gr.Checkbox(
                        label="Convert Text to Lowercase",
                        value=False,
                    )

            apply_cleaning_button = gr.Button(
                "Apply Cleaning to Current Data",
                variant="primary",
            )

            cleaning_result = gr.Textbox(
                label="Cleaning Result",
                lines=16,
                interactive=False,
            )

            cleaning_audit = gr.Textbox(
                label="Cleaning Status",
                interactive=False,
            )

            cleaned_dataset = gr.Dataframe(
                label="Current Cleaned Dataset",
                interactive=False,
                wrap=True,
            )

            reset_cleaning_button = gr.Button(
                "Reset Cleaning to Original",
                variant="secondary",
            )

        # ====================================================
        # FEATURES
        # ====================================================

        with gr.Tab("Features"):

            gr.Markdown("## Feature Analysis")

            feature_button = gr.Button(
                "Analyze Features",
                variant="primary",
            )

            feature_table = gr.Dataframe(
                label="Feature Analysis",
                interactive=False,
                wrap=True,
            )

            gr.Markdown("### Remove Columns")

            remove_columns_select = gr.Dropdown(
                label="Columns to Remove",
                choices=[],
                multiselect=True,
            )

            remove_columns_button = gr.Button(
                "Remove Selected Columns",
            )

            remove_columns_status = gr.Textbox(
                label="Column Operation Status",
                interactive=False,
            )

        # ====================================================
        # OUTLIERS
        # ====================================================

        with gr.Tab("Outliers"):

            gr.Markdown("## Outlier Detection")

            outlier_button = gr.Button(
                "Analyze Outliers",
                variant="primary",
            )

            outlier_table = gr.Dataframe(
                label="Outlier Report",
                interactive=False,
            )

            outlier_status = gr.Textbox(
                label="Outlier Status",
                interactive=False,
            )

        # ====================================================
        # TARGET
        # ====================================================

        with gr.Tab("Target"):

            gr.Markdown("## Target Analysis")

            target_column = gr.Dropdown(
                label="Target Column",
                choices=[],
            )

            target_button = gr.Button(
                "Analyze Target",
                variant="primary",
            )

            target_table = gr.Dataframe(
                label="Target Values",
                interactive=False,
            )

            target_plot = gr.Plot(
                label="Target Distribution"
            )

            target_status = gr.Textbox(
                label="Target Status",
                interactive=False,
            )

        # ====================================================
        # TRAINING
        # ====================================================

        with gr.Tab("Training"):

            gr.Markdown("## Machine Learning Training")

            with gr.Row():

                problem_type = gr.Radio(
                    label="Problem Type",
                    choices=[
                        "Classification",
                        "Regression",
                    ],
                    value="Classification",
                )

                training_target = gr.Dropdown(
                    label="Target Column",
                    choices=[],
                )

            with gr.Row():

                model_name = gr.Dropdown(
                    label="Model",
                    choices=[
                        "Random Forest",
                        "Logistic Regression",
                        "Gradient Boosting",
                        "KNN",
                    ],
                    value="Random Forest",
                )

                scaling_method = gr.Dropdown(
                    label="Scaling",
                    choices=[
                        "No Scaling",
                        "StandardScaler",
                    ],
                    value="StandardScaler",
                )

            with gr.Row():

                test_size = gr.Slider(
                    label="Test Size",
                    minimum=0.1,
                    maximum=0.5,
                    value=0.2,
                    step=0.05,
                )

                random_state = gr.Number(
                    label="Random State",
                    value=42,
                    precision=0,
                )

            train_button = gr.Button(
                "Train Model",
                variant="primary",
            )

            training_result = gr.Textbox(
                label="Training Results",
                lines=18,
                interactive=False,
            )

            training_metrics = gr.Dataframe(
                label="Model Metrics",
                interactive=False,
            )

            training_plot = gr.Plot(
                label="Model Evaluation Plot"
            )

        # ====================================================
        # MODEL COMPARISON
        # ====================================================

        with gr.Tab("Model Comparison"):

            gr.Markdown("## Model Comparison")

            compare_button = gr.Button(
                "Compare Models",
                variant="primary",
            )

            comparison_table = gr.Dataframe(
                label="Model Comparison",
                interactive=False,
            )

            comparison_status = gr.Textbox(
                label="Comparison Status",
                interactive=False,
            )

        # ====================================================
        # MODEL EVALUATION
        # ====================================================

        with gr.Tab("Model Evaluation"):

            gr.Markdown("## Cross Validation")

            evaluation_target = gr.Dropdown(
                label="Target Column",
                choices=[],
            )

            evaluation_model = gr.Dropdown(
                label="Model",
                choices=[
                    "Random Forest",
                    "Logistic Regression",
                    "Gradient Boosting",
                    "KNN",
                ],
                value="Random Forest",
            )

            evaluation_problem = gr.Radio(
                label="Problem Type",
                choices=[
                    "Classification",
                    "Regression",
                ],
                value="Classification",
            )

            evaluation_scaling = gr.Dropdown(
                label="Scaling",
                choices=[
                    "No Scaling",
                    "StandardScaler",
                ],
                value="StandardScaler",
            )

            evaluation_button = gr.Button(
                "Run Cross Validation",
                variant="primary",
            )

            evaluation_table = gr.Dataframe(
                label="Cross Validation Scores",
                interactive=False,
            )

            evaluation_result = gr.Textbox(
                label="Evaluation Result",
                lines=8,
                interactive=False,
            )

        # ====================================================
        # PREDICTION
        # ====================================================

        with gr.Tab("Prediction"):

            gr.Markdown("## Prediction")

            gr.Markdown(
                """
                First train a model in the Training section.
                Then upload another dataset containing the same input features.
                """
            )

            prediction_file = gr.File(
                label="Prediction Dataset",
                file_types=[
                    ".csv",
                    ".xlsx",
                    ".xls",
                ],
                type="filepath",
            )

            prediction_button = gr.Button(
                "Generate Predictions",
                variant="primary",
            )

            prediction_status = gr.Textbox(
                label="Prediction Status",
                interactive=False,
            )

            prediction_table = gr.Dataframe(
                label="Prediction Results",
                interactive=False,
            )

            prediction_download = gr.File(
                label="Download Predictions",
                interactive=False,
            )

        # ====================================================
        # EXPLAINABILITY
        # ====================================================

        with gr.Tab("Explainability"):

            gr.Markdown("## Model Explainability")

            explain_target = gr.Textbox(
                label="Target",
                value="Target",
            )

            explain_button = gr.Button(
                "Explain Trained Model",
                variant="primary",
            )

            explain_status = gr.Textbox(
                label="Explainability Status",
                interactive=False,
            )

            importance_table = gr.Dataframe(
                label="Feature Importance",
                interactive=False,
            )

            importance_plot = gr.Plot(
                label="Feature Importance Plot"
            )

        # ====================================================
        # DOWNLOAD
        # ====================================================

        with gr.Tab("Download"):

            gr.Markdown("## Download Cleaned Dataset")

            download_format = gr.Radio(
                label="Select Download Format",
                choices=[
                    "CSV",
                    "Excel",
                ],
                value="CSV",
            )

            download_button = gr.Button(
                "Prepare Cleaned Dataset",
                variant="primary",
            )

            download_status = gr.Textbox(
                label="Download Status",
                interactive=False,
            )

            cleaned_download = gr.File(
                label="Download File",
                interactive=False,
            )

        # ====================================================
        # AUDIT LOG
        # ====================================================

        with gr.Tab("Audit Log"):

            gr.Markdown("## Audit Log")

            audit_button = gr.Button(
                "Refresh Audit Information",
            )

            audit_log = gr.Textbox(
                label="Dashboard Activity",
                lines=20,
                interactive=False,
            )

    # ========================================================
    # UPLOAD EVENTS
    # ========================================================

    upload_outputs = [
        original_state,
        current_state,
        filename_state,
        loaded_file,
        dataset_preview,
        status_box,
    ]

    upload.change(
        fn=load_dataset,
        inputs=upload,
        outputs=upload_outputs,
    )

    # Update all column selectors after upload
    upload.change(
        fn=update_column_choices,
        inputs=current_state,
        outputs=[
            histogram_column,
            box_column,
            bar_column,
            scatter_x,
            scatter_y,
            target_column,
        ],
    )

    upload.change(
        fn=update_column_choices,
        inputs=current_state,
        outputs=[
            quality_column,
            training_target,
            evaluation_target,
            remove_columns_select,
            outlier_column,
            explain_target,
        ],
    )

    # ========================================================
    # RESET EVENTS
    # ========================================================

    reset_button.click(
        fn=reset_dataset,
        inputs=original_state,
        outputs=[
            current_state,
            dataset_preview,
            status_box,
        ],
    )

    reset_cleaning_button.click(
        fn=reset_dataset,
        inputs=original_state,
        outputs=[
            current_state,
            cleaned_dataset,
            cleaning_audit,
        ],
    )

    reset_button.click(
        fn=update_column_choices,
        inputs=original_state,
        outputs=[
            histogram_column,
            box_column,
            bar_column,
            scatter_x,
            scatter_y,
            target_column,
        ],
    )

    reset_button.click(
        fn=update_column_choices,
        inputs=original_state,
        outputs=[
            quality_column,
            training_target,
            evaluation_target,
            remove_columns_select,
            outlier_column,
            explain_target,
        ],
    )

    # ========================================================
    # OVERVIEW EVENTS
    # ========================================================

    overview_button.click(
        fn=overview_analysis,
        inputs=current_state,
        outputs=[
            overview_text,
            overview_columns,
            overview_preview,
        ],
    )

    # ========================================================
    # STATISTICS EVENTS
    # ========================================================

    statistics_button.click(
        fn=statistics_analysis,
        inputs=current_state,
        outputs=[
            statistics_table,
            advanced_statistics_table,
            status_box,
        ],
    )

    # ========================================================
    # QUALITY EVENTS
    # ========================================================

    quality_button.click(
        fn=data_quality_analysis,
        inputs=current_state,
        outputs=[
            missing_table,
            duplicate_table,
            quality_table,
            status_box,
        ],
    )

    value_count_button.click(
        fn=value_counts_analysis,
        inputs=[
            current_state,
            quality_column,
        ],
        outputs=value_count_table,
    )

    # ========================================================
    # EDA EVENTS
    # ========================================================

    histogram_button.click(
        fn=histogram_plot,
        inputs=[
            current_state,
            histogram_column,
        ],
        outputs=histogram_output,
    )

    box_button.click(
        fn=box_plot,
        inputs=[
            current_state,
            box_column,
        ],
        outputs=box_output,
    )

    bar_button.click(
        fn=bar_plot,
        inputs=[
            current_state,
            bar_column,
        ],
        outputs=bar_output,
    )

    correlation_button.click(
        fn=correlation_plot,
        inputs=current_state,
        outputs=correlation_output,
    )

    scatter_button.click(
        fn=scatter_plot,
        inputs=[
            current_state,
            scatter_x,
            scatter_y,
        ],
        outputs=scatter_output,
    )

    # ========================================================
    # CLEANING EVENTS
    # ========================================================

    apply_cleaning_button.click(
        fn=apply_cleaning,
        inputs=[
            current_state,
            duplicate_treatment,
            numeric_missing,
            categorical_missing,
            constant_value,
            numeric_outlier,
            outlier_column,
            outlier_threshold,
            convert_numeric,
            convert_datetime,
            trim_text,
            standardize_text,
        ],
        outputs=[
            current_state,
            cleaned_dataset,
            cleaning_result,
            cleaning_audit,
        ],
    )

    # Update column selectors after cleaning
    apply_cleaning_button.click(
        fn=update_column_choices,
        inputs=current_state,
        outputs=[
            histogram_column,
            box_column,
            bar_column,
            scatter_x,
            scatter_y,
            target_column,
        ],
    )

    apply_cleaning_button.click(
        fn=update_column_choices,
        inputs=current_state,
        outputs=[
            quality_column,
            training_target,
            evaluation_target,
            remove_columns_select,
            outlier_column,
            explain_target,
        ],
    )

    # ========================================================
    # FEATURE EVENTS
    # ========================================================

    feature_button.click(
        fn=feature_analysis,
        inputs=current_state,
        outputs=[
            feature_table,
            status_box,
        ],
    )

    remove_columns_button.click(
        fn=remove_columns,
        inputs=[
            current_state,
            remove_columns_select,
        ],
        outputs=[
            current_state,
            dataset_preview,
            remove_columns_status,
        ],
    )

    remove_columns_button.click(
        fn=update_column_choices,
        inputs=current_state,
        outputs=[
            histogram_column,
            box_column,
            bar_column,
            scatter_x,
            scatter_y,
            target_column,
        ],
    )

    remove_columns_button.click(
        fn=update_column_choices,
        inputs=current_state,
        outputs=[
            quality_column,
            training_target,
            evaluation_target,
            remove_columns_select,
            outlier_column,
            explain_target,
        ],
    )

    # ========================================================
    # OUTLIER EVENTS
    # ========================================================

    outlier_button.click(
        fn=outlier_report,
        inputs=current_state,
        outputs=[
            outlier_table,
            outlier_status,
        ],
    )

    # ========================================================
    # TARGET EVENTS
    # ========================================================

    target_button.click(
        fn=target_analysis,
        inputs=[
            current_state,
            target_column,
        ],
        outputs=[
            target_table,
            target_plot,
            target_status,
        ],
    )

    # ========================================================
    # TRAINING EVENTS
    # ========================================================

    train_button.click(
        fn=train_model,
        inputs=[
            current_state,
            problem_type,
            training_target,
            model_name,
            test_size,
            scaling_method,
            random_state,
        ],
        outputs=[
            training_result,
            training_metrics,
            training_plot,
            trained_model_state,
        ],
    )

    # ========================================================
    # MODEL COMPARISON EVENTS
    # ========================================================

    compare_button.click(
        fn=compare_models,
        inputs=[
            current_state,
            problem_type,
            training_target,
            test_size,
            scaling_method,
            random_state,
        ],
        outputs=[
            comparison_table,
            comparison_status,
        ],
    )

    # ========================================================
    # MODEL EVALUATION EVENTS
    # ========================================================

    evaluation_button.click(
        fn=evaluate_model,
        inputs=[
            current_state,
            evaluation_problem,
            evaluation_target,
            evaluation_model,
            evaluation_scaling,
            random_state,
        ],
        outputs=[
            evaluation_table,
            evaluation_result,
        ],
    )

    # ========================================================
    # PREDICTION EVENTS
    # ========================================================

    prediction_button.click(
        fn=predict_from_uploaded_data,
        inputs=[
            trained_model_state,
            prediction_file,
        ],
        outputs=[
            prediction_table,
            prediction_status,
            prediction_download,
        ],
    )

    # ========================================================
    # EXPLAINABILITY EVENTS
    # ========================================================

    explain_button.click(
        fn=explain_model,
        inputs=[
            trained_model_state,
            explain_target,
        ],
        outputs=[
            importance_table,
            explain_status,
            importance_plot,
        ],
    )

    # ========================================================
    # DOWNLOAD EVENTS
    # ========================================================

    download_button.click(
        fn=download_cleaned_data,
        inputs=[
            current_state,
            download_format,
        ],
        outputs=[
            cleaned_download,
            download_status,
        ],
    )

    # ========================================================
    # AUDIT LOG
    # ========================================================

    def generate_audit_log(
        df,
        filename,
    ):
        if df is None:
            return "No dataset has been loaded."

        return (
            "AI AND ML ANALYTICS DASHBOARD AUDIT LOG\n\n"
            f"Loaded file: {filename or 'Unknown'}\n"
            f"Current rows: {len(df):,}\n"
            f"Current columns: {len(df.columns):,}\n"
            f"Current missing cells: {int(df.isna().sum().sum()):,}\n"
            f"Current duplicate rows: {int(df.duplicated().sum()):,}\n\n"
            "The dashboard keeps the uploaded original dataset "
            "separate from the current working dataset.\n\n"
            "Cleaning operations are cumulative. Each time "
            "Apply Cleaning to Current Data is pressed, the "
            "selected operation is applied to the current "
            "working dataset.\n\n"
            "Reset to Original restores the original uploaded "
            "dataset and removes all previous cleaning changes."
        )

    audit_button.click(
        fn=generate_audit_log,
        inputs=[
            current_state,
            filename_state,
        ],
        outputs=audit_log,
    )


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":
    dashboard.launch()