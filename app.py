import io
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

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
# PROFESSIONAL GLOBAL STYLE
# ============================================================

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
        background-color: #ffffff;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: "Times New Roman", Times, serif !important;
        color: #1f2937;
        font-weight: 700;
    }

    .main-title {
        text-align: center;
        font-size: 34px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 4px;
    }

    .main-subtitle {
        text-align: center;
        font-size: 17px;
        color: #667085;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #1f2937;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 8px;
        margin-top: 10px;
        margin-bottom: 18px;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #d9dee7;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        min-height: 95px;
    }

    .metric-label {
        color: #667085;
        font-size: 15px;
    }

    .metric-value {
        color: #1f2937;
        font-size: 25px;
        font-weight: 700;
        margin-top: 5px;
    }

    .info-box {
        background: #f8fafc;
        border: 1px solid #d9dee7;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 15px;
    }

    .stButton > button {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border-radius: 7px !important;
    }

    .stDownloadButton > button {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border-radius: 7px !important;
    }

    .stSelectbox label,
    .stRadio label,
    .stSlider label,
    .stNumberInput label,
    .stTextInput label,
    .stFileUploader label,
    .stMultiSelect label {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        font-family: "Times New Roman", Times, serif !important;
    }

    [data-testid="stMetricLabel"] {
        font-family: "Times New Roman", Times, serif !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 16px !important;
        font-weight: 600 !important;
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
    "working_df": None,
    "filename": "",
    "audit_log": [],
    "trained_model": None,
    "trained_task": None,
    "trained_target": None,
    "trained_model_name": None,
    "trained_metrics": None,
    "last_cleaning_done": False,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_copy(df):
    if df is None:
        return None
    return df.copy(deep=True)


def add_audit_log(action, before_df, after_df):
    st.session_state.audit_log.append(
        {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Action": action,
            "Rows Before": len(before_df),
            "Rows After": len(after_df),
            "Columns Before": len(before_df.columns),
            "Columns After": len(after_df.columns),
            "Missing Before": int(before_df.isna().sum().sum()),
            "Missing After": int(after_df.isna().sum().sum()),
            "Duplicates Before": int(before_df.duplicated().sum()),
            "Duplicates After": int(after_df.duplicated().sum()),
        }
    )


def reset_to_original():
    if st.session_state.original_df is not None:
        before = safe_copy(st.session_state.working_df)
        after = safe_copy(st.session_state.original_df)

        st.session_state.working_df = after
        st.session_state.last_cleaning_done = False

        if before is not None:
            add_audit_log(
                "Reset dataset to original uploaded data",
                before,
                after,
            )

        st.session_state.trained_model = None
        st.session_state.trained_task = None
        st.session_state.trained_target = None
        st.session_state.trained_model_name = None
        st.session_state.trained_metrics = None


def read_uploaded_file(uploaded_file):
    filename = uploaded_file.name.lower()

    uploaded_file.seek(0)

    if filename.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin1")

    if filename.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, engine="openpyxl")

    if filename.endswith(".xls"):
        return pd.read_excel(uploaded_file, engine="xlrd")

    raise ValueError("Only CSV, XLSX and XLS files are supported.")


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
            sheet_name="Cleaned Data",
        )

    output.seek(0)
    return output.getvalue()


def create_audit_bytes():
    if not st.session_state.audit_log:
        return b""

    audit_df = pd.DataFrame(st.session_state.audit_log)
    return audit_df.to_csv(index=False).encode("utf-8")


def missing_report(df):
    result = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Count": df.isna().sum().values,
            "Missing Percentage": (
                df.isna().mean() * 100
            ).round(2).values,
            "Unique Values": df.nunique(dropna=True).values,
        }
    )

    return result.sort_values(
        "Missing Count",
        ascending=False,
    ).reset_index(drop=True)


def duplicate_report(df):
    duplicates = int(df.duplicated().sum())

    return pd.DataFrame(
        {
            "Metric": [
                "Total Rows",
                "Duplicate Rows",
                "Duplicate Percentage",
            ],
            "Value": [
                len(df),
                duplicates,
                round(
                    duplicates / max(len(df), 1) * 100,
                    2,
                ),
            ],
        }
    )


def statistical_profile(df):
    rows = []

    for column in df.columns:
        series = df[column]

        row = {
            "Column": column,
            "Data Type": str(series.dtype),
            "Count": int(series.notna().sum()),
            "Missing": int(series.isna().sum()),
            "Missing %": round(
                series.isna().mean() * 100,
                2,
            ),
            "Unique": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(
                series,
                errors="coerce",
            ).dropna()

            if len(values) > 0:
                row.update(
                    {
                        "Mean": round(values.mean(), 4),
                        "Median": round(values.median(), 4),
                        "Std": round(values.std(), 4),
                        "Min": round(values.min(), 4),
                        "Max": round(values.max(), 4),
                        "Skewness": round(
                            values.skew(),
                            4,
                        )
                        if len(values) > 2
                        else np.nan,
                        "Kurtosis": round(
                            values.kurtosis(),
                            4,
                        )
                        if len(values) > 3
                        else np.nan,
                    }
                )
            else:
                row.update(
                    {
                        "Mean": np.nan,
                        "Median": np.nan,
                        "Std": np.nan,
                        "Min": np.nan,
                        "Max": np.nan,
                        "Skewness": np.nan,
                        "Kurtosis": np.nan,
                    }
                )
        else:
            row.update(
                {
                    "Mean": np.nan,
                    "Median": np.nan,
                    "Std": np.nan,
                    "Min": np.nan,
                    "Max": np.nan,
                    "Skewness": np.nan,
                    "Kurtosis": np.nan,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def validity_report(df):
    rows = []

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        rows.append(
            {
                "Column": column,
                "Negative Values": int(
                    (values < 0).sum()
                ),
                "Zero Values": int(
                    (values == 0).sum()
                ),
                "Infinite Values": int(
                    np.isinf(values).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def correlation_pairs(df):
    numeric = df.select_dtypes(
        include=np.number
    )

    if numeric.shape[1] < 2:
        return pd.DataFrame()

    correlation = numeric.corr()

    rows = []
    columns = list(correlation.columns)

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            value = correlation.iloc[i, j]

            rows.append(
                {
                    "Feature 1": columns[i],
                    "Feature 2": columns[j],
                    "Pearson Correlation": round(
                        value,
                        4,
                    ),
                    "Absolute Correlation": round(
                        abs(value),
                        4,
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "Absolute Correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def outlier_report(df):
    rows = []

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if len(values) < 2:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        iqr_outliers = int(
            (
                (values < lower)
                | (values > upper)
            ).sum()
        )

        z_outliers = 0

        if len(values) >= 3:
            z_scores = np.abs(
                stats.zscore(values)
            )
            z_outliers = int(
                (z_scores > 3).sum()
            )

        rows.append(
            {
                "Column": column,
                "Q1": round(q1, 4),
                "Q3": round(q3, 4),
                "IQR": round(iqr, 4),
                "Lower Bound": round(
                    lower,
                    4,
                ),
                "Upper Bound": round(
                    upper,
                    4,
                ),
                "IQR Outliers": iqr_outliers,
                "IQR Outlier %": round(
                    iqr_outliers
                    / max(len(values), 1)
                    * 100,
                    2,
                ),
                "Z Score > 3": z_outliers,
                "Z Outlier %": round(
                    z_outliers
                    / max(len(values), 1)
                    * 100,
                    2,
                ),
            }
        )

    return pd.DataFrame(rows)


def convert_numeric_text(df):
    output = df.copy()
    converted = []

    for column in output.columns:
        if output[column].dtype == "object":
            numeric = pd.to_numeric(
                output[column],
                errors="coerce",
            )

            non_null = int(
                output[column].notna().sum()
            )

            if (
                non_null > 0
                and numeric.notna().sum()
                / non_null
                >= 0.80
            ):
                output[column] = numeric
                converted.append(column)

    return output, converted


# ============================================================
# CLEANING FUNCTION
# ============================================================

def apply_cleaning(
    df,
    duplicate_method,
    missing_method,
    constant_value,
    numeric_method,
    round_digits,
    categorical_method,
    invalid_method,
    outlier_method,
):
    output = df.copy()
    changes = []

    # Duplicate handling
    if duplicate_method == "Remove duplicates":
        before_rows = len(output)

        output = (
            output
            .drop_duplicates()
            .reset_index(drop=True)
        )

        removed = before_rows - len(output)

        changes.append(
            f"Removed {removed:,} duplicate rows"
        )

    # Missing values
    if missing_method == "Drop rows with missing values":
        before_rows = len(output)

        output = (
            output
            .dropna()
            .reset_index(drop=True)
        )

        changes.append(
            f"Dropped {before_rows - len(output):,} rows with missing values"
        )

    elif missing_method == "Drop columns with more than 50% missing":
        before_columns = set(output.columns)

        minimum_non_missing = int(
            np.ceil(len(output) * 0.50)
        )

        output = output.dropna(
            axis=1,
            thresh=minimum_non_missing,
        )

        removed_columns = (
            before_columns
            - set(output.columns)
        )

        changes.append(
            f"Dropped {len(removed_columns)} columns with more than 50% missing values"
        )

    elif missing_method == "Mean imputation":
        numeric_columns = output.select_dtypes(
            include=np.number
        ).columns

        for column in numeric_columns:
            if output[column].isna().any():
                output[column] = output[column].fillna(
                    output[column].mean()
                )

        changes.append(
            "Applied mean imputation to numeric columns"
        )

    elif missing_method == "Median imputation":
        numeric_columns = output.select_dtypes(
            include=np.number
        ).columns

        for column in numeric_columns:
            if output[column].isna().any():
                output[column] = output[column].fillna(
                    output[column].median()
                )

        changes.append(
            "Applied median imputation to numeric columns"
        )

    elif missing_method == "Mode imputation":
        for column in output.columns:
            mode = output[column].mode(
                dropna=True
            )

            if not mode.empty:
                output[column] = output[column].fillna(
                    mode.iloc[0]
                )

        changes.append(
            "Applied mode imputation"
        )

    elif missing_method == "Constant imputation":
        for column in output.columns:
            if pd.api.types.is_numeric_dtype(
                output[column]
            ):
                try:
                    value = float(
                        constant_value
                    )
                except Exception:
                    value = 0.0
            else:
                value = constant_value

            output[column] = output[column].fillna(
                value
            )

        changes.append(
            "Applied constant-value imputation"
        )

    elif missing_method == "Forward fill":
        output = output.ffill()

        changes.append(
            "Applied forward fill"
        )

    elif missing_method == "Backward fill":
        output = output.bfill()

        changes.append(
            "Applied backward fill"
        )

    elif missing_method == "KNN imputation":
        try:
            from sklearn.impute import KNNImputer

            numeric_columns = output.select_dtypes(
                include=np.number
            ).columns

            if len(numeric_columns) > 0:
                imputer = KNNImputer(
                    n_neighbors=5
                )

                output[numeric_columns] = (
                    imputer.fit_transform(
                        output[numeric_columns]
                    )
                )

                changes.append(
                    "Applied KNN imputation to numeric columns"
                )
            else:
                changes.append(
                    "KNN imputation skipped because no numeric columns were found"
                )

        except Exception as error:
            changes.append(
                f"KNN imputation skipped: {error}"
            )

    elif missing_method == "Iterative imputation":
        try:
            from sklearn.experimental import (
                enable_iterative_imputer,
            )
            from sklearn.impute import (
                IterativeImputer,
            )

            numeric_columns = output.select_dtypes(
                include=np.number
            ).columns

            if len(numeric_columns) > 0:
                imputer = IterativeImputer(
                    random_state=42
                )

                output[numeric_columns] = (
                    imputer.fit_transform(
                        output[numeric_columns]
                    )
                )

                changes.append(
                    "Applied iterative imputation to numeric columns"
                )
            else:
                changes.append(
                    "Iterative imputation skipped because no numeric columns were found"
                )

        except Exception as error:
            changes.append(
                f"Iterative imputation skipped: {error}"
            )

    # Numeric cleaning
    if numeric_method == "Convert numeric-looking text":
        output, converted = convert_numeric_text(
            output
        )

        changes.append(
            f"Converted {len(converted)} numeric-looking text columns"
        )

    elif numeric_method == "Round numeric values":
        numeric_columns = output.select_dtypes(
            include=np.number
        ).columns

        output[numeric_columns] = (
            output[numeric_columns].round(
                int(round_digits)
            )
        )

        changes.append(
            f"Rounded numeric values to {int(round_digits)} decimal places"
        )

    # Categorical cleaning
    text_columns = output.select_dtypes(
        include="object"
    ).columns

    if categorical_method == "Remove extra spaces":
        for column in text_columns:
            output[column] = output[column].apply(
                lambda value:
                value.strip()
                if isinstance(value, str)
                else value
            )

        changes.append(
            "Removed extra spaces from text columns"
        )

    elif categorical_method == "Lowercase text":
        for column in text_columns:
            output[column] = output[column].apply(
                lambda value:
                value.strip().lower()
                if isinstance(value, str)
                else value
            )

        changes.append(
            "Converted text columns to lowercase"
        )

    elif categorical_method == "Uppercase text":
        for column in text_columns:
            output[column] = output[column].apply(
                lambda value:
                value.strip().upper()
                if isinstance(value, str)
                else value
            )

        changes.append(
            "Converted text columns to uppercase"
        )

    # Invalid values
    if invalid_method == "Replace negative numeric values with missing":
        count = 0

        numeric_columns = output.select_dtypes(
            include=np.number
        ).columns

        for column in numeric_columns:
            negative_count = int(
                (output[column] < 0).sum()
            )

            count += negative_count

            output.loc[
                output[column] < 0,
                column,
            ] = np.nan

        changes.append(
            f"Replaced {count:,} negative numeric values with missing values"
        )

    # Outliers
    numeric_columns = list(
        output.select_dtypes(
            include=np.number
        ).columns
    )

    if (
        outlier_method != "Do nothing"
        and numeric_columns
    ):
        if outlier_method in [
            "Remove IQR outliers",
            "Cap IQR outliers",
        ]:
            for column in numeric_columns:
                series = output[column]

                if series.dropna().empty:
                    continue

                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)

                iqr = q3 - q1

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                if outlier_method == "Remove IQR outliers":
                    keep = (
                        series.isna()
                        | series.between(
                            lower,
                            upper,
                        )
                    )

                    output = (
                        output.loc[keep]
                        .reset_index(drop=True)
                    )

                else:
                    output[column] = series.clip(
                        lower,
                        upper,
                    )

            changes.append(
                outlier_method
            )

        elif outlier_method == "Remove Z-score > 3 outliers":
            keep = pd.Series(
                True,
                index=output.index,
            )

            for column in numeric_columns:
                series = output[column]

                if series.notna().sum() >= 3:
                    filled = series.fillna(
                        series.median()
                    )

                    z_scores = np.abs(
                        stats.zscore(
                            filled
                        )
                    )

                    keep &= (
                        pd.Series(
                            z_scores,
                            index=output.index,
                        )
                        <= 3
                    )

            output = (
                output.loc[keep]
                .reset_index(drop=True)
            )

            changes.append(
                "Removed Z-score greater than 3 outliers"
            )

    return output, changes


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def feature_engineering(
    df,
    method,
    column,
    new_name,
):
    output = df.copy()

    if method == "Do nothing":
        return output, "No feature engineering was applied."

    if column not in output.columns:
        return output, "Selected column does not exist."

    if method == "Drop selected column":
        output = output.drop(
            columns=[column]
        )

        return (
            output,
            f"Dropped column '{column}'.",
        )

    if not pd.api.types.is_numeric_dtype(
        output[column]
    ):
        return (
            output,
            "The selected column must be numeric.",
        )

    name = (
        new_name.strip()
        if new_name.strip()
        else f"{column}_transformed"
    )

    if method == "Create log transformation":
        if (output[column] <= 0).any():
            return (
                output,
                "Log transformation requires positive values.",
            )

        output[name] = np.log1p(
            output[column]
        )

        return (
            output,
            f"Created '{name}' using log transformation.",
        )

    if method == "Create squared feature":
        output[name] = (
            output[column] ** 2
        )

        return (
            output,
            f"Created '{name}' as a squared feature.",
        )

    return output, "No change applied."


# ============================================================
# VISUALIZATION
# ============================================================

GRAPH_TYPES = [
    "Histogram",
    "Box Plot",
    "Violin Plot",
    "Count Plot",
    "Bar Chart",
    "Scatter Plot",
    "Line Chart",
    "Area Chart",
    "KDE Plot",
    "Pie Chart",
    "Correlation Heatmap",
    "Covariance Heatmap",
    "Missing Values Chart",
    "Q-Q Plot",
]


def make_visualization(
    df,
    graph_type,
    x_column,
    y_column,
    hue_column,
):
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    try:
        hue = (
            None
            if hue_column in [
                None,
                "",
                "None",
            ]
            else hue_column
        )

        if graph_type == "Histogram":
            sns.histplot(
                data=df,
                x=x_column,
                hue=hue,
                kde=True,
                ax=ax,
            )

        elif graph_type == "Box Plot":
            if y_column == "None":
                sns.boxplot(
                    data=df,
                    y=x_column,
                    ax=ax,
                )
            else:
                sns.boxplot(
                    data=df,
                    x=x_column,
                    y=y_column,
                    hue=hue,
                    ax=ax,
                )

        elif graph_type == "Violin Plot":
            if y_column == "None":
                sns.violinplot(
                    data=df,
                    y=x_column,
                    ax=ax,
                )
            else:
                sns.violinplot(
                    data=df,
                    x=x_column,
                    y=y_column,
                    hue=hue,
                    ax=ax,
                )

        elif graph_type == "Count Plot":
            sns.countplot(
                data=df,
                x=x_column,
                hue=hue,
                ax=ax,
            )

            ax.tick_params(
                axis="x",
                rotation=45,
            )

        elif graph_type == "Bar Chart":
            if (
                y_column != "None"
                and y_column in df.columns
                and pd.api.types.is_numeric_dtype(
                    df[y_column]
                )
            ):
                values = (
                    df.groupby(
                        x_column,
                        dropna=False,
                    )[y_column]
                    .mean()
                    .sort_values(
                        ascending=False
                    )
                    .head(20)
                )

                values.plot(
                    kind="bar",
                    ax=ax,
                )

            else:
                (
                    df[x_column]
                    .astype(str)
                    .value_counts()
                    .head(20)
                    .plot(
                        kind="bar",
                        ax=ax,
                    )
                )

            ax.tick_params(
                axis="x",
                rotation=45,
            )

        elif graph_type == "Scatter Plot":
            if y_column == "None":
                raise ValueError(
                    "Select a Y column."
                )

            sns.scatterplot(
                data=df,
                x=x_column,
                y=y_column,
                hue=hue,
                ax=ax,
            )

        elif graph_type == "Line Chart":
            if y_column == "None":
                raise ValueError(
                    "Select a Y column."
                )

            sns.lineplot(
                data=df,
                x=x_column,
                y=y_column,
                hue=hue,
                ax=ax,
            )

        elif graph_type == "Area Chart":
            if y_column == "None":
                raise ValueError(
                    "Select a Y column."
                )

            temp = (
                df[
                    [
                        x_column,
                        y_column,
                    ]
                ]
                .dropna()
                .head(500)
            )

            if temp.empty:
                raise ValueError(
                    "No usable data."
                )

            if pd.api.types.is_numeric_dtype(
                temp[x_column]
            ):
                temp = temp.sort_values(
                    x_column
                )

                ax.fill_between(
                    temp[x_column],
                    temp[y_column],
                    alpha=0.3,
                )

                ax.plot(
                    temp[x_column],
                    temp[y_column],
                )

            else:
                positions = range(
                    len(temp)
                )

                ax.fill_between(
                    positions,
                    temp[y_column],
                    alpha=0.3,
                )

                ax.plot(
                    positions,
                    temp[y_column],
                )

        elif graph_type == "KDE Plot":
            sns.kdeplot(
                data=df,
                x=x_column,
                hue=hue,
                fill=True,
                ax=ax,
            )

        elif graph_type == "Pie Chart":
            values = (
                df[x_column]
                .astype(str)
                .value_counts()
                .head(10)
            )

            ax.pie(
                values.values,
                labels=values.index,
                autopct="%1.1f%%",
                startangle=90,
            )

        elif graph_type == "Correlation Heatmap":
            numeric = df.select_dtypes(
                include=np.number
            )

            if numeric.shape[1] < 2:
                raise ValueError(
                    "At least two numeric columns are required."
                )

            sns.heatmap(
                numeric.corr(),
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                ax=ax,
            )

        elif graph_type == "Covariance Heatmap":
            numeric = df.select_dtypes(
                include=np.number
            )

            if numeric.shape[1] < 2:
                raise ValueError(
                    "At least two numeric columns are required."
                )

            sns.heatmap(
                numeric.cov(),
                annot=True,
                fmt=".2f",
                cmap="viridis",
                ax=ax,
            )

        elif graph_type == "Missing Values Chart":
            missing = (
                df.isna()
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            missing = missing[
                missing > 0
            ]

            if missing.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No missing values found.",
                    ha="center",
                    va="center",
                )

                ax.axis("off")

            else:
                missing.plot(
                    kind="bar",
                    ax=ax,
                )

                ax.tick_params(
                    axis="x",
                    rotation=45,
                )

        elif graph_type == "Q-Q Plot":
            values = pd.to_numeric(
                df[x_column],
                errors="coerce",
            ).dropna()

            if len(values) < 3:
                raise ValueError(
                    "At least three numeric values are required."
                )

            stats.probplot(
                values,
                dist="norm",
                plot=ax,
            )

        ax.set_title(
            graph_type,
            fontsize=16,
            fontweight="bold",
        )

        fig.tight_layout()

        return fig

    except Exception as error:
        plt.close(fig)

        error_fig, error_ax = plt.subplots(
            figsize=(10, 5)
        )

        error_ax.text(
            0.5,
            0.5,
            f"Visualization could not be generated.\n\n{error}",
            ha="center",
            va="center",
            wrap=True,
        )

        error_ax.axis("off")

        error_fig.tight_layout()

        return error_fig


# ============================================================
# MACHINE LEARNING
# ============================================================

def classification_models():
    return {
        "Logistic Regression":
            LogisticRegression(
                max_iter=2000
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),

        "K-Nearest Neighbors":
            KNeighborsClassifier(),

        "Support Vector Machine":
            SVC(
                probability=True,
                random_state=42,
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                random_state=42
            ),
    }


def regression_models():
    return {
        "Linear Regression":
            LinearRegression(),

        "Ridge Regression":
            Ridge(),

        "Lasso Regression":
            Lasso(),

        "Decision Tree":
            DecisionTreeRegressor(
                random_state=42
            ),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),

        "Support Vector Regression":
            SVR(),

        "Gradient Boosting":
            GradientBoostingRegressor(
                random_state=42
            ),
    }


def prepare_ml_data(
    df,
    target,
    task,
):
    if target not in df.columns:
        raise ValueError(
            "Target column was not found."
        )

    data = (
        df.copy()
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    data = data.dropna(
        subset=[target]
    )

    X = data.drop(
        columns=[target]
    )

    X = X.dropna(
        axis=1,
        how="all",
    )

    if X.empty:
        raise ValueError(
            "No usable feature columns were found."
        )

    y = data[target]

    if task == "Classification":
        y = y.astype(str)

        if y.nunique() < 2:
            raise ValueError(
                "Classification requires at least two target classes."
            )

    else:
        y = pd.to_numeric(
            y,
            errors="coerce",
        )

        valid = y.notna()

        X = X.loc[valid]
        y = y.loc[valid]

        if len(y) < 10:
            raise ValueError(
                "Regression requires at least 10 valid target values."
            )

    return X, y


def create_preprocessor(
    X,
    scaling,
):
    numeric_columns = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        )
    ]

    if scaling == "StandardScaler":
        numeric_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    elif scaling == "MinMaxScaler":
        numeric_steps.append(
            (
                "scaler",
                MinMaxScaler(),
            )
        )

    numeric_pipeline = Pipeline(
        numeric_steps
    )

    categorical_pipeline = Pipeline(
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
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError(
            "No usable numeric or categorical features were found."
        )

    return ColumnTransformer(
        transformers=transformers
    )


def train_single_model(
    df,
    target,
    task,
    model_name,
    test_size,
    scaling,
):
    X, y = prepare_ml_data(
        df,
        target,
        task,
    )

    if task == "Classification":
        model = classification_models()[
            model_name
        ]
    else:
        model = regression_models()[
            model_name
        ]

    preprocessor = create_preprocessor(
        X,
        scaling,
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

    stratify = None

    if (
        task == "Classification"
        and y.value_counts().min() >= 2
    ):
        stratify = y

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(
        X_test
    )

    if task == "Classification":
        metrics = {
            "Accuracy":
                accuracy_score(
                    y_test,
                    predictions,
                ),

            "Precision":
                precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),

            "Recall":
                recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),

            "F1 Score":
                f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
        }

    else:
        metrics = {
            "MAE":
                mean_absolute_error(
                    y_test,
                    predictions,
                ),

            "RMSE":
                np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions,
                    )
                ),

            "R2":
                r2_score(
                    y_test,
                    predictions,
                ),
        }

    return (
        pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions,
        metrics,
    )


def feature_importance_table(
    pipeline,
):
    try:
        preprocessor = (
            pipeline
            .named_steps[
                "preprocessor"
            ]
        )

        model = (
            pipeline
            .named_steps[
                "model"
            ]
        )

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        if hasattr(
            model,
            "feature_importances_",
        ):
            values = (
                model.feature_importances_
            )

        elif hasattr(
            model,
            "coef_",
        ):
            coefficients = np.asarray(
                model.coef_
            )

            if coefficients.ndim > 1:
                values = np.mean(
                    np.abs(
                        coefficients
                    ),
                    axis=0,
                )
            else:
                values = np.abs(
                    coefficients
                )

        else:
            return pd.DataFrame(
                columns=[
                    "Feature",
                    "Importance",
                ]
            )

        return (
            pd.DataFrame(
                {
                    "Feature":
                        feature_names,
                    "Importance":
                        values,
                }
            )
            .sort_values(
                "Importance",
                ascending=False,
            )
            .head(30)
            .reset_index(
                drop=True
            )
        )

    except Exception:
        return pd.DataFrame(
            columns=[
                "Feature",
                "Importance",
            ]
        )


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">AI and ML Analytics Dashboard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-subtitle">Data Analysis, Data Cleaning, Visualization, Machine Learning and Prediction</div>',
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="section-title">Dataset</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=[
            "csv",
            "xlsx",
            "xls",
        ],
        help="Upload a CSV, XLSX or XLS dataset.",
    )

    if (
        uploaded_file is not None
        and st.session_state.filename
        != uploaded_file.name
    ):
        try:
            loaded_df = read_uploaded_file(
                uploaded_file
            )

            if loaded_df.empty:
                st.error(
                    "The uploaded dataset is empty."
                )

            else:
                st.session_state.original_df = (
                    loaded_df.copy()
                )

                st.session_state.working_df = (
                    loaded_df.copy()
                )

                st.session_state.filename = (
                    uploaded_file.name
                )

                st.session_state.audit_log = []

                st.session_state.audit_log.append(
                    {
                        "Time":
                            datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                        "Action":
                            "Dataset uploaded",
                        "Rows Before":
                            len(loaded_df),
                        "Rows After":
                            len(loaded_df),
                        "Columns Before":
                            len(loaded_df.columns),
                        "Columns After":
                            len(loaded_df.columns),
                        "Missing Before":
                            int(
                                loaded_df.isna()
                                .sum()
                                .sum()
                            ),
                        "Missing After":
                            int(
                                loaded_df.isna()
                                .sum()
                                .sum()
                            ),
                        "Duplicates Before":
                            int(
                                loaded_df.duplicated()
                                .sum()
                            ),
                        "Duplicates After":
                            int(
                                loaded_df.duplicated()
                                .sum()
                            ),
                    }
                )

                st.session_state.trained_model = None
                st.session_state.trained_task = None
                st.session_state.trained_target = None
                st.session_state.trained_model_name = None
                st.session_state.trained_metrics = None
                st.session_state.last_cleaning_done = False

                st.rerun()

        except Exception as error:
            st.error(
                f"Could not load the dataset: {error}"
            )

    if st.session_state.working_df is not None:
        st.success(
            f"Loaded file: {st.session_state.filename}"
        )

        st.button(
            "Reset to Original Dataset",
            on_click=reset_to_original,
            use_container_width=True,
        )

        st.caption(
            "Reset restores the exact dataset that was uploaded."
        )


# ============================================================
# CHECK DATASET
# ============================================================

if st.session_state.working_df is None:
    st.info(
        "Upload a dataset from the sidebar to begin."
    )

    st.markdown(
        """
        ### Dashboard Functions

        This dashboard includes:

        - Dataset overview and preview
        - Data type analysis
        - Missing-value analysis
        - Duplicate analysis
        - Value counts
        - Statistical analysis
        - Skewness and kurtosis
        - Correlation analysis
        - Outlier detection
        - Multiple visualizations
        - Data cleaning
        - Missing-value imputation
        - Feature engineering
        - Target analysis
        - Classification
        - Regression
        - Model comparison
        - Model evaluation
        - Prediction
        - Feature importance
        - Cleaned-data download
        - Audit log
        """
    )

    st.stop()


df = st.session_state.working_df


# ============================================================
# TOP SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">Dataset Summary</div>',
    unsafe_allow_html=True,
)

summary_columns = st.columns(6)

summary_values = [
    ("Rows", f"{len(df):,}"),
    ("Columns", f"{len(df.columns):,}"),
    (
        "Numeric Columns",
        f"{len(df.select_dtypes(include=np.number).columns):,}",
    ),
    (
        "Categorical Columns",
        f"{len(df.select_dtypes(exclude=np.number).columns):,}",
    ),
    (
        "Missing Values",
        f"{int(df.isna().sum().sum()):,}",
    ),
    (
        "Duplicate Rows",
        f"{int(df.duplicated().sum()):,}",
    ),
]

for column, (
    label,
    value,
) in zip(
    summary_columns,
    summary_values,
):
    column.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")


# ============================================================
# MAIN DASHBOARD
# ============================================================

tabs = st.tabs(
    [
        "Overview",
        "Statistics",
        "Data Quality",
        "EDA",
        "Cleaning",
        "Feature Engineering",
        "Outliers",
        "Target Analysis",
        "Model Training",
        "Model Comparison",
        "Model Evaluation",
        "Prediction",
        "Explainability",
        "Download",
        "Audit Log",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:
    st.markdown(
        '<div class="section-title">Dataset Overview</div>',
        unsafe_allow_html=True,
    )

    preview_count = st.slider(
        "Number of rows to display",
        min_value=5,
        max_value=min(
            max(len(df), 5),
            100,
        ),
        value=min(
            max(len(df), 5),
            20,
        ),
    )

    st.dataframe(
        df.head(preview_count),
        use_container_width=True,
        height=400,
    )

    st.subheader(
        "Column Information"
    )

    column_info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type":
                df.dtypes.astype(str).values,
            "Non-Null":
                df.notna().sum().values,
            "Missing":
                df.isna().sum().values,
            "Missing %":
                (
                    df.isna().mean()
                    * 100
                ).round(2).values,
            "Unique":
                df.nunique(
                    dropna=True
                ).values,
        }
    )

    st.dataframe(
        column_info,
        use_container_width=True,
        height=450,
    )


# ============================================================
# STATISTICS
# ============================================================

with tabs[1]:
    st.markdown(
        '<div class="section-title">Statistical Analysis</div>',
        unsafe_allow_html=True,
    )

    st.subheader(
        "Descriptive Statistics"
    )

    try:
        description = (
            df.describe(
                include="all"
            )
            .T
            .reset_index()
            .rename(
                columns={
                    "index": "Column"
                }
            )
        )

        st.dataframe(
            description,
            use_container_width=True,
            height=450,
        )

    except Exception as error:
        st.warning(
            f"Descriptive statistics could not be generated: {error}"
        )

    st.subheader(
        "Detailed Statistical Profile"
    )

    st.dataframe(
        statistical_profile(df),
        use_container_width=True,
        height=500,
    )

    numeric_columns = list(
        df.select_dtypes(
            include=np.number
        ).columns
    )

    if numeric_columns:
        selected_column = st.selectbox(
            "Numeric Column",
            numeric_columns,
        )

        values = pd.to_numeric(
            df[selected_column],
            errors="coerce",
        ).dropna()

        if len(values) > 0:
            metric_columns = st.columns(6)

            metric_columns[0].metric(
                "Mean",
                f"{values.mean():.4f}",
            )

            metric_columns[1].metric(
                "Median",
                f"{values.median():.4f}",
            )

            metric_columns[2].metric(
                "Standard Deviation",
                f"{values.std():.4f}",
            )

            metric_columns[3].metric(
                "Minimum",
                f"{values.min():.4f}",
            )

            metric_columns[4].metric(
                "Maximum",
                f"{values.max():.4f}",
            )

            metric_columns[5].metric(
                "Skewness",
                f"{values.skew():.4f}",
            )


# ============================================================
# DATA QUALITY
# ============================================================

with tabs[2]:
    st.markdown(
        '<div class="section-title">Data Quality Analysis</div>',
        unsafe_allow_html=True,
    )

    st.subheader(
        "Missing Values"
    )

    st.dataframe(
        missing_report(df),
        use_container_width=True,
        height=400,
    )

    st.subheader(
        "Duplicate Analysis"
    )

    st.dataframe(
        duplicate_report(df),
        use_container_width=True,
    )

    st.subheader(
        "Value Counts"
    )

    quality_column = st.selectbox(
        "Select Column",
        list(df.columns),
        key="quality_value_counts_column",
    )

    value_counts = (
        df[quality_column]
        .value_counts(
            dropna=False
        )
        .head(50)
        .reset_index()
    )

    value_counts.columns = [
        "Value",
        "Count",
    ]

    value_counts[
        "Percentage"
    ] = (
        value_counts["Count"]
        / max(len(df), 1)
        * 100
    ).round(2)

    st.dataframe(
        value_counts,
        use_container_width=True,
    )

    st.subheader(
        "Validity Checks"
    )

    validity = validity_report(
        df
    )

    if validity.empty:
        st.info(
            "No numeric columns are available for validity checks."
        )
    else:
        st.dataframe(
            validity,
            use_container_width=True,
        )

    st.subheader(
        "Correlation Strength"
    )

    correlations = correlation_pairs(
        df
    )

    if correlations.empty:
        st.info(
            "At least two numeric columns are required."
        )
    else:
        st.dataframe(
            correlations.head(50),
            use_container_width=True,
        )


# ============================================================
# EDA
# ============================================================

with tabs[3]:
    st.markdown(
        '<div class="section-title">Exploratory Data Analysis</div>',
        unsafe_allow_html=True,
    )

    graph_type = st.selectbox(
        "Visualization Type",
        GRAPH_TYPES,
    )

    visualization_columns = st.columns(
        3
    )

    with visualization_columns[0]:
        x_column = st.selectbox(
            "X Column",
            list(df.columns),
            key="eda_x_column",
        )

    with visualization_columns[1]:
        y_column = st.selectbox(
            "Y Column",
            ["None"] + list(df.columns),
            key="eda_y_column",
        )

    with visualization_columns[2]:
        hue_column = st.selectbox(
            "Grouping Column",
            ["None"] + list(df.columns),
            key="eda_hue_column",
        )

    if st.button(
        "Generate Visualization",
        type="primary",
        use_container_width=True,
    ):
        figure = make_visualization(
            df,
            graph_type,
            x_column,
            y_column,
            hue_column,
        )

        st.pyplot(
            figure,
            clear_figure=True,
        )


# ============================================================
# CLEANING
# ============================================================

with tabs[4]:
    st.markdown(
        '<div class="section-title">Data Cleaning</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Select the cleaning operations you need, then select Apply Cleaning. The cleaned dataset will remain active throughout the dashboard."
    )

    cleaning_col_1, cleaning_col_2 = st.columns(
        2
    )

    with cleaning_col_1:
        duplicate_method = st.selectbox(
            "Duplicate Treatment",
            [
                "Do nothing",
                "Remove duplicates",
            ],
            key="clean_duplicate_method",
        )

        missing_method = st.selectbox(
            "Missing Value Treatment",
            [
                "Do nothing",
                "Drop rows with missing values",
                "Drop columns with more than 50% missing",
                "Mean imputation",
                "Median imputation",
                "Mode imputation",
                "Constant imputation",
                "Forward fill",
                "Backward fill",
                "KNN imputation",
                "Iterative imputation",
            ],
            key="clean_missing_method",
        )

        constant_value = st.text_input(
            "Constant Value",
            value="0",
            key="clean_constant_value",
        )

        numeric_method = st.selectbox(
            "Numeric Data Treatment",
            [
                "Do nothing",
                "Convert numeric-looking text",
                "Round numeric values",
            ],
            key="clean_numeric_method",
        )

        round_digits = st.number_input(
            "Number of Decimal Places",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            key="clean_round_digits",
        )

    with cleaning_col_2:
        categorical_method = st.selectbox(
            "Categorical/Text Treatment",
            [
                "Do nothing",
                "Remove extra spaces",
                "Lowercase text",
                "Uppercase text",
            ],
            key="clean_categorical_method",
        )

        invalid_method = st.selectbox(
            "Invalid Value Treatment",
            [
                "Do nothing",
                "Replace negative numeric values with missing",
            ],
            key="clean_invalid_method",
        )

        outlier_method = st.selectbox(
            "Outlier Treatment",
            [
                "Do nothing",
                "Remove IQR outliers",
                "Cap IQR outliers",
                "Remove Z-score > 3 outliers",
            ],
            key="clean_outlier_method",
        )

    st.subheader(
        "Apply Cleaning"
    )

    if st.button(
        "Apply Cleaning to Current Dataset",
        type="primary",
        use_container_width=True,
    ):
        try:
            before_df = safe_copy(
                st.session_state.working_df
            )

            cleaned_df, changes = apply_cleaning(
                before_df,
                duplicate_method,
                missing_method,
                constant_value,
                numeric_method,
                round_digits,
                categorical_method,
                invalid_method,
                outlier_method,
            )

            if cleaned_df.empty:
                st.error(
                    "Cleaning produced an empty dataset. The original dataset has been kept unchanged."
                )

            else:
                st.session_state.working_df = (
                    cleaned_df.copy()
                )

                st.session_state.last_cleaning_done = (
                    True
                )

                st.session_state.trained_model = None
                st.session_state.trained_task = None
                st.session_state.trained_target = None
                st.session_state.trained_model_name = None
                st.session_state.trained_metrics = None

                action_text = (
                    "; ".join(changes)
                    if changes
                    else "No cleaning changes were selected"
                )

                add_audit_log(
                    action_text,
                    before_df,
                    cleaned_df,
                )

                st.success(
                    "Cleaning completed successfully."
                )

                if changes:
                    st.write(
                        "Changes applied:"
                    )

                    for change in changes:
                        st.write(
                            f"- {change}"
                        )

                st.rerun()

        except Exception as error:
            st.error(
                f"Cleaning could not be completed: {error}"
            )

    st.subheader(
        "Current Cleaned Dataset"
    )

    st.dataframe(
        st.session_state.working_df,
        use_container_width=True,
        height=450,
    )

    st.subheader(
        "Download Cleaned Dataset"
    )

    st.write(
        "Choose the required file format and download the current cleaned dataset."
    )

    download_format = st.radio(
        "Download Format",
        [
            "CSV",
            "Excel",
        ],
        horizontal=True,
        key="clean_download_format",
    )

    if download_format == "CSV":
        st.download_button(
            "Download Cleaned CSV",
            data=create_csv_bytes(
                st.session_state.working_df
            ),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

    else:
        st.download_button(
            "Download Cleaned Excel",
            data=create_excel_bytes(
                st.session_state.working_df
            ),
            file_name="cleaned_dataset.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
        )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

with tabs[5]:
    st.markdown(
        '<div class="section-title">Feature Engineering</div>',
        unsafe_allow_html=True,
    )

    feature_method = st.selectbox(
        "Feature Engineering Method",
        [
            "Do nothing",
            "Drop selected column",
            "Create log transformation",
            "Create squared feature",
        ],
    )

    feature_column = st.selectbox(
        "Column",
        list(df.columns),
        key="feature_engineering_column",
    )

    new_feature_name = st.text_input(
        "New Feature Name",
        value="",
    )

    if st.button(
        "Apply Feature Engineering",
        type="primary",
        use_container_width=True,
    ):
        try:
            before_df = safe_copy(
                st.session_state.working_df
            )

            after_df, message = (
                feature_engineering(
                    before_df,
                    feature_method,
                    feature_column,
                    new_feature_name,
                )
            )

            if after_df.equals(
                before_df
            ):
                st.warning(
                    message
                )
            else:
                st.session_state.working_df = (
                    after_df
                )

                add_audit_log(
                    message,
                    before_df,
                    after_df,
                )

                st.session_state.trained_model = None

                st.success(
                    message
                )

                st.rerun()

        except Exception as error:
            st.error(
                f"Feature engineering failed: {error}"
            )


# ============================================================
# OUTLIERS
# ============================================================

with tabs[6]:
    st.markdown(
        '<div class="section-title">Outlier Analysis</div>',
        unsafe_allow_html=True,
    )

    outlier_table = outlier_report(
        df
    )

    if outlier_table.empty:
        st.info(
            "No numeric columns are available for outlier analysis."
        )

    else:
        st.dataframe(
            outlier_table,
            use_container_width=True,
            height=450,
        )

        numeric_columns = list(
            df.select_dtypes(
                include=np.number
            ).columns
        )

        selected_outlier_column = st.selectbox(
            "Select Numeric Column",
            numeric_columns,
            key="outlier_box_column",
        )

        figure, axis = plt.subplots(
            figsize=(10, 4)
        )

        sns.boxplot(
            x=df[
                selected_outlier_column
            ],
            ax=axis,
        )

        axis.set_title(
            f"Box Plot: {selected_outlier_column}"
        )

        st.pyplot(
            figure,
            clear_figure=True,
        )


# ============================================================
# TARGET ANALYSIS
# ============================================================

with tabs[7]:
    st.markdown(
        '<div class="section-title">Target Analysis</div>',
        unsafe_allow_html=True,
    )

    target_column = st.selectbox(
        "Target Column",
        list(df.columns),
        key="target_analysis_column",
    )

    target_counts = (
        df[target_column]
        .value_counts(
            dropna=False
        )
    )

    target_table = pd.DataFrame(
        {
            "Value":
                target_counts.index.astype(
                    str
                ),
            "Count":
                target_counts.values,
            "Percentage":
                (
                    target_counts.values
                    / max(len(df), 1)
                    * 100
                ).round(2),
        }
    )

    st.dataframe(
        target_table.head(50),
        use_container_width=True,
    )

    figure, axis = plt.subplots(
        figsize=(10, 4)
    )

    if pd.api.types.is_numeric_dtype(
        df[target_column]
    ):
        sns.histplot(
            df[target_column].dropna(),
            kde=True,
            ax=axis,
        )

    else:
        target_counts.head(
            20
        ).plot(
            kind="bar",
            ax=axis,
        )

        axis.tick_params(
            axis="x",
            rotation=45,
        )

    axis.set_title(
        f"Target Distribution: {target_column}"
    )

    st.pyplot(
        figure,
        clear_figure=True,
    )


# ============================================================
# MODEL TRAINING
# ============================================================

with tabs[8]:
    st.markdown(
        '<div class="section-title">Machine Learning Model Training</div>',
        unsafe_allow_html=True,
    )

    training_target = st.selectbox(
        "Target Column",
        list(df.columns),
        key="training_target_column",
    )

    training_task = st.radio(
        "Problem Type",
        [
            "Classification",
            "Regression",
        ],
        horizontal=True,
        key="training_task_type",
    )

    if training_task == "Classification":
        available_models = list(
            classification_models().keys()
        )
    else:
        available_models = list(
            regression_models().keys()
        )

    training_model_name = st.selectbox(
        "Model",
        available_models,
        key="training_model_name",
    )

    training_scaling = st.selectbox(
        "Feature Scaling",
        [
            "StandardScaler",
            "MinMaxScaler",
            "No Scaling",
        ],
        key="training_scaling",
    )

    training_test_size = st.slider(
        "Test Data Percentage",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        key="training_test_size",
    )

    if st.button(
        "Train Model",
        type="primary",
        use_container_width=True,
    ):
        try:
            (
                pipeline,
                X_train,
                X_test,
                y_train,
                y_test,
                predictions,
                metrics,
            ) = train_single_model(
                df,
                training_target,
                training_task,
                training_model_name,
                training_test_size / 100,
                training_scaling,
            )

            st.session_state.trained_model = (
                pipeline
            )

            st.session_state.trained_task = (
                training_task
            )

            st.session_state.trained_target = (
                training_target
            )

            st.session_state.trained_model_name = (
                training_model_name
            )

            st.session_state.trained_metrics = (
                metrics
            )

            st.success(
                "Model trained successfully."
            )

            metrics_table = pd.DataFrame(
                {
                    "Metric":
                        list(metrics.keys()),
                    "Score":
                        [
                            round(
                                float(value),
                                5,
                            )
                            for value in metrics.values()
                        ],
                }
            )

            st.dataframe(
                metrics_table,
                use_container_width=True,
            )

            if (
                training_task
                == "Classification"
            ):
                st.subheader(
                    "Classification Report"
                )

                report = classification_report(
                    y_test,
                    predictions,
                    output_dict=True,
                    zero_division=0,
                )

                st.dataframe(
                    pd.DataFrame(
                        report
                    ).T,
                    use_container_width=True,
                )

                st.subheader(
                    "Confusion Matrix"
                )

                figure, axis = plt.subplots(
                    figsize=(7, 5)
                )

                sns.heatmap(
                    confusion_matrix(
                        y_test,
                        predictions,
                    ),
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    ax=axis,
                )

                axis.set_xlabel(
                    "Predicted"
                )

                axis.set_ylabel(
                    "Actual"
                )

                st.pyplot(
                    figure,
                    clear_figure=True,
                )

                if (
                    hasattr(
                        pipeline,
                        "predict_proba",
                    )
                    and y_test.nunique()
                    == 2
                ):
                    try:
                        probabilities = (
                            pipeline.predict_proba(
                                X_test
                            )[:, 1]
                        )

                        model_classes = (
                            pipeline
                            .named_steps[
                                "model"
                            ]
                            .classes_
                        )

                        positive_class = (
                            model_classes[1]
                        )

                        binary_target = (
                            y_test
                            == positive_class
                        ).astype(int)

                        auc_score = (
                            roc_auc_score(
                                binary_target,
                                probabilities,
                            )
                        )

                        fpr, tpr, _ = (
                            roc_curve(
                                binary_target,
                                probabilities,
                            )
                        )

                        figure, axis = (
                            plt.subplots(
                                figsize=(7, 5)
                            )
                        )

                        axis.plot(
                            fpr,
                            tpr,
                            label=(
                                f"AUC = {auc_score:.3f}"
                            ),
                        )

                        axis.plot(
                            [0, 1],
                            [0, 1],
                            linestyle="--",
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

                        axis.legend()

                        st.pyplot(
                            figure,
                            clear_figure=True,
                        )

                    except Exception:
                        pass

            else:
                st.subheader(
                    "Actual versus Predicted"
                )

                actual_predicted = pd.DataFrame(
                    {
                        "Actual":
                            y_test.values,
                        "Predicted":
                            predictions,
                    }
                )

                st.dataframe(
                    actual_predicted.head(
                        100
                    ),
                    use_container_width=True,
                )

                figure, axis = plt.subplots(
                    figsize=(7, 5)
                )

                axis.scatter(
                    y_test,
                    predictions,
                    alpha=0.65,
                )

                lower = min(
                    float(
                        y_test.min()
                    ),
                    float(
                        np.min(
                            predictions
                        )
                    ),
                )

                upper = max(
                    float(
                        y_test.max()
                    ),
                    float(
                        np.max(
                            predictions
                        )
                    ),
                )

                axis.plot(
                    [lower, upper],
                    [lower, upper],
                    linestyle="--",
                )

                axis.set_xlabel(
                    "Actual"
                )

                axis.set_ylabel(
                    "Predicted"
                )

                axis.set_title(
                    "Actual versus Predicted"
                )

                st.pyplot(
                    figure,
                    clear_figure=True,
                )

                residuals = (
                    y_test
                    - predictions
                )

                figure, axis = plt.subplots(
                    figsize=(7, 5)
                )

                axis.scatter(
                    predictions,
                    residuals,
                    alpha=0.65,
                )

                axis.axhline(
                    0,
                    linestyle="--",
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

                st.pyplot(
                    figure,
                    clear_figure=True,
                )

        except Exception as error:
            st.error(
                f"Model training could not be completed: {error}"
            )


# ============================================================
# MODEL COMPARISON
# ============================================================

with tabs[9]:
    st.markdown(
        '<div class="section-title">Model Comparison</div>',
        unsafe_allow_html=True,
    )

    comparison_target = st.selectbox(
        "Target Column",
        list(df.columns),
        key="comparison_target_column",
    )

    comparison_task = st.radio(
        "Problem Type",
        [
            "Classification",
            "Regression",
        ],
        horizontal=True,
        key="comparison_task_type",
    )

    comparison_scaling = st.selectbox(
        "Feature Scaling",
        [
            "StandardScaler",
            "MinMaxScaler",
            "No Scaling",
        ],
        key="comparison_scaling",
    )

    comparison_test_size = st.slider(
        "Test Data Percentage",
        10,
        40,
        20,
        5,
        key="comparison_test_size",
    )

    if st.button(
        "Compare All Models",
        type="primary",
        use_container_width=True,
    ):
        if comparison_task == "Classification":
            model_dictionary = (
                classification_models()
            )
        else:
            model_dictionary = (
                regression_models()
            )

        comparison_rows = []

        progress = st.progress(0)

        total_models = len(
            model_dictionary
        )

        for index, model_name in enumerate(
            model_dictionary
        ):
            try:
                (
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    metrics,
                ) = train_single_model(
                    df,
                    comparison_target,
                    comparison_task,
                    model_name,
                    comparison_test_size
                    / 100,
                    comparison_scaling,
                )

                comparison_rows.append(
                    {
                        "Model":
                            model_name,
                        **{
                            key:
                                round(
                                    float(
                                        value
                                    ),
                                    4,
                                )
                            for key, value
                            in metrics.items()
                        },
                        "Status":
                            "Success",
                    }
                )

            except Exception as error:
                comparison_rows.append(
                    {
                        "Model":
                            model_name,
                        "Status":
                            f"Failed: {error}",
                    }
                )

            progress.progress(
                (index + 1)
                / total_models
            )

        comparison_result = pd.DataFrame(
            comparison_rows
        )

        if (
            comparison_task
            == "Classification"
            and "Accuracy"
            in comparison_result.columns
        ):
            comparison_result = (
                comparison_result.sort_values(
                    "Accuracy",
                    ascending=False,
                    na_position="last",
                )
            )

            successful = (
                comparison_result[
                    comparison_result[
                        "Status"
                    ]
                    == "Success"
                ]
            )

            if not successful.empty:
                best = successful.iloc[0]

                st.success(
                    f"Best model by Accuracy: {best['Model']} with {best['Accuracy']:.4f}"
                )

        elif (
            comparison_task
            == "Regression"
            and "R2"
            in comparison_result.columns
        ):
            comparison_result = (
                comparison_result.sort_values(
                    "R2",
                    ascending=False,
                    na_position="last",
                )
            )

            successful = (
                comparison_result[
                    comparison_result[
                        "Status"
                    ]
                    == "Success"
                ]
            )

            if not successful.empty:
                best = successful.iloc[0]

                st.success(
                    f"Best model by R2: {best['Model']} with {best['R2']:.4f}"
                )

        st.dataframe(
            comparison_result,
            use_container_width=True,
        )


# ============================================================
# MODEL EVALUATION
# ============================================================

with tabs[10]:
    st.markdown(
        '<div class="section-title">Model Evaluation</div>',
        unsafe_allow_html=True,
    )

    if (
        st.session_state.trained_model
        is None
    ):
        st.info(
            "Train a model first to view model evaluation."
        )

    else:
        st.write(
            f"Model: {st.session_state.trained_model_name}"
        )

        st.write(
            f"Problem Type: {st.session_state.trained_task}"
        )

        st.write(
            f"Target: {st.session_state.trained_target}"
        )

        metrics = (
            st.session_state.trained_metrics
        )

        metrics_table = pd.DataFrame(
            {
                "Metric":
                    list(metrics.keys()),
                "Score":
                    [
                        round(
                            float(value),
                            5,
                        )
                        for value in metrics.values()
                    ],
            }
        )

        st.dataframe(
            metrics_table,
            use_container_width=True,
        )


# ============================================================
# PREDICTION
# ============================================================

with tabs[11]:
    st.markdown(
        '<div class="section-title">New Record Prediction</div>',
        unsafe_allow_html=True,
    )

    pipeline = (
        st.session_state.trained_model
    )

    if pipeline is None:
        st.info(
            "Train a model first."
        )

    else:
        try:
            X, _ = prepare_ml_data(
                df,
                st.session_state.trained_target,
                st.session_state.trained_task,
            )

            prediction_values = {}

            prediction_columns = st.columns(
                2
            )

            for index, column in enumerate(
                X.columns
            ):
                container = prediction_columns[
                    index % 2
                ]

                if pd.api.types.is_numeric_dtype(
                    X[column]
                ):
                    median = X[column].median()

                    default_value = (
                        float(median)
                        if pd.notna(median)
                        else 0.0
                    )

                    prediction_values[
                        column
                    ] = container.number_input(
                        column,
                        value=default_value,
                        key=(
                            f"prediction_"
                            f"{column}"
                        ),
                    )

                else:
                    choices = (
                        X[column]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    if not choices:
                        choices = [""]

                    prediction_values[
                        column
                    ] = container.selectbox(
                        column,
                        choices[:100],
                        key=(
                            f"prediction_"
                            f"{column}"
                        ),
                    )

            if st.button(
                "Generate Prediction",
                type="primary",
                use_container_width=True,
            ):
                input_row = pd.DataFrame(
                    [prediction_values]
                )

                prediction = (
                    pipeline.predict(
                        input_row
                    )[0]
                )

                st.success(
                    f"Prediction: {prediction}"
                )

                if (
                    st.session_state.trained_task
                    == "Classification"
                    and hasattr(
                        pipeline,
                        "predict_proba",
                    )
                ):
                    try:
                        probabilities = (
                            pipeline.predict_proba(
                                input_row
                            )[0]
                        )

                        classes = (
                            pipeline
                            .named_steps[
                                "model"
                            ]
                            .classes_
                        )

                        probability_table = pd.DataFrame(
                            {
                                "Class":
                                    classes,
                                "Probability %":
                                    np.round(
                                        probabilities
                                        * 100,
                                        2,
                                    ),
                            }
                        ).sort_values(
                            "Probability %",
                            ascending=False,
                        )

                        st.dataframe(
                            probability_table,
                            use_container_width=True,
                        )

                    except Exception:
                        pass

        except Exception as error:
            st.error(
                f"Prediction could not be generated: {error}"
            )


# ============================================================
# EXPLAINABILITY
# ============================================================

with tabs[12]:
    st.markdown(
        '<div class="section-title">Model Explainability</div>',
        unsafe_allow_html=True,
    )

    pipeline = (
        st.session_state.trained_model
    )

    if pipeline is None:
        st.info(
            "Train a model first."
        )

    else:
        importance = feature_importance_table(
            pipeline
        )

        if importance.empty:
            st.info(
                "Feature importance is not available for this model."
            )

        else:
            st.dataframe(
                importance,
                use_container_width=True,
            )

            top_features = (
                importance.head(15)
                .sort_values(
                    "Importance"
                )
            )

            figure, axis = plt.subplots(
                figsize=(10, 7)
            )

            axis.barh(
                top_features[
                    "Feature"
                ],
                top_features[
                    "Importance"
                ],
            )

            axis.set_xlabel(
                "Importance"
            )

            axis.set_title(
                "Top Feature Importance"
            )

            st.pyplot(
                figure,
                clear_figure=True,
            )


# ============================================================
# DOWNLOAD
# ============================================================

with tabs[13]:
    st.markdown(
        '<div class="section-title">Download Dataset</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Download the current version of the dataset. If you have applied cleaning, this contains the cleaned dataset."
    )

    st.dataframe(
        df.head(100),
        use_container_width=True,
        height=400,
    )

    selected_download_format = st.radio(
        "Select Download Format",
        [
            "CSV",
            "Excel",
        ],
        horizontal=True,
        key="main_download_format",
    )

    if selected_download_format == "CSV":
        st.download_button(
            "Download Current Dataset as CSV",
            data=create_csv_bytes(
                df
            ),
            file_name="analytics_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

    else:
        st.download_button(
            "Download Current Dataset as Excel",
            data=create_excel_bytes(
                df
            ),
            file_name="analytics_dataset.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
        )

    st.subheader(
        "Download Cleaning Audit Log"
    )

    if st.session_state.audit_log:
        st.download_button(
            "Download Audit Log",
            data=create_audit_bytes(),
            file_name="cleaning_audit_log.csv",
            mime="text/csv",
            use_container_width=True,
        )

    else:
        st.info(
            "No audit records are available."
        )


# ============================================================
# AUDIT LOG
# ============================================================

with tabs[14]:
    st.markdown(
        '<div class="section-title">Cleaning Audit Log</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.audit_log:
        audit_df = pd.DataFrame(
            st.session_state.audit_log
        )

        st.dataframe(
            audit_df,
            use_container_width=True,
            height=500,
        )

    else:
        st.info(
            "No cleaning actions have been performed."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI and ML Analytics Dashboard | Data Analysis | Data Cleaning | Machine Learning"
)