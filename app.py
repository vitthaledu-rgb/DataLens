
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
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

st.set_page_config(
    page_title="AI & ML Analytics Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SESSION STATE
# ============================================================

STATE_DEFAULTS = {
    "df_original": None,
    "df_working": None,
    "filename": "",
    "audit_log": [],
    "trained_pipeline": None,
    "trained_task": None,
    "trained_target": None,
    "trained_model_name": None,
    "trained_metrics": None,
}

for key, value in STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# BASIC HELPERS
# ============================================================

def add_audit(action, before, after):
    st.session_state.audit_log.append({
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Action": action,
        "Rows Before": len(before),
        "Rows After": len(after),
        "Columns Before": len(before.columns),
        "Columns After": len(after.columns),
        "Missing Before": int(before.isna().sum().sum()),
        "Missing After": int(after.isna().sum().sum()),
        "Duplicates Before": int(before.duplicated().sum()),
        "Duplicates After": int(after.duplicated().sum()),
    })


def read_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin1")
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    if name.endswith(".xls"):
        return pd.read_excel(uploaded_file, engine="xlrd")
    raise ValueError("Only CSV, XLSX and XLS files are supported.")


def missing_report(df):
    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Count": df.isna().sum().values,
        "Missing %": (df.isna().mean() * 100).round(2).values,
        "Unique Values": df.nunique(dropna=True).values,
    }).sort_values("Missing Count", ascending=False).reset_index(drop=True)


def duplicate_report(df):
    n = int(df.duplicated().sum())
    return pd.DataFrame({
        "Metric": ["Total Rows", "Duplicate Rows", "Duplicate %"],
        "Value": [len(df), n, round(n / max(len(df), 1) * 100, 2)]
    })


def statistical_profile(df):
    rows = []
    for col in df.columns:
        s = df[col]
        row = {
            "Column": col,
            "Type": str(s.dtype),
            "Count": int(s.notna().sum()),
            "Missing": int(s.isna().sum()),
            "Missing %": round(s.isna().mean() * 100, 2),
            "Unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            x = pd.to_numeric(s, errors="coerce").dropna()
            row.update({
                "Mean": round(x.mean(), 4) if len(x) else np.nan,
                "Median": round(x.median(), 4) if len(x) else np.nan,
                "Std": round(x.std(), 4) if len(x) else np.nan,
                "Min": round(x.min(), 4) if len(x) else np.nan,
                "Max": round(x.max(), 4) if len(x) else np.nan,
                "Skewness": round(x.skew(), 4) if len(x) > 2 else np.nan,
                "Kurtosis": round(x.kurtosis(), 4) if len(x) > 3 else np.nan,
            })
        else:
            row.update({
                "Mean": np.nan, "Median": np.nan, "Std": np.nan,
                "Min": np.nan, "Max": np.nan,
                "Skewness": np.nan, "Kurtosis": np.nan,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def outlier_report(df):
    rows = []
    for col in df.select_dtypes(include=np.number).columns:
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(x) < 2:
            continue
        q1, q3 = x.quantile(.25), x.quantile(.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_n = int(((x < lo) | (x > hi)).sum())
        z_n = 0
        if len(x) >= 3:
            z = np.abs(stats.zscore(x))
            z_n = int((z > 3).sum())
        rows.append({
            "Column": col,
            "Q1": round(q1, 4),
            "Q3": round(q3, 4),
            "IQR": round(iqr, 4),
            "Lower Bound": round(lo, 4),
            "Upper Bound": round(hi, 4),
            "IQR Outliers": iqr_n,
            "IQR Outlier %": round(iqr_n / len(x) * 100, 2),
            "Z > 3": z_n,
            "Z Outlier %": round(z_n / len(x) * 100, 2),
        })
    return pd.DataFrame(rows)


def correlation_pairs(df):
    num = df.select_dtypes(include=np.number)
    if num.shape[1] < 2:
        return pd.DataFrame()
    corr = num.corr()
    result = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            result.append({
                "Feature 1": cols[i],
                "Feature 2": cols[j],
                "Pearson": round(v, 4),
                "Absolute": round(abs(v), 4),
            })
    return pd.DataFrame(result).sort_values("Absolute", ascending=False)


def validity_report(df):
    rows = []
    for col in df.select_dtypes(include=np.number).columns:
        x = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "Column": col,
            "Negative Values": int((x < 0).sum()),
            "Zero Values": int((x == 0).sum()),
            "Infinite Values": int(np.isinf(x).sum()),
        })
    return pd.DataFrame(rows)


def convert_numeric_text(df):
    out = df.copy()
    converted = []
    for col in out.columns:
        if out[col].dtype == "object":
            x = pd.to_numeric(out[col], errors="coerce")
            original_non_null = int(out[col].notna().sum())
            if original_non_null and x.notna().sum() / original_non_null >= 0.80:
                out[col] = x
                converted.append(col)
    return out, converted


# ============================================================
# CLEANING
# ============================================================

def clean_data(
    df,
    missing_method,
    duplicate_method,
    outlier_method,
    numeric_method,
    categorical_method,
    invalid_method,
    constant_value,
    round_digits,
):
    out = df.copy()
    changes = []

    if duplicate_method == "Remove duplicates":
        before = len(out)
        out = out.drop_duplicates().reset_index(drop=True)
        changes.append(f"removed {before - len(out):,} duplicate rows")

    if missing_method == "Drop rows":
        before = len(out)
        out = out.dropna().reset_index(drop=True)
        changes.append(f"dropped {before - len(out):,} rows with missing values")
    elif missing_method == "Drop columns >50% missing":
        before_cols = set(out.columns)
        threshold = int(np.ceil(len(out) * 0.50))
        out = out.dropna(axis=1, thresh=threshold)
        changes.append(f"dropped {len(before_cols - set(out.columns))} columns")
    elif missing_method in {"Mean imputation", "Median imputation"}:
        for col in out.select_dtypes(include=np.number).columns:
            fill = out[col].mean() if missing_method == "Mean imputation" else out[col].median()
            out[col] = out[col].fillna(fill)
        changes.append(missing_method.lower())
    elif missing_method == "Mode imputation":
        for col in out.columns:
            mode = out[col].mode(dropna=True)
            if not mode.empty:
                out[col] = out[col].fillna(mode.iloc[0])
        changes.append("mode imputation")
    elif missing_method == "Constant imputation":
        for col in out.columns:
            if pd.api.types.is_numeric_dtype(out[col]):
                try:
                    value = float(constant_value)
                except Exception:
                    value = 0.0
            else:
                value = constant_value
            out[col] = out[col].fillna(value)
        changes.append("constant imputation")
    elif missing_method == "Forward fill":
        out = out.ffill()
        changes.append("forward fill")
    elif missing_method == "Backward fill":
        out = out.bfill()
        changes.append("backward fill")
    elif missing_method == "KNN imputation":
        from sklearn.impute import KNNImputer
        numeric = out.select_dtypes(include=np.number).columns
        if len(numeric):
            out[numeric] = KNNImputer(n_neighbors=5).fit_transform(out[numeric])
        changes.append("KNN imputation for numeric columns")
    elif missing_method == "Iterative imputation":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        numeric = out.select_dtypes(include=np.number).columns
        if len(numeric):
            out[numeric] = IterativeImputer(random_state=42).fit_transform(out[numeric])
        changes.append("iterative imputation for numeric columns")

    if numeric_method == "Convert numeric-looking text":
        out, cols = convert_numeric_text(out)
        changes.append(f"converted {len(cols)} numeric-looking text columns")
    elif numeric_method == "Round numeric values":
        numeric = out.select_dtypes(include=np.number).columns
        out[numeric] = out[numeric].round(int(round_digits))
        changes.append(f"rounded numeric values to {int(round_digits)} decimals")

    text_cols = out.select_dtypes(include="object").columns
    if categorical_method == "Remove extra spaces":
        for col in text_cols:
            out[col] = out[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
        changes.append("trimmed text")
    elif categorical_method == "Lowercase":
        for col in text_cols:
            out[col] = out[col].apply(lambda v: v.strip().lower() if isinstance(v, str) else v)
        changes.append("lowercased text")
    elif categorical_method == "Uppercase":
        for col in text_cols:
            out[col] = out[col].apply(lambda v: v.strip().upper() if isinstance(v, str) else v)
        changes.append("uppercased text")

    if invalid_method == "Replace negative numeric values":
        count = 0
        for col in out.select_dtypes(include=np.number).columns:
            count += int((out[col] < 0).sum())
            out.loc[out[col] < 0, col] = np.nan
        changes.append(f"replaced {count:,} negative values with missing")

    numeric_cols = list(out.select_dtypes(include=np.number).columns)
    if outlier_method != "Do Nothing" and numeric_cols:
        if outlier_method in {"Remove IQR outliers", "Cap IQR outliers"}:
            for col in numeric_cols:
                x = out[col]
                if x.dropna().empty:
                    continue
                q1, q3 = x.quantile(.25), x.quantile(.75)
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                if outlier_method == "Remove IQR outliers":
                    out = out.loc[x.isna() | x.between(lo, hi)].reset_index(drop=True)
                else:
                    out[col] = x.clip(lo, hi)
            changes.append(outlier_method.lower())
        else:
            keep = pd.Series(True, index=out.index)
            for col in numeric_cols:
                x = out[col]
                if x.notna().sum() >= 3:
                    z = np.abs(stats.zscore(x.fillna(x.median())))
                    keep &= pd.Series(z <= 3, index=out.index)
            out = out.loc[keep].reset_index(drop=True)
            changes.append("removed Z-score > 3 outliers")

    return out, changes


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def feature_engineering(df, method, column, new_name):
    out = df.copy()
    if method == "Drop selected column":
        if column in out.columns:
            out = out.drop(columns=[column])
            return out, f"dropped column '{column}'"
        return out, "column not found"

    if column not in out.columns:
        return out, "column not found"

    if not pd.api.types.is_numeric_dtype(out[column]):
        return out, "selected column must be numeric"

    name = new_name.strip() or f"{column}_transformed"
    if method == "Create log transformation":
        if (out[column] <= 0).any():
            return out, "log transformation requires positive values"
        out[name] = np.log1p(out[column])
        return out, f"created '{name}' using log1p"

    if method == "Create squared feature":
        out[name] = out[column] ** 2
        return out, f"created '{name}' as squared feature"

    return out, "no feature-engineering change"


# ============================================================
# VISUALIZATION
# ============================================================

GRAPH_TYPES = [
    "Histogram", "Box Plot", "Violin Plot", "Count Plot",
    "Bar Chart", "Scatter Plot", "Line Chart", "Area Chart",
    "KDE", "Pie Chart", "Correlation Heatmap", "Covariance Heatmap",
    "Missing Values", "Q-Q Plot",
]


def make_plot(df, graph_type, x_col, y_col, hue_col):
    fig, ax = plt.subplots(figsize=(10, 5))
    hue = None if hue_col in [None, "", "None"] else hue_col

    try:
        if graph_type == "Histogram":
            sns.histplot(data=df, x=x_col, hue=hue, kde=True, ax=ax)
        elif graph_type == "Box Plot":
            if y_col == "None":
                sns.boxplot(data=df, y=x_col, ax=ax)
            else:
                sns.boxplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
        elif graph_type == "Violin Plot":
            if y_col == "None":
                sns.violinplot(data=df, y=x_col, ax=ax)
            else:
                sns.violinplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
        elif graph_type == "Count Plot":
            sns.countplot(data=df, x=x_col, hue=hue, ax=ax)
            ax.tick_params(axis="x", rotation=45)
        elif graph_type == "Bar Chart":
            if y_col != "None" and pd.api.types.is_numeric_dtype(df[y_col]):
                temp = df.groupby(x_col, dropna=False)[y_col].mean().sort_values(ascending=False).head(20)
                temp.plot(kind="bar", ax=ax)
            else:
                df[x_col].astype(str).value_counts().head(20).plot(kind="bar", ax=ax)
            ax.tick_params(axis="x", rotation=45)
        elif graph_type == "Scatter Plot":
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
        elif graph_type == "Line Chart":
            sns.lineplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
        elif graph_type == "Area Chart":
            temp = df[[x_col, y_col]].dropna().head(300)
            if temp.empty:
                raise ValueError("No data available.")
            if pd.api.types.is_numeric_dtype(temp[x_col]):
                temp = temp.sort_values(x_col)
                ax.fill_between(temp[x_col], temp[y_col], alpha=.35)
                ax.plot(temp[x_col], temp[y_col])
            else:
                ax.fill_between(range(len(temp)), temp[y_col], alpha=.35)
                ax.plot(range(len(temp)), temp[y_col])
        elif graph_type == "KDE":
            sns.kdeplot(data=df, x=x_col, hue=hue, fill=True, ax=ax)
        elif graph_type == "Pie Chart":
            counts = df[x_col].astype(str).value_counts().head(10)
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
        elif graph_type == "Correlation Heatmap":
            numeric = df.select_dtypes(include=np.number)
            if numeric.shape[1] < 2:
                raise ValueError("At least 2 numeric columns are required.")
            sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        elif graph_type == "Covariance Heatmap":
            numeric = df.select_dtypes(include=np.number)
            if numeric.shape[1] < 2:
                raise ValueError("At least 2 numeric columns are required.")
            sns.heatmap(numeric.cov(), annot=True, fmt=".2f", cmap="viridis", ax=ax)
        elif graph_type == "Missing Values":
            missing = df.isna().sum().sort_values(ascending=False)
            missing = missing[missing > 0]
            if missing.empty:
                ax.text(.5, .5, "No missing values", ha="center", va="center")
                ax.axis("off")
            else:
                missing.plot(kind="bar", ax=ax)
                ax.tick_params(axis="x", rotation=45)
        elif graph_type == "Q-Q Plot":
            x = pd.to_numeric(df[x_col], errors="coerce").dropna()
            if len(x) < 3:
                raise ValueError("At least 3 numeric values are required.")
            stats.probplot(x, dist="norm", plot=ax)

        ax.set_title(graph_type, fontweight="bold")
        fig.tight_layout()
        return fig
    except Exception as exc:
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(.5, .5, f"Visualization error:\n{exc}", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig


# ============================================================
# MACHINE LEARNING
# ============================================================

def models_for(task):
    if task == "Classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(probability=True, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "SVR": SVR(),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }


def prepare_ml(df, target, task):
    if target not in df.columns:
        raise ValueError("Target column not found.")

    data = df.replace([np.inf, -np.inf], np.nan).copy()
    data = data.dropna(subset=[target])

    X = data.drop(columns=[target]).dropna(axis=1, how="all")
    y = data[target]

    if X.empty:
        raise ValueError("No usable feature columns.")

    if task == "Classification":
        y = y.astype(str)
        if y.nunique() < 2:
            raise ValueError("Classification needs at least two classes.")
    else:
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
        if len(y) < 10:
            raise ValueError("Regression needs at least 10 valid target values.")

    return X, y


def build_preprocessor(X, scaling):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaling == "StandardScaler":
        steps.append(("scaler", StandardScaler()))
    elif scaling == "MinMaxScaler":
        steps.append(("scaler", MinMaxScaler()))

    num_pipe = Pipeline(steps)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric:
        transformers.append(("numeric", num_pipe, numeric))
    if categorical:
        transformers.append(("categorical", cat_pipe, categorical))

    if not transformers:
        raise ValueError("No usable numeric or categorical features.")

    return ColumnTransformer(transformers=transformers)


def train_model(df, target, task, model_name, test_size, scaling):
    X, y = prepare_ml(df, target, task)
    model = models_for(task)[model_name]
    preprocessor = build_preprocessor(X, scaling)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    stratify = None
    if task == "Classification" and y.value_counts().min() >= 2:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    if task == "Classification":
        metrics = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, pred, average="weighted", zero_division=0),
        }
    else:
        metrics = {
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R²": r2_score(y_test, pred),
        }

    return pipe, X_train, X_test, y_train, y_test, pred, metrics


def get_importance(pipe):
    try:
        pre = pipe.named_steps["preprocessor"]
        model = pipe.named_steps["model"]
        names = pre.get_feature_names_out()

        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            values = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            return pd.DataFrame(columns=["Feature", "Importance"])

        return (
            pd.DataFrame({"Feature": names, "Importance": values})
            .sort_values("Importance", ascending=False)
            .head(30)
            .reset_index(drop=True)
        )
    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance"])


# ============================================================
# PAGE STYLE
# ============================================================

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.title {text-align:center; font-size:2.15rem; font-weight:800;}
.subtitle {text-align:center; color:#667085; margin-bottom:1rem;}
.kpi {border:1px solid #e4e7ec; border-radius:14px; padding:13px; text-align:center; background:#fff;}
.kpi-label {font-size:.75rem; color:#667085; text-transform:uppercase;}
.kpi-value {font-size:1.5rem; font-weight:800;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 AI & ML Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">End-to-End Data Science • EDA • Cleaning • Machine Learning • Prediction • Explainability</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR / UPLOAD
# ============================================================

with st.sidebar:
    st.header("📂 Dataset")
    uploaded = st.file_uploader("Upload CSV / XLSX / XLS", type=["csv", "xlsx", "xls"])

    if uploaded is not None and st.session_state.filename != uploaded.name:
        try:
            data = read_file(uploaded)
            if data.empty:
                st.error("The dataset is empty.")
            else:
                st.session_state.df_original = data.copy()
                st.session_state.df_working = data.copy()
                st.session_state.filename = uploaded.name
                st.session_state.audit_log = [{
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Action": "Dataset loaded",
                    "Rows Before": len(data),
                    "Rows After": len(data),
                    "Columns Before": len(data.columns),
                    "Columns After": len(data.columns),
                    "Missing Before": int(data.isna().sum().sum()),
                    "Missing After": int(data.isna().sum().sum()),
                    "Duplicates Before": int(data.duplicated().sum()),
                    "Duplicates After": int(data.duplicated().sum()),
                }]
                st.session_state.trained_pipeline = None
                st.rerun()
        except Exception as exc:
            st.error(f"File loading error: {exc}")

    if st.session_state.df_working is not None:
        st.success(f"Loaded: {st.session_state.filename}")
        if st.button("🔄 Reset to Original", use_container_width=True):
            before = st.session_state.df_working.copy()
            st.session_state.df_working = st.session_state.df_original.copy()
            add_audit("Reset to original dataset", before, st.session_state.df_working)
            st.session_state.trained_pipeline = None
            st.rerun()

if st.session_state.df_working is None:
    st.info("👈 Upload a dataset from the sidebar to begin.")
    st.markdown("""
### Included features
- 📋 Overview and preview
- 📈 Statistics, skewness and kurtosis
- 🔎 Missing values, duplicates and validity checks
- 🔗 Correlation and covariance
- 📊 Histogram, box, violin, count, bar, scatter, line, area, KDE, pie, heatmaps and Q-Q plot
- 🧹 Advanced cleaning and imputation
- ⚙️ Feature engineering
- 🚨 IQR and Z-score outlier analysis
- 🎯 Target analysis
- 🤖 Classification and regression
- 🏆 Model comparison
- 📊 Evaluation
- 🎯 New-record prediction
- 🧠 Feature importance
- 💾 CSV / Excel download
- 🧾 Cleaning audit log
""")
    st.stop()

df = st.session_state.df_working

# ============================================================
# KPI ROW
# ============================================================

kpi_values = [
    ("Rows", f"{len(df):,}"),
    ("Columns", f"{len(df.columns):,}"),
    ("Numeric", f"{len(df.select_dtypes(include=np.number).columns):,}"),
    ("Categorical", f"{len(df.select_dtypes(exclude=np.number).columns):,}"),
    ("Missing", f"{int(df.isna().sum().sum()):,}"),
    ("Duplicates", f"{int(df.duplicated().sum()):,}"),
]
kcols = st.columns(6)
for col, (label, value) in zip(kcols, kpi_values):
    col.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

st.caption(f"Current working dataset: **{st.session_state.filename}**")


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "📋 Overview", "📈 Statistics", "🔎 Data Quality", "📊 EDA",
    "🧹 Cleaning", "⚙️ Features", "🚨 Outliers", "🎯 Target",
    "🤖 Training", "🏆 Comparison", "📊 Evaluation", "🎯 Prediction",
    "🧠 Explainability", "💾 Download", "🧾 Audit Log"
])


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:
    st.header("📋 Dataset Overview")
    max_rows = min(100, max(5, len(df)))
    preview_rows = st.slider("Rows to preview", 5, max_rows, min(15, max_rows))
    st.dataframe(df.head(preview_rows), use_container_width=True, height=380)

    st.subheader("Column Information")
    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null": df.notna().sum().values,
        "Missing": df.isna().sum().values,
        "Missing %": (df.isna().mean() * 100).round(2).values,
        "Unique": df.nunique(dropna=True).values,
    })
    st.dataframe(info, use_container_width=True, height=400)


# ============================================================
# STATISTICS
# ============================================================

with tabs[1]:
    st.header("📈 Statistics")
    st.dataframe(
        df.describe(include="all").T.reset_index().rename(columns={"index": "Column"}),
        use_container_width=True,
        height=500,
    )
    st.subheader("Detailed Statistical Profile")
    st.dataframe(statistical_profile(df), use_container_width=True, height=500)

    numeric = list(df.select_dtypes(include=np.number).columns)
    if numeric:
        selected = st.selectbox("Distribution column", numeric)
        x = pd.to_numeric(df[selected], errors="coerce").dropna()
        a, b, c, d = st.columns(4)
        a.metric("Mean", f"{x.mean():.4f}" if len(x) else "—")
        b.metric("Median", f"{x.median():.4f}" if len(x) else "—")
        c.metric("Skewness", f"{x.skew():.4f}" if len(x) > 2 else "—")
        d.metric("Kurtosis", f"{x.kurtosis():.4f}" if len(x) > 3 else "—")


# ============================================================
# DATA QUALITY
# ============================================================

with tabs[2]:
    st.header("🔎 Data Quality")
    st.subheader("Missing Values")
    st.dataframe(missing_report(df), use_container_width=True, height=420)

    st.subheader("Duplicates")
    st.dataframe(duplicate_report(df), use_container_width=True)

    st.subheader("Value Counts")
    col = st.selectbox("Column", list(df.columns), key="quality_value_col")
    vc = df[col].value_counts(dropna=False).head(50).reset_index()
    vc.columns = [col, "Count"]
    vc["Percentage"] = (vc["Count"] / max(len(df), 1) * 100).round(2)
    st.dataframe(vc, use_container_width=True)

    st.subheader("Validity Checks")
    st.dataframe(validity_report(df), use_container_width=True)

    st.subheader("Correlation Strength")
    corr = correlation_pairs(df)
    if corr.empty:
        st.info("At least two numeric columns are required.")
    else:
        st.dataframe(corr.head(50), use_container_width=True)


# ============================================================
# EDA
# ============================================================

with tabs[3]:
    st.header("📊 Exploratory Data Analysis")
    graph_type = st.selectbox("📊 Graph Type", GRAPH_TYPES)

    c1, c2, c3 = st.columns(3)
    with c1:
        x_col = st.selectbox("X / Numeric Column", list(df.columns), key="eda_x")
    with c2:
        y_col = st.selectbox("Y Column", ["None"] + list(df.columns), key="eda_y")
    with c3:
        hue_col = st.selectbox("Group / Hue", ["None"] + list(df.columns), key="eda_hue")

    if st.button("📊 Generate Visualization", type="primary"):
        fig = make_plot(df, graph_type, x_col, y_col, hue_col)
        st.pyplot(fig, clear_figure=True)

    st.info("Heatmaps require numeric columns. Scatter/line/area charts require a suitable Y column.")


# ============================================================
# CLEANING
# ============================================================

with tabs[4]:
    st.header("🧹 Advanced Data Cleaning")

    c1, c2 = st.columns(2)
    with c1:
        missing_method = st.selectbox(
            "Missing Value Treatment",
            [
                "Do Nothing", "Drop rows", "Drop columns >50% missing",
                "Mean imputation", "Median imputation", "Mode imputation",
                "Constant imputation", "Forward fill", "Backward fill",
                "KNN imputation", "Iterative imputation"
            ],
        )
    with c2:
        constant_value = st.text_input("Constant value", "0")

    c1, c2 = st.columns(2)
    with c1:
        duplicate_method = st.selectbox("Duplicate Treatment", ["Do Nothing", "Remove duplicates"])
        numeric_method = st.selectbox(
            "Numeric Cleaning",
            ["Do Nothing", "Convert numeric-looking text", "Round numeric values"]
        )
    with c2:
        outlier_method = st.selectbox(
            "Outlier Treatment",
            ["Do Nothing", "Remove IQR outliers", "Cap IQR outliers", "Remove Z-score >3"]
        )
        categorical_method = st.selectbox(
            "Categorical Cleaning",
            ["Do Nothing", "Remove extra spaces", "Lowercase", "Uppercase"]
        )

    c1, c2 = st.columns(2)
    with c1:
        invalid_method = st.selectbox(
            "Invalid Value Treatment",
            ["Do Nothing", "Replace negative numeric values"]
        )
    with c2:
        round_digits = st.number_input("Round decimals", 0, 10, 2)

    if st.button("🧹 Apply Cleaning", type="primary"):
        before = st.session_state.df_working.copy()
        try:
            after, changes = clean_data(
                before, missing_method, duplicate_method, outlier_method,
                numeric_method, categorical_method, invalid_method,
                constant_value, round_digits
            )
            st.session_state.df_working = after
            add_audit(" • ".join(changes) if changes else "No changes", before, after)
            st.session_state.trained_pipeline = None
            st.success("Cleaning completed.")
            if changes:
                st.write("Changes:", " • ".join(changes))
            st.rerun()
        except Exception as exc:
            st.error(f"Cleaning error: {exc}")

    st.subheader("Cleaned Dataset Preview")
    st.dataframe(st.session_state.df_working.head(100), use_container_width=True, height=400)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

with tabs[5]:
    st.header("⚙️ Feature Engineering")
    st.write("Create simple reusable features before training.")

    feature_method = st.selectbox(
        "Feature Method",
        [
            "Select method",
            "Drop selected column",
            "Create log transformation",
            "Create squared feature"
        ]
    )
    feature_col = st.selectbox("Column", list(df.columns), key="feature_col")
    new_feature_name = st.text_input("New feature name", "")

    if st.button("⚙️ Apply Feature Engineering"):
        before = st.session_state.df_working.copy()
        try:
            after, message = feature_engineering(
                before, feature_method, feature_col, new_feature_name
            )
            if after.equals(before):
                st.warning(message)
            else:
                st.session_state.df_working = after
                add_audit(message, before, after)
                st.session_state.trained_pipeline = None
                st.success(message)
                st.rerun()
        except Exception as exc:
            st.error(f"Feature engineering error: {exc}")


# ============================================================
# OUTLIERS
# ============================================================

with tabs[6]:
    st.header("🚨 Outlier Analysis")
    report = outlier_report(df)
    if report.empty:
        st.info("No numeric columns available.")
    else:
        st.dataframe(report, use_container_width=True, height=450)
        numeric = list(df.select_dtypes(include=np.number).columns)
        selected = st.selectbox("Box plot column", numeric, key="outlier_col")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(x=df[selected], ax=ax)
        ax.set_title(f"IQR Box Plot — {selected}")
        st.pyplot(fig, clear_figure=True)


# ============================================================
# TARGET ANALYSIS
# ============================================================

with tabs[7]:
    st.header("🎯 Target Analysis")
    target = st.selectbox("Target Column", list(df.columns), key="target_analysis")
    values = df[target].value_counts(dropna=False)
    table = pd.DataFrame({
        "Category": values.index.astype(str),
        "Count": values.values,
        "Percentage": (values.values / max(len(df), 1) * 100).round(2),
    })
    st.dataframe(table.head(50), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    if pd.api.types.is_numeric_dtype(df[target]):
        sns.histplot(df[target].dropna(), kde=True, ax=ax)
    else:
        values.head(20).plot(kind="bar", ax=ax)
        ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"Target Analysis — {target}")
    st.pyplot(fig, clear_figure=True)


# ============================================================
# TRAINING
# ============================================================

with tabs[8]:
    st.header("🤖 Machine Learning Training")

    target = st.selectbox("🎯 Target Column", list(df.columns), key="training_target")
    task = st.radio("ML Task", ["Classification", "Regression"], horizontal=True)

    model_name = st.selectbox("🤖 Model", list(models_for(task).keys()), key=f"model_{task}")
    scaling = st.selectbox("Scaling", ["StandardScaler", "MinMaxScaler", "No Scaling"])
    test_size = st.slider("Test Size", 0.10, 0.40, 0.20, 0.05)

    if st.button("🚀 Train Model", type="primary"):
        try:
            pipe, X_train, X_test, y_train, y_test, pred, metrics = train_model(
                df, target, task, model_name, test_size, scaling
            )

            st.session_state.trained_pipeline = pipe
            st.session_state.trained_task = task
            st.session_state.trained_target = target
            st.session_state.trained_model_name = model_name
            st.session_state.trained_metrics = metrics

            st.success(f"Model trained successfully: {model_name}")
            st.dataframe(
                pd.DataFrame({
                    "Metric": list(metrics.keys()),
                    "Score": [round(float(v), 5) for v in metrics.values()]
                }),
                use_container_width=True
            )

            if task == "Classification":
                st.subheader("Classification Report")
                st.dataframe(
                    pd.DataFrame(
                        classification_report(
                            y_test, pred, output_dict=True, zero_division=0
                        )
                    ).T,
                    use_container_width=True
                )

                fig, ax = plt.subplots(figsize=(7, 5))
                sns.heatmap(
                    confusion_matrix(y_test, pred),
                    annot=True, fmt="d", cmap="Blues", ax=ax
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig, clear_figure=True)

                if hasattr(pipe, "predict_proba") and y_test.nunique() == 2:
                    try:
                        probs = pipe.predict_proba(X_test)[:, 1]
                        positive_class = pipe.named_steps["model"].classes_[1]
                        binary_y = (y_test == positive_class).astype(int)
                        auc = roc_auc_score(binary_y, probs)
                        fpr, tpr, _ = roc_curve(binary_y, probs)
                        fig, ax = plt.subplots(figsize=(7, 5))
                        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
                        ax.plot([0, 1], [0, 1], linestyle="--")
                        ax.legend()
                        ax.set_xlabel("False Positive Rate")
                        ax.set_ylabel("True Positive Rate")
                        ax.set_title("ROC Curve")
                        st.pyplot(fig, clear_figure=True)
                    except Exception as exc:
                        st.info(f"ROC unavailable: {exc}")

            else:
                pred_df = pd.DataFrame({"Actual": y_test.values, "Predicted": pred})
                st.subheader("Actual vs Predicted")
                st.dataframe(pred_df.head(100), use_container_width=True)

                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(y_test, pred, alpha=.65)
                lo = min(float(y_test.min()), float(np.min(pred)))
                hi = max(float(y_test.max()), float(np.max(pred)))
                ax.plot([lo, hi], [lo, hi], linestyle="--")
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted")
                st.pyplot(fig, clear_figure=True)

                residuals = y_test - pred
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(pred, residuals, alpha=.65)
                ax.axhline(0, linestyle="--")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Residual")
                ax.set_title("Residual Plot")
                st.pyplot(fig, clear_figure=True)

        except Exception as exc:
            st.error(f"Training could not be completed: {exc}")


# ============================================================
# MODEL COMPARISON
# ============================================================

with tabs[9]:
    st.header("🏆 Model Comparison")
    target = st.selectbox("Target", list(df.columns), key="comparison_target")
    task = st.radio("Task", ["Classification", "Regression"], horizontal=True, key="comparison_task")
    scaling = st.selectbox("Scaling", ["StandardScaler", "MinMaxScaler", "No Scaling"], key="comparison_scaling")
    test_size = st.slider("Test Size", 0.10, 0.40, 0.20, 0.05, key="comparison_test")

    if st.button("🏆 Compare All Models", type="primary"):
        rows = []
        for name in models_for(task):
            try:
                _, _, _, _, _, _, metrics = train_model(
                    df, target, task, name, test_size, scaling
                )
                rows.append({
                    "Model": name,
                    **{k: round(float(v), 4) for k, v in metrics.items()},
                    "Status": "Success"
                })
            except Exception as exc:
                rows.append({
                    "Model": name,
                    "Status": f"Failed: {exc}"
                })

        result = pd.DataFrame(rows)
        if task == "Classification" and "Accuracy" in result.columns:
            result = result.sort_values("Accuracy", ascending=False, na_position="last")
            metric = "Accuracy"
        else:
            if "R²" in result.columns:
                result = result.sort_values("R²", ascending=False, na_position="last")
            metric = "R²"

        st.dataframe(result, use_container_width=True)

        if metric in result.columns:
            good = result[result["Status"] == "Success"]
            if not good.empty:
                st.success(
                    f"Best model by {metric}: {good.iloc[0]['Model']} "
                    f"({good.iloc[0][metric]:.4f})"
                )


# ============================================================
# EVALUATION
# ============================================================

with tabs[10]:
    st.header("📊 Model Evaluation")
    pipe = st.session_state.trained_pipeline

    if pipe is None:
        st.info("Train a model first.")
    else:
        st.write(
            f"**Model:** {st.session_state.trained_model_name}  |  "
            f"**Task:** {st.session_state.trained_task}  |  "
            f"**Target:** {st.session_state.trained_target}"
        )
        st.dataframe(
            pd.DataFrame({
                "Metric": list(st.session_state.trained_metrics.keys()),
                "Score": [round(float(v), 5) for v in st.session_state.trained_metrics.values()]
            }),
            use_container_width=True
        )


# ============================================================
# PREDICTION
# ============================================================

with tabs[11]:
    st.header("🎯 New Record Prediction")
    pipe = st.session_state.trained_pipeline

    if pipe is None:
        st.info("Train a model first.")
    else:
        try:
            X, _ = prepare_ml(
                df,
                st.session_state.trained_target,
                st.session_state.trained_task
            )
            input_data = {}

            for col in X.columns:
                key = f"prediction_{col}"
                if pd.api.types.is_numeric_dtype(X[col]):
                    median = X[col].median()
                    default = float(median) if pd.notna(median) else 0.0
                    input_data[col] = st.number_input(
                        col, value=default, key=key
                    )
                else:
                    values = X[col].dropna().astype(str).unique().tolist()
                    input_data[col] = st.selectbox(
                        col,
                        values[:100] if values else [""],
                        key=key
                    )

            if st.button("🎯 Predict", type="primary"):
                row = pd.DataFrame([input_data])
                prediction = pipe.predict(row)[0]
                st.success(f"Prediction: **{prediction}**")

                if (
                    st.session_state.trained_task == "Classification"
                    and hasattr(pipe, "predict_proba")
                ):
                    try:
                        probabilities = pipe.predict_proba(row)[0]
                        classes = pipe.named_steps["model"].classes_
                        st.dataframe(
                            pd.DataFrame({
                                "Class": classes,
                                "Probability %": np.round(probabilities * 100, 2)
                            }).sort_values("Probability %", ascending=False),
                            use_container_width=True
                        )
                    except Exception:
                        pass

        except Exception as exc:
            st.error(f"Prediction setup error: {exc}")


# ============================================================
# EXPLAINABILITY
# ============================================================

with tabs[12]:
    st.header("🧠 Model Explainability")
    pipe = st.session_state.trained_pipeline

    if pipe is None:
        st.info("Train a model first.")
    else:
        importance = get_importance(pipe)
        if importance.empty:
            st.info("Feature importance is not available for this model.")
        else:
            st.dataframe(importance, use_container_width=True)
            fig, ax = plt.subplots(figsize=(10, 7))
            top = importance.head(15).sort_values("Importance")
            ax.barh(top["Feature"], top["Importance"])
            ax.set_xlabel("Importance")
            ax.set_title("Top Feature Importance")
            st.pyplot(fig, clear_figure=True)


# ============================================================
# DOWNLOAD
# ============================================================

with tabs[13]:
    st.header("💾 Download")
    st.dataframe(df.head(100), use_container_width=True, height=400)

    csv_bytes = df.to_csv(index=False).encode("utf-8")

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned Data")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Download CSV",
            csv_bytes,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c2:
        st.download_button(
            "⬇️ Download Excel",
            excel_buffer.getvalue(),
            file_name="cleaned_dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    if st.session_state.audit_log:
        st.download_button(
            "⬇️ Download Audit Log",
            pd.DataFrame(st.session_state.audit_log).to_csv(index=False).encode("utf-8"),
            file_name="cleaning_audit_log.csv",
            mime="text/csv"
        )


# ============================================================
# AUDIT LOG
# ============================================================

with tabs[14]:
    st.header("🧾 Cleaning Audit Log")
    if st.session_state.audit_log:
        st.dataframe(
            pd.DataFrame(st.session_state.audit_log),
            use_container_width=True,
            height=500
        )
    else:
        st.info("No cleaning actions recorded yet.")

st.divider()
st.caption(
    "AI & ML Analytics Dashboard • Streamlit • Pandas • SciPy • Seaborn • Scikit-learn"
)
