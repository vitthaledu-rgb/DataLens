
import os, io, json, math, tempfile, warnings, traceback
from datetime import datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder,
    OrdinalEncoder, PolynomialFeatures
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, cross_val_score,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False

APP_TITLE = "AI Analytics & Healthcare Dashboard"

STATE = {
    "original": pd.DataFrame(),
    "processed": pd.DataFrame(),
    "viewed": pd.DataFrame(),
    "history": [],
    "model": None,
    "pipeline": None,
    "target": None,
    "task": None,
    "feature_columns": [],
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "pred": None,
    "proba": None,
    "model_results": pd.DataFrame(),
    "comparison": pd.DataFrame(),
    "feature_importance": pd.DataFrame(),
    "last_stats": pd.DataFrame(),
    "last_quality": pd.DataFrame(),
    "last_missing": pd.DataFrame(),
    "last_duplicates": pd.DataFrame(),
    "last_outliers": pd.DataFrame(),
    "last_prediction": pd.DataFrame(),
    "last_cleaning": pd.DataFrame(),
}

CSS = r"""
* { font-family: "Times New Roman", Times, serif !important; }
body { background:#eef3f8 !important; color:#162033 !important; }
.gradio-container { max-width: 1500px !important; margin:auto !important; }
h1,h2,h3,h4 { color:#102a56 !important; }
.app-header { background:linear-gradient(135deg,#102a56,#2459a6); color:white;
  padding:22px 26px; border-radius:14px; margin-bottom:14px; }
.app-header h1 { color:white !important; margin:0; font-size:30px; }
.app-header p { margin:6px 0 0; color:#e6f0ff; font-size:16px; }
.card { background:white; border:1px solid #d6e0eb; border-radius:12px; padding:16px; }
.status { background:#f7fbff; border-left:4px solid #1479c9; padding:10px 14px; }
button.primary { background:#1557a6 !important; color:white !important; }
.tabs { border-bottom:1px solid #cbd7e5; }
footer { display:none !important; }
.dataframe { font-size:13px !important; }
"""

def msg(text, kind="info"):
    return f"**{kind.upper()}**  {text}"

def safe_df(df):
    if df is None:
        return pd.DataFrame()
    x = df.copy()
    if len(x.columns):
        cols, seen = [], {}
        for i, c in enumerate(x.columns):
            c = str(c).strip() or f"Column_{i+1}"
            n = seen.get(c, 0)
            seen[c] = n + 1
            cols.append(c if n == 0 else f"{c}_{n}")
        x.columns = cols
    return x

def read_file(file):
    if file is None:
        raise ValueError("Please upload a CSV, XLSX, or XLS file.")
    path = file if isinstance(file, str) else getattr(file, "name", None)
    if not path or not os.path.exists(path):
        raise ValueError("Uploaded file could not be accessed.")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".xlsx":
        df = pd.read_excel(path, engine="openpyxl")
    elif ext == ".xls":
        df = pd.read_excel(path, engine="xlrd")
    else:
        raise ValueError("Supported formats: CSV, XLSX, XLS.")
    return safe_df(df)

def fmt_df(df, n=1000):
    if df is None or df.empty:
        return pd.DataFrame({"Message":["No data available"]})
    return df.head(n).copy()

def load_dataset(file):
    try:
        df = read_file(file)
        STATE["original"] = df.copy(deep=True)
        STATE["processed"] = df.copy(deep=True)
        STATE["viewed"] = df.head(50).copy()
        STATE["history"] = []
        for k in ["model","pipeline","target","task","feature_columns","X_train","X_test",
                  "y_train","y_test","pred","proba"]:
            STATE[k] = None if k not in ["feature_columns"] else []
        STATE["model_results"] = pd.DataFrame()
        STATE["comparison"] = pd.DataFrame()
        STATE["feature_importance"] = pd.DataFrame()
        return (
            msg(f"Loaded {len(df):,} rows × {len(df.columns):,} columns."),
            fmt_df(df.head(20)),
            f"Rows: {len(df):,} | Columns: {len(df.columns):,}",
            gr.update(choices=list(df.columns), value=list(df.columns)[:1]),
            gr.update(choices=list(df.columns), value=list(df.columns)[:1]),
            gr.update(choices=list(df.columns), value=list(df.columns)[:1]),
            gr.update(choices=list(df.columns), value=df.columns[0] if len(df.columns) else None),
            gr.update(choices=list(df.columns), value=df.columns[0] if len(df.columns) else None),
        )
    except Exception as e:
        return (msg(str(e),"error"), pd.DataFrame(), "", *[gr.update(choices=[], value=None)]*5)

def reset_data():
    if STATE["original"].empty:
        return msg("No original dataset is loaded.","error"), pd.DataFrame(), "", gr.update(choices=[], value=None)
    STATE["processed"] = STATE["original"].copy(deep=True)
    STATE["viewed"] = STATE["original"].head(50).copy()
    STATE["history"] = []
    return msg("Processed data reset to the original dataset."), fmt_df(STATE["processed"].head(20)), f"Rows: {len(STATE['processed']):,} | Columns: {len(STATE['processed'].columns):,}", gr.update(choices=list(STATE["processed"].columns), value=STATE["processed"].columns[0] if len(STATE["processed"].columns) else None)

def dataset_for(source):
    return STATE["original"] if source == "Original" else STATE["processed"]

def view_data(source, view_mode, n, start, end, search):
    try:
        df = dataset_for(source)
        if df.empty: raise ValueError("Load a dataset first.")
        total = len(df)
        n = int(n)
        if view_mode == "Head": out = df.head(n)
        elif view_mode == "Tail": out = df.tail(n)
        elif view_mode == "Random": out = df.sample(min(n,total), random_state=42)
        elif view_mode == "Entire": out = df
        elif view_mode == "Custom Range":
            a, b = max(1,int(start)), min(total,int(end))
            if a > b: raise ValueError("Start row must be <= end row.")
            out = df.iloc[a-1:b]
        else: out = df.head(n)
        if search and str(search).strip():
            q = str(search).lower()
            mask = out.astype(str).apply(lambda c: c.str.lower().str.contains(q, na=False))
            out = out[mask.any(axis=1)]
        STATE["viewed"] = out.copy()
        first = 0 if out.empty else (out.index[0] + 1 if hasattr(out.index[0],"__int__") else 1)
        last = 0 if out.empty else first + len(out) - 1
        return fmt_df(out), f"Showing rows {first:,}–{last:,} of {total:,}"
    except Exception as e:
        return pd.DataFrame(), msg(str(e),"error")

def statistics(source, columns, percentiles):
    try:
        df = dataset_for(source)
        nums = df.select_dtypes(include=np.number)
        if columns:
            nums = nums[[c for c in columns if c in nums.columns]]
        if nums.empty: raise ValueError("No numeric columns available.")
        rows=[]
        ps = [float(x)/100 for x in percentiles]
        for c in nums.columns:
            s=nums[c].dropna()
            if s.empty: continue
            mode = s.mode()
            rows.append({
                "Feature":c,"Count":s.size,"Mean":s.mean(),"Median":s.median(),
                "Mode":mode.iloc[0] if len(mode) else np.nan,"Min":s.min(),"Q1":s.quantile(.25),
                "Q2":s.quantile(.50),"Q3":s.quantile(.75),"Max":s.max(),
                "Range":s.max()-s.min(),"Variance":s.var(),"Std":s.std(),
                "IQR":s.quantile(.75)-s.quantile(.25),"Skewness":s.skew(),
                "Kurtosis":s.kurtosis(),"MAD":np.mean(np.abs(s-s.mean()))
            })
            for p in ps:
                rows[-1][f"P{int(p*100)}"] = s.quantile(p)
        out=pd.DataFrame(rows); STATE["last_stats"]=out
        return out, msg("Statistics calculated.")
    except Exception as e: return pd.DataFrame(), msg(str(e),"error")

def quality_report():
    try:
        df=STATE["processed"]
        if df.empty: raise ValueError("Load a dataset first.")
        n=max(len(df),1)
        missing=df.isna().sum()
        duplicate_rows=int(df.duplicated().sum())
        numeric=df.select_dtypes(include=np.number).columns
        invalid={}
        healthcare={}
        for c in numeric:
            s=df[c]
            invalid[c]=int((~np.isfinite(s.dropna())).sum())
            name=c.lower()
            if "age" in name: healthcare[c]=int(((s<0)|(s>120)).sum())
            elif "blood pressure" in name or name in ["bp","bloodpressure","systolic","diastolic"]: healthcare[c]=int((s<=0).sum())
            elif "heart rate" in name or name in ["hr","heartrate","pulse"]: healthcare[c]=int((s<=0).sum())
            elif "cholesterol" in name: healthcare[c]=int((s<0).sum())
            elif "bmi" in name: healthcare[c]=int(((s<=0)|(s>100)).sum())
        rows=[]
        for c in df.columns:
            s=df[c]; nonnull=s.dropna()
            nun=s.nunique(dropna=True)
            flags=[]
            if s.isna().mean()>.5: flags.append("high missing")
            if nun<=1: flags.append("constant")
            if nun/max(len(s),1)>.9: flags.append("high cardinality")
            if pd.api.types.is_object_dtype(s):
                converted=pd.to_numeric(s,errors="coerce")
                if converted.notna().mean()>.8: flags.append("numeric-looking text")
            if c.lower() in [x.lower() for x in df.columns if x!=c]: flags.append("duplicate column name")
            if c in healthcare and healthcare[c]>0: flags.append("healthcare validity warning")
            rows.append({"Column":c,"Type":str(s.dtype),"Missing":int(s.isna().sum()),
                         "Missing %":round(s.isna().mean()*100,2),"Unique":nun,
                         "Duplicates":int(s.duplicated().sum()),"Invalid/Inf":invalid.get(c,0),
                         "Healthcare warnings":healthcare.get(c,0),"Flags":", ".join(flags)})
        out=pd.DataFrame(rows); STATE["last_quality"]=out
        completeness=100*(1-missing.sum()/(len(df)*max(len(df.columns),1)))
        consistency=100*(1-duplicate_rows/n)
        validity=100*(1-sum(invalid.values())/max(len(df)*max(len(numeric),1),1))
        uniqueness=100*(1-duplicate_rows/n)
        score=max(0,min(100,round(.4*completeness+.25*consistency+.2*validity+.15*uniqueness,1)))
        summary=pd.DataFrame({"Metric":["Quality Score","Completeness","Consistency","Validity","Uniqueness","Duplicate Rows"],
                              "Value":[score,round(completeness,1),round(consistency,1),round(validity,1),round(uniqueness,1),duplicate_rows]})
        return summary,out,msg("Quality report generated. Score is a heuristic.")
    except Exception as e: return pd.DataFrame(),pd.DataFrame(),msg(str(e),"error")

def missing_report():
    df=STATE["processed"]
    out=pd.DataFrame({"Column":df.columns,"Missing":df.isna().sum().values,
                      "Missing %":(df.isna().mean()*100).round(2).values})
    STATE["last_missing"]=out
    return out

def duplicate_report():
    df=STATE["processed"]
    dup=df[df.duplicated(keep=False)].copy()
    STATE["last_duplicates"]=dup
    return fmt_df(dup)

def outlier_report(columns=None):
    df=STATE["processed"]
    nums=df.select_dtypes(include=np.number)
    if columns: nums=nums[[c for c in columns if c in nums.columns]]
    rows=[]
    for c in nums:
        s=nums[c].dropna()
        if len(s)<4: continue
        q1,q3=s.quantile(.25),s.quantile(.75); i=q3-q1
        rows.append({"Column":c,"IQR Outliers":int(((s<q1-1.5*i)|(s>q3+1.5*i)).sum()),
                     "Lower":q1-1.5*i,"Upper":q3+1.5*i})
    out=pd.DataFrame(rows); STATE["last_outliers"]=out; return out

def clean_preview(operation, columns, value, threshold, method, outlier_action, preview_n):
    try:
        df=STATE["processed"].copy()
        before=df.copy()
        cols=columns or list(df.columns)
        if operation=="Drop Rows With Missing":
            result=df.dropna(subset=cols)
        elif operation=="Drop Selected Columns":
            result=df.drop(columns=[c for c in cols if c in df])
        elif operation=="Drop Columns Above Missing %":
            result=df.drop(columns=df.columns[df.isna().mean()*100 > float(threshold)])
        elif operation in ["Mean Impute","Median Impute","Mode Impute","Constant Impute","Forward Fill","Backward Fill","Interpolation"]:
            result=df.copy()
            for c in cols:
                if c not in result: continue
                if operation=="Mean Impute" and pd.api.types.is_numeric_dtype(result[c]): result[c]=result[c].fillna(result[c].mean())
                elif operation=="Median Impute" and pd.api.types.is_numeric_dtype(result[c]): result[c]=result[c].fillna(result[c].median())
                elif operation=="Mode Impute":
                    m=result[c].mode()
                    if len(m): result[c]=result[c].fillna(m.iloc[0])
                elif operation=="Constant Impute": result[c]=result[c].fillna(value if value!="" else "Unknown")
                elif operation=="Forward Fill": result[c]=result[c].ffill()
                elif operation=="Backward Fill": result[c]=result[c].bfill()
                elif operation=="Interpolation" and pd.api.types.is_numeric_dtype(result[c]): result[c]=result[c].interpolate()
        elif operation=="Remove Duplicates": result=df.drop_duplicates()
        elif operation in ["Strip","Lowercase","Uppercase","Titlecase"]:
            result=df.copy()
            for c in cols:
                if c in result and pd.api.types.is_string_dtype(result[c]):
                    f={"Strip":str.strip,"Lowercase":str.lower,"Uppercase":str.upper,"Titlecase":str.title}[operation]
                    result[c]=result[c].astype("string").map(f)
        elif operation=="Replace Values":
            result=df.replace(str(value), str(threshold))
        elif operation=="Numeric Conversion":
            result=df.copy()
            for c in cols: result[c]=pd.to_numeric(result[c],errors="coerce")
        elif operation=="Date Conversion":
            result=df.copy()
            for c in cols: result[c]=pd.to_datetime(result[c],errors="coerce")
        elif operation=="Remove Inf":
            result=df.replace([np.inf,-np.inf],np.nan)
        elif operation=="Common Null Strings":
            result=df.replace(["","NA","N/A","na","n/a","null","NULL","None","none","-"],np.nan)
        elif operation in ["IQR Remove","IQR Cap","Z-score Remove","Z-score Cap"]:
            result=df.copy()
            for c in cols:
                if c not in result or not pd.api.types.is_numeric_dtype(result[c]): continue
                s=result[c]
                if operation.startswith("IQR"):
                    q1,q3=s.quantile(.25),s.quantile(.75); lo=q1-1.5*(q3-q1); hi=q3+1.5*(q3-q1)
                else:
                    mu,sd=s.mean(),s.std(); lo=mu-3*sd; hi=mu+3*sd
                mask=(s<lo)|(s>hi)
                if operation.endswith("Remove"): result=result.loc[~mask].copy()
                else: result[c]=s.clip(lo,hi)
        else: result=df.copy()
        changes={"Rows before":len(before),"Rows after":len(result),
                 "Columns before":len(before.columns),"Columns after":len(result.columns),
                 "Cells changed":int((before.reindex(index=result.index,columns=result.columns).fillna("__NA__").astype(str)!=result.fillna("__NA__").astype(str)).sum().sum())}
        return fmt_df(result, int(preview_n)), pd.DataFrame([changes]), json.dumps(changes,indent=2)
    except Exception as e: return pd.DataFrame(),pd.DataFrame(),msg(str(e),"error")

def apply_clean(operation, columns, value, threshold, method, outlier_action):
    try:
        preview, summary, _ = clean_preview(operation,columns,value,threshold,method,outlier_action,1000000)
        if preview.empty and operation not in ["Drop Rows With Missing","Drop Selected Columns","Drop Columns Above Missing %","Remove Duplicates"]:
            raise ValueError("Operation produced no data.")
        old=STATE["processed"].copy()
        # Reuse preview logic by assigning the full result through a small internal duplication.
        # For deterministic behavior, repeat operation with a large preview and take that table.
        new=preview.copy()
        STATE["processed"]=new
        STATE["history"].append({"time":datetime.now().isoformat(timespec="seconds"),"operation":operation,
                                 "rows_before":len(old),"rows_after":len(new),
                                 "columns_before":len(old.columns),"columns_after":len(new.columns)})
        return fmt_df(new.head(20)), summary, pd.DataFrame(STATE["history"])
    except Exception as e: return pd.DataFrame(),pd.DataFrame(),msg(str(e),"error")

def plot_eda(plot_type, x, y, hue, bins):
    try:
        df=STATE["processed"]
        if df.empty: raise ValueError("Load data first.")
        fig=plt.figure(figsize=(9,5))
        if plot_type=="Histogram": sns.histplot(data=df,x=x,bins=int(bins),hue=hue or None)
        elif plot_type=="KDE": sns.kdeplot(data=df,x=x,hue=hue or None,fill=True)
        elif plot_type=="Box": sns.boxplot(data=df,x=x,y=y)
        elif plot_type=="Violin": sns.violinplot(data=df,x=x,y=y)
        elif plot_type=="Count": sns.countplot(data=df,x=x,hue=hue or None)
        elif plot_type=="Bar": df.groupby(x)[y].mean().plot(kind="bar")
        elif plot_type=="Pie": df[x].value_counts().head(15).plot.pie(autopct="%1.1f%%")
        elif plot_type=="Scatter": sns.scatterplot(data=df,x=x,y=y,hue=hue or None)
        elif plot_type=="Line": sns.lineplot(data=df,x=x,y=y,hue=hue or None)
        elif plot_type=="Area": df.plot.area(x=x,y=y)
        elif plot_type=="Pearson Heatmap": sns.heatmap(df.select_dtypes("number").corr(),annot=False,cmap="Blues")
        elif plot_type=="Covariance Heatmap": sns.heatmap(df.select_dtypes("number").cov(),annot=False,cmap="Blues")
        elif plot_type=="Pair Plot":
            p=sns.pairplot(df.select_dtypes("number").dropna().head(500)); return p.fig
        elif plot_type=="QQ":
            stats.probplot(pd.to_numeric(df[x],errors="coerce").dropna(),dist="norm",plot=plt)
        elif plot_type=="ECDF":
            s=pd.to_numeric(df[x],errors="coerce").dropna().sort_values()
            plt.plot(s,np.arange(1,len(s)+1)/len(s))
        elif plot_type=="Missing Visualization":
            sns.heatmap(df.isna(),cbar=False)
        elif plot_type=="3D Scatter":
            from mpl_toolkits.mplot3d import Axes3D
            ax=fig.add_subplot(111,projection="3d")
            ax.scatter(df[x],df[y],df[hue])
            ax.set_xlabel(x); ax.set_ylabel(y); ax.set_zlabel(hue)
        elif plot_type=="Categorical Association":
            ct=pd.crosstab(df[x],df[y],normalize="index"); sns.heatmap(ct,annot=True,fmt=".2f",cmap="Blues")
        plt.tight_layout()
        return fig
    except Exception as e:
        fig=plt.figure(figsize=(8,2)); plt.text(.02,.5,str(e)); plt.axis("off"); return fig

def target_analysis(target):
    try:
        df=STATE["processed"].copy()
        if target not in df: raise ValueError("Select a valid target.")
        y=df[target].dropna()
        if pd.api.types.is_numeric_dtype(y) and y.nunique()>10:
            task="Regression"
            out=pd.DataFrame({"Metric":["Task","Count","Mean","Median","Std","Skewness","Kurtosis","Min","Max"],
                              "Value":[task,len(y),y.mean(),y.median(),y.std(),y.skew(),y.kurtosis(),y.min(),y.max()]})
        else:
            task="Classification"
            counts=y.value_counts()
            out=pd.DataFrame({"Class":counts.index.astype(str),"Count":counts.values,
                              "Percent":(counts.values/len(y)*100).round(2)})
            out["Imbalance Ratio"]=round(counts.max()/max(counts.min(),1),2)
        STATE["target"]=target; STATE["task"]=task
        return out, task, msg(f"Detected {task} target: {target}")
    except Exception as e: return pd.DataFrame(), "", msg(str(e),"error")

def feature_engineer(operation, columns, k):
    try:
        df=STATE["processed"].copy(); cols=columns or list(df.select_dtypes("number").columns)
        if operation=="Log": 
            for c in cols: df[c+"_log"]=np.log1p(np.clip(pd.to_numeric(df[c],errors="coerce"),0,None))
        elif operation=="Square":
            for c in cols: df[c+"_sq"]=pd.to_numeric(df[c],errors="coerce")**2
        elif operation=="Sqrt":
            for c in cols: df[c+"_sqrt"]=np.sqrt(np.clip(pd.to_numeric(df[c],errors="coerce"),0,None))
        elif operation=="StandardScaler":
            for c in cols: df[c]=StandardScaler().fit_transform(df[[c]].fillna(df[c].median()))
        elif operation=="MinMaxScaler":
            for c in cols: df[c]=MinMaxScaler().fit_transform(df[[c]].fillna(df[c].median()))
        elif operation=="RobustScaler":
            for c in cols: df[c]=RobustScaler().fit_transform(df[[c]].fillna(df[c].median()))
        elif operation=="One Hot Encode":
            df=pd.get_dummies(df,columns=[c for c in cols if c in df],dummy_na=True)
        elif operation=="Frequency Encode":
            for c in cols:
                freq=df[c].value_counts(normalize=True); df[c+"_freq"]=df[c].map(freq)
        elif operation=="Variance Filter":
            nums=df.select_dtypes("number"); keep=nums.columns[nums.var()>float(k or 0)]
            df=df.drop(columns=[c for c in nums.columns if c not in keep])
        elif operation=="Correlation Filter":
            nums=df.select_dtypes("number").corr().abs(); upper=nums.where(np.triu(np.ones(nums.shape),1).astype(bool))
            drop=[c for c in upper.columns if any(upper[c]>.9)]
            df=df.drop(columns=drop)
        elif operation=="PCA":
            nums=df.select_dtypes("number").columns
            if len(nums)<2: raise ValueError("PCA needs at least two numeric columns.")
            X=df[nums].fillna(df[nums].median())
            z=StandardScaler().fit_transform(X); pca=PCA(n_components=min(int(k or 2),len(nums))).fit_transform(z)
            df=df.drop(columns=list(nums))
            for i in range(pca.shape[1]): df[f"PC{i+1}"]=pca[:,i]
        else: raise ValueError("Choose a feature engineering operation.")
        STATE["processed"]=safe_df(df)
        return fmt_df(df.head(20)), msg(f"Feature engineering applied: {operation}")
    except Exception as e: return pd.DataFrame(),msg(str(e),"error")

def build_preprocessor(X):
    nums=X.select_dtypes(include=np.number).columns.tolist()
    cats=[c for c in X.columns if c not in nums]
    transformers=[]
    if nums: transformers.append(("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums))
    if cats: transformers.append(("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cats))
    return ColumnTransformer(transformers=transformers)

def model_object(name, task):
    if task=="Classification":
        models={
        "Logistic Regression":LogisticRegression(max_iter=2000),
        "KNN":KNeighborsClassifier(),
        "Naive Bayes":GaussianNB(),
        "Decision Tree":DecisionTreeClassifier(random_state=42),
        "Random Forest":RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1),
        "SVM":SVC(probability=True,random_state=42),
        "Gradient Boosting":GradientBoostingClassifier(random_state=42)}
        if XGB_OK: models["XGBoost"]=XGBClassifier(n_estimators=200,max_depth=4,learning_rate=.08,eval_metric="logloss",random_state=42)
    else:
        models={
        "Linear Regression":LinearRegression(),"Ridge":Ridge(),"Lasso":Lasso(),
        "Decision Tree":DecisionTreeRegressor(random_state=42),
        "Random Forest":RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1),
        "SVR":SVR(),"KNN":KNeighborsRegressor(),"Gradient Boosting":GradientBoostingRegressor(random_state=42)}
        if XGB_OK: models["XGBoost"]=XGBRegressor(n_estimators=200,max_depth=4,learning_rate=.08,objective="reg:squarederror",random_state=42)
    if name not in models: raise ValueError("Unsupported model.")
    return models[name]

def train_model(target, model_name, test_size, folds):
    try:
        df=STATE["processed"].copy()
        if target not in df: raise ValueError("Select a valid target.")
        df=df.dropna(subset=[target])
        y=df[target]
        task="Regression" if pd.api.types.is_numeric_dtype(y) and y.nunique()>10 else "Classification"
        X=df.drop(columns=[target])
        if X.empty: raise ValueError("No features remain after selecting the target.")
        if task=="Classification" and y.nunique()<2: raise ValueError("Classification requires at least two classes.")
        model=model_object(model_name,task)
        prep=build_preprocessor(X)
        pipe=Pipeline([("preprocessor",prep),("model",model)])
        strat=y if task=="Classification" else None
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=float(test_size),random_state=42,stratify=strat)
        pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
        proba=pipe.predict_proba(Xte)[:,1] if task=="Classification" and hasattr(pipe,"predict_proba") and len(np.unique(yte))==2 else None
        if task=="Classification":
            res={"Model":model_name,"Task":task,"Accuracy":accuracy_score(yte,pred),
                 "Precision":precision_score(yte,pred,average="weighted",zero_division=0),
                 "Recall":recall_score(yte,pred,average="weighted",zero_division=0),
                 "F1":f1_score(yte,pred,average="weighted",zero_division=0)}
            if proba is not None: res["ROC-AUC"]=roc_auc_score(yte,proba)
        else:
            mse=mean_squared_error(yte,pred)
            res={"Model":model_name,"Task":task,"MAE":mean_absolute_error(yte,pred),
                 "MSE":mse,"RMSE":math.sqrt(mse),"R2":r2_score(yte,pred)}
        cv=StratifiedKFold(n_splits=int(folds),shuffle=True,random_state=42) if task=="Classification" else KFold(n_splits=int(folds),shuffle=True,random_state=42)
        scoring="f1_weighted" if task=="Classification" else "neg_root_mean_squared_error"
        cvscore=cross_val_score(pipe,X,y,cv=cv,scoring=scoring,n_jobs=-1)
        res["CV Mean"]=cvscore.mean()
        STATE.update({"model":model,"pipeline":pipe,"target":target,"task":task,"feature_columns":list(X.columns),
                      "X_train":Xtr,"X_test":Xte,"y_train":ytr,"y_test":yte,"pred":pred,"proba":proba})
        one=pd.DataFrame([res]); STATE["model_results"]=one
        return one,msg(f"{model_name} trained successfully. Task: {task}")
    except Exception as e: return pd.DataFrame(),msg(str(e),"error")

def compare_models(target, models, test_size, folds):
    try:
        names=models or []
        if not names: raise ValueError("Select at least one model.")
        rows=[]
        for name in names:
            r,_=train_model(target,name,test_size,folds)
            if not r.empty: rows.append(r.iloc[0].to_dict())
        out=pd.DataFrame(rows); STATE["comparison"]=out
        return out,msg(f"Compared {len(out)} models.")
    except Exception as e: return pd.DataFrame(),msg(str(e),"error")

def evaluate_model():
    try:
        if STATE["pipeline"] is None: raise ValueError("Train a model first.")
        task=STATE["task"]; y=STATE["y_test"]; p=STATE["pred"]
        fig=plt.figure(figsize=(9,5))
        if task=="Classification":
            cm=confusion_matrix(y,p); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
        else:
            plt.scatter(y,p,alpha=.7); lo,hi=min(y.min(),p.min()),max(y.max(),p.max()); plt.plot([lo,hi],[lo,hi]); plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Actual vs Predicted")
        plt.tight_layout()
        return fig, pd.DataFrame(classification_report(y,p,output_dict=True)).T if task=="Classification" else pd.DataFrame([{"MAE":mean_absolute_error(y,p),"RMSE":math.sqrt(mean_squared_error(y,p)),"R2":r2_score(y,p)}])
    except Exception as e:
        fig=plt.figure(figsize=(8,2)); plt.text(.02,.5,str(e)); plt.axis("off"); return fig,pd.DataFrame()

def tune_model(target, model_name, search_type, folds):
    try:
        df=STATE["processed"].dropna(subset=[target]).copy()
        y=df[target]; X=df.drop(columns=[target])
        task="Regression" if pd.api.types.is_numeric_dtype(y) and y.nunique()>10 else "Classification"
        pipe=Pipeline([("preprocessor",build_preprocessor(X)),("model",model_object(model_name,task))])
        grids={
          "Random Forest":{"model__n_estimators":[100,200],"model__max_depth":[None,5,10]},
          "Decision Tree":{"model__max_depth":[None,3,5,10],"model__min_samples_split":[2,5,10]},
          "Logistic Regression":{"model__C":[.1,1,10]},
          "Ridge":{"model__alpha":[.1,1,10]},
          "Lasso":{"model__alpha":[.001,.01,.1]},
          "Gradient Boosting":{"model__n_estimators":[100,200],"model__learning_rate":[.03,.1],"model__max_depth":[2,3]}
        }
        params=grids.get(model_name)
        if params is None: raise ValueError("No safe tuning grid defined for this model.")
        scoring="f1_weighted" if task=="Classification" else "neg_root_mean_squared_error"
        if search_type=="GridSearchCV": search=GridSearchCV(pipe,params,cv=int(folds),scoring=scoring,n_jobs=-1)
        else: search=RandomizedSearchCV(pipe,params,n_iter=min(8,math.prod([len(v) for v in params.values()])),cv=int(folds),scoring=scoring,random_state=42,n_jobs=-1)
        search.fit(X,y); STATE["pipeline"]=search.best_estimator_; STATE["model"]=search.best_estimator_.named_steps["model"]
        return pd.DataFrame([{"Best Score":search.best_score_,"Best Params":str(search.best_params_)}]),msg("Hyperparameter tuning completed.")
    except Exception as e: return pd.DataFrame(),msg(str(e),"error")

def predict_single(target, values_json):
    try:
        pipe=STATE["pipeline"]
        if pipe is None: raise ValueError("Train a model first.")
        features=STATE["feature_columns"]
        vals=json.loads(values_json) if isinstance(values_json,str) else values_json
        row={c:vals.get(c,np.nan) for c in features}
        X=pd.DataFrame([row])
        pred=pipe.predict(X)[0]
        out=pd.DataFrame({"Feature":["Prediction"],"Value":[str(pred)]})
        if STATE["task"]=="Classification" and hasattr(pipe,"predict_proba"):
            try: out.loc[len(out)]=["Probability",str(float(np.max(pipe.predict_proba(X)[0])))]
            except Exception: pass
        STATE["last_prediction"]=out
        return out,msg("Prediction generated.")
    except Exception as e: return pd.DataFrame(columns=["Feature","Value"]),msg(str(e),"error")

def explain_model():
    try:
        pipe=STATE["pipeline"]
        if pipe is None: raise ValueError("Train a model first.")
        model=pipe.named_steps["model"]
        pre=pipe.named_steps["preprocessor"]
        names=list(pre.get_feature_names_out())
        vals=None
        if hasattr(model,"feature_importances_"): vals=model.feature_importances_
        elif hasattr(model,"coef_"): vals=np.abs(np.ravel(model.coef_)) if np.ravel(model.coef_).size==len(names) else None
        if vals is None:
            X=STATE["X_test"]; y=STATE["y_test"]
            pi=permutation_importance(pipe,X,y,n_repeats=5,random_state=42,n_jobs=-1)
            out=pd.DataFrame({"Feature":X.columns,"Importance":pi.importances_mean}).sort_values("Importance",ascending=False)
        else:
            out=pd.DataFrame({"Feature":names,"Importance":vals}).sort_values("Importance",ascending=False)
        STATE["feature_importance"]=out
        return out.head(100),msg("Model explainability generated.")
    except Exception as e: return pd.DataFrame(),msg(str(e),"error")

def export_data(kind, fmt):
    try:
        mapping={
          "Original":STATE["original"],"Processed":STATE["processed"],"Viewed":STATE["viewed"],
          "Statistics":STATE["last_stats"],"Missing":STATE["last_missing"],
          "Duplicate":STATE["last_duplicates"],"Outlier":STATE["last_outliers"],
          "Quality":STATE["last_quality"],"Model Metrics":STATE["model_results"],
          "Model Comparison":STATE["comparison"],"Prediction":STATE["last_prediction"],
          "Feature Importance":STATE["feature_importance"],
          "Cleaning Change Log":pd.DataFrame(STATE["history"])
        }
        if kind=="Complete Analysis Summary":
            mapping[kind]=pd.DataFrame({"Section":["Rows","Columns","Target","Task","Models"],
              "Value":[len(STATE["processed"]),len(STATE["processed"].columns),STATE["target"],STATE["task"],
                       ", ".join(STATE["comparison"]["Model"].astype(str)) if not STATE["comparison"].empty else ""]})
        df=mapping.get(kind,pd.DataFrame())
        if df is None: df=pd.DataFrame()
        suffix=".csv" if fmt=="CSV" else ".xlsx"
        path=os.path.join(tempfile.gettempdir(),f"analytics_export_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{suffix}")
        if fmt=="CSV": df.to_csv(path,index=False)
        else: df.to_excel(path,index=False,engine="openpyxl")
        return path,msg(f"Export created: {os.path.basename(path)}")
    except Exception as e: return None,msg(str(e),"error")

def model_choices():
    return ["Logistic Regression","KNN","Naive Bayes","Decision Tree","Random Forest","SVM","Gradient Boosting"] + (["XGBoost"] if XGB_OK else [])

def refresh_choices():
    cols=list(STATE["processed"].columns)
    return [gr.update(choices=cols,value=cols[:1]) for _ in range(12)]

with gr.Blocks(css=CSS, title=APP_TITLE, theme=gr.themes.Soft()) as app:
    gr.HTML('<div class="app-header"><h1>AI Analytics & Healthcare Dashboard</h1><p>Data loading, quality, cleaning, EDA, statistics, feature engineering, machine learning, evaluation, tuning, prediction, explainability and export.</p></div>')

    with gr.Row():
        with gr.Column(scale=3):
            file_input=gr.File(label="Upload Dataset",file_types=[".csv",".xlsx",".xls"],type="filepath")
        with gr.Column(scale=1):
            load_btn=gr.Button("Load Dataset",variant="primary")
            reset_btn=gr.Button("Reset to Original")
    status=gr.Markdown("Load a dataset to begin.",elem_classes=["status"])
    data_info=gr.Markdown("Rows: 0 | Columns: 0")
    preview=gr.Dataframe(label="Dataset Preview",interactive=False)

    with gr.Tabs():
        with gr.Tab("Overview"):
            gr.Markdown("### Dataset Overview")
            overview=gr.Markdown("Upload a dataset to populate the dashboard.")
            qbtn=gr.Button("Generate Quality Overview")
            qsummary=gr.Dataframe(interactive=False)
            qdetail=gr.Dataframe(interactive=False)

        with gr.Tab("Viewer"):
            source=gr.Dropdown(["Original","Processed"],value="Processed",label="Dataset")
            view_mode=gr.Dropdown(["Head","Tail","Random","Entire","Custom Range"],value="Head",label="View")
            n=gr.Dropdown([10,20,50,100,147,500],value=50,label="Rows")
            with gr.Row():
                start=gr.Number(value=1,label="Start Row")
                end=gr.Number(value=50,label="End Row")
            search=gr.Textbox(label="Search all columns")
            view_btn=gr.Button("View Data",variant="primary")
            view_table=gr.Dataframe(interactive=False)
            view_message=gr.Markdown()

        with gr.Tab("Statistics"):
            stat_source=gr.Dropdown(["Original","Processed"],value="Processed",label="Dataset")
            stat_cols=gr.Dropdown([],multiselect=True,label="Numeric Columns")
            percentile=gr.CheckboxGroup(["1","5","10","25","50","75","90","95","99"],value=["25","50","75"],label="Percentiles")
            stat_btn=gr.Button("Calculate Statistics",variant="primary")
            stat_table=gr.Dataframe(interactive=False)

        with gr.Tab("Quality"):
            quality_btn=gr.Button("Run Data Quality Audit",variant="primary")
            quality_summary=gr.Dataframe(interactive=False)
            quality_detail=gr.Dataframe(interactive=False)
            missing_btn=gr.Button("Missing Values")
            missing_table=gr.Dataframe(interactive=False)
            duplicate_btn=gr.Button("Duplicate Rows")
            duplicate_table=gr.Dataframe(interactive=False)
            outlier_btn=gr.Button("Outlier Report")
            outlier_table=gr.Dataframe(interactive=False)

        with gr.Tab("Cleaning"):
            clean_op=gr.Dropdown(["Drop Rows With Missing","Drop Selected Columns","Drop Columns Above Missing %",
              "Mean Impute","Median Impute","Mode Impute","Constant Impute","Forward Fill","Backward Fill","Interpolation",
              "Remove Duplicates","Strip","Lowercase","Uppercase","Titlecase","Replace Values","Numeric Conversion","Date Conversion",
              "Remove Inf","Common Null Strings","IQR Remove","IQR Cap","Z-score Remove","Z-score Cap"],label="Cleaning Operation")
            clean_cols=gr.Dropdown([],multiselect=True,label="Columns")
            with gr.Row():
                clean_value=gr.Textbox(label="Value")
                clean_threshold=gr.Number(value=50,label="Threshold / Replacement")
            clean_method=gr.Dropdown(["Mean","Median","Most Frequent"],value="Median",label="Method")
            outlier_action=gr.Dropdown(["Remove","Cap"],value="Remove",label="Outlier Action")
            with gr.Row():
                clean_preview_btn=gr.Button("Preview Changes")
                clean_apply_btn=gr.Button("Apply Cleaning",variant="primary")
            clean_table=gr.Dataframe(interactive=False,label="Preview / Result")
            clean_summary=gr.Dataframe(interactive=False,label="Before / After")
            history_table=gr.Dataframe(interactive=False,label="Cleaning History")

        with gr.Tab("EDA"):
            plot_type=gr.Dropdown(["Histogram","KDE","Box","Violin","Count","Bar","Pie","Scatter","Line","Area",
                "Pearson Heatmap","Covariance Heatmap","Pair Plot","QQ","ECDF","Missing Visualization","3D Scatter","Categorical Association"],
                value="Histogram",label="Plot")
            with gr.Row():
                eda_x=gr.Dropdown([],label="X / Feature")
                eda_y=gr.Dropdown([],label="Y / Value")
                eda_hue=gr.Dropdown([],label="Hue / Third Variable",allow_custom_value=True)
                bins=gr.Number(value=30,label="Bins")
            eda_btn=gr.Button("Generate Plot",variant="primary")
            eda_plot=gr.Plot()

        with gr.Tab("Target"):
            target=gr.Dropdown([],label="Target")
            target_btn=gr.Button("Analyze Target",variant="primary")
            target_task=gr.Textbox(label="Detected Task",interactive=False)
            target_table=gr.Dataframe(interactive=False)

        with gr.Tab("Features"):
            feat_op=gr.Dropdown(["Log","Square","Sqrt","StandardScaler","MinMaxScaler","RobustScaler","One Hot Encode","Frequency Encode","Variance Filter","Correlation Filter","PCA"],label="Feature Engineering")
            feat_cols=gr.Dropdown([],multiselect=True,label="Columns")
            feat_k=gr.Number(value=2,label="K / Components / Threshold")
            feat_btn=gr.Button("Apply Feature Engineering",variant="primary")
            feat_table=gr.Dataframe(interactive=False)

        with gr.Tab("Training"):
            train_target=gr.Dropdown([],label="Target")
            train_model_name=gr.Dropdown(model_choices(),value="Random Forest",label="Model")
            with gr.Row():
                test_size=gr.Slider(.1,.5,.2,step=.05,label="Test Size")
                folds=gr.Dropdown([3,5,10],value=5,label="CV Folds")
            train_btn=gr.Button("Train Model",variant="primary")
            train_result=gr.Dataframe(interactive=False)

        with gr.Tab("Evaluation"):
            eval_btn=gr.Button("Evaluate Trained Model",variant="primary")
            eval_plot=gr.Plot()
            eval_table=gr.Dataframe(interactive=False)

        with gr.Tab("Comparison"):
            cmp_target=gr.Dropdown([],label="Target")
            cmp_models=gr.CheckboxGroup(model_choices(),value=["Logistic Regression","Random Forest","Gradient Boosting"],label="Models")
            cmp_btn=gr.Button("Compare Models",variant="primary")
            cmp_table=gr.Dataframe(interactive=False)

        with gr.Tab("Tuning"):
            tune_target=gr.Dropdown([],label="Target")
            tune_model=gr.Dropdown(model_choices(),value="Random Forest",label="Model")
            tune_search=gr.Dropdown(["GridSearchCV","RandomizedSearchCV"],value="GridSearchCV",label="Search")
            tune_folds=gr.Dropdown([3,5,10],value=5,label="CV Folds")
            tune_btn=gr.Button("Tune Model",variant="primary")
            tune_table=gr.Dataframe(interactive=False)

        with gr.Tab("Prediction"):
            pred_target=gr.Dropdown([],label="Target")
            gr.Markdown("Enter feature values as JSON. Example: {\"Age\":45,\"BMI\":27.2,\"Gender\":\"Male\"}")
            pred_json=gr.Textbox(value="{}",label="Feature Values JSON",lines=5)
            pred_btn=gr.Button("Predict",variant="primary")
            pred_table=gr.Dataframe(headers=["Feature","Value"],datatype=["str","str"],interactive=False)

        with gr.Tab("Explainability"):
            explain_btn=gr.Button("Generate Feature Importance",variant="primary")
            explain_table=gr.Dataframe(interactive=False)

        with gr.Tab("Export"):
            export_kind=gr.Dropdown(["Original","Processed","Viewed","Statistics","Missing","Duplicate","Outlier","Quality",
                "Model Metrics","Model Comparison","Prediction","Feature Importance","Cleaning Change Log","Complete Analysis Summary"],
                value="Processed",label="Export Dataset / Result")
            export_fmt=gr.Dropdown(["CSV","Excel"],value="CSV",label="Format")
            export_btn=gr.Button("Generate Download",variant="primary")
            export_file=gr.File(label="Download")
            export_status=gr.Markdown()

    load_btn.click(load_dataset,file_input,[status,preview,data_info,stat_cols,clean_cols,feat_cols,target,train_target])
    reset_btn.click(reset_data,None,[status,preview,data_info,clean_cols])
    view_btn.click(view_data,[source,view_mode,n,start,end,search],[view_table,view_message])
    stat_btn.click(statistics,[stat_source,stat_cols,percentile],[stat_table,status])
    quality_btn.click(quality_report,None,[quality_summary,quality_detail,status])
    missing_btn.click(missing_report,None,missing_table)
    duplicate_btn.click(duplicate_report,None,duplicate_table)
    outlier_btn.click(outlier_report,[clean_cols],outlier_table)
    clean_preview_btn.click(clean_preview,[clean_op,clean_cols,clean_value,clean_threshold,clean_method,outlier_action, n],[clean_table,clean_summary,status])
    clean_apply_btn.click(apply_clean,[clean_op,clean_cols,clean_value,clean_threshold,clean_method,outlier_action],[clean_table,clean_summary,history_table])
    eda_btn.click(plot_eda,[plot_type,eda_x,eda_y,eda_hue,bins],eda_plot)
    target_btn.click(target_analysis,target,[target_table,target_task,status])
    feat_btn.click(feature_engineer,[feat_op,feat_cols,feat_k],[feat_table,status])
    train_btn.click(train_model,[train_target,train_model_name,test_size,folds],[train_result,status])
    eval_btn.click(evaluate_model,None,[eval_plot,eval_table])
    cmp_btn.click(compare_models,[cmp_target,cmp_models,test_size,folds],[cmp_table,status])
    tune_btn.click(tune_model,[tune_target,tune_model,tune_search,tune_folds],[tune_table,status])
    pred_btn.click(predict_single,[pred_target,pred_json],[pred_table,status])
    explain_btn.click(explain_model,None,[explain_table,status])
    export_btn.click(export_data,[export_kind,export_fmt],[export_file,export_status])

    # Synchronize column selectors after loading and after cleaning/feature engineering.
    def refresh_all():
        cols=list(STATE["processed"].columns)
        val=cols[0] if cols else None
        return [gr.update(choices=cols,value=([val] if val else [])) for _ in range(7)] + [
            gr.update(choices=cols,value=val) for _ in range(4)
        ]
    load_btn.click(lambda: refresh_all(),
                   None,[eda_x,eda_y,feat_cols,clean_cols,stat_cols,target,train_target,cmp_target,tune_target,pred_target,eda_hue])
    clean_apply_btn.click(lambda: refresh_all(),
                   None,[eda_x,eda_y,feat_cols,clean_cols,stat_cols,target,train_target,cmp_target,tune_target,pred_target,eda_hue])
    feat_btn.click(lambda: refresh_all(),
                   None,[eda_x,eda_y,feat_cols,clean_cols,stat_cols,target,train_target,cmp_target,tune_target,pred_target,eda_hue])

if __name__ == "__main__":
    app.launch()
