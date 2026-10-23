import io
import os
import tempfile
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI & ML Analytics Studio", page_icon="", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Times New Roman', Times, serif !important; }
.stApp { background: #f3f7fb; color: #172033; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1600px; }
[data-testid="stSidebar"] { background: #102a56; }
[data-testid="stSidebar"] * { color: white !important; }
.hero { background: linear-gradient(135deg,#102a56,#1d4ed8); color:white; padding:22px 26px; border-radius:16px; margin-bottom:16px; box-shadow:0 8px 24px rgba(16,42,86,.16); }
.hero h1 { margin:0; font-size:30px; }
.hero p { margin:6px 0 0; opacity:.9; }
.card { background:white; border:1px solid #d8e2ef; border-radius:14px; padding:16px; box-shadow:0 3px 12px rgba(20,50,90,.05); margin-bottom:12px; }
.section-title { color:#123c78; font-size:22px; font-weight:700; margin:6px 0 12px; }
.metric-card { background:white; border:1px solid #d8e2ef; border-radius:12px; padding:12px; text-align:center; }
.metric-value { color:#1d4ed8; font-size:25px; font-weight:700; }
.metric-label { color:#52647d; font-size:13px; }
div.stButton > button { border-radius:8px; border:1px solid #1d4ed8; }
[data-testid="stDataFrame"] { border:1px solid #d8e2ef; }
</style>
""", unsafe_allow_html=True)

# ---------------------------- State ----------------------------
def init_state():
    defaults = {
        "original_df": None,
        "processed_df": None,
        "viewed_df": None,
        "cleaning_history": [],
        "trained_pipeline": None,
        "trained_model_name": None,
        "target_column": None,
        "task_type": None,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "predictions": None,
        "model_results": pd.DataFrame(),
        "feature_names": [],
        "prediction_result": None,
        "last_evaluation": {},
        "tuned_model": None,
        "tuned_info": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ---------------------------- Helpers ----------------------------
def clear_model_state():
    for k in ["trained_pipeline", "trained_model_name", "target_column", "task_type", "X_train", "X_test", "y_train", "y_test", "predictions", "feature_names", "prediction_result", "last_evaluation", "tuned_model", "tuned_info"]:
        st.session_state[k] = None if k not in ["feature_names"] else []
    st.session_state.model_results = pd.DataFrame()


def fmt_num(x):
    if pd.isna(x): return "NA"
    if isinstance(x, (int, np.integer)): return str(int(x))
    if isinstance(x, (float, np.floating)): return f"{x:.4g}"
    return str(x)


def df_summary(df):
    if df is None:
        return {"Rows": 0, "Columns": 0, "Missing Values": 0, "Duplicates": 0, "Numeric Columns": 0, "Categorical Columns": 0, "Outlier Cells": 0}
    num = df.select_dtypes(include=np.number)
    outliers = 0
    for c in num.columns:
        s = num[c].dropna()
        if len(s) > 3:
            q1, q3 = s.quantile(.25), s.quantile(.75)
            iqr = q3-q1
            if iqr > 0: outliers += int(((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).sum())
    return {"Rows": len(df), "Columns": df.shape[1], "Missing Values": int(df.isna().sum().sum()), "Duplicates": int(df.duplicated().sum()), "Numeric Columns": len(num.columns), "Categorical Columns": df.shape[1]-len(num.columns), "Outlier Cells": outliers}


def quality_report(df):
    if df is None or df.empty: return pd.DataFrame(), 0, "No dataset loaded."
    rows=[]
    total_cells=max(1,df.shape[0]*df.shape[1])
    missing=int(df.isna().sum().sum())
    dup=int(df.duplicated().sum())
    for c in df.columns:
        s=df[c]
        nonnull=s.dropna()
        text_ratio=float(nonnull.map(lambda x:isinstance(x,str)).mean()) if len(nonnull) else 0
        numeric_like=0
        if text_ratio>0.5:
            numeric_like=float(pd.to_numeric(nonnull.astype(str).str.replace(',','',regex=False), errors='coerce').notna().mean()) if len(nonnull) else 0
        rows.append({
            "Column": c, "Dtype": str(s.dtype), "Missing": int(s.isna().sum()), "Missing %": round(100*s.isna().mean(),2),
            "Unique": int(s.nunique(dropna=True)), "Unique %": round(100*s.nunique(dropna=True)/max(1,len(s)),2),
            "Constant": bool(s.nunique(dropna=True)<=1), "Numeric-looking text": bool(text_ratio>0.5 and numeric_like>0.8),
            "Infinite": int(np.isinf(s.to_numpy(dtype=float)).sum()) if pd.api.types.is_numeric_dtype(s) else 0,
        })
    rep=pd.DataFrame(rows)
    completeness=100*(1-missing/total_cells)
    consistency=100*(1-min(1,dup/max(1,len(df))))
    validity=100*(1-min(1,float(rep["Missing %"].mean())/100))
    uniqueness=100*(1-min(1,dup/max(1,len(df))))
    score=round(.4*completeness+.25*consistency+.2*validity+.15*uniqueness,1)
    warnings_list=[]
    for c in df.columns:
        lc=c.lower().replace("_"," ")
        s=df[c]
        if pd.api.types.is_numeric_dtype(s):
            if "age" in lc and ((s<0).any() or (s>120).any()): warnings_list.append(f"{c}: values outside typical human age range detected.")
            if any(x in lc for x in ["blood pressure","systolic","diastolic"]) and (s<=0).any(): warnings_list.append(f"{c}: non-positive blood-pressure values detected.")
            if any(x in lc for x in ["heart rate","pulse"]) and (s<=0).any(): warnings_list.append(f"{c}: non-positive heart-rate values detected.")
            if "cholesterol" in lc and (s<0).any(): warnings_list.append(f"{c}: negative cholesterol values detected.")
            if "bmi" in lc and ((s<=0).any() or (s>100).any()): warnings_list.append(f"{c}: suspicious BMI values detected.")
    return rep, score, "\n".join(warnings_list) if warnings_list else "No healthcare-specific heuristic warnings detected."


def read_uploaded(file):
    if file is None: return None
    name = getattr(file, "name", "uploaded.csv")
    try:
        if name.lower().endswith(('.xlsx','.xls')): return pd.read_excel(file)
        if name.lower().endswith('.json'): return pd.read_json(file)
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Could not read dataset: {e}")
        return None


def make_export(df, name, excel=False):
    if df is None: return None
    folder=os.path.join(tempfile.gettempdir(),"ai_ml_exports"); os.makedirs(folder,exist_ok=True)
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path=os.path.join(folder,f"{name}_{stamp}.{'xlsx' if excel else 'csv'}")
    try:
        if excel: df.to_excel(path,index=False)
        else: df.to_csv(path,index=False)
        return path
    except Exception as e:
        st.error(f"Export failed: {e}"); return None


def apply_cleaning(df, operation, columns, params):
    out=df.copy(); cols=columns if columns else list(out.columns)
    before=df_summary(out)
    if operation=="Drop rows with missing": out=out.dropna(subset=cols)
    elif operation=="Drop columns with missing above threshold":
        threshold=float(params.get("threshold",50)); drop=[c for c in cols if 100*out[c].isna().mean()>threshold]; out=out.drop(columns=drop)
    elif operation in ["Mean imputation","Median imputation","Mode imputation","Constant imputation","Forward fill","Backward fill","Interpolation"]:
        for c in cols:
            if c not in out: continue
            if operation=="Mean imputation" and pd.api.types.is_numeric_dtype(out[c]): out[c]=out[c].fillna(out[c].mean())
            elif operation=="Median imputation" and pd.api.types.is_numeric_dtype(out[c]): out[c]=out[c].fillna(out[c].median())
            elif operation=="Mode imputation":
                m=out[c].mode(dropna=True); out[c]=out[c].fillna(m.iloc[0] if len(m) else "Unknown")
            elif operation=="Constant imputation": out[c]=out[c].fillna(params.get("constant","Unknown"))
            elif operation=="Forward fill": out[c]=out[c].ffill()
            elif operation=="Backward fill": out[c]=out[c].bfill()
            elif operation=="Interpolation" and pd.api.types.is_numeric_dtype(out[c]): out[c]=out[c].interpolate().ffill().bfill()
    elif operation=="Remove exact duplicates": out=out.drop_duplicates()
    elif operation=="Strip whitespace":
        for c in cols:
            if c in out and out[c].dtype=="object": out[c]=out[c].map(lambda x:x.strip() if isinstance(x,str) else x)
    elif operation=="Lowercase":
        for c in cols:
            if c in out and out[c].dtype=="object": out[c]=out[c].map(lambda x:x.lower() if isinstance(x,str) else x)
    elif operation=="Uppercase":
        for c in cols:
            if c in out and out[c].dtype=="object": out[c]=out[c].map(lambda x:x.upper() if isinstance(x,str) else x)
    elif operation=="Title case":
        for c in cols:
            if c in out and out[c].dtype=="object": out[c]=out[c].map(lambda x:x.title() if isinstance(x,str) else x)
    elif operation=="Replace values":
        old=params.get("old",""); new=params.get("new","")
        for c in cols: out[c]=out[c].replace(old,new)
    elif operation=="Convert numeric":
        for c in cols: out[c]=pd.to_numeric(out[c],errors="coerce")
    elif operation=="Convert category":
        for c in cols: out[c]=out[c].astype("category")
    elif operation=="Convert date":
        for c in cols: out[c]=pd.to_datetime(out[c],errors="coerce")
    elif operation=="Replace negative numeric with missing":
        for c in cols:
            if c in out and pd.api.types.is_numeric_dtype(out[c]): out.loc[out[c]<0,c]=np.nan
    elif operation=="Replace infinite with missing":
        for c in cols:
            if c in out and pd.api.types.is_numeric_dtype(out[c]): out[c]=out[c].replace([np.inf,-np.inf],np.nan)
    elif operation=="Replace empty strings with missing":
        for c in cols:
            if c in out and out[c].dtype=="object": out[c]=out[c].replace(r"^\s*$",np.nan,regex=True)
    elif operation=="Replace common null strings":
        nulls=["null","none","nan","n/a","na","missing","unknown","-","--",""]
        for c in cols: out[c]=out[c].replace(nulls,np.nan,regex=False)
    elif operation in ["IQR remove","IQR cap","Z-score remove","Z-score cap"]:
        for c in cols:
            if c not in out or not pd.api.types.is_numeric_dtype(out[c]): continue
            s=out[c]
            if operation.startswith("IQR"):
                q1,q3=s.quantile(.25),s.quantile(.75); iqr=q3-q1; lo,hi=q1-1.5*iqr,q3+1.5*iqr
            else:
                mu,sd=s.mean(),s.std(); lo,hi=mu-3*sd,mu+3*sd
            if operation.endswith("remove"): out=out[(out[c].isna())|(out[c].between(lo,hi))]
            else: out[c]=out[c].clip(lo,hi)
    after=df_summary(out)
    return out,before,after


def stats_table(df, cols, percentiles):
    if df is None or not cols: return pd.DataFrame()
    rows=[]
    for c in cols:
        s=pd.to_numeric(df[c],errors="coerce").dropna()
        if s.empty: continue
        q=s.quantile([p/100 for p in percentiles]).to_dict()
        mad=float(np.mean(np.abs(s-s.mean())))
        rows.append({"Column":c,"Mean":s.mean(),"Median":s.median(),"Mode":s.mode().iloc[0] if not s.mode().empty else np.nan,"Min":s.min(),"Max":s.max(),"Range":s.max()-s.min(),"Variance":s.var(),"Std":s.std(),"Q1":s.quantile(.25),"Q3":s.quantile(.75),"IQR":s.quantile(.75)-s.quantile(.25),"Skewness":s.skew(),"Kurtosis":s.kurtosis(),"MAD":mad,**{f"P{p}":q.get(p/100,np.nan) for p in percentiles}})
    return pd.DataFrame(rows)

# ---------------------------- Models ----------------------------
def get_models(task):
    if task=="Classification":
        return {
            "Logistic Regression":LogisticRegression(max_iter=2000),
            "KNN":KNeighborsClassifier(), "Naive Bayes":GaussianNB(), "Decision Tree":DecisionTreeClassifier(random_state=42),
            "Random Forest":RandomForestClassifier(n_estimators=250,random_state=42,n_jobs=-1), "SVM":SVC(probability=True,random_state=42),
            "Gradient Boosting":GradientBoostingClassifier(random_state=42)
        }
    return {
        "Linear Regression":LinearRegression(), "Polynomial Regression":Pipeline([("poly",PolynomialFeatures(degree=2,include_bias=False)),("model",LinearRegression())]),
        "Ridge":Ridge(), "Lasso":Lasso(max_iter=5000), "Decision Tree":DecisionTreeRegressor(random_state=42),
        "Random Forest":RandomForestRegressor(n_estimators=250,random_state=42,n_jobs=-1), "SVR":SVR(), "Gradient Boosting":GradientBoostingRegressor(random_state=42)
    }


def build_pipeline(X, model, scaler_name="StandardScaler"):
    num=list(X.select_dtypes(include=np.number).columns); cat=[c for c in X.columns if c not in num]
    scaler={"StandardScaler":StandardScaler(),"MinMaxScaler":MinMaxScaler(),"RobustScaler":RobustScaler(),"None": "passthrough"}.get(scaler_name,StandardScaler())
    numeric=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",scaler)])
    categorical=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
    transformers=[]
    if num: transformers.append(("num",numeric,num))
    if cat: transformers.append(("cat",categorical,cat))
    prep=ColumnTransformer(transformers,remainder="drop",verbose_feature_names_out=False)
    return Pipeline([("preprocessor",prep),("model",model)])


def train_one(df,target,task,model_name,test_size,seed,scaler):
    work=df.copy(); y=work[target]
    X=work.drop(columns=[target])
    if task=="Regression": y=pd.to_numeric(y,errors="coerce"); mask=y.notna(); X=X.loc[mask]; y=y.loc[mask]
    else:
        mask=y.notna(); X=X.loc[mask]; y=y.loc[mask]
    if len(X)<20: raise ValueError("At least 20 usable rows are recommended for training.")
    strat=y if task=="Classification" and y.nunique()<20 else None
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=seed,stratify=strat)
    model=get_models(task)[model_name]
    pipe=build_pipeline(X,model,scaler)
    pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
    metrics={}
    if task=="Classification":
        avg="binary" if y.nunique()==2 else "weighted"
        metrics={"Accuracy":accuracy_score(yte,pred),"Precision":precision_score(yte,pred,average=avg,zero_division=0),"Recall":recall_score(yte,pred,average=avg,zero_division=0),"F1":f1_score(yte,pred,average=avg,zero_division=0)}
        if hasattr(pipe,"predict_proba") and y.nunique()==2:
            try: metrics["ROC-AUC"]=roc_auc_score(yte,pipe.predict_proba(Xte)[:,1])
            except Exception: pass
    else:
        mse=mean_squared_error(yte,pred); n=len(yte); p=max(1,X.shape[1]); r2=r2_score(yte,pred)
        metrics={"MAE":mean_absolute_error(yte,pred),"MSE":mse,"RMSE":np.sqrt(mse),"R2":r2,"Adjusted R2":1-(1-r2)*(n-1)/max(1,n-p-1)}
    return pipe,Xtr,Xte,ytr,yte,pred,metrics

# ---------------------------- Header ----------------------------
st.markdown('<div class="hero"><h1>AI & ML Analytics Studio</h1><p>Healthcare-focused data quality, cleaning, exploratory analysis, machine learning, evaluation, explainability and export.</p></div>',unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Dataset")
    uploaded=st.file_uploader("Upload CSV, XLSX, XLS or JSON",type=["csv","xlsx","xls","json"])
    if uploaded is not None and st.button("Load Dataset",use_container_width=True):
        df=read_uploaded(uploaded)
        if df is not None:
            st.session_state.original_df=df.copy(); st.session_state.processed_df=df.copy(); st.session_state.viewed_df=df.head(10).copy(); st.session_state.cleaning_history=[]; clear_model_state(); st.success(f"Loaded {len(df):,} rows × {df.shape[1]:,} columns.")
    if st.session_state.processed_df is not None:
        st.markdown("### Session")
        if st.button("Reset to Original",use_container_width=True):
            st.session_state.processed_df=st.session_state.original_df.copy(); st.session_state.cleaning_history=[]; st.session_state.viewed_df=st.session_state.processed_df.head(10).copy(); clear_model_state(); st.rerun()
        s=df_summary(st.session_state.processed_df)
        st.write(f"Rows: {s['Rows']:,}")
        st.write(f"Columns: {s['Columns']:,}")
        st.write(f"Missing: {s['Missing Values']:,}")

if st.session_state.processed_df is None:
    st.info("Upload a dataset from the left sidebar to begin.")
    st.stop()

df=st.session_state.processed_df

# ---------------------------- Tabs ----------------------------
tabs=st.tabs(["Overview","Viewer","Statistics","Quality","Cleaning","EDA","Target","Features","Training","Evaluation","Comparison","Tuning","Prediction","Explainability","Export"])

# Overview
with tabs[0]:
    s=df_summary(df); rep,score,warn=quality_report(df)
    cols=st.columns(6)
    for col,(label,val) in zip(cols,[("Rows",s["Rows"]),("Columns",s["Columns"]),("Missing",s["Missing Values"]),("Duplicates",s["Duplicates"]),("Numeric",s["Numeric Columns"]),("Quality Score",f"{score}/100")]):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>',unsafe_allow_html=True)
    st.markdown("### Dataset snapshot")
    st.dataframe(df.head(10),use_container_width=True)
    st.markdown("### Healthcare-aware quality signals")
    st.info(warn)

# Viewer
with tabs[1]:
    c1,c2,c3=st.columns(3)
    source=c1.selectbox("Dataset",["Processed","Original"])
    view=c2.selectbox("View",["Head","Tail","Random","Entire Dataset","Custom Range"])
    n=c3.selectbox("Rows",[10,20,50,100,147,500],index=0)
    base=st.session_state.processed_df if source=="Processed" else st.session_state.original_df
    if view=="Custom Range":
        a,b=st.columns(2); start=a.number_input("Start row",min_value=0,max_value=max(0,len(base)-1),value=0); end=b.number_input("End row",min_value=1,max_value=len(base),value=min(50,len(base)))
        viewed=base.iloc[int(start):int(end)].copy()
    elif view=="Head": viewed=base.head(n).copy()
    elif view=="Tail": viewed=base.tail(n).copy()
    elif view=="Random": viewed=base.sample(min(n,len(base)),random_state=42).copy()
    else: viewed=base.head(min(len(base),5000)).copy()
    q=st.text_input("Search across all columns")
    if q:
        mask=viewed.astype(str).apply(lambda col:col.str.contains(q,case=False,na=False)).any(axis=1); viewed=viewed[mask]
    st.session_state.viewed_df=viewed.copy()
    st.caption(f"Displaying {len(viewed):,} rows × {viewed.shape[1]:,} columns from {source} dataset.")
    st.dataframe(viewed,use_container_width=True,height=520)
    p=make_export(viewed,"currently_viewed_dataset",False)
    if p: st.download_button("Download Current View CSV",open(p,"rb"),file_name="currently_viewed_dataset.csv",mime="text/csv")

# Statistics
with tabs[2]:
    nums=list(df.select_dtypes(include=np.number).columns)
    selected=st.multiselect("Numerical columns",nums,default=nums[:min(5,len(nums))])
    perc=st.multiselect("Percentiles",[1,5,10,25,50,75,90,95,99],default=[25,50,75,90,95])
    st.dataframe(stats_table(df,selected,perc),use_container_width=True)

# Quality
with tabs[3]:
    rep,score,warn=quality_report(df)
    st.metric("Overall quality score",f"{score}/100")
    st.dataframe(rep,use_container_width=True)
    st.info(warn)
    st.markdown("### Dataset-level checks")
    checks=pd.DataFrame([{"Check":"Duplicate rows","Count":int(df.duplicated().sum())},{"Check":"Duplicate column names","Count":int(df.columns.duplicated().sum())},{"Check":"Empty columns","Count":int((df.notna().sum()==0).sum())},{"Check":"Constant columns","Count":int((df.nunique(dropna=True)<=1).sum())},{"Check":"Infinite numeric values","Count":int(np.isinf(df.select_dtypes(include=np.number).to_numpy()).sum())}])
    st.dataframe(checks,use_container_width=True)

# Cleaning
with tabs[4]:
    st.markdown("### Cleaning workflow: Original → Preview → Apply → Processed")
    ops=["Drop rows with missing","Drop columns with missing above threshold","Mean imputation","Median imputation","Mode imputation","Constant imputation","Forward fill","Backward fill","Interpolation","Remove exact duplicates","Strip whitespace","Lowercase","Uppercase","Title case","Replace values","Convert numeric","Convert category","Convert date","Replace negative numeric with missing","Replace infinite with missing","Replace empty strings with missing","Replace common null strings","IQR remove","IQR cap","Z-score remove","Z-score cap"]
    op=st.selectbox("Operation",ops)
    eligible=list(df.columns)
    cols=st.multiselect("Columns (leave empty for all applicable columns)",eligible)
    p1,p2=st.columns(2)
    params={}
    if op=="Drop columns with missing above threshold": params["threshold"]=p1.slider("Missing threshold %",0,100,50)
    if op=="Constant imputation": params["constant"]=p1.text_input("Constant value","Unknown")
    if op=="Replace values": params["old"]=p1.text_input("Old value"); params["new"]=p2.text_input("New value")
    try:
        preview,before,after=apply_cleaning(df,op,cols,params)
        b,a=st.columns(2)
        with b: st.markdown("**Before Cleaning**"); st.dataframe(df.head(100),use_container_width=True,height=300)
        with a: st.markdown("**After / Preview**"); st.dataframe(preview.head(100),use_container_width=True,height=300)
        delta=pd.DataFrame([before,after],index=["Before","After"]); st.dataframe(delta,use_container_width=True)
        st.caption(f"Preview changes rows from {before['Rows']:,} to {after['Rows']:,} and missing values from {before['Missing Values']:,} to {after['Missing Values']:,}. Preview does not modify the processed dataset.")
        if st.button("Apply Change",type="primary"):
            st.session_state.processed_df=preview.copy(); st.session_state.cleaning_history.append({"Step":len(st.session_state.cleaning_history)+1,"Operation":op,"Column":", ".join(cols) if cols else "All applicable","Before Rows":before["Rows"],"After Rows":after["Rows"],"Before Missing":before["Missing Values"],"After Missing":after["Missing Values"],"Status":"Applied","Timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}); clear_model_state(); st.success("Cleaning change applied."); st.rerun()
    except Exception as e: st.error(f"Preview failed: {e}")
    if st.session_state.cleaning_history:
        st.markdown("### Cleaning history")
        st.dataframe(pd.DataFrame(st.session_state.cleaning_history),use_container_width=True)

# EDA
with tabs[5]:
    num=list(df.select_dtypes(include=np.number).columns); allc=list(df.columns)
    chart=st.selectbox("Analysis",["Histogram","KDE","Box","Violin","Count","Bar","Pie","Scatter","Line","Area","Correlation","Covariance","Pair Plot","Q-Q","ECDF","Missing","3D Scatter"])
    fig=None
    try:
        if chart in ["Histogram","KDE","Box","Violin","Q-Q","ECDF"]:
            c=st.selectbox("Numerical column",num)
            s=pd.to_numeric(df[c],errors="coerce").dropna()
            fig,ax=plt.subplots(figsize=(9,5))
            if chart=="Histogram": ax.hist(s,bins=30); ax.set_title(f"Histogram: {c}")
            elif chart=="KDE": sns.kdeplot(s,ax=ax,fill=True); ax.set_title(f"KDE: {c}")
            elif chart=="Box": sns.boxplot(x=s,ax=ax); ax.set_title(f"Box Plot: {c}")
            elif chart=="Violin": sns.violinplot(x=s,ax=ax); ax.set_title(f"Violin: {c}")
            elif chart=="Q-Q": stats.probplot(s,dist="norm",plot=ax); ax.set_title(f"Q-Q: {c}")
            elif chart=="ECDF": x=np.sort(s); y=np.arange(1,len(x)+1)/len(x); ax.plot(x,y); ax.set_title(f"ECDF: {c}")
        elif chart=="Count":
            c=st.selectbox("Categorical column",[x for x in allc if x not in num] or allc); fig,ax=plt.subplots(figsize=(9,5)); df[c].astype(str).value_counts().head(30).plot.bar(ax=ax); ax.set_title(f"Count: {c}")
        elif chart in ["Bar","Pie"]:
            c=st.selectbox("Column",allc); counts=df[c].astype(str).value_counts().head(20); fig,ax=plt.subplots(figsize=(9,5)); (counts.plot.pie(ax=ax,autopct="%1.1f%%") if chart=="Pie" else counts.plot.bar(ax=ax)); ax.set_title(chart)
        elif chart=="Scatter":
            x=st.selectbox("X",num); y=st.selectbox("Y",[c for c in num if c!=x] or num); fig,ax=plt.subplots(figsize=(9,5)); ax.scatter(df[x],df[y],alpha=.65); ax.set_xlabel(x); ax.set_ylabel(y)
        elif chart in ["Line","Area"]:
            x=st.selectbox("X",allc); y=st.selectbox("Y",num); tmp=df[[x,y]].dropna(); fig,ax=plt.subplots(figsize=(9,5)); ax.plot(tmp[x],tmp[y]) if chart=="Line" else ax.fill_between(np.arange(len(tmp)),tmp[y].to_numpy()); ax.set_title(chart)
        elif chart in ["Correlation","Covariance"]:
            mat=df[num].corr() if chart=="Correlation" else df[num].cov(); fig,ax=plt.subplots(figsize=(9,7)); sns.heatmap(mat,ax=ax,cmap="Blues",annot=len(num)<=10); ax.set_title(chart)
        elif chart=="Pair Plot":
            use=num[:6]; fig=sns.pairplot(df[use].dropna().sample(min(500,len(df)),random_state=42)).fig
        elif chart=="Missing":
            miss=df.isna().sum().sort_values(ascending=False); fig,ax=plt.subplots(figsize=(9,5)); miss[miss>0].plot.bar(ax=ax); ax.set_ylabel("Missing values")
        elif chart=="3D Scatter":
            from mpl_toolkits.mplot3d import Axes3D  # noqa
            x,y,z=st.selectbox("X",num),st.selectbox("Y",num),st.selectbox("Z",num); fig=plt.figure(figsize=(9,6)); ax=fig.add_subplot(111,projection="3d"); ax.scatter(df[x],df[y],df[z],alpha=.6); ax.set_xlabel(x); ax.set_ylabel(y); ax.set_zlabel(z)
        if fig is not None: st.pyplot(fig,use_container_width=True); plt.close(fig)
    except Exception as e: st.error(f"EDA could not be generated: {e}")

# Target
with tabs[6]:
    target=st.selectbox("Target column",list(df.columns))
    inferred="Regression" if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique()>10 else "Classification"
    task=st.radio("Task",["Classification","Regression"],index=0 if inferred=="Classification" else 1,horizontal=True)
    if task=="Classification":
        counts=df[target].value_counts(dropna=False); st.dataframe(pd.DataFrame({"Count":counts,"Percent":100*counts/counts.sum()}),use_container_width=True); st.bar_chart(counts)
    else:
        y=pd.to_numeric(df[target],errors="coerce").dropna(); c1,c2,c3=st.columns(3); c1.metric("Skewness",fmt_num(y.skew())); c2.metric("Kurtosis",fmt_num(y.kurtosis())); c3.metric("Outliers",int(((y<y.quantile(.25)-1.5*(y.quantile(.75)-y.quantile(.25)))|(y>y.quantile(.75)+1.5*(y.quantile(.75)-y.quantile(.25)))).sum())); fig,ax=plt.subplots(); ax.hist(y,bins=30); st.pyplot(fig); plt.close(fig)

# Features
with tabs[7]:
    st.markdown("Feature engineering operates on a copy and never changes the original dataset.")
    feature_cols=st.multiselect("Columns",list(df.columns),default=list(df.select_dtypes(include=np.number).columns[:3]))
    feat_op=st.selectbox("Feature operation",["Log","Square","Square root","Standardization","MinMax scaling","Robust scaling","Frequency encoding","Variance filtering","Correlation filtering","SelectKBest","PCA","One-hot encoding","Ordinal encoding"])
    if st.button("Preview Feature Engineering"):
        try:
            tmp=df.copy()
            if feat_op in ["Log","Square","Square root"]:
                for c in feature_cols:
                    x=pd.to_numeric(tmp[c],errors="coerce")
                    tmp[c+"__"+feat_op.replace(" ","_")]=np.log1p(x.clip(lower=0)) if feat_op=="Log" else x**2 if feat_op=="Square" else np.sqrt(x.clip(lower=0))
            elif feat_op in ["Standardization","MinMax scaling","Robust scaling"]:
                scaler={"Standardization":StandardScaler(),"MinMax scaling":MinMaxScaler(),"Robust scaling":RobustScaler()}[feat_op]; tmp[feature_cols]=scaler.fit_transform(tmp[feature_cols].apply(pd.to_numeric,errors="coerce").fillna(tmp[feature_cols].median(numeric_only=True)))
            elif feat_op=="Frequency encoding":
                for c in feature_cols: tmp[c+"__freq"]=tmp[c].map(tmp[c].value_counts(normalize=True))
            elif feat_op=="One-hot encoding": tmp=pd.get_dummies(tmp,columns=feature_cols,dummy_na=True)
            elif feat_op=="Ordinal encoding":
                for c in feature_cols: tmp[c]=pd.factorize(tmp[c])[0]
            elif feat_op=="Variance filtering":
                threshold=st.number_input("Variance threshold",0.0,100.0,0.0); numtmp=tmp.select_dtypes(include=np.number); keep=numtmp.var()[numtmp.var()>threshold].index; tmp=tmp[list(keep)]
            elif feat_op=="Correlation filtering":
                corr=tmp.select_dtypes(include=np.number).corr().abs(); upper=corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool)); drop=[c for c in upper.columns if any(upper[c]>.9)]; tmp=tmp.drop(columns=drop)
            elif feat_op=="SelectKBest":
                from sklearn.feature_selection import SelectKBest, f_classif
                k=min(10,max(1,len(feature_cols))); tmp=tmp[feature_cols].select_dtypes(include=np.number); tmp=tmp.iloc[:,SelectKBest(f_classif,k=min(k,tmp.shape[1])).fit(tmp.fillna(tmp.median()),np.arange(len(tmp))).get_support()] if len(tmp)>2 else tmp
            elif feat_op=="PCA":
                from sklearn.decomposition import PCA
                x=tmp[feature_cols].apply(pd.to_numeric,errors="coerce").fillna(tmp[feature_cols].median(numeric_only=True)); n=min(5,x.shape[1]); pcs=PCA(n_components=n).fit_transform(StandardScaler().fit_transform(x)); tmp=pd.DataFrame(pcs,index=tmp.index,columns=[f"PC{i+1}" for i in range(n)])
            st.dataframe(tmp.head(100),use_container_width=True)
            st.session_state.feature_preview=tmp
            if st.button("Apply Feature Engineering"):
                st.session_state.processed_df=tmp.copy(); clear_model_state(); st.success("Feature engineering applied."); st.rerun()
        except Exception as e: st.error(f"Feature engineering failed: {e}")

# Training
with tabs[8]:
    st.markdown("### Model training")
    target_train=st.selectbox("Target",list(df.columns),index=list(df.columns).index(st.session_state.target_column) if st.session_state.target_column in df.columns else 0,key="train_target")
    inferred="Regression" if pd.api.types.is_numeric_dtype(df[target_train]) and df[target_train].nunique()>10 else "Classification"
    task_train=st.radio("Task type",["Classification","Regression"],index=0 if inferred=="Classification" else 1,horizontal=True,key="train_task")
    models=list(get_models(task_train).keys()); model_name=st.selectbox("Model",models)
    a,b,c=st.columns(3); test_size=a.slider("Test size",0.1,0.4,0.2,0.05); seed=b.number_input("Random seed",0,9999,42); scaler=c.selectbox("Numeric scaler",["StandardScaler","MinMaxScaler","RobustScaler","None"])
    if st.button("Train Model",type="primary"):
        try:
            pipe,Xtr,Xte,ytr,yte,pred,metrics=train_one(df,target_train,task_train,model_name,test_size,seed,scaler)
            st.session_state.update({"trained_pipeline":pipe,"trained_model_name":model_name,"target_column":target_train,"task_type":task_train,"X_train":Xtr,"X_test":Xte,"y_train":ytr,"y_test":yte,"predictions":pred,"last_evaluation":metrics})
            st.success(f"{model_name} trained successfully.")
            st.dataframe(pd.DataFrame([metrics]),use_container_width=True)
        except Exception as e: st.error(f"Training failed: {e}")

# Evaluation
with tabs[9]:
    if st.session_state.trained_pipeline is None: st.info("Train a model first.")
    else:
        task=st.session_state.task_type; Xte=st.session_state.X_test; yte=st.session_state.y_test; pred=st.session_state.predictions; pipe=st.session_state.trained_pipeline
        st.dataframe(pd.DataFrame([st.session_state.last_evaluation]),use_container_width=True)
        fig,ax=plt.subplots(figsize=(8,5))
        if task=="Classification":
            cm=confusion_matrix(yte,pred); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
        else:
            ax.scatter(yte,pred,alpha=.65); lo=min(yte.min(),pred.min()); hi=max(yte.max(),pred.max()); ax.plot([lo,hi],[lo,hi],"--"); ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title("Actual vs Predicted")
        st.pyplot(fig); plt.close(fig)
        if task=="Classification" and yte.nunique()==2 and hasattr(pipe,"predict_proba"):
            try:
                prob=pipe.predict_proba(Xte)[:,1]; fpr,tpr,_=roc_curve(yte,prob); fig,ax=plt.subplots(); ax.plot(fpr,tpr); ax.plot([0,1],[0,1],"--"); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curve"); st.pyplot(fig); plt.close(fig)
                pr,rc,_=precision_recall_curve(yte,prob); fig,ax=plt.subplots(); ax.plot(rc,pr); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall Curve"); st.pyplot(fig); plt.close(fig)
            except Exception as e: st.warning(f"ROC/PR unavailable: {e}")
        else:
            residual=np.asarray(yte)-np.asarray(pred); fig,ax=plt.subplots(); ax.scatter(pred,residual,alpha=.65); ax.axhline(0,ls="--"); ax.set_xlabel("Predicted"); ax.set_ylabel("Residual"); ax.set_title("Residual Plot"); st.pyplot(fig); plt.close(fig)

# Comparison
with tabs[10]:
    if st.button("Run Model Comparison"):
        target=st.session_state.get("target_column") or st.selectbox("Comparison target",list(df.columns),key="cmp_target")
        task=st.session_state.get("task_type") or ("Regression" if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique()>10 else "Classification")
        results=[]
        for name in get_models(task):
            try:
                _,_,_,_,_,_,m=train_one(df,target,task,name,.2,42,"StandardScaler"); results.append({"Model":name,**m})
            except Exception as e: results.append({"Model":name,"Error":str(e)})
        st.session_state.model_results=pd.DataFrame(results); 
    if not st.session_state.model_results.empty:
        st.dataframe(st.session_state.model_results,use_container_width=True)
        metric="F1" if "F1" in st.session_state.model_results.columns else "R2"
        valid=st.session_state.model_results.dropna(subset=[metric])
        if not valid.empty: st.success(f"Best model by {metric}: {valid.sort_values(metric,ascending=False).iloc[0]['Model']}")

# Tuning
with tabs[11]:
    if st.session_state.trained_pipeline is None: st.info("Train a model first, then tune it here.")
    else:
        task=st.session_state.task_type; name=st.session_state.trained_model_name; X=st.session_state.X_train; y=st.session_state.y_train
        method=st.radio("Search",["GridSearchCV","RandomizedSearchCV"],horizontal=True)
        folds=st.selectbox("CV folds",[3,5,10],index=1)
        grids={
            "Logistic Regression":{"model__C":[.1,1,10]},"KNN":{"model__n_neighbors":[3,5,9]},"Decision Tree":{"model__max_depth":[None,4,8,12]},"Random Forest":{"model__n_estimators":[100,200],"model__max_depth":[None,8,16]},"SVM":{"model__C":[.1,1,10],"model__gamma":["scale","auto"]},"Gradient Boosting":{"model__n_estimators":[50,100],"model__learning_rate":[.03,.1]},
            "Ridge":{"model__alpha":[.1,1,10]},"Lasso":{"model__alpha":[.001,.01,.1]},"Decision Tree":{"model__max_depth":[None,4,8,12]},"Random Forest":{"model__n_estimators":[100,200],"model__max_depth":[None,8,16]},"SVR":{"model__C":[.1,1,10],"model__gamma":["scale","auto"]},"Gradient Boosting":{"model__n_estimators":[50,100],"model__learning_rate":[.03,.1]}}
        if st.button("Run Tuning"):
            try:
                pipe=build_pipeline(X,get_models(task)[name],"StandardScaler")
                cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=42) if task=="Classification" else KFold(n_splits=folds,shuffle=True,random_state=42)
                scoring="f1_weighted" if task=="Classification" else "r2"
                search=GridSearchCV(pipe,grids.get(name,{}),cv=cv,scoring=scoring,n_jobs=-1) if method=="GridSearchCV" else RandomizedSearchCV(pipe,grids.get(name,{}),n_iter=min(8,max(1,np.prod([len(v) for v in grids.get(name,{}).values()]))),cv=cv,scoring=scoring,random_state=42,n_jobs=-1)
                search.fit(X,y); st.session_state.tuned_model=search.best_estimator_; st.session_state.tuned_info={"Best Params":search.best_params_,"Best CV Score":search.best_score_}; st.json(st.session_state.tuned_info); st.success("Tuning completed.")
            except Exception as e: st.error(f"Tuning failed: {e}")

# Prediction
with tabs[12]:
    if st.session_state.trained_pipeline is None: st.info("Train a model first.")
    else:
        pipe=st.session_state.trained_pipeline; target=st.session_state.target_column; features=[c for c in st.session_state.X_train.columns]
        pred_input=pd.DataFrame({"Feature":features,"Value":["" for _ in features]})
        edited=st.data_editor(pred_input,use_container_width=True,num_rows="fixed",key="prediction_table")
        st.caption("Enter values in the Value column. Numeric features are converted automatically; categorical values should match values used in the training data.")
        if st.button("Generate Prediction",type="primary"):
            try:
                row={}
                for _,r in edited.iterrows():
                    f=str(r["Feature"]); v=r["Value"]
                    if v is None or str(v).strip()=="": raise ValueError(f"Missing value for {f}.")
                    if pd.api.types.is_numeric_dtype(st.session_state.X_train[f]): v=float(v)
                    row[f]=v
                one=pd.DataFrame([row],columns=features); pred=pipe.predict(one)[0]; result={"Prediction":pred}
                if st.session_state.task_type=="Classification" and hasattr(pipe,"predict_proba"):
                    probs=pipe.predict_proba(one)[0]; result.update({f"Probability: {cls}":float(p) for cls,p in zip(pipe.classes_,probs)})
                st.session_state.prediction_result=result; st.success(f"Prediction: {pred}"); st.dataframe(pd.DataFrame([result]),use_container_width=True)
            except Exception as e: st.error(f"Prediction failed: {e}")

# Explainability
with tabs[13]:
    if st.session_state.trained_pipeline is None: st.info("Train a model first.")
    else:
        pipe=st.session_state.trained_pipeline; Xte=st.session_state.X_test; yte=st.session_state.y_test
        st.markdown("### Model explainability")
        if st.button("Calculate Permutation Importance"):
            try:
                scoring="f1_weighted" if st.session_state.task_type=="Classification" else "r2"; pi=permutation_importance(pipe,Xte,yte,n_repeats=5,random_state=42,scoring=scoring)
                names=list(Xte.columns); imp=pd.DataFrame({"Feature":names,"Importance":pi.importances_mean}).sort_values("Importance",ascending=False); st.session_state.feature_importance=imp
            except Exception as e: st.error(f"Permutation importance failed: {e}")
        if "feature_importance" in st.session_state: st.dataframe(st.session_state.feature_importance,use_container_width=True); st.bar_chart(st.session_state.feature_importance.set_index("Feature").head(20))
        try:
            model=pipe.named_steps["model"]; prep=pipe.named_steps["preprocessor"]; names=prep.get_feature_names_out()
            if hasattr(model,"feature_importances_"):
                imp=pd.DataFrame({"Feature":names,"Importance":model.feature_importances_}).sort_values("Importance",ascending=False); st.markdown("### Model feature importance"); st.dataframe(imp.head(50),use_container_width=True)
            elif hasattr(model,"coef_"):
                coef=np.ravel(model.coef_); imp=pd.DataFrame({"Feature":names[:len(coef)],"Coefficient":coef}).sort_values("Coefficient",key=np.abs,ascending=False); st.markdown("### Model coefficients"); st.dataframe(imp.head(50),use_container_width=True)
        except Exception as e: st.caption(f"Native model importance unavailable: {e}")

# Export
with tabs[14]:
    st.markdown("### Export Center")
    export_items={"Original Dataset":st.session_state.original_df,"Processed Dataset":st.session_state.processed_df,"Currently Viewed Dataset":st.session_state.viewed_df,"Statistics":stats_table(df,list(df.select_dtypes(include=np.number).columns),[25,50,75,90,95]),"Missing Value Report":df.isna().sum().rename("Missing").reset_index().rename(columns={"index":"Column"}),"Duplicate Report":pd.DataFrame({"Duplicate Rows":[int(df.duplicated().sum())]}),"Data Quality Report":quality_report(df)[0],"Model Metrics":pd.DataFrame([st.session_state.last_evaluation]) if st.session_state.last_evaluation else pd.DataFrame(),"Model Comparison":st.session_state.model_results,"Prediction Result":pd.DataFrame([st.session_state.prediction_result]) if st.session_state.prediction_result else pd.DataFrame(),"Feature Importance":st.session_state.get("feature_importance",pd.DataFrame()),"Cleaning Change Log":pd.DataFrame(st.session_state.cleaning_history)}
    choice=st.selectbox("Artifact",list(export_items.keys()))
    fmt=st.radio("Format",["CSV","Excel"],horizontal=True)
    data=export_items[choice]
    if data is not None and not data.empty:
        path=make_export(data,"export",fmt=="Excel")
        if path:
            with open(path,"rb") as f: st.download_button("Download",f,file_name=os.path.basename(path),mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if fmt=="Excel" else "text/csv")
    else: st.info("No data is available for this export yet.")
    complete=pd.DataFrame([{"Section":"Dataset","Rows":len(df),"Columns":df.shape[1]},{"Section":"Quality","Rows":quality_report(df)[1],"Columns":None},{"Section":"Model","Rows":1 if st.session_state.trained_pipeline else 0,"Columns":st.session_state.trained_model_name or ""}])
    st.download_button("Download Complete Analysis Summary",complete.to_csv(index=False).encode(),file_name="complete_analysis_summary.csv",mime="text/csv")
