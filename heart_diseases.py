# streamlit_dashboard.py
"""
Professional EDA dashboard (single file).
- Place source3.csv in same folder OR it will try heart.csv OR you can upload a CSV.
- Run with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Source3 — EDA Dashboard", layout="wide", page_icon="📊")

# ---- Helper functions ----
@st.cache_data
def load_csv_try(paths):
    """Try a list of paths in order and return DataFrame or None."""
    for p in paths:
        try:
            df = pd.read_csv(p)
            return df, p
        except Exception:
            continue
    return None, None

def basic_overview(df):
    c1, c2, c3 = st.columns([2, 2, 6])
    with c1:
        st.metric("Rows", f"{df.shape[0]:,}")
        st.metric("Columns", f"{df.shape[1]:,}")
    with c2:
        n_num = df.select_dtypes(include=np.number).shape[1]
        n_cat = df.select_dtypes(exclude=np.number).shape[1]
        st.metric("Numeric cols", n_num)
        st.metric("Categorical cols", n_cat)
    with c3:
        memory = df.memory_usage(deep=True).sum() / (1024**2)
        st.metric("Memory usage (MB)", f"{memory:.2f}")

def missing_summary(df):
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        st.success("No missing values detected.")
        return
    st.subheader("Missing values")
    st.dataframe(pd.DataFrame({'missing_count': miss, 'missing_pct': (miss / len(df) * 100).round(2)}))

def dtype_summary(df):
    st.subheader("Column types")
    dtypes = df.dtypes.astype(str).value_counts()
    st.bar_chart(dtypes)

def quick_stats(df):
    st.subheader("Quick statistics")
    st.dataframe(df.describe(include='all').T)

def corr_heatmap(df, numeric_cols):
    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns for correlation heatmap.")
        return

    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(min(14, len(numeric_cols)), min(8, len(numeric_cols))))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='vlag', center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    st.pyplot(fig)

def encode_for_model(X):
    """Simple encoding: label encode categoricals (inplace copy), fill NA with median/most frequent."""
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == 'object' or X[c].dtype.name == 'category':
            X[c] = X[c].astype(str).fillna("___missing___")
            le = LabelEncoder()
            try:
                X[c] = le.fit_transform(X[c])
            except Exception:
                # fallback to factorize
                X[c] = pd.factorize(X[c])[0]
        else:
            if X[c].isna().any():
                imputer = SimpleImputer(strategy='median')
                X[c] = imputer.fit_transform(X[[c]])
    return X

# ---- Page header ----
st.title("📊 Source3 — Exploratory Data Analysis (EDA) Dashboard")
st.markdown(
    """
A professional EDA workspace:
- Auto-loads `source3.csv` (fallback `heart.csv`) or upload your CSV.
- Interactive filters, distributions, correlations, feature importance, PCA and download of filtered data.
""")

# ---- Data loading ----
local_paths = ["source3.csv", "source3.zip", "heart.csv"]
df, used_path = load_csv_try(local_paths)

st.sidebar.header("Data input")
if df is None:
    st.sidebar.info("No default CSV found. Upload one below.")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv", "txt", "zip"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            used_path = "uploaded file"
        except Exception as e:
            st.sidebar.error(f"Could not read uploaded file: {e}")
            st.stop()
else:
    st.sidebar.success(f"Loaded: {used_path}")

if df is None:
    st.stop()

# small cleaning for display
df_original = df.copy()
st.sidebar.write(f"Rows: {df.shape[0]}, Cols: {df.shape[1]}")

# ---- Sidebar: interactive controls ----
st.sidebar.header("Interactive filters & options")
all_columns = df.columns.tolist()
default_target = "target" if "target" in all_columns else all_columns[-1]

target_column = st.sidebar.selectbox("Select target/column of interest (for supervised insights)", options=all_columns, index=all_columns.index(default_target) if default_target in all_columns else 0)

# column selectors
cols_to_show = st.sidebar.multiselect("Columns to show in preview", options=all_columns, default=all_columns[:8])

# quick filter builder for numeric cols
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
filters = {}
if numeric_cols:
    st.sidebar.markdown("### Numeric filters (range)")
    for col in numeric_cols[:8]:  # keep sidebar reasonable; you can extend
        vmin, vmax = float(df[col].min()), float(df[col].max())
        step = (vmax - vmin) / 100 if vmax != vmin else 1
        lo, hi = st.sidebar.slider(f"{col}", min_value=vmin, max_value=vmax, value=(vmin, vmax), step=step)
        filters[col] = (lo, hi)

# categorical filters (top categories)
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
cat_filters = {}
if cat_cols:
    st.sidebar.markdown("### Categorical filters")
    for col in cat_cols[:6]:
        choices = df[col].astype(str).fillna("___missing___").value_counts().index.tolist()
        default_choices = choices[:5] if len(choices) > 5 else choices
        sel = st.sidebar.multiselect(f"{col}", options=choices, default=default_choices)
        cat_filters[col] = sel

# apply filters to create filtered_df
filtered_df = df.copy()
for c, (lo, hi) in filters.items():
    filtered_df = filtered_df[(pd.to_numeric(filtered_df[c], errors='coerce') >= lo) & (pd.to_numeric(filtered_df[c], errors='coerce') <= hi)]
for c, sel in cat_filters.items():
    if sel:
        filtered_df = filtered_df[filtered_df[c].astype(str).isin(sel)]

st.sidebar.markdown("---")
st.sidebar.markdown("📥 Download")
st.sidebar.download_button("Download filtered CSV", data=filtered_df.to_csv(index=False).encode('utf-8'), file_name="filtered_source3.csv", mime="text/csv")

# ---- Main layout ----
st.header("Dataset preview & structure")
with st.expander("Preview and schema", expanded=True):
    st.subheader(f"Preview — first 10 rows (from `{used_path}`)")
    st.dataframe(filtered_df[cols_to_show].head(10), use_container_width=True)
    st.write("Column types:")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]).astype(str))

# Overview metrics
basic_overview(filtered_df)
st.markdown("---")

# Missing values & dtypes
missing_summary(filtered_df)
dtype_summary(filtered_df)
st.markdown("---")

# Descriptive statistics
quick_stats(filtered_df)
st.markdown("---")

# ---- Visualizations ----
st.header("Visual analysis")

# 1) Correlation heatmap (numeric)
st.subheader("Correlation — numeric features")
corr_heatmap(filtered_df, numeric_cols)
st.markdown("---")

# 2) Interactive distribution & boxplot
st.subheader("Distribution explorer")
dist_col = st.selectbox("Choose numeric column for distribution", options=numeric_cols) if numeric_cols else None

if dist_col:
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown(f"**Histogram & KDE — {dist_col}**")
        fig = px.histogram(filtered_df, x=dist_col, nbins=50, marginal="box", title=f"Distribution of {dist_col}")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # show summary stats
        s = filtered_df[dist_col].dropna()
        st.metric("Mean", f"{s.mean():.3f}")
        st.metric("Median", f"{s.median():.3f}")
        st.metric("Std", f"{s.std():.3f}")
st.markdown("---")

# 3) Categorical counts
st.subheader("Categorical counts")
if cat_cols:
    cat_col = st.selectbox("Choose categorical column", options=cat_cols)
    vc = filtered_df[cat_col].astype(str).value_counts().reset_index()
    vc.columns = [cat_col, 'count']
    fig = px.bar(vc, x=cat_col, y='count', title=f"Value counts for {cat_col}", text='count')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No categorical columns found.")

st.markdown("---")

# 4) Scatter matrix (numeric) / Pairwise scatter
st.subheader("Scatter matrix (multivariate)")
if len(numeric_cols) >= 2:
    limited = numeric_cols[:8]
    fig = px.scatter_matrix(filtered_df[limited].dropna(), dimensions=limited, title="Scatter matrix (first 8 numeric cols)")
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Need at least 2 numeric columns to show scatter matrix.")

st.markdown("---")

# 5) Feature importance (if user wants)
st.subheader("Feature importance (model-guided)")
if st.checkbox("Compute feature importance (Random Forest)", value=False):
    # try to build a supervised problem based on selected target
    st.info("Encoding data and training a Random Forest to estimate feature importance.")
    target = target_column
    if filtered_df[target].isna().all():
        st.warning("Selected target column is all NaN — pick another column or uncheck.")
    else:
        X = filtered_df.drop(columns=[target]).copy()
        y = filtered_df[target].copy()

        # drop columns with single unique value
        nunique = X.nunique(dropna=False)
        to_drop = nunique[nunique <= 1].index.tolist()
        X = X.drop(columns=to_drop)
        if X.shape[1] == 0:
            st.warning("No usable feature columns remain after dropping constant columns.")
        else:
            X_enc = encode_for_model(X)
            # if target is numeric and has many unique values, treat as regression; otherwise classification
            if y.dtype.kind in 'iuf' and y.nunique() > 10:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                problem_type = "regression"
            else:
                # encode y for classification if needed
                if y.dtype == 'object' or y.dtype.name == 'category':
                    le_y = LabelEncoder()
                    y_enc = le_y.fit_transform(y.fillna("___missing___").astype(str))
                else:
                    y_enc = y.fillna(y.median()).values
                model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
                problem_type = "classification"
                y = y_enc

            # align X and y (drop rows where y is NaN)
            mask = pd.notna(filtered_df[target])
            X_enc = X_enc.loc[mask].fillna(0)
            y = pd.Series(y).loc[mask].values

            if len(y) < 10:
                st.warning("Too few rows with target values to train a reliable model.")
            else:
                model.fit(X_enc, y)
                importances = pd.Series(model.feature_importances_, index=X_enc.columns).sort_values(ascending=False)
                st.write(f"Top features (by importance) — problem type: {problem_type}")
                st.dataframe(importances.head(30).round(4))
                fig = px.bar(importances.head(20).reset_index().rename(columns={'index':'feature', 0:'importance'}), x='importance', y='feature', orientation='h')
                st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 6) PCA for numeric features
st.subheader("PCA — numeric projection (2D)")
if st.checkbox("Compute PCA (2 components)", value=False):
    numeric_only = filtered_df.select_dtypes(include=np.number).dropna(axis=1, how='all')
    if numeric_only.shape[1] < 2:
        st.info("Need at least 2 numeric columns for PCA.")
    else:
        Xnum = numeric_only.fillna(numeric_only.median())
        X_scaled = (Xnum - Xnum.mean()) / Xnum.std().replace(0, 1)
        pca = PCA(n_components=2, random_state=42)
        pcs = pca.fit_transform(X_scaled)
        pc_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'], index=X_scaled.index)
        # color by target if reasonable
        color_col = None
        if target_column in filtered_df.columns:
            color_col = filtered_df.loc[pc_df.index, target_column].astype(str)
        fig = px.scatter(pc_df, x='PC1', y='PC2', color=color_col, title="PCA (PC1 vs PC2)", hover_data=[pc_df.index])
        st.plotly_chart(fig, use_container_width=True)
        st.write("Explained variance ratios:", pca.explained_variance_ratio_.round(4))

st.markdown("---")

# ---- Final notes and export ----
st.header("Final notes & export")
st.markdown(
    """
**Recommended next steps**
- Inspect top feature importances shown above and deep-dive into those columns.
- If target exists and this is a supervised task, consider more robust modeling with cross-validation.
- Clean any columns with high missingness, or add better imputation tailored to domain knowledge.
"""
)
st.markdown("### Filtered dataset sample")
st.dataframe(filtered_df.head(50), use_container_width=True)

# small footer
st.write("---")
st.caption("Built with Streamlit • EDA pipeline: quick summaries, visualizations, and model-guided feature ranking.")
