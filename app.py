"""
app.py
------
Streamlit dashboard that visualizes the outputs produced by spark_processor.py.
Run with:  streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CSV Analyzer Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

DATA_DIR = "data"

@st.cache_data(show_spinner=False)
def load_csv(filename: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def require(df, name):
    if df is None:
        st.warning(f"⚠️ `{name}` not found. Run `spark_processor.py` first.")
        st.stop()
    return df


# ─────────────────────────────────────────────
# Load all outputs
# ─────────────────────────────────────────────
summary_df  = load_csv("summary.csv")
ts_df       = load_csv("time_series.csv")
top_df      = load_csv("top_n.csv")
sample_df   = load_csv("cleaned_sample.csv")

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.title("🔧 Controls")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📈 Time Series", "🏆 Top N", "🔍 Explore Data"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data files detected:**")
for fname in ["summary.csv", "time_series.csv", "top_n.csv", "cleaned_sample.csv"]:
    exists = os.path.exists(os.path.join(DATA_DIR, fname))
    icon = "✅" if exists else "❌"
    st.sidebar.markdown(f"{icon} `{fname}`")

st.sidebar.markdown("---")
st.sidebar.info(
    "Run the pipeline first:\n```\npython src/spark_processor.py \\\n  --input data/your_file.csv\n```"
)

# ─────────────────────────────────────────────
# Page: Overview
# ─────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("📊 CSV Analyzer Dashboard")
    st.markdown("Built with **PySpark** + **Streamlit**")

    df = require(summary_df, "summary.csv")

    # Detect category col (first non-numeric col)
    cat_col = df.select_dtypes(include="object").columns[0]
    num_cols = df.select_dtypes(include="number").columns.tolist()

    # ── KPI row ──────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Categories", df[cat_col].nunique())
    if num_cols:
        first_num = num_cols[0]
        col2.metric("Total (col 1)", f"{df[first_num].sum():,.0f}")
        col3.metric("Avg (col 1)",   f"{df[first_num].mean():,.2f}")
        col4.metric("Max (col 1)",   f"{df[first_num].max():,.0f}")

    st.markdown("---")

    # ── Bar chart ─────────────────────────────
    if num_cols:
        selected_metric = st.selectbox("Metric to visualize", num_cols)
        fig = px.bar(
            df.sort_values(selected_metric, ascending=False),
            x=cat_col,
            y=selected_metric,
            color=cat_col,
            title=f"{selected_metric} by {cat_col}",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Summary table ─────────────────────────
    st.subheader("Summary Table")
    st.dataframe(df, use_container_width=True, height=300)


# ─────────────────────────────────────────────
# Page: Time Series
# ─────────────────────────────────────────────
elif page == "📈 Time Series":
    st.title("📈 Time Series Analysis")

    df = require(ts_df, "time_series.csv")

    num_cols = [c for c in df.select_dtypes(include="number").columns if c != "row_count"]

    if not num_cols:
        st.error("No numeric columns found in time_series.csv")
        st.stop()

    selected = st.multiselect("Select metrics", num_cols, default=num_cols[:2])

    if selected:
        fig = go.Figure()
        for col in selected:
            fig.add_trace(go.Scatter(x=df["month"], y=df[col], mode="lines+markers", name=col))
        fig.update_layout(
            title="Monthly Trends",
            xaxis_title="Month",
            template="plotly_dark",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row count over time
    if "row_count" in df.columns:
        st.subheader("Transaction Volume Over Time")
        fig2 = px.area(df, x="month", y="row_count", template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────
# Page: Top N
# ─────────────────────────────────────────────
elif page == "🏆 Top N":
    st.title("🏆 Top N Records")

    df = require(top_df, "top_n.csv")

    n = st.slider("Show top N rows", min_value=5, max_value=len(df), value=10)
    display_df = df.head(n)

    num_cols = display_df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        metric = st.selectbox("Sort / color by", num_cols)
        fig = px.bar(
            display_df,
            x=display_df.columns[0],
            y=metric,
            color=metric,
            color_continuous_scale="viridis",
            template="plotly_dark",
            title=f"Top {n} by {metric}",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(display_df, use_container_width=True)


# ─────────────────────────────────────────────
# Page: Explore Data
# ─────────────────────────────────────────────
elif page == "🔍 Explore Data":
    st.title("🔍 Explore Cleaned Data")

    df = require(sample_df, "cleaned_sample.csv")

    # ── Filters ───────────────────────────────
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()

    with st.expander("🔧 Filters", expanded=True):
        filters = {}
        cols = st.columns(min(len(cat_cols), 3))
        for i, col in enumerate(cat_cols[:3]):
            with cols[i]:
                selected = st.multiselect(col, df[col].dropna().unique(), default=df[col].dropna().unique()[:5])
                filters[col] = selected

    # Apply filters
    filtered = df.copy()
    for col, vals in filters.items():
        if vals:
            filtered = filtered[filtered[col].isin(vals)]

    st.markdown(f"**{len(filtered):,} rows** after filters")

    # ── Scatter plot ──────────────────────────
    if len(num_cols) >= 2:
        st.subheader("Scatter Plot")
        c1, c2, c3 = st.columns(3)
        x_col = c1.selectbox("X axis", num_cols, index=0)
        y_col = c2.selectbox("Y axis", num_cols, index=1 if len(num_cols) > 1 else 0)
        color_col = c3.selectbox("Color", [None] + cat_cols)

        fig = px.scatter(
            filtered.sample(min(2000, len(filtered))),
            x=x_col,
            y=y_col,
            color=color_col,
            opacity=0.6,
            template="plotly_dark",
            title=f"{y_col} vs {x_col}",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Histogram ─────────────────────────────
    if num_cols:
        st.subheader("Distribution")
        hist_col = st.selectbox("Column", num_cols)
        fig2 = px.histogram(filtered, x=hist_col, nbins=40, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Raw table ─────────────────────────────
    st.subheader("Data Table")
    st.dataframe(filtered.head(500), use_container_width=True, height=400)

    # ── Download ──────────────────────────────
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered data as CSV", csv_bytes, "filtered_data.csv", "text/csv")
