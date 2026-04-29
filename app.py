"""
app.py
------
Three pages:
  🌐 Kaggle Browser  — search and download any dataset
  🔬 Analyse         — generic dynamic explorer (pandas, any CSV)
  🚕 Taxi Analysis   — dedicated page for the NYC Taxi Spark pipeline

Run with:  streamlit run app.py
"""

import os, glob, subprocess, threading, sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
st.set_page_config(page_title="CSV Analyzer", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  [data-testid="stSidebar"] { background:#0f1117; }
  .card {
    background:#1a1d27; border:1px solid #2d3145;
    border-radius:10px; padding:1rem 1.2rem; margin-bottom:.6rem;
  }
  .card-title { color:#e2e8f0; font-weight:600; font-size:1rem; }
  .card-meta  { color:#8b8fa8; font-size:.8rem; margin-top:4px; }
  .kpi { background:#1a1d27; border:1px solid #2d3145; border-radius:10px;
         padding:1rem; text-align:center; }
  .kpi-label { color:#8b8fa8; font-size:.75rem; text-transform:uppercase; letter-spacing:1px; }
  .kpi-value { color:#e2e8f0; font-size:1.7rem; font-weight:700; }
  .output-card {
    background:#1a1d27; border:1px solid #2d3145; border-radius:12px;
    padding:1.2rem 1.5rem; margin-bottom:1rem;
  }
  .output-title { color:#e2e8f0; font-size:1.1rem; font-weight:700; margin-bottom:.3rem; }
  .output-desc  { color:#8b8fa8; font-size:.85rem; margin-bottom:.8rem; }
</style>
""", unsafe_allow_html=True)

DATA_DIR      = "data"
TAXI_DIR      = "data/taxi"
TAXI_OUTPUTS  = {
    "outliers.csv":       ("🔍 Outlier Detection",    "Suspicious trips flagged by mean ± 3σ on fare, impossible passenger counts, sub-$1 fares."),
    "ranked.csv":         ("🏆 Driver Rankings",       "Window function: payment types ranked by daily revenue within each date."),
    "running_totals.csv": ("📈 Running Totals",        "Window function: cumulative revenue and trip count over time per payment type."),
    "sessions.csv":       ("⏰ Sessionization",         "Trips grouped into Morning Rush, Daytime, Evening Rush, Night and Early Morning."),
    "pivot.csv":          ("📊 Pivot Table",            "Revenue pivot: hour of day × time-of-day session. Spot the busiest hours."),
    "time_series.csv":    ("📅 Time Series",            "Daily trip count, total revenue, avg fare and avg distance trends."),
    "joined.csv":         ("🔗 Enriched Trips",         "Trips joined with vendor and payment type lookup tables for human-readable labels."),
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def fmt_size(b):
    if not b: return "?"
    for u in ["B","KB","MB","GB"]:
        if b < 1024: return f"{b:.0f} {u}"
        b /= 1024
    return f"{b:.1f} TB"

def num_cols(df): return df.select_dtypes(include="number").columns.tolist()
def cat_cols(df): return df.select_dtypes(include="object").columns.tolist()

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


# ─────────────────────────────────────────────
# Kaggle helpers
# ─────────────────────────────────────────────

def setup_kaggle_credentials() -> bool:
    import json
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_json):
        return True
    try:
        username = st.secrets["KAGGLE_USERNAME"]
        key      = st.secrets["KAGGLE_KEY"]
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        with open(kaggle_json, "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(kaggle_json, 0o600)
        return True
    except Exception:
        return False

def kaggle_ok() -> bool:
    if not setup_kaggle_credentials():
        return False
    try:
        import kaggle; kaggle.api.authenticate(); return True
    except Exception: return False

@st.cache_data(show_spinner=False, ttl=300)
def kaggle_search(query, sort_by, n):
    try:
        import kaggle; kaggle.api.authenticate()
        rows = kaggle.api.dataset_list(search=query, sort_by=sort_by,
                                       file_type="csv")
        return [{"ref": str(r.ref), "title": str(r.title),
                 "size": fmt_size(getattr(r,"totalBytes",None)),
                 "downloads": getattr(r,"downloadCount",0),
                 "votes": getattr(r,"voteCount",0),
                 "updated": str(getattr(r,"lastUpdated",""))[:10],
                 "desc": str(getattr(r,"description","") or "")[:180],
                 "url": f"https://www.kaggle.com/datasets/{r.ref}"}
                for r in list(rows)[:n]]
    except Exception as e:
        return {"error": str(e)}

def kaggle_download(ref: str) -> tuple:
    os.makedirs(DATA_DIR, exist_ok=True)
    r = subprocess.run(["kaggle","datasets","download", ref,
                        "--unzip","-p", DATA_DIR],
                       capture_output=True, text=True)
    return r.returncode == 0, r.stdout + r.stderr


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

st.sidebar.title("📊 CSV Analyzer")
st.sidebar.markdown("---")

# session_state tracks current page so we can switch programmatically
if "current_page" not in st.session_state:
    st.session_state.current_page = "🌐 Kaggle Browser"

page = st.sidebar.radio(
    "**Navigate:**",
    ["🌐 Kaggle Browser", "🔬 Analyse", "🚕 Taxi Analysis"],
    index=["🌐 Kaggle Browser", "🔬 Analyse", "🚕 Taxi Analysis"].index(
        st.session_state.current_page
    ),
)
st.session_state.current_page = page

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
if csv_files:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Downloaded files:**")
    for f in csv_files:
        st.sidebar.markdown(f"📄 `{os.path.basename(f)}` — {fmt_size(os.path.getsize(f))}")


# ═════════════════════════════════════════════
# PAGE 1 — Kaggle Browser
# ═════════════════════════════════════════════

if page == "🌐 Kaggle Browser":
    st.title("🌐 Kaggle Dataset Browser")
    st.caption("Search, preview and download any public CSV dataset in one click.")

    if not kaggle_ok():
        st.error("""
**Kaggle credentials not found.** One-time setup:
1. Go to **kaggle.com → Settings → API → Create New Token**
2. Move `kaggle.json` to `~/.kaggle/kaggle.json` (Mac/Linux) or `C:\\Users\\YOU\\.kaggle\\` (Windows)
3. Restart the app
        """)
        st.stop()

    c1, c2, c3 = st.columns([3,1,1])
    query   = c1.text_input("Search", placeholder="sales, netflix, weather, titanic…")
    sort_by = c2.selectbox("Sort by", ["hottest","votes","updated","active"])
    n       = c3.slider("Results", 5, 50, 15)

    if not query:
        st.info("👆 Type a keyword to search Kaggle's 50,000+ public CSV datasets.")
        st.markdown("**Quick start:**")
        suggestions = ["titanic","netflix","world happiness","covid","sales","spotify","airbnb"]
        cols = st.columns(len(suggestions))
        for i, s in enumerate(suggestions):
            if cols[i].button(s, key=f"qs_{s}"): query = s

    if not query: st.stop()

    with st.spinner(f'Searching "{query}"…'):
        results = kaggle_search(query, sort_by, n)

    if isinstance(results, dict): st.error(f"API error: {results['error']}"); st.stop()
    if not results: st.warning("No results."); st.stop()

    st.markdown(f"**{len(results)} datasets** for `{query}`")
    st.markdown("---")

    for r in results:
        st.markdown(f"""
        <div class="card">
          <div class="card-title">📦 {r['title']}</div>
          <div class="card-meta">🔖 <code>{r['ref']}</code> &nbsp;|&nbsp;
            💾 {r['size']} &nbsp;|&nbsp; ⬇️ {r['downloads']:,} &nbsp;|&nbsp;
            👍 {r['votes']:,} &nbsp;|&nbsp; 🕒 {r['updated']}</div>
          <div class="card-meta" style="margin-top:6px">{r['desc']}…</div>
        </div>""", unsafe_allow_html=True)
        b1, b2, _ = st.columns([1,1,5])
        if b1.button("⬇️ Download", key=f"dl_{r['ref']}"):
            with st.spinner(f"Downloading `{r['ref']}`…"):
                ok, log = kaggle_download(r["ref"])
            if ok:
                # Find the CSV that was just downloaded
                new_csvs = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
                if new_csvs:
                    newest = os.path.relpath(max(new_csvs, key=os.path.getmtime), DATA_DIR)
                    st.session_state.preselected_file = newest
                st.cache_data.clear()
                st.session_state.current_page = "🔬 Analyse"
                st.rerun()
            else:
                st.error(f"Failed:\n```\n{log}\n```")
        b2.markdown(f"[🔗 Kaggle]({r['url']})")
        st.write("")


# ═════════════════════════════════════════════
# PAGE 2 — Analyse (generic pandas explorer)
# ═════════════════════════════════════════════

elif page == "🔬 Analyse":
    st.title("🔬 Analyse")

    all_csvs = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")) +
                      glob.glob(os.path.join(TAXI_DIR, "*.csv")))

    if not all_csvs:
        st.warning("No CSV files found. Use **🌐 Kaggle Browser** to download one, "
                   "or **🚕 Taxi Analysis** to run the Spark pipeline.")
        st.stop()

    names = [os.path.relpath(f, DATA_DIR) for f in all_csvs]

    # Preselect file if coming from Kaggle Browser download
    preselected = st.session_state.pop("preselected_file", None)
    default_idx = 0
    if preselected and preselected in names:
        default_idx = names.index(preselected)

    chosen = st.selectbox("**Choose a file:**", names, index=default_idx)
    df_raw = load_csv(os.path.join(DATA_DIR, chosen))

    st.caption(f"{len(df_raw):,} rows × {len(df_raw.columns)} columns — "
               f"{', '.join(f'`{c}`' for c in df_raw.columns)}")
    st.markdown("---")

    nc = num_cols(df_raw)
    cc = cat_cols(df_raw)

    # Step 1 — Filter
    with st.expander("**Step 1 — Filter rows** (optional)"):
        filtered = df_raw.copy()
        low_card = [c for c in cc if df_raw[c].nunique() <= 50]
        if low_card:
            st.markdown("**By category:**")
            fcols = st.columns(min(len(low_card), 3))
            for i, col in enumerate(low_card[:6]):
                vals = df_raw[col].dropna().unique().tolist()
                sel  = fcols[i%3].multiselect(col, vals, default=vals, key=f"f_{col}")
                if sel: filtered = filtered[filtered[col].isin(sel)]
        if nc:
            st.markdown("**By numeric range:**")
            rcols = st.columns(min(len(nc), 3))
            for i, col in enumerate(nc[:6]):
                lo_b, hi_b = float(df_raw[col].min()), float(df_raw[col].max())
                if lo_b < hi_b:
                    lo, hi = rcols[i%3].slider(col, lo_b, hi_b, (lo_b, hi_b), key=f"r_{col}")
                    filtered = filtered[(filtered[col]>=lo)&(filtered[col]<=hi)]
        st.caption(f"→ **{len(filtered):,} rows** after filters")

    # Step 2 — Group
    with st.expander("**Step 2 — Group & aggregate** (optional)"):
        result_df = filtered.copy()
        do_group  = st.toggle("Enable grouping")
        if do_group:
            g1, g2 = st.columns(2)
            group_by = g1.multiselect("Group by", cc, default=[cc[0]] if cc else [])
            agg_cols_sel = g2.multiselect("Aggregate", nc, default=nc[:3])
            agg_func = st.selectbox("Function", ["sum","mean","count","min","max","median"])
            if group_by and agg_cols_sel:
                result_df = (filtered.groupby(group_by, as_index=False)
                                     .agg({c: agg_func for c in agg_cols_sel})
                                     .sort_values(agg_cols_sel[0], ascending=False))
                st.caption(f"→ **{len(result_df):,} groups**")
    st.markdown("---")

    # Step 3 — Visualise
    st.subheader("📊 Visualise")
    rnc = num_cols(result_df); rcc = cat_cols(result_df)
    v1,v2,v3,v4 = st.columns(4)
    chart_type = v1.selectbox("Chart type", ["Bar","Line","Scatter","Area","Histogram","Box","Pie"])
    x_col      = v2.selectbox("X axis", result_df.columns.tolist())
    y_col      = v3.selectbox("Y axis", rnc) if rnc else None
    color_col  = v4.selectbox("Color by", [None]+rcc+rnc)

    if y_col:
        pdf = result_df.sample(min(3000,len(result_df))) if chart_type=="Scatter" else result_df
        try:
            kw = dict(template="plotly_dark", color=color_col)
            if   chart_type=="Bar":        fig=px.bar(pdf,x=x_col,y=y_col,**kw)
            elif chart_type=="Line":       fig=px.line(pdf,x=x_col,y=y_col,**kw)
            elif chart_type=="Scatter":    fig=px.scatter(pdf,x=x_col,y=y_col,opacity=.6,**kw)
            elif chart_type=="Area":       fig=px.area(pdf,x=x_col,y=y_col,**kw)
            elif chart_type=="Histogram":  fig=px.histogram(pdf,x=x_col,nbins=50,**kw)
            elif chart_type=="Box":        fig=px.box(pdf,x=color_col,y=y_col,template="plotly_dark")
            elif chart_type=="Pie":        fig=px.pie(pdf,names=x_col,values=y_col,template="plotly_dark")
            fig.update_layout(margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Can't draw: {e}")

    st.markdown("---")
    st.subheader("📋 Result table")
    st.dataframe(result_df, use_container_width=True, height=400)
    st.download_button("⬇️ Download result as CSV",
                       result_df.to_csv(index=False).encode(),
                       "result.csv", "text/csv")


# ═════════════════════════════════════════════
# PAGE 3 — Taxi Analysis
# ═════════════════════════════════════════════

elif page == "🚕 Taxi Analysis":
    st.title("🚕 NYC Taxi — Spark Pipeline")
    st.caption("Runs 7 PySpark transformations on the NYC Yellow Taxi dataset "
               "and produces one downloadable result file per transformation.")

    # ── Pipeline launcher ─────────────────────
    st.markdown("---")
    st.subheader("▶️ Run the Pipeline")

    # Session state tracks whether pipeline is currently running
    if "pipeline_running" not in st.session_state:
        st.session_state.pipeline_running = False

    already_ran = any(
        os.path.exists(os.path.join(TAXI_DIR, f))
        for f in TAXI_OUTPUTS.keys()
    )

    # Detect if pyspark is available (local) or not (Streamlit Cloud)
    try:
        import pyspark
        is_local = True
    except ImportError:
        is_local = False

    if not is_local:
        # On Streamlit Cloud — show instructions, no button
        st.info("""
**ℹ️ Spark runs on your local machine, not on the cloud.**

PySpark needs 4GB+ RAM and Java — too heavy for Streamlit Cloud free tier.

Run this locally instead:
```bash
python src/spark_processor.py
```
Then push the output files:
```bash
git add data/taxi/
git commit -m "Add Spark outputs"
git push
```
Streamlit redeploys automatically and shows results below.
        """)
    else:
        # On local machine — show the run button
        if already_ran and not st.session_state.pipeline_running:
            st.success("✅ Pipeline already run. Scroll down for results.")

        col_btn, col_info = st.columns([1, 3])
        col_info.markdown("""
        - Downloads the NYC Yellow Taxi dataset from Kaggle (~1.5M rows)
        - Runs 7 Spark transformations
        - Writes one CSV per transformation to `data/taxi/`
        ⏱️ 2–5 minutes on first run.
        """)

        if not st.session_state.pipeline_running:
            run_clicked = col_btn.button("🚀 Run Spark Pipeline", type="primary")
        else:
            col_btn.warning("⏳ Running…")
            run_clicked = False

        if run_clicked:
            if not kaggle_ok():
                st.error("Kaggle credentials not found. Set up `~/.kaggle/kaggle.json` first.")
                st.stop()
            st.session_state.pipeline_running = True
            st.rerun()

        if st.session_state.pipeline_running:
            log_placeholder = st.empty()
            progress_bar    = st.progress(0)
            log_lines       = []

            process = subprocess.Popen(
                [sys.executable, "src/spark_processor.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            steps = ["Downloading", "Loading", "Outlier", "Window",
                     "Session", "Pivot", "Time Series", "Join", "Done"]
            step_i = 0

            for line in process.stdout:
                log_lines.append(line.rstrip())
                log_placeholder.code("\n".join(log_lines[-20:]), language="bash")
                for i, keyword in enumerate(steps):
                    if keyword.lower() in line.lower() and i > step_i:
                        step_i = i
                        progress_bar.progress(min(int((step_i / len(steps)) * 100), 99))

            process.wait()
            progress_bar.progress(100)
            st.session_state.pipeline_running = False

            if process.returncode == 0:
                st.success("✅ Pipeline complete!")
                st.cache_data.clear()
            else:
                st.error("❌ Pipeline failed. Check the log above.")
                st.stop()

    # ── Results ───────────────────────────────
    available_outputs = {
        fname: meta for fname, meta in TAXI_OUTPUTS.items()
        if os.path.exists(os.path.join(TAXI_DIR, fname))
    }

    if not available_outputs:
        st.info("Run the pipeline above to generate results.")
        st.stop()

    st.markdown("---")
    st.subheader("📂 Results")
    st.caption(f"{len(available_outputs)} / {len(TAXI_OUTPUTS)} outputs generated")

    for fname, (title, description) in TAXI_OUTPUTS.items():
        fpath = os.path.join(TAXI_DIR, fname)
        if not os.path.exists(fpath):
            continue

        df = load_csv(fpath)
        nc = num_cols(df)
        cc = cat_cols(df)

        with st.expander(f"{title} — `{fname}` ({len(df):,} rows)", expanded=False):
            st.markdown(f"_{description}_")

            # Auto-chart per output type
            try:
                if fname == "outliers.csv" and "outlier_reason" in df.columns:
                    fig = px.histogram(df, x="outlier_reason",
                                       color="outlier_reason", template="plotly_dark",
                                       title="Outlier types")
                    st.plotly_chart(fig, use_container_width=True)

                elif fname == "ranked.csv" and nc:
                    fig = px.bar(df.head(50), x=df.columns[1],
                                 y=nc[0], color=df.columns[1],
                                 template="plotly_dark",
                                 title="Daily revenue by type (first 50 dates)")
                    st.plotly_chart(fig, use_container_width=True)

                elif fname == "running_totals.csv":
                    if "cumulative_revenue" in df.columns and "date" in df.columns:
                        fig = px.line(df, x="date", y="cumulative_revenue",
                                      color=cc[1] if len(cc)>1 else None,
                                      template="plotly_dark",
                                      title="Cumulative revenue over time")
                        st.plotly_chart(fig, use_container_width=True)

                elif fname == "sessions.csv" and "session" in df.columns:
                    y = "trip_count" if "trip_count" in df.columns else (nc[0] if nc else None)
                    if y:
                        fig = px.bar(df, x="session", y=y,
                                     color="session", template="plotly_dark",
                                     title="Trips by time-of-day session")
                        st.plotly_chart(fig, use_container_width=True)

                elif fname == "pivot.csv" and "hour" in df.columns:
                    pivot_cols = [c for c in df.columns if c != "hour"]
                    if pivot_cols:
                        fig = go.Figure()
                        for col in pivot_cols:
                            fig.add_trace(go.Bar(name=col, x=df["hour"], y=df[col]))
                        fig.update_layout(barmode="stack", template="plotly_dark",
                                          title="Revenue by hour × session",
                                          xaxis_title="Hour of day",
                                          yaxis_title="Revenue ($)")
                        st.plotly_chart(fig, use_container_width=True)

                elif fname == "time_series.csv" and "date" in df.columns:
                    metrics = [c for c in ["trip_count","total_revenue"] if c in df.columns]
                    if metrics:
                        fig = go.Figure()
                        for m in metrics:
                            fig.add_trace(go.Scatter(x=df["date"], y=df[m],
                                                     mode="lines", name=m))
                        fig.update_layout(template="plotly_dark",
                                          title="Daily trends",
                                          hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)

                elif fname == "joined.csv":
                    if "vendor_name" in df.columns and "payment_name" in df.columns:
                        counts = df.groupby(["vendor_name","payment_name"]).size().reset_index(name="count")
                        fig = px.bar(counts, x="vendor_name", y="count",
                                     color="payment_name", barmode="group",
                                     template="plotly_dark",
                                     title="Trips by vendor and payment type")
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not render chart: {e}")

            # Data preview + download
            st.dataframe(df.head(100), use_container_width=True, height=250)
            st.download_button(
                label=f"⬇️ Download {fname}",
                data=df.to_csv(index=False).encode(),
                file_name=fname,
                mime="text/csv",
                key=f"dl_{fname}",
            )