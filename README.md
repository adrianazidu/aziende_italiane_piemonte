# 📊 PySpark CSV Analyzer

A full data pipeline that processes large CSV files with **PySpark** and visualizes results with an interactive **Streamlit** dashboard.

---

## 🗂️ Project Structure

```
spark-project/
├── data/                        # CSVs (gitignored except small sample)
├── src/
│   ├── spark_processor.py       # Main PySpark pipeline
│   └── transformations.py       # Reusable transform functions
├── app.py                       # Streamlit dashboard
├── generate_sample_data.py      # Generate test data
├── requirements.txt
└── README.md
```

---

## ⚡ Quickstart

### 1. Install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Java required:** PySpark needs Java 8 or 11.
> Check with `java -version`. Install from https://adoptium.net if needed.

### 2. Generate sample data (optional)
```bash
python generate_sample_data.py --rows 100000
```

### 3. Run the PySpark pipeline
```bash
python src/spark_processor.py --input data/sample_data.csv
```
Outputs written to `data/`:
- `summary.csv` — aggregated stats by category
- `time_series.csv` — monthly trends
- `top_n.csv` — top 20 records
- `cleaned_sample.csv` — cleaned sample (up to 5,000 rows)

### 4. Launch the dashboard
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🌐 Deploy on Streamlit Cloud

1. Push your repo to GitHub (CSV files are gitignored — outputs must be pre-generated or loaded from a URL)
2. Go to https://share.streamlit.io
3. Connect your repo → set **Main file**: `app.py`
4. Click **Deploy**

> **Tip:** For cloud deployment, pre-run the Spark pipeline locally, commit the output CSVs (remove them from `.gitignore`), then re-deploy.

---

## 🔧 Customizing for Your Data

| File | What to change |
|------|----------------|
| `src/transformations.py` | Update `compute_summary()` to use your actual column names |
| `src/spark_processor.py` | Add/remove pipeline steps |
| `app.py` | Add new pages, charts, or filters |

---

## 📦 Tech Stack

- [PySpark](https://spark.apache.org/docs/latest/api/python/) — distributed data processing
- [Streamlit](https://streamlit.io) — interactive web dashboard
- [Plotly](https://plotly.com/python/) — interactive charts
- [Pandas](https://pandas.pydata.org) — bridge between Spark and Streamlit
