import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
from dateutil.parser import parse
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, PageBreak
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from wordcloud import WordCloud

# ML / NLP libs
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pyod.models.knn import KNN
import spacy
from textblob import TextBlob
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # deterministic langdetect

# Try to load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

# -----------------------------
# Validation Helpers
# -----------------------------
def is_valid_date(value):
    try:
        parse(str(value), fuzzy=False)
        return True
    except Exception:
        return False

def is_likely_date_column(col_name, sample_values):
    keywords = ['date', 'time', 'created', 'updated', 'timestamp', 'dob', 'joined', 'registered']
    col_name_match = any(k in col_name.lower() for k in keywords)
    try:
        if pd.api.types.is_numeric_dtype(sample_values):
            return False
    except Exception:
        pass
    sample_valid_dates = sum(is_valid_date(val) for val in sample_values if pd.notnull(val))
    sample_ratio = sample_valid_dates / (len(sample_values) if len(sample_values) > 0 else 1)
    return col_name_match or sample_ratio > 0.5

def is_valid_email(value):
    pattern = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(value))) if pd.notnull(value) else False

def is_valid_phone(value):
    pattern = r'^\+?[0-9\s\-\(\)]{7,20}$'
    return bool(re.match(pattern, str(value))) if pd.notnull(value) else False

def is_likely_email_column(col_name):
    return 'email' in col_name.lower()

def is_likely_phone_column(col_name):
    keywords = ['phone', 'mobile', 'contact', 'tel']
    return any(k in col_name.lower() for k in keywords)

# -----------------------------
# NLP Helpers
# -----------------------------
def analyze_text_sample(series, sample_size=100):
    results = {
        "language_counts": {},
        "sentiment_avg": None,
        "entities": {},
        "avg_length": None,
        "sample_size_used": 0,
        "top_words": {}
    }

    texts = series.dropna().astype(str)
    if texts.empty:
        return results

    sample = texts.head(sample_size)
    sentiments = []
    lengths = []
    lang_counts = {}
    entity_counts = {}
    all_words = []

    for t in sample:
        try:
            lang = detect(t)
        except Exception:
            lang = "unknown"
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

        try:
            blob = TextBlob(t)
            sentiments.append(blob.sentiment.polarity)
        except Exception:
            pass

        lengths.append(len(t))

        tokens = re.findall(r"\b\w+\b", t.lower())
        all_words.extend(tokens)

        if nlp is not None:
            try:
                doc = nlp(t)
                for ent in doc.ents:
                    entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1
            except Exception:
                pass

    results["language_counts"] = lang_counts
    results["sentiment_avg"] = sum(sentiments) / len(sentiments) if sentiments else None
    results["entities"] = entity_counts
    results["avg_length"] = sum(lengths) / len(lengths) if lengths else None
    results["sample_size_used"] = len(sample)
    results["top_words"] = dict(Counter(all_words).most_common(20))
    return results

# -----------------------------
# ML Helpers
# -----------------------------
def run_isolation_forest(df_numeric, contamination=0.05):
    if df_numeric.shape[0] == 0 or df_numeric.shape[1] == 0:
        return pd.Series([], dtype=bool)
    X = df_numeric.fillna(df_numeric.median())
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(Xs)
    return pd.Series(preds == -1, index=df_numeric.index)

def run_pyod_knn(df_numeric, contamination=0.05):
    if df_numeric.shape[0] == 0 or df_numeric.shape[1] == 0:
        return pd.Series([], dtype=bool)
    X = df_numeric.fillna(df_numeric.median())
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = KNN(contamination=contamination)
    clf.fit(Xs)
    preds = clf.labels_
    return pd.Series(preds == 1, index=df_numeric.index)

# -----------------------------
# Main Validation
# -----------------------------
def validate_data(df, enable_nlp=False, enable_ml=False, contamination=0.05):
    report = {"dates": {}, "emails": {}, "phones": {}, "outliers": {}, "unique_counts": {}, "nlp": {}, "ml_outliers": {}}

    for col in df.columns:
        report["unique_counts"][col] = int(df[col].nunique())

    for col in df.columns:
        sample_values = df[col].dropna().head(20)
        if is_likely_date_column(col, sample_values):
            null_count = int(df[col].isnull().sum())
            invalid_count = int(sum(not is_valid_date(val) for val in df[col] if pd.notnull(val)))
            valid_count = int(len(df[col]) - invalid_count - null_count)
            report["dates"][col] = {"valid": valid_count, "invalid": invalid_count, "missing": null_count}

    for col in df.columns:
        if is_likely_email_column(col):
            null_count = int(df[col].isnull().sum())
            invalid_count = int(sum(not is_valid_email(val) for val in df[col] if pd.notnull(val)))
            duplicates_count = int(df[col][df[col].duplicated(keep=False)].nunique())
            valid_count = int(len(df[col]) - invalid_count - null_count)
            report["emails"][col] = {"valid": valid_count, "invalid": invalid_count, "missing": null_count, "duplicates": duplicates_count}

    for col in df.columns:
        if is_likely_phone_column(col):
            null_count = int(df[col].isnull().sum())
            invalid_count = int(sum(not is_valid_phone(val) for val in df[col] if pd.notnull(val)))
            duplicates_count = int(df[col][df[col].duplicated(keep=False)].nunique())
            valid_count = int(len(df[col]) - invalid_count - null_count)
            report["phones"][col] = {"valid": valid_count, "invalid": invalid_count, "missing": null_count, "duplicates": duplicates_count}

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_count = int(df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0])
        report["outliers"][col] = outliers_count

    if enable_nlp:
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            try:
                avg_len = df[col].dropna().astype(str).str.len().mean()
            except Exception:
                avg_len = 0
            if avg_len and avg_len > 10:
                report["nlp"][col] = analyze_text_sample(df[col], sample_size=200)

    if enable_ml:
        if len(numeric_cols) > 0:
            df_numeric = df[numeric_cols].copy()
            try:
                iso_flags = run_isolation_forest(df_numeric, contamination=contamination)
                knn_flags = run_pyod_knn(df_numeric, contamination=contamination)
                report["ml_outliers"]["isolation_forest_count"] = int(iso_flags.sum())
                report["ml_outliers"]["knn_count"] = int(knn_flags.sum())
                report["ml_outliers"]["isolation_forest_indices"] = iso_flags[iso_flags].index.tolist()[:20]
                report["ml_outliers"]["knn_indices"] = knn_flags[knn_flags].index.tolist()[:20]
            except Exception as e:
                report["ml_outliers"]["error"] = str(e)
        else:
            report["ml_outliers"]["note"] = "No numeric columns for ML-based detection."

    return report

# -----------------------------
# Utility: Quality Scores
# -----------------------------
def compute_quality_scores(df, report):
    scores = {}
    for col in df.columns:
        score = 100
        try:
            missing = df[col].isnull().sum()
            pct_missing = missing / len(df) if len(df) > 0 else 0
            score -= pct_missing * 50
        except Exception:
            pass

        if col in report.get("emails", {}):
            inval = report["emails"][col]["invalid"]
            total = report["emails"][col]["valid"] + report["emails"][col]["invalid"] + report["emails"][col]["missing"]
            if total > 0:
                score -= (inval / total) * 30

        if col in report.get("phones", {}):
            inval = report["phones"][col]["invalid"]
            total = report["phones"][col]["valid"] + report["phones"][col]["invalid"] + report["phones"][col]["missing"]
            if total > 0:
                score -= (inval / total) * 20

        if col in report.get("outliers", {}):
            count_out = report["outliers"][col]
            pct_out = count_out / len(df) if len(df) > 0 else 0
            score -= pct_out * 20

        if col in report.get("nlp", {}):
            entities = report["nlp"][col].get("entities", {})
            if entities:
                score += min(10, sum(entities.values()) * 0.5)

        score = max(0, min(100, score))
        scores[col] = round(score, 2)
    return scores

# -----------------------------
# Streamlit App
# -----------------------------
st.sidebar.title("⚙️ Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xls", "xlsx"])

st.sidebar.markdown("### Display Settings")
show_raw = st.sidebar.checkbox("Show Raw Data", value=True)
show_summary = st.sidebar.checkbox("Show Validation Summary", value=True)
show_charts = st.sidebar.checkbox("Show Charts", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### Advanced (ML/NLP)")
enable_nlp = st.sidebar.checkbox("Enable NLP", value=True)
enable_ml = st.sidebar.checkbox("Enable ML Outlier Detection", value=True)
contamination = st.sidebar.slider("Outlier contamination", min_value=0.01, max_value=0.2, value=0.05, step=0.01)

st.title("📊 Data Validator + 🧠 NLP + 🔍 ML Outliers")

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, on_bad_lines="skip")
        else:
            df = pd.read_excel(uploaded_file)

        st.success(" File loaded successfully!")

        if show_raw:
            st.write("### Raw Data")
            st.dataframe(df)

        report = validate_data(df, enable_nlp=enable_nlp, enable_ml=enable_ml, contamination=contamination)
        col_scores = compute_quality_scores(df, report)

        if show_summary:
            st.write("### Validation Summary")
            st.json(report)
            st.write("### Column Quality Scores")
            st.dataframe(pd.DataFrame.from_dict(col_scores, orient="index", columns=["Quality Score"]).sort_values("Quality Score"))

        def plot_report(report_section, title):
            if report_section:
                df_chart = pd.DataFrame(report_section).T.reset_index()
                df_chart.rename(columns={"index": "column"}, inplace=True)
                fig, ax = plt.subplots(figsize=(10, 5))
                ycols = [c for c in df_chart.columns if c != "column"]
                if ycols:
                    df_chart.plot(x="column", y=ycols, kind="bar", stacked=True, ax=ax, colormap="Set2")
                    ax.set_ylabel("Count")
                    ax.set_title(title)
                    for container in ax.containers:
                        try:
                            ax.bar_label(container, label_type='center', fontsize=9, weight='bold')
                        except Exception:
                            pass
                    st.pyplot(fig)

        if show_charts:
            plot_report(report.get("dates"), "Date Validation")
            plot_report(report.get("emails"), "Email Validation")
            plot_report(report.get("phones"), "Phone Validation")
            if report.get("ml_outliers"):
                st.write("### ML Outlier Summary")
                st.json(report["ml_outliers"])

        if report.get("outliers"):
            st.write("### Numeric Outliers (IQR)")
            outliers_df = pd.DataFrame(list(report["outliers"].items()), columns=["Column", "Outliers"])
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            bars = ax2.bar(outliers_df["Column"], outliers_df["Outliers"])
            ax2.set_ylabel("Count")
            ax2.set_title("Outliers per Numeric Column")
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height, str(int(height)), ha='center', va='bottom', fontsize=9, fontweight='bold')
            st.pyplot(fig2)

        if enable_nlp and report.get("nlp"):
            st.write("### NLP Analysis of Text Columns")
            for col, nlp_res in report["nlp"].items():
                st.write(f"#### Column: {col}")
                st.write(f"- Sample rows analyzed: {nlp_res.get('sample_size_used')}")
                st.write(f"- Avg text length: {nlp_res.get('avg_length')}")
                st.write(f"- Language counts: {nlp_res.get('language_counts')}")
                st.write(f"- Sentiment average: {nlp_res.get('sentiment_avg')}")
                st.write(f"- Entities found (top types): {nlp_res.get('entities')}")

                if nlp_res.get("top_words"):
                    st.write("#### 🔤 Most Frequent Words")
                    df_words = pd.DataFrame(list(nlp_res["top_words"].items()), columns=["Word", "Count"])
                    st.bar_chart(df_words.set_index("Word"))

                st.markdown("---")

        # 🔍 Search & Duplicates
        st.write("### Search ")
        all_columns = df.columns.tolist()
        selected_col = st.selectbox("Select Column to Analyze ", all_columns)
        col_series = df[selected_col]
        duplicates_count = int(col_series[col_series.duplicated(keep=False)].nunique())
        missing_count = int(col_series.isnull().sum())
        unique_count = int(col_series.nunique())
        st.write(f"**Unique:** {unique_count}, **Missing:** {missing_count}, **Duplicates:** {duplicates_count}")

        if duplicates_count > 0:
            show_dup = st.checkbox(f"Show Duplicated Rows in '{selected_col}' ")
            if show_dup:
                duplicates_df = df[col_series.duplicated(keep=False)].sort_values(by=selected_col)
                st.dataframe(duplicates_df.head(5))
                st.write("### Download Duplicates")
                download_format = st.radio("Choose format", ["CSV", "PDF"])
                if download_format == "CSV":
                    buffer = BytesIO()
                    duplicates_df.to_csv(buffer, index=False)
                    buffer.seek(0)
                    st.download_button(
                        label=f"Download Duplicates CSV for '{selected_col}'",
                        data=buffer,
                        file_name=f"duplicates_{selected_col}.csv",
                        mime="text/csv"
                    )
                else:
                    buffer = BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4))
                    styles = getSampleStyleSheet()
                    df_t = duplicates_df.transpose().reset_index()
                    df_t.columns = ["Field"] + [f"Row {i}" for i in range(1, len(df_t.columns))]
                    data = [df_t.columns.tolist()] + df_t.astype(str).values.tolist()
                    wrapped_data = []
                    for row in data:
                        new_row = [Paragraph(str(cell), styles["Normal"]) for cell in row]
                        wrapped_data.append(new_row)
                    table = Table(wrapped_data, repeatRows=1)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.gray),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('GRID', (0,0), (-1,-1), 0.25, colors.black),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0,0), (-1,-1), 7),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ]))
                    elements = [table, PageBreak()]
                    doc.build(elements)
                    buffer.seek(0)
                    st.download_button(
                        label=f"Download Duplicates PDF for '{selected_col}'",
                        data=buffer,

                        file_name=f"duplicates_{selected_col}.pdf",
                        mime="application/pdf"
                    )

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Upload a CSV or Excel file using the sidebar to begin validation.")

