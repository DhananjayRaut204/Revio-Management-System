from pathlib import Path
import hashlib

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except ImportError:
    px = None

from utils.classifier import load_model, predict
from utils.information_extractor import extract_info
from utils.priority_detector import detect_priority
from utils.response_generator import generate_response

ROOT = Path(__file__).resolve().parent
PAGES = ["Overview", "Analyze Inquiry", "Bulk Analysis", "Analytics", "About"]
PAGE_COPY = {
    "Overview": ("Your inquiry workspace", "A clear view of incoming requests and the next step."),
    "Analyze Inquiry": ("Analyze an inquiry", "Turn a project message into a clear, reviewable summary."),
    "Bulk Analysis": ("Analyze a CSV", "Review multiple inquiries together and export the results."),
    "Analytics": ("Inquiry analytics", "Explore the inquiries analyzed in this session."),
    "About": ("About this prototype", "What the model does, what it does not do, and where its data comes from."),
}

st.set_page_config(page_title="Cynbit Revio | Inquiry Intelligence", page_icon="🔷", layout="wide")
st.markdown("""
<style>
:root { --ink:#172b4d; --muted:#586b84; --line:#dce5f0; --blue:#2764d8; --teal:#11a8a5; --paper:#f0f5fc; }
.stApp { background:linear-gradient(135deg,#edf4fc 0%,#f7f8fc 52%,#edf7f8 100%); color:var(--ink); }
header[data-testid="stHeader"] { display:none !important; }
div[data-testid="stToolbar"], .stDeployButton, #MainMenu { display:none !important; }
footer { display:none !important; }
.block-container { max-width:1280px; padding-top:2rem; padding-bottom:3rem; }
.stApp p, .stApp label, .stApp [data-testid="stMarkdownContainer"] { font-size:16px; }
.stApp h1 { font-size:2.15rem; letter-spacing:-.025em; }
.stApp h2 { font-size:1.55rem; }
.stApp h3 { font-size:1.2rem; }
[data-testid="stMetricLabel"] p { font-size:14px; }
[data-testid="stMetricValue"] { font-size:29px; }
.stTextArea textarea { font-size:16px; line-height:1.55; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#f7faff 0%,#eef4fb 100%); border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top:1.4rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#52647d !important; }
.brand { display:flex; align-items:center; gap:11px; padding:4px 2px 26px; }
.brand-mark { width:44px; height:44px; border-radius:14px; display:grid; place-items:center; background:linear-gradient(135deg,#2764d8,#11a8a5); color:#fff; font-weight:800; font-size:14px; letter-spacing:-.06em; box-shadow:0 5px 14px #2764d833; }
.brand-name { color:var(--ink); font-weight:750; font-size:16px; line-height:1.15; }
.brand-sub { color:var(--muted); font-size:10px; letter-spacing:.12em; text-transform:uppercase; margin-top:4px; }
.section-label { color:#8996a8; font-size:10px; letter-spacing:.13em; text-transform:uppercase; font-weight:700; margin:4px 0 8px; }
.page-eyebrow { color:var(--blue); font-size:11px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin-bottom:7px; }
.page-subtitle { color:var(--muted); margin-top:-8px; margin-bottom:24px; }
.page-heading { position:relative; overflow:hidden; display:flex; align-items:center; gap:17px; padding:22px 26px; margin:4px 0 24px; border:1px solid #dce7f5; border-radius:18px; background:linear-gradient(115deg,#ffffff 0%,#eef5ff 72%,#eaf8f7 100%); box-shadow:0 8px 24px #284a7010; }
.page-heading:after { content:""; position:absolute; width:155px; height:155px; right:-38px; top:-70px; border-radius:50%; background:radial-gradient(circle,#3c91ed20 0%,#11a8a510 55%,transparent 70%); pointer-events:none; }
.heading-mark { flex:0 0 48px; height:48px; display:grid; place-items:center; border-radius:15px; color:#fff; font-weight:750; font-size:13px; letter-spacing:.04em; background:linear-gradient(145deg,#2764d8,#11a8a5); box-shadow:0 5px 14px #2764d82b; }
.page-heading h1 { margin:0 0 4px; font-size:1.9rem; color:var(--ink); }
.page-heading .page-subtitle { margin:0; color:#586b84; }
.stApp h2 { color:var(--ink); }
[data-testid="stVerticalBlockBorderWrapper"] { background:rgba(255,255,255,.88); border-color:#dce5f0 !important; border-radius:16px !important; box-shadow:0 5px 18px #284a700a; }
[data-testid="stFileUploader"] { background:#fff; border:1px dashed #b9cbe1; border-radius:15px; padding:14px; }
[data-testid="stDataFrame"] { border:1px solid #e1e8f1; border-radius:12px; overflow:hidden; }
[data-testid="stPlotlyChart"] { background:#fff; border-radius:12px; }
.empty-state { padding:24px 26px; border-radius:16px; border:1px solid #dce7f5; background:linear-gradient(110deg,#fff,#eff6ff); }
.empty-state h3 { margin-top:0; color:#203b62; }
.kicker { color:#3470bd; font-size:11px; font-weight:750; letter-spacing:.12em; text-transform:uppercase; }

.welcome { background:linear-gradient(115deg,#2764d8,#168f9e); border-radius:16px; padding:28px 30px; color:#fff; margin:8px 0 22px; }
.welcome h2 { color:#fff; font-size:24px; margin:0 0 7px; }
.welcome p { color:#d7e5f5; margin:0; max-width:720px; line-height:1.55; }
.welcome-tag { color:#d9f5f4; font-size:11px; letter-spacing:.12em; text-transform:uppercase; margin-bottom:10px; }
.stMetric { background:#fff; border:1px solid var(--line); border-radius:13px; padding:14px 16px; }
[data-testid="stMetricLabel"] p { color:var(--muted); font-size:14px; }
[data-testid="stMetricValue"] { color:var(--ink); font-size:29px; }
.stButton > button, .stDownloadButton > button { border-radius:9px; font-weight:650; min-height:42px; }
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button[kind="primary"] { background:var(--blue) !important; border-color:var(--blue) !important; color:#fff !important; }
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button:hover { background:#1d50b4 !important; border-color:#1d50b4 !important; }
.stTextArea textarea, .stTextInput input { background:#fff !important; color:#1e2e46 !important; border:1px solid #cbd7e6 !important; border-radius:10px !important; font-size:16px !important; line-height:1.55 !important; }
.stTextArea textarea::placeholder, .stTextInput input::placeholder { color:#718198 !important; opacity:1; }
[data-testid="stWidgetLabel"] p { color:#344761 !important; font-weight:600; }
[data-testid="stCaptionContainer"] { color:#52647d !important; font-size:14px !important; }
input[type="radio"] { accent-color:var(--blue); }
[data-testid="stRadio"] label { color:#344761 !important; font-size:15px !important; }
div[data-testid="stForm"] { background:#fff; border:1px solid var(--line); border-radius:14px; padding:18px 20px; }
div[data-testid="stFileUploader"] section { border-radius:12px; }
hr { border-color:var(--line); }
.small-note { color:var(--muted); font-size:12px; }
.pill { display:inline-block; border-radius:999px; background:#edf3ff; color:#2457a8; padding:5px 10px; font-size:11px; font-weight:650; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_model()

try:
    model = get_model()
except Exception as exc:
    st.error(f"The model could not be loaded: {exc}")
    st.stop()

if "records" not in st.session_state:
    st.session_state.records = []
if "latest_analysis" not in st.session_state:
    st.session_state.latest_analysis = None

model_name = "Naive Bayes (offline)" if model.__class__.__name__ == "SimpleNB" else "Logistic Regression"

def clear_session():
    st.session_state.records = []
    st.session_state.latest_analysis = None
    st.session_state.demo_loaded = False
    st.session_state.last_bulk_fingerprint = None


with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">CR</div><div><div class="brand-name">Cynbit Revio</div><div class="brand-sub">Inquiry intelligence</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">Workspace</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", PAGES, label_visibility="collapsed", key="Navigation")
    st.divider()
    st.markdown('<div class="section-label">Prototype status</div>', unsafe_allow_html=True)
    st.caption("Local demonstration")
    st.caption("Synthetic training examples")
    st.caption("Human review recommended")
    st.divider()
    st.button("Clear session", use_container_width=True, disabled=not st.session_state.records, on_click=clear_session)


def analyze(text):
    classification = predict(text, model)
    info = extract_info(text)
    priority = detect_priority(text, info)
    reply = generate_response(text, classification["category"], info, priority)
    return {
        "inquiry": text,
        "category": classification["category"],
        "model_confidence": round(classification["confidence"], 4),
        "priority": priority["priority"],
        "priority_reason": "; ".join(priority["reasons"]),
        "budget": info["budget"],
        "timeline": info["timeline"],
        "requested_services": "; ".join(info["requested_services"]),
        "requirements": "; ".join(info["requirements"]),
        "response_suggestion": reply,
        "confidence_warning": classification["warning"],
    }


def load_sample_dashboard():
    sample_df = pd.read_csv(ROOT / "data" / "sample_inquiries.csv")
    preferred = ["AI/ML", "Web Development", "Cloud Solutions"]
    examples = sample_df[sample_df["category"].isin(preferred)].drop_duplicates("category")
    for _, sample in examples.iterrows():
        row = analyze(str(sample["inquiry"]))
        row["record_source"] = "Synthetic demo"
        st.session_state.records.append(row)
    st.session_state["demo_loaded"] = True


def open_manual_analysis():
    st.session_state["Navigation"] = "Analyze Inquiry"


def page_header(page_name):
    title, subtitle = PAGE_COPY[page_name]
    marks = {"Overview": "CR", "Analyze Inquiry": "AI", "Bulk Analysis": "CSV", "Analytics": "CH", "About": "CR"}
    mark = marks[page_name]
    if page_name == "Overview":
        st.markdown(f'<div class="page-eyebrow">Cynbit Revio / Workspace</div><h1>{title}</h1><div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="page-heading"><div class="heading-mark">{mark}</div><div><div class="kicker">Cynbit Revio / Workspace</div><h1>{title}</h1><div class="page-subtitle">{subtitle}</div></div></div>', unsafe_allow_html=True)


def render_result(row):
    st.markdown("#### Analysis result")
    col1, col2, col3 = st.columns(3)
    col1.metric("Suggested category", row["category"])
    col2.metric("Model confidence", f'{row["model_confidence"]:.0%}')
    col3.metric("Priority", row["priority"])
    if row["confidence_warning"]:
        st.warning(row["confidence_warning"])
    st.progress(row["model_confidence"], text="Classifier confidence (review uncertain results)")
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### Project details")
            st.write("**Budget**", row["budget"])
            st.write("**Timeline**", row["timeline"])
            st.write("**Requested services**", row["requested_services"])
            st.write("**Requirements detected**", row["requirements"])
        with st.container(border=True):
            st.markdown("#### Priority reasoning")
            st.write(row["priority_reason"])
            st.caption("Priority is a transparent rule-based heuristic, separate from the ML classifier.")
    with right:
        with st.container(border=True):
            st.markdown("#### Suggested reply")
            st.caption("Template-based draft. Review and edit before sending.")
            st.text_area("Reply draft", row["response_suggestion"], height=220, label_visibility="collapsed")


page_header(page)

if page == "Overview":
    st.markdown(
        '<div class="welcome"><div class="welcome-tag">A clearer start to every conversation</div><h2>Understand the request. Plan the follow-up.</h2><p>Classify project inquiries, capture the details that were mentioned, and prepare a thoughtful first reply. Every suggestion stays reviewable.</p></div>',
        unsafe_allow_html=True,
    )
    rows = st.session_state.records
    categories = len(getattr(model, "classes_", []))
    a, b, c, d = st.columns(4)
    a.metric("Inquiries analyzed", len(rows))
    b.metric("High priority", sum(item["priority"] == "High" for item in rows))
    c.metric("Needs review", sum(bool(item.get("confidence_warning")) for item in rows))
    d.metric("Service categories", categories)
    st.markdown("### Get started")
    start, info = st.columns([1, 1], gap="large")
    with start:
        with st.container(border=True):
            st.markdown("#### Analyze one inquiry")
            st.write("Paste a project message to see its category, key details, and a draft response.")
            st.button("Open inquiry analysis", type="primary", use_container_width=True, on_click=open_manual_analysis)
    with info:
        with st.container(border=True):
            st.markdown("#### Review a batch")
            st.write("Upload a CSV with an `inquiry` column and download the analyzed file.")
            st.caption(f"Current classifier: {model_name}. Training examples are synthetic.")
    if rows:
        if st.session_state.get("demo_loaded"):
            st.info("Demo records are synthetic examples included with the project. They are not real customer inquiries.")
        st.markdown("### Activity in this session")
        chart_col, table_col = st.columns([.85, 1.4], gap="large")
        recent = pd.DataFrame(rows[-5:])[ ["category", "priority", "budget", "timeline", "inquiry"] ]
        counts = pd.DataFrame(rows)["category"].value_counts().rename_axis("Category").reset_index(name="Inquiries")
        with chart_col:
            with st.container(border=True):
                st.markdown("#### Inquiries by category")
                if px:
                    fig = px.bar(counts, x="Inquiries", y="Category", orientation="h", color="Category", template="plotly_white")
                    fig.update_layout(showlegend=False, margin=dict(l=0, r=5, t=5, b=5), height=250)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.bar_chart(counts.set_index("Category"))
        with table_col:
            with st.container(border=True):
                st.markdown("#### Recent inquiries")
                st.dataframe(recent, width="stretch", hide_index=True)
    else:
        st.info("Your workspace is ready. Load a few sample inquiries to preview the dashboard, or start with your own message.")
        st.button("Load sample dashboard", type="primary", on_click=load_sample_dashboard)
        st.caption("Sample messages are synthetic and clearly labeled. They help you preview the charts without using customer data.")
        st.markdown("### What happens to an inquiry?")
        step1, step2, step3 = st.columns(3)
        step1.markdown("**01 · Classify**\n\nA text model suggests a service category and confidence score.")
        step2.markdown("**02 · Extract**\n\nSimple rules find budget, timeline, and project details.")
        step3.markdown("**03 · Review**\n\nA person checks the result and edits the suggested reply.")

elif page == "Analyze Inquiry":
    with st.container(border=True):
        st.markdown("#### Start with the customer’s message")
        st.caption("The classifier suggests a category. The details and priority are based on visible rules, so you can check how each result was produced.")
        with st.form("manual_inquiry", border=False):
            inquiry_text = st.text_area(
                "Inquiry message",
                height=160,
                placeholder="We need an AI dashboard connected to our CRM. Budget is $20,000 and we hope to launch in six weeks.",
            )
            submitted = st.form_submit_button("Analyze inquiry", type="primary")
    if submitted:
        if not inquiry_text.strip():
            st.warning("Add an inquiry message to analyze.")
            st.session_state.latest_analysis = None
        else:
            result = analyze(inquiry_text)
            result["record_source"] = "Manual entry"
            st.session_state.latest_analysis = result
            st.session_state.records.append(result)
    if st.session_state.latest_analysis:
        render_result(st.session_state.latest_analysis)
    else:
        st.caption("Your inquiry is processed locally with the included model and extraction rules.")

elif page == "Bulk Analysis":
    with st.container(border=True):
        st.markdown("#### Bring in a batch of inquiries")
        st.caption("Upload a CSV with a column named `inquiry`. Other columns are kept in the output, including any original labels.")
        sample_csv = (ROOT / "data" / "sample_inquiries.csv").read_bytes()
        st.download_button("Download sample CSV", sample_csv, file_name="sample_inquiries.csv", mime="text/csv")
        upload = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")
    if upload is not None:
        try:
            input_df = pd.read_csv(upload)
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
            st.stop()
        if "inquiry" not in input_df.columns:
            st.error("This file needs a column named `inquiry`.")
        elif input_df.empty:
            st.info("The CSV has a header but no inquiry rows yet.")
        else:
            results = []
            for value in input_df["inquiry"].fillna(""):
                if not str(value).strip():
                    results.append({
                        "inquiry": "", "category": "Other", "model_confidence": 0.0,
                        "priority": "Low", "priority_reason": "Empty inquiry",
                        "budget": "Not stated", "timeline": "Not stated",
                        "requested_services": "Needs clarification", "requirements": "None detected",
                        "response_suggestion": "No reply generated for an empty inquiry.",
                        "confidence_warning": "Empty inquiry; review manually.", "record_source": "CSV upload",
                    })
                else:
                    row = analyze(str(value))
                    row["record_source"] = "CSV upload"
                    results.append(row)
            generated = pd.DataFrame(results).drop(columns=["inquiry"])
            source_df = input_df.reset_index(drop=True).copy()
            overlaps = [name for name in generated.columns if name in source_df.columns]
            source_df = source_df.rename(columns={name: f"input_{name}" for name in overlaps})
            output_df = pd.concat([source_df, generated], axis=1)
            fingerprint = hashlib.sha256(upload.getvalue()).hexdigest()
            if st.session_state.get("last_bulk_fingerprint") != fingerprint:
                st.session_state.records.extend(results)
                st.session_state.last_bulk_fingerprint = fingerprint
            st.success(f"Analyzed {len(output_df)} inquiries.")
            st.dataframe(output_df, width="stretch", hide_index=True)
            st.download_button(
                "Download analyzed CSV",
                output_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="analyzed_inquiries.csv",
                mime="text/csv",
                type="primary",
            )

elif page == "Analytics":
    rows = st.session_state.records
    if not rows:
        with st.container(border=True):
            st.markdown('<div class="empty-state"><div class="kicker">Your workspace is ready</div><h3>Charts appear after the first analysis</h3><p>Use your own inquiry or load the clearly marked demo examples from Overview. Nothing here is pre-filled as real customer activity.</p></div>', unsafe_allow_html=True)
            st.button("Analyze an inquiry", type="primary", on_click=open_manual_analysis)
    else:
        df = pd.DataFrame(rows)
        metric_a, metric_b = st.columns(2)
        metric_a.metric("Total inquiries", len(df))
        metric_b.metric("Most common category", df["category"].mode().iloc[0])
        chart_a, chart_b = st.columns(2, gap="large")
        category_counts = df["category"].value_counts().rename_axis("Category").reset_index(name="Inquiries")
        priority_counts = df["priority"].value_counts().rename_axis("Priority").reset_index(name="Inquiries")
        with chart_a:
            with st.container(border=True):
                st.markdown("#### By service category")
                if px:
                    fig = px.bar(category_counts, x="Category", y="Inquiries", color="Category", template="plotly_white")
                    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=300)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.bar_chart(category_counts.set_index("Category"))
        with chart_b:
            with st.container(border=True):
                st.markdown("#### By priority")
                if px:
                    fig = px.bar(priority_counts, x="Priority", y="Inquiries", color="Priority", template="plotly_white")
                    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=300)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.bar_chart(priority_counts.set_index("Priority"))
        st.markdown("#### Most requested services")
        service_counts = df["requested_services"].str.split("; ").explode().value_counts().rename_axis("Service").reset_index(name="Requests")
        if px:
            fig = px.bar(service_counts, x="Service", y="Requests", color="Service", template="plotly_white")
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=280)
            st.plotly_chart(fig, width="stretch")
        else:
            st.bar_chart(service_counts.set_index("Service"))

elif page == "About":
    st.markdown("This is an independent assessment prototype. The Cynbit Revio name is a project label; the app is not an official Cynbit or Revio feature and is not connected to Cynbit systems.")
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### What the app does")
            st.markdown("- Suggests a service category from an inquiry message.\n- Extracts budget, timeline, services, and selected requirements.\n- Explains a rule-based priority suggestion.\n- Drafts a reply for a person to review and edit.")
        with st.container(border=True):
            st.markdown("#### Model in this build")
            st.markdown(f"**{model_name}**\n\nRuns locally on CPU. The standard training path uses TF-IDF and Logistic Regression when scikit-learn is installed.")
    with right:
        with st.container(border=True):
            st.markdown("#### Data and reliability")
            st.markdown("- Training examples are synthetic; no customer data is included.\n- Confidence is a model score, not a guarantee.\n- Priority comes from transparent rules, not ML.\n- A person should review every classification and draft.")
        with st.container(border=True):
            st.markdown("#### Current limits")
            st.markdown("This prototype does not store inquiries after the session, connect to a CRM, or send messages. The sample evaluation score is not evidence of real-world performance.")
    st.caption("For the data, model setup, and research sources, see the two reports in the `reports` folder.")
