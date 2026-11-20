from pathlib import Path
import pandas as pd
try:
 import plotly.express as px
except ImportError:
 px=None
import streamlit as st
from utils.classifier import load_model,predict
from utils.information_extractor import extract_info
from utils.priority_detector import detect_priority
from utils.response_generator import generate_response
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title="Cynbit Inquiry Intelligence",page_icon="◈",layout="wide")
st.markdown("""<style> .stApp{background:#f5f7fb}.block-container{padding-top:2rem;max-width:1400px}.hero{padding:1.5rem 1.8rem;background:linear-gradient(120deg,#10213d,#1e4966);color:white;border-radius:18px;margin-bottom:1.2rem}.hero h1{color:white;margin:0}.muted{color:#637083}.stMetric{background:white;padding:14px;border:1px solid #e8edf3;border-radius:12px}</style>""",unsafe_allow_html=True)
@st.cache_resource
def model_load(): return load_model()
try: model=model_load()
except Exception as e: st.error(f"Model unavailable: {e}"); st.stop()
if "records" not in st.session_state: st.session_state.records=[]
page=st.sidebar.radio("Workspace",["Overview","Analyze Inquiry","Bulk Inquiry Analysis","Analytics","About the Project"]); st.sidebar.caption("Local prototype • synthetic training data")
def analyze(text):
 c=predict(text,model); info=extract_info(text); pri=detect_priority(text,info); response=generate_response(text,c["category"],info,pri); return {"inquiry":text,"category":c["category"],"model_confidence":round(c["confidence"],4),"priority":pri["priority"],"priority_reason":"; ".join(pri["reasons"]),"budget":info["budget"],"timeline":info["timeline"],"requested_services":"; ".join(info["requested_services"]),"requirements":"; ".join(info["requirements"]),"response_suggestion":response,"confidence_warning":c["warning"]}
def render_result(row):
 a,b,c=st.columns(3); a.metric("Predicted category",row["category"]); b.metric("Model confidence",f"{row['model_confidence']:.0%}"); c.metric("Priority (rule-based)",row["priority"])
 if row["confidence_warning"]: st.warning(row["confidence_warning"])
 l,r=st.columns(2)
 with l:
  st.subheader("Extracted details"); st.write("**Budget:**",row["budget"]); st.write("**Timeline:**",row["timeline"]); st.write("**Requested services:**",row["requested_services"]); st.write("**Requirements:**",row["requirements"]); st.caption("Priority reason: "+row["priority_reason"]+". Priority is a transparent rule-based heuristic, not an ML score.")
 with r: st.subheader("Suggested reply"); st.text_area("Template-based draft",row["response_suggestion"],height=270)
if page=="Overview":
 st.markdown('<div class="hero"><h1>Cynbit Inquiry Intelligence</h1><p>Triage incoming project inquiries with an explainable text classifier and transparent extraction rules.</p></div>',unsafe_allow_html=True)
 rows=st.session_state.records; a,b,c,d=st.columns(4); a.metric("Analyzed inquiries",len(rows)); b.metric("High priority",sum(x["priority"]=="High" for x in rows)); b2=model.classes_; c.metric("Service categories",len(b2)); d.metric("Model", "TF-IDF + Logistic Regression")
 st.subheader("Get started"); st.write("Analyze a message manually or upload a CSV with an `inquiry` column. Predictions are experimental and trained only on labeled synthetic examples.")
 if rows: st.dataframe(pd.DataFrame(rows)[["category","priority","budget","timeline","inquiry"]],use_container_width=True,hide_index=True)
elif page=="Analyze Inquiry":
 st.title("Analyze an inquiry"); txt=st.text_area("Paste a prospect inquiry",height=180,placeholder="We need an AI dashboard integrated with our CRM. Budget is $20,000; delivery in 6 weeks.")
 if st.button("Analyze inquiry",type="primary"):
  if not txt.strip(): st.warning("Please enter an inquiry before analysis.")
  else:
   row=analyze(txt); st.session_state.records.append(row); render_result(row)
elif page=="Bulk Inquiry Analysis":
 st.title("Bulk inquiry analysis"); st.write("Upload a CSV file containing an `inquiry` column. Results remain in this browser session."); up=st.file_uploader("CSV file",type=["csv"])
 if up:
  try: df=pd.read_csv(up)
  except Exception as e: st.error(f"Could not read CSV: {e}"); st.stop()
  if "inquiry" not in df.columns: st.error("CSV must include a column named `inquiry`.")
  elif df.empty: st.warning("The uploaded CSV has no rows.")
  else:
   out=[]
   for val in df["inquiry"].fillna(""):
    if not str(val).strip(): out.append({"inquiry":"","category":"Other","model_confidence":0.0,"priority":"Low","priority_reason":"empty input","budget":"Not stated","timeline":"Not stated","requested_services":"Needs clarification","requirements":"None detected","response_suggestion":"No response generated for empty input.","confidence_warning":"Empty inquiry; review manually."})
    else: out.append(analyze(str(val)))
   generated=pd.DataFrame(out).drop(columns=["inquiry"]); source=df.reset_index(drop=True).copy(); collisions=[c for c in generated.columns if c in source.columns]; source=source.rename(columns={c:f"input_{c}" for c in collisions}); result=pd.concat([source,generated],axis=1); st.dataframe(result,use_container_width=True,hide_index=True); st.download_button("Download analyzed CSV",result.to_csv(index=False).encode("utf-8-sig"),"analyzed_inquiries.csv","text/csv")
   st.session_state.records.extend(out)
elif page=="Analytics":
 st.title("Inquiry analytics"); rows=st.session_state.records
 if not rows: st.info("Analyze inquiries or upload a CSV to populate session analytics.")
 else:
  df=pd.DataFrame(rows); a,b=st.columns(2)
  if px:
   a.plotly_chart(px.histogram(df,x="category",color="category",title="Category distribution"),use_container_width=True)
   b.plotly_chart(px.histogram(df,x="priority",category_orders={"priority":["High","Medium","Low"]},color="priority",title="Priority distribution"),use_container_width=True)
  else:
   a.bar_chart(df.category.value_counts()); b.bar_chart(df.priority.value_counts())
  st.subheader("Most requested services"); services=df.requested_services.str.split("; ").explode().value_counts().rename_axis("service").reset_index(name="count")
  if px: st.plotly_chart(px.bar(services,x="service",y="count",color="service"),use_container_width=True)
  else: st.bar_chart(services.set_index("service"))
elif page=="About the Project":
 st.title("About this prototype"); st.markdown("""**Proposed workflow aid.** Cynbit's public site describes AI automation, software/cloud services and demo inquiries; this prototype demonstrates a possible internal inquiry triage tool. It is not affiliated with or deployed by Cynbit.

- **Classifier:** TF-IDF + Logistic Regression when scikit-learn is available; this offline run uses multinomial Naive Bayes.
- **Data:** 192 deterministic synthetic examples across six classes; no company/customer records.
- **Extraction:** regular-expression budget/timeline parsing and transparent keyword rules.
- **Priority:** explicit heuristic based on urgency, short timelines, and stated budget.
- **Response:** editable template, no external model/API.
- **Limits:** synthetic test scores do not establish real-world performance; human review is required.

See `reports/` for opportunity and technical reports.""")
