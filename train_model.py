from pathlib import Path
import json, math, re
from collections import Counter, defaultdict
import pandas as pd
from utils.preprocessing import clean_text
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data"/"sample_inquiries.csv"; MODEL=ROOT/"models"/"inquiry_classifier.joblib"
def metrics(y, pred, labels):
 n=len(y); acc=sum(a==b for a,b in zip(y,pred))/n; prs=[]; recs=[]; f1s=[]; cm=[]
 for c in labels:
  tp=sum(a==c and b==c for a,b in zip(y,pred)); fp=sum(a!=c and b==c for a,b in zip(y,pred)); fn=sum(a==c and b!=c for a,b in zip(y,pred)); pr=tp/(tp+fp) if tp+fp else 0; re=tp/(tp+fn) if tp+fn else 0; prs.append(pr); recs.append(re); f1s.append(2*pr*re/(pr+re) if pr+re else 0); cm.append([sum(a==c and b==d for a,b in zip(y,pred)) for d in labels])
 return {"accuracy":acc,"precision_macro":sum(prs)/len(labels),"recall_macro":sum(recs)/len(labels),"f1_macro":sum(f1s)/len(labels),"labels":labels,"confusion_matrix":cm}
def main():
 df=pd.read_csv(DATA).dropna(subset=["inquiry","category"]); df["inquiry"]=df.inquiry.map(clean_text); df=df[df.inquiry.str.len()>0].reset_index(drop=True); labels=sorted(df.category.unique())
 # deterministic stratified split: each class sorted by original row order, 25% last rows held out
 train=[]; test=[]
 for c in labels:
  ids=list(df.index[df.category==c]); cut=max(1,int(len(ids)*.75)); train+=ids[:cut]; test+=ids[cut:]
 xtr=list(df.loc[train,"inquiry"]); ytr=list(df.loc[train,"category"]); xte=list(df.loc[test,"inquiry"]); yte=list(df.loc[test,"category"])
 try:
  import joblib
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.linear_model import LogisticRegression
  from sklearn.pipeline import Pipeline
  from sklearn.model_selection import train_test_split
  Xtr,Xte,Ytr,Yte=train_test_split(df.inquiry,df.category,test_size=.25,random_state=42,stratify=df.category)
  model=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=12000,sublinear_tf=True)),("classifier",LogisticRegression(max_iter=1200,class_weight="balanced",random_state=42))]); model.fit(Xtr,Ytr); pred=model.predict(Xte); m=metrics(list(Yte),list(pred),labels); joblib.dump(model,MODEL); backend="scikit-learn TF-IDF + Logistic Regression"; split="stratified 75/25, random_state=42"; nt=len(Xtr); nv=len(Xte)
 except ImportError:
  # Offline fallback: multinomial Naive Bayes with train-only smoothed TF-IDF weights.
  vocab=set(); docs={c:[] for c in labels};
  for x,c in zip(xtr,ytr):
   toks=re.findall(r"\b\w+\b",x); vocab.update(toks); docs[c].append(toks)
  N=len(xtr); dfreq=Counter(t for x in xtr for t in set(re.findall(r"\b\w+\b",x))); idf={t:math.log((1+N)/(1+n))+1 for t,n in dfreq.items()}; ll={}; priors={}
  for c in labels:
   counts=Counter(t for doc in docs[c] for t in doc); total=sum(counts.values()); priors[c]=len(docs[c])/N; denom=total+len(vocab); ll[c]={t:math.log((counts[t]+1)/denom) for t in vocab}
  data={"classes":labels,"priors":priors,"idf":idf,"log_likelihoods":ll}; (ROOT/"models"/"inquiry_classifier.json").write_text(json.dumps(data),encoding="utf-8")
  from utils.classifier import SimpleNB
  model=SimpleNB(data); pred=[labels[max(range(len(q:=model.predict_proba([x])[0])),key=lambda j:q[j])] for x in xte]; m=metrics(yte,pred,labels); backend="Offline multinomial Naive Bayes fallback (scikit-learn unavailable)"; split="stratified per-class 75/25, deterministic order"; nt=len(xtr); nv=len(xte)
 ev={"dataset":"Synthetic examples only; not Cynbit customer data","n_total":len(df),"n_train":nt,"n_test":nv,"split":split,"model_backend":backend,**m,"note":"Template-based synthetic holdout; metrics are an implementation check, not a real-world generalization estimate."}; (ROOT/"models"/"evaluation.json").write_text(json.dumps(ev,indent=2),encoding="utf-8"); print(json.dumps(ev,indent=2))
if __name__=="__main__":main()
