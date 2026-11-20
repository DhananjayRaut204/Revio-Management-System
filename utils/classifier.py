from pathlib import Path
import json, math
from collections import Counter
from .preprocessing import clean_text
ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/"models"/"inquiry_classifier.joblib"
FALLBACK_PATH=ROOT/"models"/"inquiry_classifier.json"
class SimpleNB:
    """Offline fallback multinomial Naive Bayes over smoothed TF-IDF features."""
    def __init__(self, data): self.data=data; self.classes_=data["classes"]; self.priors=data["priors"]; self.logs=data["log_likelihoods"]; self.idf=data["idf"]
    def predict_proba(self, texts):
        result=[]
        for text in texts:
            toks=Counter(clean_text(text).split()); scores=[]
            for c in self.classes_: scores.append(math.log(self.priors[c])+sum(n*self.idf.get(t,0)*self.logs[c].get(t,-12) for t,n in toks.items()))
            mx=max(scores); ex=[math.exp(x-mx) for x in scores]; s=sum(ex); result.append([x/s for x in ex])
        return result
def load_model(path=MODEL_PATH):
    if Path(path).exists():
        try:
            import joblib
            return joblib.load(path)
        except ModuleNotFoundError:
            if not FALLBACK_PATH.exists():
                raise
    if FALLBACK_PATH.exists(): return SimpleNB(json.loads(FALLBACK_PATH.read_text(encoding="utf-8")))
    raise FileNotFoundError("Trained model missing. Run: python train_model.py")
def predict(text, model=None, threshold=0.45):
    model=model or load_model(); cleaned=clean_text(text)
    if not cleaned: return {"category":"Other","confidence":0.0,"warning":"Enter a non-empty inquiry.","probabilities":{}}
    probs=model.predict_proba([cleaned])[0]; classes=model.classes_; i=int(max(range(len(probs)),key=lambda j:probs[j])); confidence=float(probs[i])
    category=str(classes[i]) if confidence>=threshold else "Other (uncertain)"
    warning="Low confidence: review manually; category may be unknown." if confidence<threshold else ""
    return {"category":category,"predicted_category":str(classes[i]),"confidence":confidence,"warning":warning,"probabilities":dict(zip(classes,map(float,probs)))}
