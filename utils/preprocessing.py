import re
import unicodedata

def clean_text(text):
    """Normalize Unicode, URLs, whitespace; preserve useful words and numbers."""
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"[^\w\s.,$€£%/-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
