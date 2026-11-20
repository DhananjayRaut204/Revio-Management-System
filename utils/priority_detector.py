def detect_priority(text,info):
    """Transparent heuristic, not an ML prediction. Score explicit urgency, timeline, budget."""
    t=str(text or "").lower(); score=0; reasons=[]
    if any(x in t for x in ["urgent","asap","immediately","this week","critical"]): score+=2; reasons.append("explicit urgency language")
    if info.get("timeline","Not stated")!="Not stated":
        import re
        m=re.search(r"(\d+)\s*(day|week|month)",info["timeline"].lower())
        if m:
            n=int(m.group(1)); unit=m.group(2)
            if (unit=="day" and n<=14) or (unit=="week" and n<=4) or (unit=="month" and n<=1): score+=2; reasons.append("short stated timeline")
            elif unit=="month" and n>=6: reasons.append("longer stated timeline")
    if info.get("budget","Not stated")!="Not stated": score+=1; reasons.append("budget provided")
    if score>=3: level="High"
    elif score>=1: level="Medium"
    else: level="Low"
    return {"priority":level,"score":score,"reasons":reasons or ["no explicit urgency, short deadline, or budget detected"],"method":"Rule-based heuristic; not an ML prediction."}
