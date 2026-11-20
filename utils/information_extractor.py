import re

def extract_info(text):
    t=str(text or ""); low=t.lower()
    money=re.search(r"(?:[$₹€£]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:k|m|million|lakh|lakhs|crore))?|\b(?:usd|inr|rs\.?|rupees)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:k|m|million|lakh|lakhs|crore))?)",t,re.I)
    time=re.search(r"\b(?:within\s+)?(?:\d+\s*(?:days?|weeks?|months?|years?)|(?:one|two|three|four|six|eight|twelve)\s+(?:days?|weeks?|months?)|Q[1-4]\s*20\d{2}|ASAP|urgent)\b",t,re.I)
    service_terms={"AI/ML":["ai","machine learning","ml","llm","model","computer vision","recommendation"],"Web Development":["website","web app","portal","frontend","backend","next.js","e-commerce"],"Mobile App Development":["mobile app","android","ios","flutter","react native"],"Cloud Solutions":["cloud","aws","azure","migration","kubernetes","hosting"],"Automation":["automation","automate","workflow","agent","data entry","crm integration"]}
    services=[k for k,terms in service_terms.items() if any(x in low for x in terms)]
    requirements=[]
    patterns=[("integration",r"\b(?:integrat\w*|api|connect\w*)\b"),("security",r"\b(?:security|secure|compliance|privacy|soc2|gdpr)\b"),("scale",r"\b(?:scale|scalable|users|traffic|high availability)\b"),("dashboard",r"\b(?:dashboard|analytics|reporting)\b"),("legacy migration",r"\b(?:legacy|monolith|migration)\b"),("AI capability",r"\b(?:ai|machine learning|llm|model)\b")]
    for label,pat in patterns:
        if re.search(pat,low): requirements.append(label)
    return {"budget":money.group(0) if money else "Not stated","timeline":time.group(0) if time else "Not stated","requested_services":services or ["Needs clarification"],"requirements":requirements or ["None detected"]}
