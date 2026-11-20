from utils.preprocessing import clean_text
from utils.information_extractor import extract_info
from utils.priority_detector import detect_priority
from utils.response_generator import generate_response
from utils.classifier import load_model,predict
def test_cleaning_empty_and_unicode():
 assert clean_text("  HELLO\nworld! ")=="hello world"
 assert clean_text("")==""
def test_extract_and_priority():
 info=extract_info("Need an AI app, $20,000 in 2 weeks, secure API integration")
 assert info["budget"]!="Not stated" and info["timeline"]!="Not stated"
 assert "AI/ML" in info["requested_services"]
 assert detect_priority("ASAP",info)["priority"]=="High"
def test_model_prediction_and_unknown_fallback():
 m=load_model(); assert predict("Build an Android mobile app",m)["confidence"]>0
 result=predict("qzxv blorf unknown jargon",m); assert "category" in result and result["confidence"]>=0
def test_response_is_template():
 i=extract_info("Need website"); pr=detect_priority("Need website",i); out=generate_response("Need website","Web Development",i,pr)
 assert "Hello" in out and "budget" in out.lower()
