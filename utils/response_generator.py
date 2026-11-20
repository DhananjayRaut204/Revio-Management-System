def generate_response(text,category,info,priority):
    services=", ".join(info["requested_services"])
    budget=info["budget"]; timeline=info["timeline"]
    questions=[]
    if budget=="Not stated": questions.append("Could you share an approximate budget range?")
    if timeline=="Not stated": questions.append("What target launch date or timeline should we plan around?")
    questions.append("Are there existing systems or integrations the solution must work with?")
    urgency="We have noted the timing and will prioritize an initial review." if priority["priority"]=="High" else ""
    return (f"Hello,\n\nThank you for reaching out about your project. We understand you are exploring {services.lower()} ({category}). "
      f"We have noted the budget as {budget} and timeline as {timeline}. {urgency}\n\n"
      f"To help us assess the right approach, could you clarify: " + " ".join(questions) + "\n\n"
      "Once we have these details, our team can review the requirements and suggest a suitable next step.\n\nBest regards,\nCynbit Team")
