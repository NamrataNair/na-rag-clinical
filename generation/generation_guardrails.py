"""
Post-generation assertion consistency check.
"""

def check_faithfulness(generated_text, evidence):
    generated_text = generated_text.lower()
    for ev in evidence:
        if ev["assertion"] == "negated" and ev["text"].lower() in generated_text:
            return False
    return True


# Example usage
"""
if __name__ == "__main__":  
    gen_text = "The patient does not have diabetes."
    sample_evidence = [
        {"text": "The patient has a history of diabetes.", "assertion": "historical"},
        {"text": "He is currently experiencing chest pain.", "assertion": "asserted"},
        {"text": "No evidence of infection was found.", "assertion": "negated"}
    ]
    is_faithful = check_faithfulness(gen_text, sample_evidence)
    print(f"Is the generated text faithful to the evidence? {is_faithful}")    
"""