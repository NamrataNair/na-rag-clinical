def build_prompt(query, evidence):
    prompt = f"Clinical query: {query}\n\nEvidence:\n"

    for ev in evidence:
        prompt += f"[{ev['assertion'].upper()}] {ev['text']}\n"

    prompt += (
        "\nInstructions:\n"
        "- Only assert conditions marked ASSERTED.\n"
        "- Explicitly state when conditions are negated or historical.\n"
        "- If no asserted evidence exists, say so.\n"
    )

    return prompt

# Example usage
"""
if __name__ == "__main__":  
    sample_query = "Does the patient have diabetes?"
    sample_evidence = [
        {"text": "The patient has a history of diabetes.", "assertion": "historical"},
        {"text": "He is currently experiencing chest pain.", "assertion": "asserted"},
        {"text": "No evidence of infection was found.", "assertion": "negated"}
    ]
    prompt = build_prompt(sample_query, sample_evidence)
    print(prompt)    
"""