import re
import numpy as np


def extract_answer(text):
    """Extract numerical answers from generated text"""
    patterns = [
        r"\\boxed{([\d.]+)}",  # LaTeX boxed
        r"Answer:\s*([\d.,]+)",  # Text answer
        r"answer is\s*([\d.,]+)",  # Informal answer
        r"=\s*([\d.,]+)(?:\s|$)"  # Equation result
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).replace(',', '')
            return float(value) if '.' in value else int(value)
    return None


def evaluate_answer(predicted, reference):
    """Compare extracted numerical answers with tolerance"""
    pred = extract_answer(predicted)
    ref = extract_answer(reference)

    if pred is None or ref is None:
        return 0

    # Allow 1% relative error or absolute 0.1 difference
    tolerance = max(0.1, 0.01 * abs(ref))
    return 1 if abs(pred - ref) <= tolerance else 0