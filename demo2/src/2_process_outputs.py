import json
import os
import re
import logging
import time

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/2_process_outputs.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

INPUT_DIR = os.path.join('data', '1_model_outputs')
OUTPUT_DIR = os.path.join('data', '2_processed')

# def parse_output(text):
#     if not text: return {"rationale_output": "", "answer_output": ""}
#     # regex matching 
#     rat_match = re.search(r"### Rationale:\s*(.*?)\s*### Answer:", text, re.DOTALL)
#     ans_match = re.search(r"### Answer:\s*(.*)", text, re.DOTALL)
#     return {
#         "rationale_output": rat_match.group(1).strip() if rat_match else "",
#         "answer_output": ans_match.group(1).strip() if ans_match else ""
#     }

def parse_output(text):
    if not text:
        return {"rationale_output": "", "answer_output": ""}

    text = text.strip()

    # Try canonical marked forms first (### Rationale: ... ### Answer: ...)
    rat_between = re.search(r"###\s*Rationale:\s*(.*?)\s*(?=###\s*Answer:)", text, re.IGNORECASE | re.DOTALL)
    rat_to_end = re.search(r"###\s*Rationale:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    ans_marked = re.search(r"###\s*Answer:\s*(.*)", text, re.IGNORECASE | re.DOTALL)

    rationale = ""
    if rat_between:
        rationale = rat_between.group(1).strip()
    elif rat_to_end:
        # no explicit Answer marker — take remainder as rationale
        rationale = rat_to_end.group(1).strip()

    answer = ""
    if ans_marked:
        answer = ans_marked.group(1).strip()
    else:
        # fallback heuristics if no "### Answer:" marker
        # 1) look for "Answer: X" or "Final Answer: X"
        m = re.search(r"(?:Final\s+Answer|Answer)[:\s]*([A-D])\b", text, re.IGNORECASE)
        if m:
            answer = m.group(1).upper()
        else:
            # 2) standalone single-letter line (A/B/C/D)
            m2 = re.search(r"^\s*([A-D])\s*$", text, re.MULTILINE)
            if m2:
                answer = m2.group(1).upper()
            else:
                # 3) occurrences like "C) ..." or "C) 1.7"
                m3 = re.search(r"\b([A-D])\)", text)
                if m3:
                    answer = m3.group(1).upper()

    return {"rationale_output": rationale, "answer_output": answer}

def parse_output(text, question_type):
    if not text:
        return {"rationale_output": "", "answer_output": ""}

    text = text.strip()

    # Extract rationale (everything before ### Answer:)
    rat_match = re.search(r"###\s*Rationale:\s*(.*?)\s*(?=###\s*Answer:)", text, re.IGNORECASE | re.DOTALL)
    rationale = rat_match.group(1).strip() if rat_match else ""

    # Extract answer (everything after ### Answer:)
    ans_match = re.search(r"###\s*Answer:\s*(.+?)$", text, re.IGNORECASE | re.DOTALL)
    answer = ans_match.group(1).strip() if ans_match else ""

    # Clean answer based on question type
    if question_type == "MC":
        # Single letter A-D
        match = re.search(r"[A-D]", answer, re.IGNORECASE)
        answer = match.group(0).upper() if match else ""
    
    elif question_type == "MS":
        # Multiple letters A-D, comma-separated
        matches = re.findall(r"[A-D]", answer, re.IGNORECASE)
        answer = ", ".join(sorted(set(m.upper() for m in matches))) if matches else ""
    
    elif question_type == "TF":
        # True or False
        if re.search(r"true", answer, re.IGNORECASE):
            answer = "True"
        elif re.search(r"false", answer, re.IGNORECASE):
            answer = "False"
        else:
            answer = ""
    
    elif question_type == "FB":
        # Fill-in-blank: keep as-is, just trim
        answer = answer.strip()
    
    elif question_type == "OE":
        # Open-ended: keep as-is, just trim
        answer = answer.strip()

    return {"rationale_output": rationale, "answer_output": answer}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all JSON files in the output directory
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith("_output.json")]
    
    if not files:
        logger.warning("No model outputs found to process.")
        return

    logger.info(f"Found {len(files)} model files to process.")

    for filename in files:
        try:
            model_name = filename.replace("_output.json", "")
            input_path = os.path.join(INPUT_DIR, filename)
            
            logger.info(f"Processing {model_name}...")
            
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            processed = []
            for entry in data:
                parsed = parse_output(entry.get('model_output', ''), entry.get('question_type', '').upper())
                entry.update(parsed)
                processed.append(entry)
                time.sleep(0.1)

            output_path = os.path.join(OUTPUT_DIR, f"{model_name}_processed.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed, f, indent=4)
                
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    main()