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


def parse_output(text, question_type):
    if not text:
        return {"rationale_output": "", "answer_output": ""}

    text = text.strip()

    rationale = ""
    answer = ""

    # ---- Try multiple marker formats ---- #

    # Format 1: "### Rationale: ... ### Answer: ..."
    # Format 2: "$Rationale: ... $Answer: ..."
    # Format 3: "$Answer: ... $Rationale: ..."
    # Format 4: Mixed / any order

    # Generic pattern: support both "###" and "$" prefixes
    # Extract answer
    ans_match = (
        re.search(r"(?:###\s*Answer|\$\s*Answer)[:\s]*(.+?)(?=(?:###\s*Rationale|\$\s*Rationale)|\Z)", text, re.IGNORECASE | re.DOTALL)
    )
    if ans_match:
        answer = ans_match.group(1).strip()

    # Extract rationale
    rat_match = (
        re.search(r"(?:###\s*Rationale|\$\s*Rationale)[:\s]*(.+?)(?=(?:###\s*Answer|\$\s*Answer)|\Z)", text, re.IGNORECASE | re.DOTALL)
    )
    if rat_match:
        rationale = rat_match.group(1).strip()

    # ---- Fallback if no markers found ---- #
    if not answer and not rationale:
        # Try "Answer: X" or "Final Answer: X"
        m = re.search(r"(?:Final\s+Answer|Answer)[:\s]*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if m:
            answer = m.group(1).strip()
            # Everything else is rationale
            rationale = text[:m.start()].strip()
        else:
            # No structure found — put everything as answer
            answer = text

    # ---- Clean answer based on question type ---- #
    if question_type == "MC":
        match = re.search(r"[A-G]", answer, re.IGNORECASE)
        answer = match.group(0).upper() if match else ""

    elif question_type == "MS":
        matches = re.findall(r"[A-G]", answer, re.IGNORECASE)
        answer = ", ".join(sorted(set(m.upper() for m in matches))) if matches else ""

    elif question_type == "TF":
        # Check single letter first (T/F), then full words
        letter_match = re.match(r"^[TF]$", answer.strip(), re.IGNORECASE)
        if letter_match:
            answer = letter_match.group(0).upper()
        elif re.search(r"true", answer, re.IGNORECASE):
            answer = "T"
        elif re.search(r"false", answer, re.IGNORECASE):
            answer = "F"
        else:
            answer = ""

    elif question_type == "FB":
        answer = answer.strip()

    elif question_type == "OE":
        answer = answer.strip()

    return {"rationale_output": rationale, "answer_output": answer}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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