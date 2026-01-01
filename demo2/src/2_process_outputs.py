import json
import os
import re
import logging

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

INPUT_DIR = os.path.join('data', '1_model_outputs')
OUTPUT_DIR = os.path.join('data', '2_processed')

def parse_output(text):
    if not text: return {"rationale_output": "", "answer_output": ""}
    # regex matching 
    rat_match = re.search(r"### Rationale:\s*(.*?)\s*### Answer:", text, re.DOTALL)
    ans_match = re.search(r"### Answer:\s*(.*)", text, re.DOTALL)
    return {
        "rationale_output": rat_match.group(1).strip() if rat_match else "",
        "answer_output": ans_match.group(1).strip() if ans_match else ""
    }

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
                parsed = parse_output(entry.get('model_output', ''))
                entry.update(parsed)
                processed.append(entry)

            output_path = os.path.join(OUTPUT_DIR, f"{model_name}_processed.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed, f, indent=4)
                
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    main()