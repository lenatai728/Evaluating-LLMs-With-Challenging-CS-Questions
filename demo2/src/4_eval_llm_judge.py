import json
import os
import logging
from config import JUDGE_MODEL

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

INPUT_DIR = os.path.join('data', '3_scores')
OUTPUT_DIR = os.path.join('data', '3_scores') # Update in place

def get_judge_score(prompt):
    # Call OpenAI API with JUDGE_MODEL here
    return 2 # Dummy

def main():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith("_scored_partial.json")]
    
    logger.info(f"Judge Model: {JUDGE_MODEL} is ready to grade {len(files)} candidates.")

    for filename in files:
        try:
            model_name = filename.replace("_scored_partial.json", "")
            logger.info(f"--- Judging Candidate: {model_name} ---")
            
            with open(os.path.join(INPUT_DIR, filename), 'r') as f:
                data = json.load(f)

            final_data = []
            for entry in data:
                # Grade Answer if needed
                if 'answer_score' not in entry:
                    entry['answer_score'] = get_judge_score("Grade Answer Prompt...")
                
                # Grade Rationale (Always)
                entry['rationale_score'] = get_judge_score("Grade Rationale Prompt...")
                
                final_data.append(entry)

            output_path = os.path.join(OUTPUT_DIR, f"{model_name}_scored_final.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=4)

        except Exception as e:
            logger.error(f"Error judging {filename}: {e}")

if __name__ == "__main__":
    main()