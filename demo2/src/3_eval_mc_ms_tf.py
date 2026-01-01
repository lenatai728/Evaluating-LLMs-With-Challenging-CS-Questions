import json
import os
import logging
import re

# --- LOGGING SETUP ---
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/3_eval_mc_ms_tf.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
INPUT_DIR = os.path.join('data', '2_processed')
OUTPUT_DIR = os.path.join('data', '3_scores')

def parse_selection_set(text):
    """
        Parses a string like "A, B" or "['A', 'C']" into a standardized set {'a', 'b'}.
    """
    if not text:
        return set()
    
    text = str(text).lower().strip()
    
    # Remove standard list brackets/quotes if the model output pseudo-code
    text = text.replace('[', '').replace(']', '').replace("'", "").replace('"', "")
    
    # Normalize delimiters (semicolons, 'and', slashes -> commas)
    for delimiter in [';', '/', ' and ', '&']:
        text = text.replace(delimiter, ',')
        
    # Split by comma and strip whitespace
    parts = [p.strip() for p in text.split(',') if p.strip()]
    
    return set(parts)

def main():
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        files = [f for f in os.listdir(INPUT_DIR) if f.endswith("_processed.json")]

        if not files:
            logger.warning(f"No processed files found in {INPUT_DIR}")
            return

        for filename in files:
            try:
                model_name = filename.replace("_processed.json", "")
                logger.info(f"Grading questions for: {model_name}")
                
                input_path = os.path.join(INPUT_DIR, filename)
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                scored_data = []
                count_ms = 0
                count_mc = 0
                count_tf = 0

                for entry in data:
                    q_type = entry.get('question_type', '').lower()
                    
                    # 1. Multiple Select (MS) - Based with 0-2 Scoring
                    if any(x in q_type for x in ['ms']):
                        m_ans_set = parse_selection_set(entry.get('answer_output', ''))
                        g_ans_set = parse_selection_set(entry.get('gold_answer', ''))
                        
                        entry['answer_eval_method'] = "exact_match"
                        
                        # Case A: Fully Correct 
                        if m_ans_set == g_ans_set:
                            entry['answer_score'] = 2
                        
                        # Case B: Partially Correct 
                        # Condition 1: Must have some overlap (intersection > 0)
                        # Condition 2: Must have SAME AMOUNT of choices
                        elif (not m_ans_set.isdisjoint(g_ans_set)) and (len(m_ans_set) == len(g_ans_set)):
                            entry['answer_score'] = 1    
                        
                        # Case C: Incorrect 
                        else:
                            entry['answer_score'] = 0
                            
                        count_ms += 1
                        scored_data.append(entry)

                    # 2. Multiple Choice (MC) / True-False (TF) - Exact String Match
                    elif any(x in q_type for x in ['mc']):
                        m_ans = str(entry.get('answer_output', '')).lower().strip()
                        g_ans = str(entry.get('gold_answer', '')).lower().strip()
                        
                        # Binary scoring for single-choice (0 or 2)
                        entry['answer_eval_method'] = "exact_match"
                        entry['answer_score'] = 2 if m_ans == g_ans else 0
                        count_mc += 1
                        scored_data.append(entry)
                    
                    elif any(x in q_type for x in ['true-or-false', 'tf']):
                        m_ans = str(entry.get('answer_output', '')).lower().strip()
                        g_ans = str(entry.get('gold_answer', '')).lower().strip()
                        
                        # Binary scoring for single-choice (0 or 2)
                        entry['answer_eval_method'] = "exact_match"
                        entry['answer_score'] = 2 if m_ans == g_ans else 0
                        count_tf += 1
                        scored_data.append(entry)
                    
                    else:
                        # Pass through other types (Open-Ended) for the LLM Judge
                        scored_data.append(entry)

                output_path = os.path.join(OUTPUT_DIR, f"{model_name}_scored_partial.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(scored_data, f, indent=4)
                
                logger.info(f"Saved {len(scored_data)} scores (Graded {count_mc} MC, {count_ms} MS, {count_tf} True-or-False).")

            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")

    except Exception as e:
        logger.critical(f"Critical Grading Error: {e}")

if __name__ == "__main__":
    main()