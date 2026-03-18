import json
import pandas as pd
import numpy as np
import os
import logging
import argparse
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime

INPUT_DIR = os.path.join('data', '3_scores')
OUTPUT_DIR = os.path.join('data', '4_final_results')

# --- LOGGING ---
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/5_generate_report.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def generate_report(input_file, output_csv_q, output_csv_m):
    
    input_path = os.path.join(INPUT_DIR, input_file)

    # 1. Load Data
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    
    for item in data:
        q_id = item.get('question_id')
        domain = item.get('question_domain')
        q_type = item.get('question_type')
        
        # 1. Determine average answer score
        raw_ans = item.get('average_answer_score', item.get('answer_score', 0.0))
        raw_rat = item.get('average_rationale_score', item.get('rationale_score', 0.0))
            
        # 2. Determine average rationale score (Decimal)
        avg_ans_score = Decimal(str(raw_ans))
        avg_rat_score = Decimal(str(raw_rat))

        # 3. Calculate Weighted Scores (wA, wR)
        # Scores are on 0-2 scale, so divide by 2 to normalize to 0-1
        wA = avg_ans_score / Decimal('2.0')
        wR = avg_rat_score / Decimal('2.0')
        
        # penalty = Decimal('0.3')
        penalty = Decimal('0.85')
        
        # 4. Calculate Final Score
        # Formula: 0.3*wA + 0.7*wR - penalty * |wA - wR|
        # score_val = (Decimal('0.3') * wA) + (Decimal('0.7') * wR) - (penalty * abs(wA - wR))
        score_val = (Decimal('0.5') * wA) + (Decimal('0.5') * wR) - (penalty * abs(wA - wR))
        
        # Prevent negative scores
        score_val = Decimal(0.0) if score_val < Decimal(0.0) else score_val

        # Round to 2 decimal places
        final_score = (score_val * Decimal('100')).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        
        # 5. Aggregate Explanation Codes
        ex_codes = set()
        for key, value in item.items():
            if key.endswith('_rationale_exCode') and value:
                codes = [c.strip() for c in value.split(',')]
                for c in codes:
                    ex_codes.add(c)
        
        ex_code_str = ", ".join(sorted(list(ex_codes)))
        
        rows.append({
            'question_id': q_id,
            'question_domain': domain,
            'question_type': q_type,
            'average_answer_score': float(avg_ans_score),
            'average_rationale_score': float(avg_rat_score),
            'wA': float(wA),
            'wR': float(wR),
            'Final Score': float(final_score),
            'Explanation Code': ex_code_str
        })
        
    # Create first DataFrame (Results by Question)
    df_q = pd.DataFrame(rows)
    cols_q = ['question_id', 'question_domain', 'question_type', 
              'average_answer_score', 'average_rationale_score', 
              'wA', 'wR', 'Final Score', 'Explanation Code']
    df_q = df_q[cols_q]
    
    # Ensure directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df_q.to_csv(output_csv_q, index=False)
    print(f"Generated {output_csv_q}")
        
    # --- CALCULATE MODEL STATISTICS ---
    model_name = data[0].get('model', 'Unknown') if data else 'Unknown'
    num_questions = len(rows)
    
    # 1. Overall Mean Score
    final_model_score = df_q['Final Score'].mean() if not df_q.empty else 0.0
    
    # 2. Component Scores (Normalized to 100)
    # wA and wR are already 0-1, so just mean() * 100
    if not df_q.empty:
        mean_ans_score = df_q['wA'].mean() * 100.0
        mean_rat_score = df_q['wR'].mean() * 100.0
    else:
        mean_ans_score = 0.0
        mean_rat_score = 0.0
    
    # 3. Question Type Scores
    if not df_q.empty:
        type_means = df_q.groupby('question_type')['Final Score'].mean()
    else:
        type_means = pd.Series(dtype=float)

    def get_type_score(qtype):
        return round(type_means.get(qtype, 0.0), 2)

    mc_score = get_type_score('MC')
    ms_score = get_type_score('MS')
    tf_score = get_type_score('TF')
    fb_score = get_type_score('FB')
    oe_score = get_type_score('OE')
    
    experiment_date = datetime.now().strftime('%Y-%m-%d')
    
    summary_rows = [{
        'Model Name': model_name,
        'Number of Question Samples': num_questions,
        'MC Score': mc_score,
        'MS Score': ms_score,
        'TF Score': tf_score,
        'FB Score': fb_score,
        'OE Score': oe_score,
        'Final Answer Score': round(mean_ans_score, 2),
        'Final Rationale Score': round(mean_rat_score, 2),
        'Final Model Score': round(final_model_score, 2),
        'Experiment Date': experiment_date
    }]
    
    df_m = pd.DataFrame(summary_rows)
    cols_m = ['Model Name', 'Number of Question Samples', 
              'MC Score', 'MS Score', 'TF Score', 'FB Score', 'OE Score', 
              'Final Answer Score', 'Final Rationale Score', 'Final Model Score',
              'Experiment Date']
    df_m = df_m[cols_m]
    
    df_m.to_csv(output_csv_m, index=False)
    print(f"Generated {output_csv_m}")
    
    # Output to 4_final_results (JSON)
    json_output_path = os.path.join(OUTPUT_DIR, input_file.replace("_scored_partial_final_answer_final_rationale.json", "_final_report.json"))
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=4)
    logger.info(f"Saved detailed report to {json_output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Filename inside data/3_scores/ (e.g., final_answer_final_rationale.json)")
    args = parser.parse_args()

    input_file = args.input_file 
    
    # extract model_name from input_file
    model_name = input_file.replace("_scored_partial_final_answer_final_rationale.json", "")
    output_csv_q = os.path.join(OUTPUT_DIR, f"{model_name}_result_by_question.csv")
    output_csv_m = os.path.join(OUTPUT_DIR, f"{model_name}_result_by_model.csv")
    
    generate_report(input_file, output_csv_q, output_csv_m)
    

if __name__ == "__main__":
    main()