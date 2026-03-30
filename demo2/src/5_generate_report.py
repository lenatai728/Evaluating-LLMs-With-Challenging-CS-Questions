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

def generate_report(input_file, output_csv_q, output_csv_m, scoring_model=None):

    input_path = os.path.join(INPUT_DIR, input_file)

    # 1. Load Data
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if scoring_model:
        logger.info(f"Using single scoring model: {scoring_model}")
    else:
        logger.info("Using average of all scoring models (default)")

    rows = []

    for item in data:
        q_id = item.get('question_id')
        domain = item.get('question_domain')
        q_type = item.get('question_type')
        eval_method = item.get('answer_eval_method', '')

        # --- Determine answer & rationale scores ---
        if scoring_model:
            # Answer score: use model-specific key if it exists, otherwise fall back to 
            # "answer_score" (exact match questions that have no per-model answer score)
            model_ans_key = f'{scoring_model}_answer_score'
            if model_ans_key in item:
                raw_ans = item[model_ans_key]
            elif 'answer_score' in item:
                # Exact-match or single-score questions
                raw_ans = item['answer_score']
                logger.debug(f"Question {q_id}: '{model_ans_key}' not found, "
                             f"using 'answer_score' (eval_method={eval_method})")
            else:
                logger.warning(f"Question {q_id}: neither '{model_ans_key}' nor "
                               f"'answer_score' found, defaulting to 0.0")
                raw_ans = 0.0

            # Rationale score: use model-specific key if it exists, otherwise fall back
            model_rat_key = f'{scoring_model}_rationale_score'
            if model_rat_key in item:
                raw_rat = item[model_rat_key]
            elif 'rationale_score' in item:
                raw_rat = item['rationale_score']
                logger.debug(f"Question {q_id}: '{model_rat_key}' not found, "
                             f"using 'rationale_score'")
            else:
                logger.warning(f"Question {q_id}: neither '{model_rat_key}' nor "
                               f"'rationale_score' found, defaulting to 0.0")
                raw_rat = 0.0
        else:
            # Default: use the pre-computed average scores
            raw_ans = item.get('average_answer_score', item.get('answer_score', 0.0))
            raw_rat = item.get('average_rationale_score', item.get('rationale_score', 0.0))

        # Convert to Decimal for precision
        avg_ans_score = Decimal(str(raw_ans))
        avg_rat_score = Decimal(str(raw_rat))

        # 3. Calculate Weighted Scores (wA, wR)
        # Scores are on 0-2 scale, so divide by 2 to normalize to 0-1
        wA = avg_ans_score / Decimal('2.0')
        wR = avg_rat_score / Decimal('2.0')

        penalty = Decimal('0.85')

        # 4. Calculate Final Score
        score_val = (Decimal('0.5') * wA) + (Decimal('0.5') * wR) - (penalty * abs(wA - wR))

        # Prevent negative scores
        score_val = Decimal(0.0) if score_val < Decimal(0.0) else score_val

        # Round to 2 decimal places
        final_score = (score_val * Decimal('100')).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

        # 5. Aggregate Explanation Codes
        ex_codes = set()
        if scoring_model:
            # Only collect exCode from the specified scoring model
            ex_key = f'{scoring_model}_rationale_exCode'
            value = item.get(ex_key, '')
            if value:
                for c in value.split(','):
                    c = c.strip()
                    if c:
                        ex_codes.add(c)
        else:
            # Collect exCodes from ALL scoring models
            for key, value in item.items():
                if key.endswith('_rationale_exCode') and value:
                    for c in value.split(','):
                        c = c.strip()
                        if c:
                            ex_codes.add(c)

        ex_code_str = ", ".join(sorted(list(ex_codes)))

        rows.append({
            'question_id': q_id,
            'question_domain': domain,
            'question_type': q_type,
            'answer_score': float(avg_ans_score),
            'rationale_score': float(avg_rat_score),
            'wA': float(wA),
            'wR': float(wR),
            'Final Score': float(final_score),
            'Explanation Code': ex_code_str
        })

    # Create first DataFrame (Results by Question)
    df_q = pd.DataFrame(rows)
    cols_q = ['question_id', 'question_domain', 'question_type',
              'answer_score', 'rationale_score',
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

    # 1. Overall Scores
    final_model_score = round(df_q['Final Score'].mean(), 2) if not df_q.empty else 0.0
    mean_ans_score = round(df_q['wA'].mean() * 100.0, 2) if not df_q.empty else 0.0
    mean_rat_score = round(df_q['wR'].mean() * 100.0, 2) if not df_q.empty else 0.0

    # 2. Question-Type counts & scores
    type_order = ['MC', 'MS', 'TF', 'FB', 'OE']
    type_counts = df_q['question_type'].value_counts() if not df_q.empty else pd.Series(dtype=int)
    type_means = df_q.groupby('question_type')['Final Score'].mean() if not df_q.empty else pd.Series(dtype=float)

    # 3. Domain counts & scores
    domain_abbr = {
        'Data Structures and Algorithms': 'DSA',
        'Computer Systems and Architecture': 'CSA',
        'Databases and Information Systems': 'DIS',
        'Software Engineering and AI Fundamentals': 'SEAI',
        'Cybersecurity': 'Cyber',
    }
    domain_order = ['DSA', 'CSA', 'DIS', 'SEAI', 'Cyber']
    if not df_q.empty:
        df_q['domain_abbr'] = df_q['question_domain'].map(domain_abbr).fillna('Other')
        domain_counts = df_q['domain_abbr'].value_counts()
        domain_means = df_q.groupby('domain_abbr')['Final Score'].mean()
    else:
        domain_counts = pd.Series(dtype=int)
        domain_means = pd.Series(dtype=float)

    # 4. Calculation vs Non-Calculation scores
    if not df_q.empty:
        calc_flags = {str(item.get('question_id')): item.get('is_calculation', 0) for item in data}
        df_q['is_calculation'] = df_q['question_id'].astype(str).map(calc_flags).fillna(0).astype(int)
        calc_count = int((df_q['is_calculation'] == 1).sum())
        non_calc_count = int((df_q['is_calculation'] == 0).sum())
        calc_mean = round(df_q.loc[df_q['is_calculation'] == 1, 'Final Score'].mean(), 2) if calc_count > 0 else 0.0
        non_calc_mean = round(df_q.loc[df_q['is_calculation'] == 0, 'Final Score'].mean(), 2) if non_calc_count > 0 else 0.0
    else:
        calc_count = non_calc_count = 0
        calc_mean = non_calc_mean = 0.0

    # 5. Error-code frequencies (E0-E11)
    error_code_counts = {f'E{i}': 0 for i in range(12)}
    for item in data:
        if scoring_model:
            # Only count exCodes from the specified scoring model
            ex_key = f'{scoring_model}_rationale_exCode'
            value = item.get(ex_key, '')
            if value:
                for code in value.split(','):
                    code = code.strip()
                    if code in error_code_counts:
                        error_code_counts[code] += 1
        else:
            # Count exCodes from ALL scoring models
            for key, value in item.items():
                if key.endswith('_exCode') and value:
                    for code in value.split(','):
                        code = code.strip()
                        if code in error_code_counts:
                            error_code_counts[code] += 1

    # 6. Answer scoring distribution (FB & OE only)
    if not df_q.empty:
        fb_oe = df_q[df_q['question_type'].isin(['FB', 'OE'])]
        n_fb_oe = len(fb_oe)
        if n_fb_oe > 0:
            ans_buckets = fb_oe['answer_score'].round(0).astype(int).clip(0, 2)
            ans_dist = ans_buckets.value_counts()
            ans_incorrect_pct = round(ans_dist.get(0, 0) / n_fb_oe * 100, 2)
            ans_partial_pct = round(ans_dist.get(1, 0) / n_fb_oe * 100, 2)
            ans_correct_pct = round(ans_dist.get(2, 0) / n_fb_oe * 100, 2)
        else:
            ans_incorrect_pct = ans_partial_pct = ans_correct_pct = 0.0
    else:
        ans_incorrect_pct = ans_partial_pct = ans_correct_pct = 0.0

    # 7. Rationale scoring distribution (all questions)
    if not df_q.empty:
        n_all = len(df_q)
        rat_buckets = df_q['rationale_score'].round(0).astype(int).clip(0, 2)
        rat_dist = rat_buckets.value_counts()
        rat_incorrect_pct = round(rat_dist.get(0, 0) / n_all * 100, 2)
        rat_partial_pct = round(rat_dist.get(1, 0) / n_all * 100, 2)
        rat_correct_pct = round(rat_dist.get(2, 0) / n_all * 100, 2)
    else:
        rat_incorrect_pct = rat_partial_pct = rat_correct_pct = 0.0

    experiment_date = datetime.now().strftime('%Y-%m-%d')

    # --- Assemble model summary row ---
    summary = {
        'Model Name': model_name,
        'Scoring Model': scoring_model if scoring_model else 'average (all)',
        'Total Questions': num_questions,
        'Final Model Score': final_model_score,
        'Final Answer Score': mean_ans_score,
        'Final Rationale Score': mean_rat_score,
    }
    for t in type_order:
        summary[f'{t} Count'] = int(type_counts.get(t, 0))
    for t in type_order:
        summary[f'{t} Score'] = round(type_means.get(t, 0.0), 2)
    for d in domain_order:
        summary[f'{d} Count'] = int(domain_counts.get(d, 0))
    for d in domain_order:
        summary[f'{d} Score'] = round(domain_means.get(d, 0.0), 2)
    summary['Calc Count'] = calc_count
    summary['Non-Calc Count'] = non_calc_count
    summary['Calc Score'] = calc_mean
    summary['Non-Calc Score'] = non_calc_mean
    for code in [f'E{i}' for i in range(12)]:
        summary[code] = error_code_counts[code]
    summary['Ans Incorrect %'] = ans_incorrect_pct
    summary['Ans Partial %'] = ans_partial_pct
    summary['Ans Correct %'] = ans_correct_pct
    summary['Rat Incorrect %'] = rat_incorrect_pct
    summary['Rat Partial %'] = rat_partial_pct
    summary['Rat Correct %'] = rat_correct_pct
    summary['Experiment Date'] = experiment_date

    df_m = pd.DataFrame([summary])
    df_m.to_csv(output_csv_m, index=False)
    print(f"Generated {output_csv_m}")

    # Output to 4_final_results (JSON)
    json_output_path = os.path.join(OUTPUT_DIR, input_file.replace('.json', '_final_report.json'))
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=4)
    logger.info(f"Saved detailed report to {json_output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        help="Filename inside data/3_scores/ (e.g., final_answer_final_rationale.json)")
    parser.add_argument("--model", type=str, required=False, default=None,
                        choices=['deepseek-reasoner', 'gpt-4o-mini'],
                        help="Use ONLY this scoring model's scores instead of the average. "
                             "If omitted, average_answer_score and average_rationale_score are used.")
    args = parser.parse_args()

    input_file = args.input_file
    scoring_model = args.model

    # Extract model_name from input_file
    model_name = input_file.replace("_scored_partial_final_answer_final_rationale.json", "")

    if scoring_model:
        suffix = f"_scored_by_{scoring_model}"
        output_csv_q = os.path.join(OUTPUT_DIR, f"{model_name}{suffix}_result_by_question.csv")
        output_csv_m = os.path.join(OUTPUT_DIR, f"{model_name}{suffix}_result_by_model.csv")
    else:
        output_csv_q = os.path.join(OUTPUT_DIR, f"{model_name}_result_by_question.csv")
        output_csv_m = os.path.join(OUTPUT_DIR, f"{model_name}_result_by_model.csv")

    generate_report(input_file, output_csv_q, output_csv_m, scoring_model=scoring_model)


if __name__ == "__main__":
    main()