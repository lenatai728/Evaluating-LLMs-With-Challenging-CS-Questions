import pandas as pd
import json
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

INPUT_PATH = os.path.join('data', 'Question_v5.xlsx')
OUTPUT_PATH = os.path.join('data', 'input.jsonl')

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    logger.info(f"Reading Excel file: {INPUT_PATH}")
    df = pd.read_excel(INPUT_PATH)

    # Verify required columns exist
    required_columns = ["id", "domain", "question_type", "question_prompt",
                        "gold_answer", "gold_rationale", "is_calculation"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logger.error(f"Missing columns in Excel file: {missing}")
        return

    # Rename columns to match desired output format
    column_mapping = {
        "id": "question_id",
        "domain": "question_domain",
        "question_type": "question_type",
        "question_prompt": "question_prompt",
        "gold_answer": "gold_answer",
        "gold_rationale": "gold_rationale",
        "is_calculation": "is_calculation",
    }

    records = []
    for _, row in df.iterrows():
        record = {}
        for excel_col, json_col in column_mapping.items():
            value = row[excel_col]

            # Handle NaN values
            if pd.isna(value):
                value = "" if json_col != "is_calculation" else 0

            # Convert question_id to string
            if json_col == "question_id":
                value = str(int(value)) if isinstance(value, float) else str(value)

            # Convert is_calculation to int
            if json_col == "is_calculation":
                value = int(value)

            record[json_col] = value

        records.append(record)

    # Write as JSONL (one JSON object per line)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info(f"Successfully converted {len(records)} rows to JSONL: {OUTPUT_PATH}")

    # Also save as a formatted JSON file for reference
    json_output_path = OUTPUT_PATH.replace('.jsonl', '.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=4, ensure_ascii=False)

    logger.info(f"Also saved formatted JSON: {json_output_path}")

if __name__ == "__main__":
    main()