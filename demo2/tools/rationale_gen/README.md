## Description
- For each question:
  - If `gold_answer` is missing, generates it using DeepSeek.
  - Generates `gold_rationale` if missing, based on the question and answer.
- Adds a `deepseek_answer` column for verification or alternative answers.
  - Optionally verifies the answer with DeepSeek (skips generates  rationale if mismatch for non-OE questions).

Install via `requirements.txt`:
```
pandas
openai
python-dotenv
tqdm
openpyxl
```

## Installation

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

Run the script with:
```
python rationale_gen.py [--skip]
```

- `--skip`: Optional flag to skip DeepSeek answer verification and generate rationales directly based on existing `gold_answer`. Useful if you trust the exist manual gold answers and want to force rationale generation.

### Input/Output
- **Input File**: `Question.xlsx` (default)
- **Output File**: `Question_rationale.xlsx` (updated Excel with filled answers and rationales).

The script processes only sheets with "en" in the name (English). Modify the condition in the code if needed.