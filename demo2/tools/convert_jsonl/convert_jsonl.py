"""
Usage:
  python3 convert_jsonl.py --input_file Question.xlsx
  python3 convert_jsonl.py --input_file Question.xlsx --type
  python3 convert_jsonl.py --input_file Question.xlsx --domain
  python3 convert_jsonl.py --input_file Question.xlsx --type --domain
"""

import pandas as pd
import json
import re
import sys
import os
import argparse


# Domain abbreviation mapping
DOMAIN_ABBR = {
    'Data Structures and Algorithms': 'DSA',
    'Computer Systems and Architecture': 'CSA',
    'Databases and Information Systems': 'DIS',
    'Software Engineering and AI Fundamentals': 'SEAI',
    'Cybersecurity': 'Cyber',
}


def parse_question_prompt(prompt):
    """
    Splits a question_prompt into the question text and a dict of choices.
    Handles any number of choices (A through Z).
    """
    if not prompt or not isinstance(prompt, str):
        return "", {}

    # Normalize line breaks and whitespace
    prompt = prompt.replace('\r\n', '\n').replace('\r', '\n').strip()

    # Pattern: Match "A." or "A)" at the start of a line or after whitespace
    choice_pattern = r'(?:^|\n)\s*([A-Z])\s*[.)]\s*'

    # Find all choice positions
    matches = list(re.finditer(choice_pattern, prompt))

    if not matches:
        # No choices found — entire prompt is the question
        return prompt.strip(), {}

    # Everything before the first choice is the question
    question_text = prompt[:matches[0].start()].strip()

    # Extract each choice
    choices = {}
    for i, match in enumerate(matches):
        label = match.group(1)
        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(prompt)

        choice_text = prompt[start:end].strip().rstrip('\n').strip()
        choices[label] = choice_text

    return question_text, choices


def build_record(row, df_columns):
    """
    Build a single JSONL record from a DataFrame row.
    """
    COL_PROMPT       = "question_prompt"
    COL_DOMAIN       = "domain"
    COL_RATIONALE    = "gold_rationale"
    COL_ANSWER       = "gold_answer"
    COL_QTYPE        = "question_type"
    COL_IS_CALC      = "is_calculation"

    prompt = str(row.get(COL_PROMPT, ""))
    question_text, choices = parse_question_prompt(prompt)

    # Determine question type
    if COL_QTYPE in df_columns and pd.notna(row.get(COL_QTYPE)):
        q_type = str(row[COL_QTYPE]).strip()
    else:
        q_type = "MCQ" if choices else "Open-ended"

    # Determine is_calculation
    if COL_IS_CALC in df_columns and pd.notna(row.get(COL_IS_CALC)):
        raw_val = row[COL_IS_CALC]
        if isinstance(raw_val, bool):
            is_calc = raw_val
        elif isinstance(raw_val, (int, float)):
            is_calc = bool(int(raw_val))
        else:
            raw_str = str(raw_val).strip().lower()
            is_calc = raw_str in ('true', '1', 'yes', 'y')
    else:
        is_calc = False

    # Determine domain
    domain_raw = str(row.get(COL_DOMAIN, "")).strip() if pd.notna(row.get(COL_DOMAIN)) else ""

    record = {
        "Domain": domain_raw,
        "Question_Type": q_type,
        "is_calculation": is_calc,
        "Question": question_text,
        "Rationale": str(row.get(COL_RATIONALE, "")).strip() if pd.notna(row.get(COL_RATIONALE)) else "",
        "Answer": str(row.get(COL_ANSWER, "")).strip() if pd.notna(row.get(COL_ANSWER)) else ""
    }

    # Only add Choices if they exist
    if choices:
        record["Choices"] = choices

    return record, q_type, domain_raw


def normalize_columns(df):
    """
    Normalize column names: strip whitespace.
    Try case-insensitive matching for required columns.
    """
    df.columns = df.columns.str.strip()

    required_cols = ["question_prompt", "domain", "gold_rationale",
                     "gold_answer", "question_type", "is_calculation"]

    for col in required_cols:
        if col not in df.columns:
            found = [c for c in df.columns if c.lower() == col.lower()]
            if found:
                df.rename(columns={found[0]: col}, inplace=True)

    return df


def write_records_to_file(filepath, records):
    """Write a list of record dicts to a JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    return len(records)


def convert_excel(excel_path, split_by_type=False, split_by_domain=False):
    """
    Reads an Excel file and converts rows to JSONL format.

    - Default (no flags): produces a single <basename>.jsonl
    - --type:   produces one JSONL per question_type (e.g., MC.jsonl, MS.jsonl)
    - --domain: produces one JSONL per domain abbreviation (e.g., DSA.jsonl, CSA.jsonl)
    - Both can be combined.
    """
    df = pd.read_excel(excel_path)
    df = normalize_columns(df)

    base_dir = os.path.dirname(excel_path) or "."
    base_name = os.path.splitext(os.path.basename(excel_path))[0]

    # Collect all records
    all_records = []
    type_buckets = {}    # question_type -> [records]
    domain_buckets = {}  # domain_abbr -> [records]
    errors = []

    for idx, row in df.iterrows():
        try:
            record, q_type, domain_raw = build_record(row, df.columns)
            all_records.append(record)

            # Bucket by question type
            if split_by_type:
                type_key = q_type if q_type else "UNKNOWN"
                type_buckets.setdefault(type_key, []).append(record)

            # Bucket by domain abbreviation
            if split_by_domain:
                domain_key = DOMAIN_ABBR.get(domain_raw, None)
                if domain_key is None:
                    # Try partial / case-insensitive match
                    for full_name, abbr in DOMAIN_ABBR.items():
                        if domain_raw.lower() == full_name.lower():
                            domain_key = abbr
                            break
                    if domain_key is None:
                        domain_key = domain_raw if domain_raw else "UNKNOWN"
                domain_buckets.setdefault(domain_key, []).append(record)

        except Exception as e:
            errors.append({"row": idx + 2, "error": str(e)})
            print(f"ERROR at row {idx + 2}: {e}")

    # ---- Write output files ----
    output_files = []

    # If neither flag is set, write a single file
    if not split_by_type and not split_by_domain:
        output_path = os.path.join(base_dir, base_name + ".jsonl")
        count = write_records_to_file(output_path, all_records)
        output_files.append((output_path, count))

    # Write per question_type
    if split_by_type:
        type_dir = os.path.join(base_dir, "by_type")
        os.makedirs(type_dir, exist_ok=True)
        for q_type, records in sorted(type_buckets.items()):
            # Sanitize filename
            safe_name = re.sub(r'[^\w\-]', '_', q_type)
            out_path = os.path.join(type_dir, f"{safe_name}.jsonl")
            count = write_records_to_file(out_path, records)
            output_files.append((out_path, count))

    # Write per domain
    if split_by_domain:
        domain_dir = os.path.join(base_dir, "by_domain")
        os.makedirs(domain_dir, exist_ok=True)
        for domain_key, records in sorted(domain_buckets.items()):
            safe_name = re.sub(r'[^\w\-]', '_', domain_key)
            out_path = os.path.join(domain_dir, f"{safe_name}.jsonl")
            count = write_records_to_file(out_path, records)
            output_files.append((out_path, count))

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"Conversion complete!")
    print(f"  Input:  {excel_path}")
    print(f"  Total records processed: {len(all_records)}")
    print(f"  Errors: {len(errors)}")
    print(f"\n  Output files:")
    for fpath, cnt in output_files:
        print(f"    {fpath}  ({cnt} records)")
    print(f"{'=' * 60}")

    return output_files, errors


# ========================
# MAIN
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Excel question bank to JSONL format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 convert_jsonl.py --input_file Question.xlsx
      → Produces Question.jsonl (single file, all records)

  python3 convert_jsonl.py --input_file Question.xlsx --type
      → Produces by_type/MC.jsonl, by_type/MS.jsonl, etc.

  python3 convert_jsonl.py --input_file Question.xlsx --domain
      → Produces by_domain/DSA.jsonl, by_domain/CSA.jsonl, etc.

  python3 convert_jsonl.py --input_file Question.xlsx --type --domain
      → Produces both by_type/ and by_domain/ directories.
        """
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input Excel file (e.g., Question.xlsx)"
    )
    parser.add_argument(
        "--type",
        action="store_true",
        dest="split_by_type",
        help="Generate separate JSONL files by question_type (e.g., MC.jsonl, MS.jsonl)"
    )
    parser.add_argument(
        "--domain",
        action="store_true",
        dest="split_by_domain",
        help="Generate separate JSONL files by domain abbreviation (e.g., DSA.jsonl, CSA.jsonl)"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f"ERROR: File not found: {args.input_file}")
        sys.exit(1)

    convert_excel(
        excel_path=args.input_file,
        split_by_type=args.split_by_type,
        split_by_domain=args.split_by_domain
    )

# """
# Usage: python3 convert_jsonl.py Question.xlsx
# """

# import pandas as pd
# import json
# import re
# import sys
# import os

# def parse_question_prompt(prompt):
#     """
#     Splits a question_prompt into the question text and a dict of choices.
#     Handles any number of choices (A through Z).
#     """
#     if not prompt or not isinstance(prompt, str):
#         return "", {}

#     # Normalize line breaks and whitespace
#     prompt = prompt.replace('\r\n', '\n').replace('\r', '\n').strip()

#     # Pattern: Match "A." or "A)" at the start of a line or after whitespace
#     # This regex finds all choice labels (single letter) followed by . or )
#     choice_pattern = r'(?:^|\n)\s*([A-Z])\s*[.)]\s*'

#     # Find all choice positions
#     matches = list(re.finditer(choice_pattern, prompt))

#     if not matches:
#         # No choices found — entire prompt is the question
#         return prompt.strip(), {}

#     # Everything before the first choice is the question
#     question_text = prompt[:matches[0].start()].strip()

#     # Extract each choice
#     choices = {}
#     for i, match in enumerate(matches):
#         label = match.group(1)  # e.g., "A", "B", "C"
#         start = match.end()     # where the choice text begins

#         # Choice text ends where the next choice starts, or at end of string
#         if i + 1 < len(matches):
#             end = matches[i + 1].start()
#         else:
#             end = len(prompt)

#         choice_text = prompt[start:end].strip().rstrip('\n').strip()
#         choices[label] = choice_text

#     return question_text, choices


# def convert_excel_to_jsonl(excel_path, output_path=None):
#     """
#     Reads an Excel file and converts rows to JSONL format.
#     """
#     if output_path is None:
#         base = os.path.splitext(excel_path)[0]
#         output_path = base + ".jsonl"

#     # --- Read Excel ---
#     df = pd.read_excel(excel_path)

#     # --- Normalize column names (strip whitespace, lowercase for matching) ---
#     df.columns = df.columns.str.strip()

#     # --- Map your actual column names here ---
#     # Adjust these if your Excel column names differ
#     COL_PROMPT    = "question_prompt"
#     COL_DOMAIN    = "domain"
#     COL_RATIONALE = "gold_rationale"
#     COL_ANSWER    = "gold_answer"
#     COL_QTYPE     = "question_type"  # optional, adjust as needed

#     # Validate required columns exist
#     required = [COL_PROMPT]
#     for col in required:
#         if col not in df.columns:
#             # Try case-insensitive match
#             found = [c for c in df.columns if c.lower() == col.lower()]
#             if found:
#                 df.rename(columns={found[0]: col}, inplace=True)
#             else:
#                 print(f"WARNING: Column '{col}' not found. Available: {list(df.columns)}")

#     records_written = 0
#     errors = []

#     with open(output_path, 'w', encoding='utf-8') as f:
#         for idx, row in df.iterrows():
#             try:
#                 prompt = str(row.get(COL_PROMPT, ""))
#                 question_text, choices = parse_question_prompt(prompt)

#                 # Determine question type based on choices or explicit column
#                 if COL_QTYPE in df.columns and pd.notna(row.get(COL_QTYPE)):
#                     q_type = str(row[COL_QTYPE]).strip()
#                 else:
#                     # Auto-detect: if choices exist → MC, else → open-ended
#                     q_type = "MCQ" if choices else "Open-ended"

#                 record = {
#                     "Domain": str(row.get(COL_DOMAIN, "")).strip() if pd.notna(row.get(COL_DOMAIN)) else "",
#                     "Question_Type": q_type,
#                     "Question": question_text,
#                     # "Choices": choices if choices else None,
#                     "Rationale": str(row.get(COL_RATIONALE, "")).strip() if pd.notna(row.get(COL_RATIONALE)) else "",
#                     "Answer": str(row.get(COL_ANSWER, "")).strip() if pd.notna(row.get(COL_ANSWER)) else ""
#                 }
                
#                 # Only add Choices if they exist
#                 if choices:
#                     record["Choices"] = choices
                
#                 # Write one JSON object per line
#                 f.write(json.dumps(record, ensure_ascii=False) + '\n')
#                 records_written += 1

#             except Exception as e:
#                 errors.append({"row": idx + 2, "error": str(e)})  # +2 for Excel row number
#                 print(f"ERROR at row {idx + 2}: {e}")

#     print(f"\n{'='*50}")
#     print(f"Conversion complete!")
#     print(f"  Input:   {excel_path}")
#     print(f"  Output:  {output_path}")
#     print(f"  Records: {records_written}")
#     print(f"  Errors:  {len(errors)}")
#     print(f"{'='*50}")

#     return output_path, errors


# def convert_multiple_files(excel_paths):
#     """Process multiple Excel files."""
#     for path in excel_paths:
#         print(f"\nProcessing: {path}")
#         convert_excel_to_jsonl(path)


# # ========================
# # USAGE
# # ========================
# if __name__ == "__main__":
#     # --- Option 1: Command line ---
#     if len(sys.argv) > 1:
#         files = sys.argv[1:]
#         convert_multiple_files(files)

#     # --- Option 2: Hardcode paths ---
#     else:
#         excel_files = [
#             "Question_v5.xlsx",
#             # "another_file.xlsx",
#         ]
#         convert_multiple_files(excel_files)