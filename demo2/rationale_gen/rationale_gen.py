import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm
import argparse

# Set your DeepSeek API key in the environment variable
load_dotenv()
api_key = os.getenv("DeepSeek_API")
if not api_key:
    raise ValueError("DeepSeek_API environment variable not set.")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)
# Prompts for rationale generation
prompts_eng = {
    "MC": "Given this multiple-choice question: {q}, explain why the choice: {answer} is the correct answer while the others are incorrect, including necessary explanations and relevant information. Make it less than forty words.",
    "MS": "Given this multi-select question: {q}, explain why the choices: {answer} are the correct answers while the others are incorrect, including necessary explanations and relevant information. Make it less than forty words.",
    "TF": "Given this True-or-False question: {q}, explain why this statement is {answer}, including necessary explanations and relevant information. Make it less than twenty words.",
    "FB": "Generate a rationale for this fill-in-blank question: {q}, the answer is {answer}, including necessary explanations and relevant information. Make it less than forty words.",
    "OE": "Generate a rationale for this open-ended question: {q}, indicate what kind of answer is satisfactory, including the coverage of knowledge points, necessary reasoning steps, well-rounded explanations, and relevant information. Make it less than forty words."
}
# Prompts for answer generation (if gold_answer is missing)
answer_prompts_eng = {
    "MC": "Given this multiple-choice question: {q} Choose the correct answer option letter only without any explanation.",
    "MS": "Given this multi-select question: {q} Choose all correct answer option letters, separated by commas and without any explanation.",
    "TF": "Given this True-or-False question: {q} Answer True or False only.",
    "FB": "Fill in the blank for this question: {q} Provide the answer, separated by commas if need and without any explanation.",
    "OE": "Answer this open-ended question: {q} Provide a necessary insights. Make it less than forty words."
}

def generate_text(prompt):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def process_excel(input_file="Question.xlsx", output_file="Question_rationale.xlsx", skip_deepseek=False): #Edited input file name if need
    dfs = pd.read_excel(input_file, sheet_name=None, engine='openpyxl')
    writer = pd.ExcelWriter(output_file, engine='openpyxl')

    for sheet_name, df in dfs.items():
        if "en" not in sheet_name.lower(): #Edited to process only English sheets, you can modify this condition based on your sheet naming convention
            continue
        df['gold_answer'] = df['gold_answer'].astype(object)
        df['gold_rationale'] = df['gold_rationale'].astype(object)

        # Add 'deepseek_answer' column after 'Difficulty' for verfication the answers 
        if 'deepseek_answer' not in df.columns:
            df.insert(7, 'deepseek_answer', None)
        df['deepseek_answer'] = df['deepseek_answer'].astype(object)
        
        print(f"Processing sheet: {sheet_name}")

        for idx in tqdm(range(len(df)), desc=f"Progress in {sheet_name}"):
            row = df.iloc[idx]
            question_type = row['question_type']
            question_prompt = row['question_prompt']

            gold_answer = row.get('gold_answer', '')
            gold_answer_str = str(gold_answer).strip() if not pd.isna(gold_answer) else ""

            # Check gold_answer (If it already exists, skip it.)
            if pd.isna(gold_answer) or str(gold_answer).strip() == "":
                ap = answer_prompts_eng.get(question_type, "").format(q=question_prompt)
                if ap:
                    deepseek_answer = generate_text(ap).strip()
                    df.at[idx, 'gold_answer'] = deepseek_answer
                    gold_answer_str = deepseek_answer
                else:
                    print(f"Unknown question type {question_type} in sheet {sheet_name}")
                    continue

            deepseek_answer = None
            if not skip_deepseek:
                # Generate deepseek_answer for verification
                ap = answer_prompts_eng.get(question_type, "").format(q=question_prompt)
                if ap:
                    deepseek_answer = generate_text(ap).strip()
                else:
                    print(f"Unknown question type {question_type} in sheet {sheet_name}")
                    continue

                if question_type == "OE":
                    # For OE, always provide deepseek_answer in the "deepseek_answer" column
                    df.at[idx, 'deepseek_answer'] = deepseek_answer
                else:
                    # compare the generated answer with gold_answer, if different, provide deepseek_answer in new column, skip rationale
                    if deepseek_answer.lower() != gold_answer_str.lower():
                        df.at[idx, 'deepseek_answer'] = deepseek_answer
                        continue

            # Check gold_rationale (If it already exists, skip it.)
            gold_rationale = row.get('gold_rationale', '')
            if pd.isna(gold_rationale) or str(gold_rationale).strip() == "":
                rp_template = prompts_eng.get(question_type, "")
                rp = rp_template.format(q=question_prompt, answer=gold_answer_str)
                if rp:
                    gold_rationale = generate_text(rp)
                    df.at[idx, 'gold_rationale'] = gold_rationale

        df.to_excel(writer, sheet_name=sheet_name, index=False)

    writer.close()
    print(f"Updated Excel saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Excel file for answers and rationales.")
    parser.add_argument('--skip', action='store_true', help='Skip generating and checking deepseek_answer, generate rationale based on gold_answer directly.')
    args = parser.parse_args()
    process_excel(skip_deepseek=args.skip)