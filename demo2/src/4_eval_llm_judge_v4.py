import json
import os
import logging
import argparse
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI  

# --- CONFIGURATION ---
INPUT_DIR = os.path.join('data', '3_scores')
OUTPUT_DIR = os.path.join('data', '3_scores')
SEMANTIC_THRESHOLD = 0.90
SAVE_INTERVAL = 5  # Save to disk every X items to prevent data loss
load_dotenv()

# --- JUDGE CONFIGURATION (Add your keys here) ---
JUDGE_CONFIGS = [
    # {
    #     "id": "gpt_5_mini", 
    #     "model_name": "openai/gpt-5-mini", 
    #     "api_key": os.getenv("GPT-5-MINI-OPENROUTER-API-KEY"), 
    #     "base_url": "https://openrouter.ai/api/v1"
    # },
    {
        "id": "deepseek-chat", 
        "model_name": "deepseek-chat", 
        "api_key": os.getenv("DEEPSEEK-CHAT-API-KEY"), 
        "base_url": "https://api.deepseek.com"
    },
    {
        "id": "gemini-2.5-flash", 
        "model_name": "google/gemini-2.5-flash", 
        "api_key": os.getenv("GEMINI-25-FLASH-OPENROUTER-API-KEY"), 
        "base_url": "https://openrouter.ai/api/v1"
    }
]

# --- LOGGING ---
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/4_eval_llm_judge.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def get_judge_response(q_id, judge_config, system_prompt, user_prompt, eval_mode):
    """
    Generic function to call any OpenAI-compatible API (DeepSeek, OpenAI, etc.)
    """
    # client = OpenAI(api_key=judge_config['api_key'], base_url=judge_config['base_url'])
    
    client = OpenAI(
        api_key=judge_config['api_key'],
        base_url=judge_config['base_url']
    )
    
    try:
        response = client.chat.completions.create(
            model=judge_config['model_name'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0, 
            max_tokens=10  # Increased to allow for score and code output
        )
        
        content = response.choices[0].message.content.strip()
        logging.info(f"[{q_id}] get_judge_response(): Judge {judge_config['id']} Response: {content}")
        
        if eval_mode == "answer":
            try:
                score = int(content)
            except ValueError:
                score = -1  # Error code
            return score, None  # No explanation code for answer mode
        
        elif eval_mode == "rationale":
            # Parse the standardized output: Score: <number>\nCode: <code>
            score = -1
            ex_code = None
            lines = content.split('\n')
            for line in lines:
                if line.startswith("Score:"):
                    try:
                        score = int(line.split("Score:")[1].strip())
                    except ValueError:
                        score = -1
                elif line.startswith("Code:"):
                    ex_code = line.split("Code:")[1].strip()
            
            return score, ex_code
        
    except Exception as e:
        logger.error(f"API Error with {judge_config['id']}: {e}")
        return -1, None  # Error code

def construct_prompts(mode, question, gold, model_output):
    """
    Constructs the prompt based on the mode (Answer vs Rationale).
    Refers to Proposal Page 12, standardized scoring.
    """
    if mode == "answer":
        system = (
            "You are an expert Computer Science evaluator. "
            "Score the Student Answer 0, 1, or 2 based on the Gold Answer.\n"
            "0: Entirely incorrect or irrelevant.\n"
            "1: Partially correct or incomplete.\n"
            "2: Fully correct and contextually appropriate.\n"
            "-1: Only if there is error for your evaluation (i.e. No Question or Gold Rationale or Student Rationale given is empty or contains error)\n"
            # f"Question: {question}\nGold Answer: {gold}\nStudent Answer: {model_output}\n\nScore:"
            "Return ONLY the number."
        )
        user = f"Question: {question}\nGold Answer: {gold}\nStudent Answer: {model_output}\n\nScore:"
        
    elif mode == "rationale": 
        system = (
            "You are an expert Computer Science evaluator. "
            "Score the Student's Reasoning Logic (Rationale) 0, 1, or 2 based on the Gold Rationale.\n"
            "0: Entirely incorrect or irrelevant, Logic contains contradictions, hallucinations, or valid steps are missing.\n"
            "1: Reasoning is partially correct but misses key steps or is vague.\n"
            "2: Fully correct, logically sound, complete, and leads to the correct conclusion.\n"
            "-1: Only if there is error for your evaluation (i.e. No Question or Gold Rationale or Student Rationale given is empty or contains error)\n\n"
            "After scoring, select the most appropriate explanation code from the following taxonomy that best explains your verdict:\n"
            "E0 - Fully Correct: Rationale is entirely correct and well-justified.\n"
            "E1 - Conceptual Misunderstanding: Core concept or principle misunderstood.\n"
            "E2 - Logical Inconsistency: Internal contradiction or invalid reasoning steps.\n"
            "E3 - Incomplete Reasoning: Rationale lacks key reasoning steps or conditions.\n"
            "E4 - Irrelevant Reasoning: Provides logically valid but unrelated explanation.\n"
            "E5 - Calculation/Procedure Error: Arithmetic or procedural mistake.\n"
            "E6 - Hallucination / Fabrication: Includes facts, functions, or terminology that do not exist.\n"
            "E7 - Misinterpretation of Question: Rationale valid for a different question.\n"
            "E8 - Terminology Error: Uses wrong or imprecise technical term.\n"
            "E9 - Incomplete Answer: Partially correct; some details missing.\n"
            "E10 - Ambiguous / Vague Explanation: Lacks specificity or clear structure.\n"
            "E11 - Correct but Poorly Justified: Rationale is right but reasoning is shallow or tautological.\n\n"
            "Return in this exact format without any additional text:\n"
            "Score: <number>\n"
            "Code: <code>"
        )
        user = f"Question: {question}\nGold Rationale: {gold}\nStudent Rationale: {model_output}"
    return system, user

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["answer", "rationale"], help="Mode: answer or rationale")
    parser.add_argument("--input_file", type=str, required=True, help="Filename inside data/3_scores/ (e.g., model_processed.json)")
    args = parser.parse_args()
    
    eval_mode = args.mode
    input_file = args.input_file
    
    input_path = os.path.join(INPUT_DIR, input_file)
    output_filename = input_file.replace(".json", f"_final_{eval_mode}.json")
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # 1. Load Data
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Load RoBERTa (Heavy, load once)
    if eval_mode == "answer":
        logger.info("Loading RoBERTa model...")
        similarity_model = SentenceTransformer('all-roberta-large-v1')

    # 3. Processing Loop
    logger.info(f"--- Starting Evaluation Mode: {eval_mode.upper()} ---")
    
    for idx, entry in enumerate(data):
        
        # Skip MC/MS/TF questions (already exact matching) in Answer Mode
        if eval_mode == "answer":
            question_type = entry.get("question_type", "")
            answer_eval_method = entry.get("answer_eval_method", "")
            if (question_type.lower() == "mc" or question_type.lower() == "ms" or question_type.lower() == "tf") and answer_eval_method == "exact_match":
                continue

        # Determine keys based on mode
        key_output = "answer_output" if eval_mode == "answer" else "rationale_output"
        key_gold = "gold_answer" if eval_mode == "answer" else "gold_rationale"
        key_sim = f"{eval_mode}_semantic_similarity"
        
        # Skip if already scored (allows resuming)
        first_judge_key = f"{JUDGE_CONFIGS[0]['id']}_{eval_mode}_score"
        if first_judge_key in entry:
            continue

        model_text = str(entry.get(key_output, ""))
        gold_text = str(entry.get(key_gold, ""))
        question_text = entry.get("question_prompt", "")

        if eval_mode == "answer":
            # === SEMANTIC SIMILARITY CHECK ===
            if not model_text.strip(): 
                cosine_score = -1 # Error
            else:
                embeddings = similarity_model.encode([model_text, gold_text], convert_to_tensor=True)
                cosine_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
            
            entry[key_sim] = round(cosine_score, 4)

            # === ROBERTa SIMILARITY FILTER ===
            # If High Similarity -> Auto-Pass ALL judges
            
            if cosine_score >= SEMANTIC_THRESHOLD:
                logger.info(f"[{idx}] High Similarity ({cosine_score:.3f}). Auto-passing.")
                entry[f"{eval_mode}_score"] = 2
                entry[f"{eval_mode}_eval_method"] = "roberta_auto_pass"
            
            # If Low Similarity -> Call APIs for EACH judge
            else:  
                # === LLM-as-the-JUDGE LOGIC ===
                logger.info(f"[{idx}] Low Similarity ({cosine_score:.3f}). Sending to Judges.")
                system_prompt, user_prompt = construct_prompts(eval_mode, question_text, gold_text, model_text)
                
                entry[f"{eval_mode}_eval_method"] = "llm_judge"
                scores = []
                for judge in JUDGE_CONFIGS:
                    score, ex_code = get_judge_response(idx, judge, system_prompt, user_prompt, eval_mode)
                    entry[f"{judge['id']}_{eval_mode}_score"] = score
                    if eval_mode == "rationale":
                        entry[f"{judge['id']}_{eval_mode}_exCode"] = ex_code
                    scores.append(score)
                    time.sleep(0.5) # Rate limit safety

        elif eval_mode == "rationale":
            # === LLM-as-the-JUDGE LOGIC ===
            # logger.info(f"[{idx}] Low Similarity ({cosine_score:.3f}). Sending to Judges.")
            system_prompt, user_prompt = construct_prompts(eval_mode, question_text, gold_text, model_text)
            
            entry[f"{eval_mode}_eval_method"] = "llm_judge"
            scores = []
            for judge in JUDGE_CONFIGS:
                score, ex_code = get_judge_response(idx, judge, system_prompt, user_prompt, eval_mode)
                entry[f"{judge['id']}_{eval_mode}_score"] = score
                if eval_mode == "rationale":
                    entry[f"{judge['id']}_{eval_mode}_exCode"] = ex_code
                scores.append(score)
                time.sleep(0.5) # Rate limit safety
        
        # === AVERAGE SCORE ===
        # If any judge returned -1 (error), average is -1
        if any(s == -1 for s in scores):
            entry[f"average_{eval_mode}_score"] = -1
        else:
            average_score = round(sum(scores) / len(JUDGE_CONFIGS), 4)
            entry[f"average_{eval_mode}_score"] = average_score
            
        
        # === INCREMENTAL SAVE ===
        if (idx + 1) % SAVE_INTERVAL == 0:
            logger.info(f"Saving progress at item {idx+1}...")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
    
    # Final Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    
    logger.info(f"Completed. Saved to {output_path}")

if __name__ == "__main__":
    main()