import json
import os
import logging
import argparse
from config import JUDGE_MODEL
from sentence_transformers import SentenceTransformer, util

INPUT_DIR = os.path.join('data', '3_scores')
OUTPUT_DIR = os.path.join('data', '3_scores')
SEMANTIC_THRESHOLD = 0.90  
SAVE_INTERVAL = 5  # Save to disk every X items to prevent data loss

# --- JUDGE CONFIGURATION (Add your keys here) ---
JUDGE_CONFIGS = [
    {
        "id": "gpt_4o_mini", 
        "model_name": "gpt-4o-mini", 
        "api_key": "YOUR_OPENAI_KEY", 
        "base_url": "https://api.openai.com/v1"
    },
    {
        "id": "deepseek_v3", 
        "model_name": "deepseek-chat", 
        "api_key": "YOUR_DEEPSEEK_KEY", 
        "base_url": "https://api.deepseek.com"
    }
]

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def construct_prompts(mode, question, gold, model_output):
    pass

# def construct_judge_answer_prompt(question, gold_answer, llm_answer):
#     pass

# def construct_judge_rationale_prompt(question, gold_answer, llm_answer):
#     pass

def get_judge_score(prompt):
    """
    Simulates calling the LLM Judge (e.g., GPT-4).
    Replace this with real API call in production.
    """
    # REAL CODE EXAMPLE:
    # response = client.chat.completions.create(
    #     model=JUDGE_MODEL, 
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return int(response.choices[0].message.content)
    
    return -1  # Dummy: "Partially Correct" (simulating a tough judge)

def main():
    # Config Args
    
    eval_mode = None 
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["answer", "rationale"], help="Mode: answer or rationale")
    args = parser.parse_args()
    
    # Get Args
    # eval_mode = args.mode
    # if eval_mode == None or eval_mode.lower() != "answer" and eval_mode.lower() != "rationale":
    #     logger.warning(f"Mode is invalid. Please enter again with --mode.")
    #     return
    
    try:
        # 1. Load RoBERTa Model (Downloads on first run)
        logger.info("Loading RoBERTa model for semantic similarity...")
        similarity_model = SentenceTransformer('all-roberta-large-v1') # 'all-roberta-large-v1' is the SOTA sentence-transformer based on RoBERTa-large
        
        files = [f for f in os.listdir(INPUT_DIR) if f.endswith("_scored_partial.json")]
        
        if not files:
            logger.warning("No partial score files found. Run Script 3 first.")
            return

        for filename in files:
            try:
                model_name = filename.replace("_scored_partial.json", "")
                logger.info(f"--- Processing Candidate: {model_name} ---")
                
                input_path = os.path.join(INPUT_DIR, filename)
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                final_data = []
                answer_judge_count = 0
                rationale_judge_count = 0
                answer_auto_pass_count = 0
                rationale_auto_pass_count = 0

                for entry in data:
                    
                    # if eval_mode == "answer":
                        
                    # --- 1. Evaluate ANSWER ---
                    # If answer_score is missing, it means it wasn't an MC/MS question.
                    # We need to grade it now using Hybrid Semantic/LLM approach.
                    if 'answer_score' not in entry:
                        
                        llm_ans = str(entry.get('answer_output', ''))
                        gold_ans = str(entry.get('gold_answer', ''))
                        
                        # Step A: Semantic Similarity Check
                        embeddings = similarity_model.encode([llm_ans, gold_ans], convert_to_tensor=True)
                        cosine_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
                        

                        # LOGIC: High Similarity -> Auto-Correct (2)
                        if cosine_score >= SEMANTIC_THRESHOLD:
                            entry['answer_eval_method'] = "roberta_semantic_pass"
                            entry['answer_score'] = 2
                            answer_auto_pass_count += 1
                        
                        # LOGIC: Low Similarity -> Send to Judge
                        else:
                            judge_prompt = (
                                f"Question: {entry.get('question_prompt')}\n"
                                f"Gold Answer: {gold_ans}\n"
                                f"LLM Answer: {llm_ans}\n"
                                "Score 0 (Wrong), 1 (Partial), or 2 (Correct). Return ONLY the number."
                            )
                            entry['answer_eval_method'] = "llm_judge"
                            entry['answer_score'] = get_judge_score(judge_prompt)
                            answer_judge_count += 1
                        
                        entry['answer_semantic_similarity'] = round(cosine_score, 4)
                        
                    # elif eval_mode == "rationale":
                    # --- 2. Evaluate RATIONALE ---
                    if 'rationale_score' not in entry:
                        
                        llm_ans = str(entry.get('rationale_output', ''))
                        gold_ans = str(entry.get('gold_rationale', ''))
                        
                        # Step A: Semantic Similarity Check
                        embeddings = similarity_model.encode([llm_ans, gold_ans], convert_to_tensor=True)
                        cosine_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

                        if cosine_score >= SEMANTIC_THRESHOLD:
                                entry['rationale_eval_method'] = "roberta_semantic_pass"
                                entry['rationale_score'] = 2
                                rationale_auto_pass_count += 1
                        # Send to LLM Judge
                        else: 
                            rationale_prompt = (
                                f"Question: {entry.get('question_prompt')}\n"
                                f"Gold Rationale: {entry.get('gold_answer')} (Proxy)\n" # Assuming gold_rationale unavailable in mock
                                f"LLM Rationale: {entry.get('rationale_output')}\n"
                                "Rate the reasoning logic 0-2."
                            )
                            entry['rationale_eval_method'] = "llm_judge"
                            entry['rationale_score'] = get_judge_score(rationale_prompt)
                            rationale_judge_count += 1

                        entry['rationale_semantic_similarity'] = round(cosine_score, 4)
                        
                    final_data.append(entry)

                output_path = os.path.join(OUTPUT_DIR, f"{model_name}_scored_final.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(final_data, f, indent=4)

                # if eval_mode == "answer":
                #     output_path = os.path.join(OUTPUT_DIR, f"{model_name}_scored_final_answer.json")
                #     with open(output_path, 'w', encoding='utf-8') as f:
                #         json.dump(final_data, f, indent=4)

                # elif eval_mode == "rationale":
                #     output_path = os.path.join(OUTPUT_DIR, f"{model_name}_scored_final_rationale.json")
                #     with open(output_path, 'w', encoding='utf-8') as f:
                #         json.dump(final_data, f, indent=4)

                logger.info(f"Finished {model_name}: {answer_auto_pass_count} Answer(s) Auto-Passed (Sim > {SEMANTIC_THRESHOLD}), {answer_judge_count} Answer(s) sent to Judge.")
                logger.info(f"Finished {model_name}: {rationale_auto_pass_count} Rationale(s) Auto-Passed (Sim > {SEMANTIC_THRESHOLD}), {rationale_judge_count} Rationale(s) sent to Judge.")

            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")

    except Exception as e:
        logger.critical(f"Critical Failure: {e}")

if __name__ == "__main__":  
    main()