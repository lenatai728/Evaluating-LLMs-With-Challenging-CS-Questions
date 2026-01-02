import json
import os
import logging
import argparse
import time
from dotenv import load_dotenv
from openai import OpenAI
# from huggingface_hub import InferenceClient
from config import CANDIDATE_MODELS

# --- LOGGING SETUP ---
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/1_get_model_responses.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
load_dotenv()
INPUT_PATH = os.path.join('data', '0_raw', 'input.json')
OUTPUT_DIR = os.path.join('data', '1_model_outputs')

def get_type_specific_instructions(question_type):
    # instruction = ""
    if question_type == "MC":
        instruction = "Provide only the single uppercase letter corresponding to the correct option (e.g., 'A'). Do not write the option text."
    elif question_type == "MS":
        instruction = "Provide the uppercase letters corresponding to the correct options, separated by commas (e.g., 'A, C'). Sort them alphabetically."
    elif question_type == "TF":
        instruction = "Provide only the word 'True' or 'False'."
    elif question_type == "FB":
        instruction = "Provide only the exact term, number, or mathematical expression that fills the blank. Do not include a full sentence. Do not add a period at the end unless it is part of the value."
    elif question_type == "OE":
        instruction = "Provide a detailed and clear final answer." 
    return instruction
    
def construct_prompt(q_set):
    # prompt = f"""
    #     You are a Computer Science expert taking a final year exam. 
    #     Domain: {q_set['question_domain']} 
    #     Question Type: {q_set['question_type']} 
    #     QUESTION:{q_set['question_prompt']} 
    #     INSTRUCTIONS: 
    #     1. Think step-by-step. If this is a calculation question, show every step of the math. 
    #     2. Provide your detailed Rationale first. 
    #     3. End your response with the final Answer clearly labeled. 
    #     FORMAT: 
    #     ### Rationale: [Your reasoning here] 
    #     ### Answer: [Your final answer here] 
    # """
    
    system_prompt = f"""
    You are a Computer Science expert taking a final year exam. 
    You are given university-level Computer Science questions. 
    You should read each question carefully and decide your best answer for every question.
    When you respond, you must follow the INSTRUCTIONS and FORMAT below.

    **CRITICAL: YOU MUST FOLLOW THIS FORMAT EXACTLY**
    Your response MUST contain BOTH sections below, or your answer will be marked invalid:
    1. A rationale section starting with "### Rationale:"
    2. An answer section starting with "### Answer:"

    GLOBAL INSTRUCTIONS:
    1. **NO LATEX**: Do not use LaTeX formatting (e.g., no `\\( ... \\)`, no `\log`).
    2. **Standard Math Notation**: Use simple, standard ASCII characters for math.
       - Use `*` for multiplication, `/` for division, `^` for power.
       - Use `log2()` for log base 2, `sqrt()` for square root.
       - Example: Write `2 * log2(4n + 1)` instead of `\( 2 \log_2(4n + 1) \)`.
    3. **Plain Text Rationale**: Write your rationale in clear, concise, plain text paragraphs.
       - **DO NOT** use bullet points, numbered lists, bold text (**text**), or headers.
       - Keep the explanation logical but free of markdown styling.

    SPECIFIC ANSWER FORMATTING:
    {get_type_specific_instructions(q_set['question_type'])}

    OUTPUT FORMAT (DO NOT DEVIATE):
    ### Rationale:
    [Your final rationale strictly following the specific formatting rules above]

    ### Answer:
    [Your final answer strictly following the specific formatting rules above]
    """
    
    user_prompt = f"""
        Domain: {q_set['question_domain']}
        Question Type: {q_set['question_type']}
        QUESTION:{q_set['question_prompt']}
    """
    
    return system_prompt, user_prompt 
    

def get_llm_response(system_prompt, user_prompt, model_name, provider):
    """
        Get response from specified LLM model. 
    """
    try:
        # FOR OPENROUTER APIs
        if provider.lower() == "openrouter":
            
            key_str = f"{model_name.split('/')[-1].upper()}-OPENROUTER-API-KEY"
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv(key_str)
                # api_key="justfortestingonly"
            )
            
            response = client.chat.completions.create(
                extra_headers={
                    "azureml-model-deployment": model_name
                },
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role":"user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            }
                        ]
                    }
                ],
                temperature=0.0,
                # max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            # return f"[Mock Output for {model_name}]\n### Rationale: Logic.\n### Answer: 42"

        # FOR HUGGING FACE APIs
        elif provider.lower() == "huggingface":
            # response = client_hf.text_generation(
            #     prompt, model=model_name, max_new_tokens=500
            # )
            # return response
            return f"[Mock Output for {model_name}]\n### Rationale: OpenSource Logic.\n### Answer: 42"

    except Exception as e:
        logger.error(f"API Error for {model_name}: {e}")
        return "ERROR_RESPONSE"

def main():
    
    # Config Args
    parser = argparse.ArgumentParser(description="Run inference for a specific LLM.")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to run")
    args = parser.parse_args()
    
    # Get Args
    current_model = args.model
    if current_model not in CANDIDATE_MODELS:
        logger.warning(f"Model '{current_model}' is not in your config list. Please enter again.")
        return

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Load Question Sets From input.json
        if not os.path.exists(INPUT_PATH):
            logger.error(f"Input file missing at {INPUT_PATH}")
            return
        with open(INPUT_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Run Inference
        results = []
        success_count = 0
        fail_count = 0
        logger.info(f"--- Starting Inference for: {current_model} ---")

        for i, q in enumerate(data):
            
            system_prompt, user_prompt = construct_prompt(q)
            
            response = get_llm_response(system_prompt, user_prompt, current_model, "openrouter")
            
            time.sleep(0.5)  # To avoid rate limits
            
            output_entry = q.copy()
            output_entry['model'] = current_model
            output_entry['model_output'] = response
            
            if response == "ERROR_RESPONSE":
                fail_count += 1
                logger.error(f"Failed to get response for [[[ Q{i+1} ]]]: {q['question_prompt'][:50]}...")
            else:
                success_count += 1
            
            results.append(output_entry)

    except Exception as e:
        logger.critical(f"Pipeline failed for {current_model}: {e}")
        
    finally:
        # Save specific file for this model
        safe_name = current_model.replace("/", "_")    # Handle "meta-llama/Llama..." -> "meta-llama_Llama..."
        out_file = os.path.join(OUTPUT_DIR, f"{safe_name}_output.json")
        
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)

        logger.info(f"Saved {len(results)} responses to {out_file} ( {fail_count} failures / {success_count} successes )")

if __name__ == "__main__":
    main()