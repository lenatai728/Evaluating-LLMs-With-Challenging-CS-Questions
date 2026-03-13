import json
import os
import logging
import argparse
import time
import asyncio
from dotenv import load_dotenv
from tqdm import tqdm
from config import CANDIDATE_MODELS
from llm_clients import (
    create_openrouter_client,
    create_huggingface_client,
    create_google_client,
    create_deepseek_client,
    create_async_openrouter_client,
    create_async_deepseek_client,
    create_async_google_client,
    get_llm_response,
    async_get_llm_response,
)

# --- LOGGING SETUP (file only, terminal stays clean) ---
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/1_get_model_responses.log")]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
load_dotenv()
INPUT_PATH = os.path.join('data', '0_raw', 'input.json')
OUTPUT_DIR = os.path.join('data', '1_model_outputs')

MAX_RETRIES = 3

def get_type_specific_instructions(question_type):
    if question_type == "MC":
        return "Provide ONLY the single uppercase letter of the correct option (e.g., 'A')."
    elif question_type == "MS":
        return "Provide ONLY uppercase letters for all correct options, sorted alphabetically and comma-separated and with one whitespace (e.g., 'A, C')."
    elif question_type == "TF":
        return "Provide ONLY one uppercase character 'T' for <True> or 'F' for <False>."
    elif question_type == "FB":
        return "Provide ONLY the exact term or value. For multiple blanks, separate with a comma (e.g., 'apple, orange'). No trailing periods."
    elif question_type == "OE":
        return "Provide a concise, technically complete final answer."
    return ""

def get_rationale_content_instruction(question_type):
    instructions = {
        "MC": "Explain why the correct choice is right and others are wrong. Include necessary explanations and relevant information. Make it less than 40 words.",
        "MS": "Explain why the selected choices are right and others are wrong. Include necessary explanations and relevant information. Make it less than 40 words.",
        "TF": "Explain the validity of the statement. Include necessary explanations and relevant information. Make it less than 20 words. Do not start with 'True' or 'False'.",
        "FB": "Explain the reasoning behind the required value or term. Include necessary explanations and relevant information. Make it less than 40 words.",
        "OE": "Explain the reasoning steps, knowledge points, and criteria for a satisfactory answer. Make it less than 40 words."
    }
    return instructions.get(question_type)

def construct_prompt(q_set):

    system_prompt = f"""
        You are a Computer Science expert solving university-level exams. Technical precision and strict adherence to formatting are mandatory.\n
        1. MANDATORY OUTPUT FORMAT
        Your response MUST follow this structure exactly:
        "### Rationale: [Your Final Rationale]\n### Answer: [Your Final Answer]"
        2. CRITICAL RULES:
        - Output ONLY the two sections above, NO missing one section for "Answer" or "Rationale". NO extra text.
        - Do NOT write drafts. Do NOT show your working out. Do NOT reconsider.
        3. GLOBAL INSTRUCTIONS:
        A. Write your rationale in plain text paragraphs ONLY.
        B. PROHIBITED: NO bullet points, NO lists, NO bolding (**), NO any markdown headers or styling, NO exceeding maximum word limit as stated.
        C. MATH: ASCII only (e.g., `2 * log2(n)`). NO LaTeX (e.g., no `\\(`, no `\log`). Use `*` for multiplication, `/` for division, `^` for power; Use `log2()` for log base 2. E.g. Write `2 * log2(4n + 1)` instead of `\( 2 \log_2(4n + 1) \)`.
        4.SPECIFIC ANSWER FORMATTING:
        {get_type_specific_instructions(q_set['question_type'])}
        5.SPECIFIC RATIONALE FORMATTING:
        {get_rationale_content_instruction(q_set['question_type'])}
    """
    
    user_prompt = f"""Domain: {q_set['question_domain']}; Question Type: {q_set['question_type']}; QUESTION:{q_set['question_prompt']}"""
    
    return system_prompt, user_prompt 
    

def save_results(results, out_file):
    """Incrementally save results to disk to prevent data loss."""
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

def retry_failed_questions(results, out_file, current_model, primary_client, provider="openrouter", hf_client=None):
    """Prompt user to retry questions that received ERROR_RESPONSE. Retries up to MAX_RETRIES times each."""
    failed_indices = [i for i, r in enumerate(results) if r.get('model_output') == 'ERROR_RESPONSE']
    
    if not failed_indices:
        logger.info("All questions answered successfully. No retries needed.")
        print("\n✅ All questions answered successfully. No retries needed.")
        return results
    
    logger.info(f"{len(failed_indices)} question(s) failed with ERROR_RESPONSE.")
    logger.info(f"Failed question IDs: {[results[i]['question_id'] for i in failed_indices]}")
    
    print(f"\n{'='*60}")
    print(f"  ⚠️  {len(failed_indices)} question(s) failed with ERROR_RESPONSE.")
    print(f"  Failed IDs: {[results[i]['question_id'] for i in failed_indices]}")
    print(f"{'='*60}")
    
    user_input = input(f"\n>>> Retry up to {MAX_RETRIES} attempts each? (y/n): ").strip().lower()
    
    if user_input != 'y':
        logger.info("User declined retry. Keeping existing results.")
        print("Keeping existing results.")
        return results
    
    logger.info(f"--- Starting Retry Phase (max {MAX_RETRIES} attempts per question) ---")
    resolved = 0
    
    with tqdm(total=len(failed_indices), desc="Retrying failed", unit="q",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        for idx in failed_indices:
            entry = results[idx]
            q_id = entry['question_id']
            
            for attempt in range(1, MAX_RETRIES + 1):
                logger.info(f"[Retry] Q{q_id} - Attempt {attempt}/{MAX_RETRIES}")
                pbar.set_postfix_str(f"Q{q_id} attempt {attempt}/{MAX_RETRIES}")
                
                system_prompt, user_prompt = construct_prompt(entry)
                response = get_llm_response(system_prompt, user_prompt, current_model, primary_client, provider, hf_client)
                
                time.sleep(0.5)  # To avoid rate limits
                
                if response != "ERROR_RESPONSE":
                    entry['model_output'] = response
                    save_results(results, out_file)
                    logger.info(f"[Retry Success] Q{q_id} succeeded on attempt {attempt}. Live-saved.")
                    resolved += 1
                    break
                else:
                    logger.warning(f"[Retry Failed] Q{q_id} - Attempt {attempt}/{MAX_RETRIES} failed.")
            else:
                logger.error(f"[Retry Exhausted] Q{q_id} still failed after {MAX_RETRIES} attempts.")
            
            pbar.update(1)
    
    # Final summary after retries
    still_failed = [i for i, r in enumerate(results) if r.get('model_output') == 'ERROR_RESPONSE']
    logger.info(f"Retry Phase Complete. Resolved: {resolved}/{len(failed_indices)}")
    
    if still_failed:
        logger.warning(f"Still failed ({len(still_failed)}): question IDs {[results[i]['question_id'] for i in still_failed]}")
        print(f"\n⚠️  Resolved {resolved}/{len(failed_indices)}. Still failed: {[results[i]['question_id'] for i in still_failed]}")
    else:
        print(f"\n✅ All {len(failed_indices)} previously failed questions resolved!")
    
    return results

# =============================================================================
# ASYNC / CONCURRENT MODE
# =============================================================================

async def async_process_question(sem, save_lock, q, idx, current_model, async_client,
                                  provider, results, results_by_id, out_file, pbar, counters):
    """
    Process a single question asynchronously with semaphore rate-limiting.
    Acquires the semaphore before making the API call.
    """
    async with sem:
        q_id = q.get('question_id')
        system_prompt, user_prompt = construct_prompt(q)

        response = await async_get_llm_response(
            system_prompt, user_prompt, current_model, async_client, provider
        )

        output_entry = q.copy()
        output_entry['model'] = current_model
        output_entry['model_output'] = response

        if response == "ERROR_RESPONSE":
            logger.error(f"[Async] Failed Q{q_id}: {q['question_prompt'][:50]}...")

        # Thread-safe update of results list
        async with save_lock:
            if q_id in results_by_id:
                results[results_by_id[q_id]] = output_entry
            else:
                results.append(output_entry)
                results_by_id[q_id] = len(results) - 1

            # Live save
            save_results(results, out_file)

        # Update progress
        if response != "ERROR_RESPONSE":
            counters['success'] += 1
        else:
            counters['fail'] += 1
        pbar.set_postfix_str(f"{counters['success']} ❌{counters['fail']}")
        pbar.update(1)

        logger.info(f"[Async][Live Save] Q{q_id} done")


async def run_concurrent_inference(data, to_process_questions, current_model, async_client,
                                    provider, results, results_by_id, out_file, max_concurrent):
    """Top-level async orchestrator: gather all tasks with semaphore rate-limiting."""
    sem = asyncio.Semaphore(max_concurrent)
    save_lock = asyncio.Lock()
    counters = {'success': 0, 'fail': 0}

    print(f"\n⚡ Concurrent mode: max {max_concurrent} simultaneous requests")
    print(f"   To process via LLM: {len(to_process_questions)}\n")
    logger.info(f"[Async] Starting concurrent inference: {len(to_process_questions)} questions, semaphore={max_concurrent}")

    with tqdm(total=len(to_process_questions), desc="Inference (async)", unit="q",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [✅{postfix}]") as pbar:

        tasks = []
        for idx, q in to_process_questions:
            task = asyncio.create_task(
                async_process_question(
                    sem, save_lock, q, idx, current_model, async_client,
                    provider, results, results_by_id, out_file, pbar, counters
                )
            )
            tasks.append(task)

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.warning("[Async] Tasks cancelled. Saving progress...")
            print("\n⏸️  Cancelled. Progress saved.")

    # Final save
    save_results(results, out_file)
    logger.info(f"[Async] Final save to {out_file}")
    print(f"\n💾 Saved to: {out_file}")
    print(f"📊 Results: {counters['success']} success / {counters['fail']} failed")


# =============================================================================
# MAIN
# =============================================================================

def main():
    
    # Config Args
    parser = argparse.ArgumentParser(description="Run inference for a specific LLM.")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to run")
    parser.add_argument("--provider", type=str, default="openrouter", choices=["openrouter", "hf", "google", "deepseek"],
                        help="Primary API provider (default: openrouter)")
    parser.add_argument("--fallback-hf", action="store_true", help="Enable HuggingFace as fallback when primary fails")
    parser.add_argument("--concurrent", type=int, default=None, metavar="N",
                        help="Enable async concurrent mode with N max simultaneous requests. "
                             "If omitted, runs sequentially (default). Supports openrouter, deepseek, and google providers.")
    args = parser.parse_args()
    
    # Get Args
    current_model = args.model
    provider = args.provider
    concurrent = args.concurrent

    if current_model not in CANDIDATE_MODELS:
        logger.warning(f"Model '{current_model}' is not in your config list. Please enter again.")
        return

    # Compute output path early so it's available for live-saving
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_name = current_model.replace("/", "_")
    out_file = os.path.join(OUTPUT_DIR, f"{safe_name}_output.json")

    # Resume support: load existing results if output file already exists
    results = []
    answered_ids = set()
    if os.path.exists(out_file):
        with open(out_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        answered_ids = {r['question_id'] for r in results if r.get('model_output') != 'ERROR_RESPONSE'}
        logger.info(f"Resuming: loaded {len(results)} existing results ({len(answered_ids)} successful)")
        print(f"📂 Resuming: {len(answered_ids)} already answered, {len(results) - len(answered_ids)} failed")

    # Build a lookup of existing results by question_id for duplicate prevention
    results_by_id = {r['question_id']: i for i, r in enumerate(results)}

    # Load Question Sets From input.json
    if not os.path.exists(INPUT_PATH):
        logger.error(f"Input file missing at {INPUT_PATH}")
        print(f"❌ Input file missing at {INPUT_PATH}")
        return
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_questions = len(data)
    to_process = total_questions - len(answered_ids)

    logger.info(f"--- Starting Inference for: {current_model} ---")
    print(f"\n🚀 Starting Inference for: {current_model}")
    print(f"   Total: {total_questions} | Already done: {len(answered_ids)} | To process: {to_process}")

    # =========================================================================
    # CONCURRENT MODE
    # =========================================================================
    if concurrent is not None and concurrent > 0:
        # Create async client based on provider
        async_client = None
        if provider == "openrouter":
            async_client = create_async_openrouter_client(current_model)
        elif provider == "deepseek":
            async_client = create_async_deepseek_client(current_model)
        elif provider == "google":
            async_client = create_async_google_client()
        else:
            print(f"❌ Provider '{provider}' is not supported in concurrent mode.")
            return

        if not async_client:
            print(f"❌ Failed to create async client for provider '{provider}'. Aborting.")
            return

        print(f"📡 Primary provider: {provider}")
        logger.info(f"--- Starting CONCURRENT Inference | model={current_model} | provider={provider} | concurrency={concurrent} ---")

        # Collect questions that still need inference
        to_process_questions = [
            (idx, q) for idx, q in enumerate(data) if q.get('question_id') not in answered_ids
        ]

        try:
            asyncio.run(
                run_concurrent_inference(
                    data, to_process_questions, current_model, async_client,
                    provider, results, results_by_id, out_file, concurrent
                )
            )
        except KeyboardInterrupt:
            logger.warning("Interrupted by user. Saving progress...")
            print(f"\n\n⏸️  Interrupted. Progress saved.")
        finally:
            save_results(results, out_file)
            success_count = sum(1 for r in results if r.get('model_output') != 'ERROR_RESPONSE')
            fail_count = len(results) - success_count
            logger.info(f"Saved {len(results)} responses to {out_file} ( {fail_count} failures / {success_count} successes )")

        # Retry phase (sequential — simple and safe)
        primary_client = None
        if provider == "openrouter":
            primary_client = create_openrouter_client(current_model)
        elif provider == "deepseek":
            primary_client = create_deepseek_client(current_model)
        elif provider == "google":
            primary_client = create_google_client()

        if primary_client:
            results = retry_failed_questions(results, out_file, current_model, primary_client, provider)
            save_results(results, out_file)
            logger.info(f"Final output saved to {out_file}")
        return

    # =========================================================================
    # SEQUENTIAL MODE (default)
    # =========================================================================

    # Create primary client based on provider
    primary_client = None
    if provider == "openrouter":
        primary_client = create_openrouter_client(current_model)
    elif provider == "hf":
        primary_client = create_huggingface_client(current_model)
        if not primary_client:
            print(f"❌ HuggingFace client creation failed (no mapping or token). Aborting.")
            return
    elif provider == "google":
        primary_client = create_google_client()
        if not primary_client:
            print(f"❌ Google clients creation failed (no API key). Aborting.")
            return
    elif provider == "deepseek":
        primary_client = create_deepseek_client(current_model)
        if not primary_client:
            print(f"❌ DeepSeek client creation failed (is DeepSeek installed and model available?). Aborting.")
            return     
    
    print(f"📡 Primary provider: {provider}")

    # Create optional HF fallback client
    hf_client = None
    if args.fallback_hf and provider != "hf":
        hf_client = create_huggingface_client(current_model)
        if hf_client:
            print(f"🔀 HuggingFace fallback enabled")
        else:
            print(f"⚠️  HuggingFace fallback requested but unavailable (no mapping or token)")

    print(f"   Total: {total_questions} | Already done: {len(answered_ids)} | To process: {to_process}\n")

    try:
        with tqdm(total=total_questions, initial=len(answered_ids), desc="Progress", unit="q",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [✅{postfix}]") as pbar:
            for i, q in enumerate(data):
                q_id = q.get('question_id')
                
                # Skip questions already answered successfully (resume support)
                if q_id in answered_ids:
                    continue
                
                system_prompt, user_prompt = construct_prompt(q)
                
                response = get_llm_response(system_prompt, user_prompt, current_model, primary_client, provider, hf_client)
                
                time.sleep(0.5)  # To avoid rate limits
                
                output_entry = q.copy()
                output_entry['model'] = current_model
                output_entry['model_output'] = response
                
                if response == "ERROR_RESPONSE":
                    logger.error(f"Failed to get response for [[[ Q{i+1} ]]]: {q['question_prompt'][:50]}...")
                
                # Replace existing ERROR entry or append new entry (prevents duplicates on resume)
                if q_id in results_by_id:
                    results[results_by_id[q_id]] = output_entry
                else:
                    results.append(output_entry)
                    results_by_id[q_id] = len(results) - 1
                
                success_count = sum(1 for r in results if r.get('model_output') != 'ERROR_RESPONSE')
                fail_count = len(results) - success_count
                
                # Live-save after each question to prevent data loss
                save_results(results, out_file)
                logger.info(f"[Live Save] Saved {len(results)} results so far to {out_file}")
                
                pbar.set_postfix_str(f"{success_count} ❌{fail_count}")
                pbar.update(1)

    except KeyboardInterrupt:
        logger.warning(f"Interrupted by user. Saving progress...")
        print(f"\n\n⏸️  Interrupted. Progress saved.")
    except Exception as e:
        logger.critical(f"Pipeline failed for {current_model}: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        
    finally:
        # Final save to ensure everything is persisted
        save_results(results, out_file)
        success_count = sum(1 for r in results if r.get('model_output') != 'ERROR_RESPONSE')
        fail_count = len(results) - success_count
        logger.info(f"Saved {len(results)} responses to {out_file} ( {fail_count} failures / {success_count} successes )")
        print(f"\n📊 Results: {success_count} success / {fail_count} failed / {len(results)} total")
        print(f"💾 Saved to: {out_file}")
    
    # --- RETRY PHASE: Re-attempt failed questions ---
    results = retry_failed_questions(results, out_file, current_model, primary_client, provider, hf_client)
    save_results(results, out_file)
    logger.info(f"Final output saved to {out_file}")

if __name__ == "__main__":
    main()