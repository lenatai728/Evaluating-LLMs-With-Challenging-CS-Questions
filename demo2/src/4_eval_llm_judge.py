import json
import os
import logging
import argparse
import time
import re
import asyncio
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from openai import AsyncOpenAI
from llm_clients import (
    create_deepseek_client,
    create_openrouter_client,
    get_llm_response,
)

# --- CONFIGURATION ---
INPUT_DIR = os.path.join('data', '3_scores')
OUTPUT_DIR = os.path.join('data', '3_scores')
SEMANTIC_THRESHOLD = 0.90
MAX_RETRIES = 3  # For retry phase
ASYNC_RETRIES = 3
ASYNC_BACKOFF_BASE = 2
load_dotenv()

# --- JUDGE CONFIGURATION (Add API keys here) ---
JUDGE_CONFIGS = [
    {
        "id": "deepseek-reasoner", 
        "model_name": "deepseek-reasoner", 
        "base_url": "https://api.deepseek.com"
    },
    # {
    #     "id": "gpt_4o_mini", 
    #     "model_name": "openai/gpt-4o-mini", 
    #     "base_url": "https://openrouter.ai/api/v1"
    # },
]

# --- LOGGING (file only, terminal stays clean for tqdm) ---
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/4_eval_llm_judge.log")]
)
logger = logging.getLogger(__name__)

# =============================================================================
# PROVIDER & CLIENT HELPERS
# =============================================================================

def get_provider_for_judge(judge_config):
    """Determine the llm_clients provider string from a judge's base_url."""
    base_url = judge_config.get("base_url", "")
    if "deepseek" in base_url:
        return "deepseek"
    elif "openrouter" in base_url:
        return "openrouter"
    return "openrouter"  # default fallback


def create_client_for_judge(judge_config):
    """Create the appropriate llm_clients client for a judge."""
    provider = get_provider_for_judge(judge_config)
    model_name = judge_config["model_name"]
    if provider == "deepseek":
        return create_deepseek_client(model_name)
    elif provider == "openrouter":
        return create_openrouter_client(model_name)
    return None


def create_async_client_for_judge(judge_config):
    """Create an AsyncOpenAI client for a judge (used in --concurrent mode)."""
    base_url = judge_config.get("base_url", "https://openrouter.ai/api/v1")
    provider = get_provider_for_judge(judge_config)
    model_name = judge_config["model_name"]

    if provider == "deepseek":
        key_str = f"{model_name.split('/')[-1].upper()}-API-KEY"
    else:
        key_str = f"{model_name.split('/')[-1].upper()}-OPENROUTER-API-KEY"

    api_key = os.getenv(key_str)
    if not api_key:
        logger.error(f"API key '{key_str}' not found in .env for async client.")
        return None

    return AsyncOpenAI(base_url=base_url, api_key=api_key)


# =============================================================================
# JUDGE RESPONSE (uses shared llm_clients — sequential mode)
# =============================================================================

def get_judge_response(q_id, judge_config, client, provider, system_prompt, user_prompt, eval_mode):
    """
    Call the judge model via `get_llm_response()` (shared retry/backoff),
    then parse the raw text into (score, explanation_code).
    Returns (-1, None) on error.
    """
    raw = get_llm_response(
        system_prompt, user_prompt,
        judge_config["model_name"], client, provider
    )

    if raw == "ERROR_RESPONSE":
        logger.error(f"[{q_id}] Judge {judge_config['id']} returned ERROR_RESPONSE")
        return -1, None

    content = raw.strip()
    logger.info(f"[{q_id}] Judge {judge_config['id']} raw response: {content}")

    return parse_judge_response(content, eval_mode)


def parse_judge_response(content, eval_mode):
    """Parse raw judge text into (score, ex_code). Shared by sync & async paths."""
    if eval_mode == "answer":
        match = re.search(r'-?\d+', content)
        score = int(match.group()) if match else -1
        return score, None

    elif eval_mode == "rationale":
        score = -1
        ex_code = None
        lines = content.split('\n')
        for line in lines:
            if line.startswith("Score:"):
                match = re.search(r'-?\d+', line)
                score = int(match.group()) if match else -1
            elif line.startswith("Code:"):
                ex_code = line.split("Code:")[1].strip()
        return score, ex_code

    return -1, None

# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================

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
            "Return ONLY the number."
        )
        user = f"Question: {question}\nGold Answer: {gold}\nStudent Answer: {model_output}\n\nScore:"

    elif mode == "rationale": 
        system = (
            "You are an expert Computer Science evaluator. Score Student Rationale vs Gold Rationale (0, 1, 2) of the Question and provide the Code that best explains your verdict.\n\n"
            "## SCORING\n"
            "0:Entirely incorrect or irrelevant, Logic contains contradictions, hallucinations, or valid steps are missing. | 1:Reasoning is partially correct but misses key steps or is vague. | 2:Fully correct, logically sound, complete, and leads to the correct conclusion. | -1:Evaluation Error (Empty/Invalid Input)\n\n"
            "## TAXONOMY\n"
            "E0:Correct | E1:Conceptual Error | E2:Logic Conflict | E3:Incomplete Reasoning | E4:Irrelevant | E5:Calculation Error | E6:Hallucination | E7:Question Misinterpret | E8:Term Error | E9:Incomplete Answer | E10:Ambiguous | E11:Shallow Justification\n\n"
            "## Return in this exact format without any additional text:\n"
            "Score: <number>\nCode: <E#>"
        )
        user = f"Question: {question}\nGold Rationale: {gold}\nStudent Rationale: {model_output}"
    return system, user

# =============================================================================
# HELPERS
# =============================================================================

def save_results(data, path):
    """Atomically save results to disk."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def update_average_score(entry, eval_mode):
    """Recalculate average score from all available judge scores."""
    available_scores = []
    for jc in JUDGE_CONFIGS:
        sk = f"{jc['id']}_{eval_mode}_score"
        if sk in entry and entry[sk] != -1:
            available_scores.append(entry[sk])
    if available_scores:
        entry[f"average_{eval_mode}_score"] = round(sum(available_scores) / len(available_scores), 4)
    else:
        entry[f"average_{eval_mode}_score"] = -1


def retry_failed_evaluations(data, output_path, eval_mode, active_judges, judge_clients, judge_providers):
    """Prompt user to retry entries where any active judge got score == -1."""
    failed_indices = []
    for idx, entry in enumerate(data):
        for judge in active_judges:
            key = f"{judge['id']}_{eval_mode}_score"
            if entry.get(key) == -1:
                failed_indices.append(idx)
                break  # one failed judge is enough to flag this entry

    if not failed_indices:
        logger.info("All evaluations succeeded. No retries needed.")
        print("\n✅ All evaluations succeeded. No retries needed.")
        return data

    logger.info(f"{len(failed_indices)} entry/entries have score == -1.")
    print(f"\n{'='*60}")
    print(f"  ⚠️  {len(failed_indices)} entry/entries have score == -1.")
    print(f"{'='*60}")

    user_input = input(f"\n>>> Retry up to {MAX_RETRIES} attempts each? (y/n): ").strip().lower()
    if user_input != 'y':
        logger.info("User declined retry.")
        print("Keeping existing results.")
        return data

    logger.info(f"--- Starting Retry Phase (max {MAX_RETRIES} attempts per entry) ---")
    resolved = 0

    with tqdm(total=len(failed_indices), desc="Retrying failed", unit="q",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        for idx in failed_indices:
            entry = data[idx]
            q_id = entry.get("question_id", idx)

            key_output = "answer_output" if eval_mode == "answer" else "rationale_output"
            key_gold = "gold_answer" if eval_mode == "answer" else "gold_rationale"

            model_text = str(entry.get(key_output, ""))
            gold_text = str(entry.get(key_gold, ""))
            question_text = entry.get("question_prompt", "")
            system_prompt, user_prompt = construct_prompts(eval_mode, question_text, gold_text, model_text)

            entry_resolved = False
            for judge in active_judges:
                score_key = f"{judge['id']}_{eval_mode}_score"
                if entry.get(score_key) != -1:
                    continue  # this judge already OK

                client = judge_clients[judge['id']]
                provider = judge_providers[judge['id']]

                for attempt in range(1, MAX_RETRIES + 1):
                    logger.info(f"[Retry] Q{q_id} Judge {judge['id']} attempt {attempt}/{MAX_RETRIES}")
                    pbar.set_postfix_str(f"Q{q_id} {judge['id']} #{attempt}")

                    score, ex_code = get_judge_response(q_id, judge, client, provider, system_prompt, user_prompt, eval_mode)
                    time.sleep(0.5)

                    if score != -1:
                        entry[score_key] = score
                        if eval_mode == "rationale":
                            entry[f"{judge['id']}_{eval_mode}_exCode"] = ex_code
                        save_results(data, output_path)
                        logger.info(f"[Retry Success] Q{q_id} {judge['id']} = {score}")
                        entry_resolved = True
                        break
                    else:
                        logger.warning(f"[Retry Failed] Q{q_id} {judge['id']} attempt {attempt} failed.")
                else:
                    logger.error(f"[Retry Exhausted] Q{q_id} {judge['id']} still -1 after {MAX_RETRIES} attempts.")

            update_average_score(entry, eval_mode)

            if entry_resolved:
                resolved += 1
            pbar.update(1)

    still_failed = sum(1 for idx in failed_indices if any(
        data[idx].get(f"{j['id']}_{eval_mode}_score") == -1 for j in active_judges
    ))
    logger.info(f"Retry Phase Complete. Resolved: {resolved}/{len(failed_indices)}")

    if still_failed:
        print(f"\n⚠️  Resolved {resolved}/{len(failed_indices)}. Still {still_failed} with errors.")
    else:
        print(f"\n✅ All {len(failed_indices)} previously failed entries resolved!")

    return data

# =============================================================================
# ASYNC / CONCURRENT MODE
# =============================================================================

async def async_call_judge(async_client, model_name, system_prompt, user_prompt):
    """
    Single async API call with retry + exponential backoff.
    Returns raw response text or "ERROR_RESPONSE".
    """
    for attempt in range(1, ASYNC_RETRIES + 1):
        try:
            response = await async_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content

            if content is None or content.strip() == "":
                logger.warning(f"[Async] Empty response for {model_name} (attempt {attempt}/{ASYNC_RETRIES})")
                if attempt < ASYNC_RETRIES:
                    wait = ASYNC_BACKOFF_BASE ** attempt
                    await asyncio.sleep(wait)
                    continue
                return "ERROR_RESPONSE"

            return content

        except Exception as e:
            logger.error(f"[Async] API Error for {model_name} (attempt {attempt}/{ASYNC_RETRIES}): {e}")
            if attempt < ASYNC_RETRIES:
                wait = ASYNC_BACKOFF_BASE ** attempt
                await asyncio.sleep(wait)
            else:
                return "ERROR_RESPONSE"

    return "ERROR_RESPONSE"


async def async_get_judge_response(q_id, judge_config, async_client, system_prompt, user_prompt, eval_mode):
    """Async version of get_judge_response: call API then parse score."""
    raw = await async_call_judge(async_client, judge_config["model_name"], system_prompt, user_prompt)

    if raw == "ERROR_RESPONSE":
        logger.error(f"[Async][{q_id}] Judge {judge_config['id']} returned ERROR_RESPONSE")
        return -1, None

    content = raw.strip()
    logger.info(f"[Async][{q_id}] Judge {judge_config['id']} raw response: {content}")
    return parse_judge_response(content, eval_mode)


async def async_process_entry(sem, save_lock, entry, idx, eval_mode, active_judges,
                              async_clients, data, output_path, pbar, counters):
    """
    Process a single entry asynchronously with semaphore rate-limiting.
    Acquires the semaphore before making any API call.
    """
    async with sem:
        key_output = "answer_output" if eval_mode == "answer" else "rationale_output"
        key_gold = "gold_answer" if eval_mode == "answer" else "gold_rationale"

        model_text = str(entry.get(key_output, ""))
        gold_text = str(entry.get(key_gold, ""))
        question_text = entry.get("question_prompt", "")

        system_prompt, user_prompt = construct_prompts(eval_mode, question_text, gold_text, model_text)
        entry[f"{eval_mode}_eval_method"] = "llm_judge"

        for judge in active_judges:
            score_key = f"{judge['id']}_{eval_mode}_score"
            if score_key in entry:
                continue  # already scored by this judge
            
            async_client = async_clients[judge['id']]
            score, ex_code = await async_get_judge_response(
                idx, judge, async_client, system_prompt, user_prompt, eval_mode
            )
            entry[score_key] = score
            if eval_mode == "rationale" and ex_code:
                entry[f"{judge['id']}_{eval_mode}_exCode"] = ex_code

        # Update average
        update_average_score(entry, eval_mode)

        # Live save (thread-safe via asyncio.Lock)
        async with save_lock:
            save_results(data, output_path)

        # Update progress
        cur_score = entry.get(f"average_{eval_mode}_score", -1)
        if cur_score != -1:
            counters['success'] += 1
        else:
            counters['fail'] += 1
        pbar.set_postfix_str(f"{counters['success']} ❌{counters['fail']}")
        pbar.update(1)

        logger.info(f"[Async][Live Save] Completed entry {idx+1}")


async def run_concurrent_evaluation(data, output_path, eval_mode, active_judges,
                                     async_clients, similarity_model, max_concurrent):
    """Top-level async orchestrator: gather all tasks with semaphore rate-limiting."""
    total = len(data)
    sem = asyncio.Semaphore(max_concurrent)
    save_lock = asyncio.Lock()
    counters = {'success': 0, 'fail': 0}

    # Pre-pass: handle answer-mode auto-pass (RoBERTa) synchronously
    # and collect indices that need LLM judging
    to_process_indices = []

    for idx, entry in enumerate(data):
        # Skip MC/MS/TF in answer mode
        if eval_mode == "answer":
            qt = entry.get("question_type", "")
            aem = entry.get("answer_eval_method", "")
            if qt.lower() in ("mc", "ms", "tf") and aem == "exact_match":
                continue

        # Skip already fully scored
        all_active_scored = all(
            f"{j['id']}_{eval_mode}_score" in entry for j in active_judges
        )
        if all_active_scored:
            continue

        if eval_mode == "answer":
            # Compute semantic similarity (synchronous, CPU-bound)
            key_sim = f"{eval_mode}_semantic_similarity"
            model_text = str(entry.get("answer_output", ""))
            gold_text = str(entry.get("gold_answer", ""))

            if key_sim in entry:
                cosine_score = entry[key_sim]
            else:
                if not model_text.strip():
                    cosine_score = -1
                else:
                    embeddings = similarity_model.encode([model_text, gold_text], convert_to_tensor=True)
                    cosine_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
                entry[key_sim] = round(cosine_score, 4)

            if cosine_score >= SEMANTIC_THRESHOLD:
                if entry.get(f"{eval_mode}_eval_method") != "roberta_auto_pass":
                    entry[f"{eval_mode}_score"] = 2
                    entry[f"{eval_mode}_eval_method"] = "roberta_auto_pass"
                for judge in active_judges:
                    entry[f"{judge['id']}_{eval_mode}_score"] = 2
                update_average_score(entry, eval_mode)
                continue  # no LLM call needed

        to_process_indices.append(idx)

    already_done = total - len(to_process_indices)

    print(f"\n⚡ Concurrent mode: max {max_concurrent} simultaneous requests")
    print(f"   Total: {total} | Already done/auto-pass: {already_done} | To process via LLM: {len(to_process_indices)}\n")
    logger.info(f"[Async] Starting concurrent evaluation: {len(to_process_indices)} entries, semaphore={max_concurrent}")

    with tqdm(total=len(to_process_indices), desc="Evaluating (async)", unit="q",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [✅{postfix}]") as pbar:

        tasks = []
        for idx in to_process_indices:
            task = asyncio.create_task(
                async_process_entry(
                    sem, save_lock, data[idx], idx, eval_mode, active_judges,
                    async_clients, data, output_path, pbar, counters
                )
            )
            tasks.append(task)

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.warning("[Async] Tasks cancelled. Saving progress...")
            print("\n⏸️  Cancelled. Progress saved.")

    # Final save
    save_results(data, output_path)
    logger.info(f"[Async] Final save to {output_path}")
    print(f"\n💾 Saved to: {output_path}")
    print(f"📊 Results: {counters['success']} success / {counters['fail']} failed")

    return data

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM responses using LLM judges.")
    parser.add_argument("--mode", type=str, required=True, choices=["answer", "rationale"], help="Mode: answer or rationale")
    parser.add_argument("--input_file", type=str, required=True, help="Filename inside data/3_scores/ (e.g., model_processed.json)")
    parser.add_argument("--judge_id", type=str, help="Specify a single judge ID from JUDGE_CONFIGS. If omitted, uses all active judges.")
    parser.add_argument("--concurrent", type=int, default=None, metavar="N",
                        help="Enable async concurrent mode with N max simultaneous requests. "
                             "If omitted, runs sequentially (default).")
    args = parser.parse_args()
    
    eval_mode = args.mode
    input_file = args.input_file
    target_judge_id = args.judge_id
    concurrent = args.concurrent
    
    input_path = os.path.join(INPUT_DIR, input_file)
    
    # Filter active judges
    if target_judge_id:
        active_judges = [j for j in JUDGE_CONFIGS if j['id'] == target_judge_id]
        if not active_judges:
            print(f"❌ Judge ID '{target_judge_id}' not found in JUDGE_CONFIGS.")
            return
        output_filename = input_file.replace(".json", f"_{target_judge_id}_{eval_mode}.json")
    else:
        active_judges = JUDGE_CONFIGS
        output_filename = input_file.replace(".json", f"_final_{eval_mode}.json")

    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # 1. Load Input Data
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Resume Support: load existing output and merge scores into data
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        existing_by_id = {e.get("question_id"): e for e in existing_data}
        for entry in data:
            q_id = entry.get("question_id")
            if q_id in existing_by_id:
                prev = existing_by_id[q_id]
                for key, val in prev.items():
                    if key not in entry:
                        entry[key] = val
        logger.info(f"Resumed: merged scores from {output_path} ({len(existing_data)} entries)")
        print(f"📂 Resuming from existing output with {len(existing_data)} entries")

    # 3. Load RoBERTa if needed
    similarity_model = None
    if eval_mode == "answer":
        print("🔄 Loading RoBERTa model...")
        logger.info("Loading RoBERTa model...")
        similarity_model = SentenceTransformer('all-roberta-large-v1')
        print("✅ RoBERTa loaded.")

    # =========================================================================
    # CONCURRENT MODE
    # =========================================================================
    if concurrent is not None and concurrent > 0:
        # Create async clients
        async_clients = {}
        for judge in active_judges:
            ac = create_async_client_for_judge(judge)
            if not ac:
                print(f"❌ Failed to create async client for judge '{judge['id']}'. Aborting.")
                return
            async_clients[judge['id']] = ac
            logger.info(f"Created async client for judge '{judge['id']}'")

        judge_names = ", ".join(j['id'] for j in active_judges)
        print(f"\n🚀 Starting Evaluation: mode={eval_mode.upper()}, judge(s)={judge_names}")
        logger.info(f"--- Starting CONCURRENT Evaluation | mode={eval_mode.upper()} | Judges: {judge_names} | concurrency={concurrent} ---")

        try:
            data = asyncio.run(
                run_concurrent_evaluation(
                    data, output_path, eval_mode, active_judges,
                    async_clients, similarity_model, concurrent
                )
            )
        except KeyboardInterrupt:
            logger.warning("Interrupted by user. Saving progress...")
            print(f"\n\n⏸️  Interrupted. Progress saved.")
        finally:
            save_results(data, output_path)

        # Retry phase (sequential — simple and safe)
        judge_clients = {}
        judge_providers = {}
        for judge in active_judges:
            provider = get_provider_for_judge(judge)
            client = create_client_for_judge(judge)
            judge_clients[judge['id']] = client
            judge_providers[judge['id']] = provider

        data = retry_failed_evaluations(data, output_path, eval_mode, active_judges, judge_clients, judge_providers)
        save_results(data, output_path)
        logger.info(f"Final output saved to {output_path}")
        return

    # =========================================================================
    # SEQUENTIAL MODE (default)
    # =========================================================================

    # Create sync clients
    judge_clients = {}
    judge_providers = {}
    for judge in active_judges:
        provider = get_provider_for_judge(judge)
        client = create_client_for_judge(judge)
        if not client:
            print(f"❌ Failed to create client for judge '{judge['id']}'. Aborting.")
            return
        judge_clients[judge['id']] = client
        judge_providers[judge['id']] = provider
        logger.info(f"Created {provider} client for judge '{judge['id']}'")

    # Count how many entries need processing
    total = len(data)
    already_done = 0
    for entry in data:
        if eval_mode == "answer":
            qt = entry.get("question_type", "")
            aem = entry.get("answer_eval_method", "")
            if qt.lower() in ("mc", "ms", "tf") and aem == "exact_match":
                already_done += 1
                continue
        all_scored = all(
            f"{j['id']}_{eval_mode}_score" in entry for j in active_judges
        )
        if all_scored:
            already_done += 1

    to_process = total - already_done
    judge_names = ", ".join(j['id'] for j in active_judges)

    print(f"\n🚀 Starting Evaluation: mode={eval_mode.upper()}, judge(s)={judge_names}")
    print(f"   Total: {total} | Already done: {already_done} | To process: {to_process}\n")
    logger.info(f"--- Starting Evaluation Mode: {eval_mode.upper()} | Judges: {judge_names} ---")

    # Processing Loop
    success_count = 0
    fail_count = 0

    try:
        with tqdm(total=total, initial=already_done, desc="Evaluating", unit="q",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [✅{postfix}]") as pbar:
            for idx, entry in enumerate(data):

                # Skip MC/MS/TF in answer mode (already exact match scored)
                if eval_mode == "answer":
                    qt = entry.get("question_type", "")
                    aem = entry.get("answer_eval_method", "")
                    if qt.lower() in ("mc", "ms", "tf") and aem == "exact_match":
                        continue

                # Determine keys
                key_output = "answer_output" if eval_mode == "answer" else "rationale_output"
                key_gold = "gold_answer" if eval_mode == "answer" else "gold_rationale"
                key_sim = f"{eval_mode}_semantic_similarity"

                # Check if all active judges already scored
                all_active_scored = all(
                    f"{j['id']}_{eval_mode}_score" in entry for j in active_judges
                )
                if all_active_scored:
                    continue

                model_text = str(entry.get(key_output, ""))
                gold_text = str(entry.get(key_gold, ""))
                question_text = entry.get("question_prompt", "")

                if eval_mode == "answer":
                    # Semantic similarity (preserve existing)
                    if key_sim in entry:
                        cosine_score = entry[key_sim]
                    else:
                        if not model_text.strip():
                            cosine_score = -1
                        else:
                            embeddings = similarity_model.encode([model_text, gold_text], convert_to_tensor=True)
                            cosine_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
                        entry[key_sim] = round(cosine_score, 4)

                    if cosine_score >= SEMANTIC_THRESHOLD:
                        if entry.get(f"{eval_mode}_eval_method") != "roberta_auto_pass":
                            logger.info(f"[{idx}] High Similarity ({cosine_score:.3f}). Auto-passing.")
                            entry[f"{eval_mode}_score"] = 2
                            entry[f"{eval_mode}_eval_method"] = "roberta_auto_pass"
                        for judge in active_judges:
                            entry[f"{judge['id']}_{eval_mode}_score"] = 2
                    else:
                        logger.debug(f"[{idx}] Low Similarity ({cosine_score:.3f}). Sending to judges.")
                        system_prompt, user_prompt = construct_prompts(eval_mode, question_text, gold_text, model_text)
                        entry[f"{eval_mode}_eval_method"] = "llm_judge"
                        for judge in active_judges:
                            score_key = f"{judge['id']}_{eval_mode}_score"
                            if score_key in entry:
                                continue
                            score, ex_code = get_judge_response(
                                idx, judge, judge_clients[judge['id']],
                                judge_providers[judge['id']], system_prompt, user_prompt, eval_mode
                            )
                            entry[score_key] = score
                            if eval_mode == "rationale":
                                entry[f"{judge['id']}_{eval_mode}_exCode"] = ex_code
                            time.sleep(0.5)

                elif eval_mode == "rationale":
                    system_prompt, user_prompt = construct_prompts(eval_mode, question_text, gold_text, model_text)
                    entry[f"{eval_mode}_eval_method"] = "llm_judge"
                    for judge in active_judges:
                        score_key = f"{judge['id']}_{eval_mode}_score"
                        if score_key in entry:
                            continue
                        score, ex_code = get_judge_response(
                            idx, judge, judge_clients[judge['id']],
                            judge_providers[judge['id']], system_prompt, user_prompt, eval_mode
                        )
                        entry[score_key] = score
                        if eval_mode == "rationale":
                            entry[f"{judge['id']}_{eval_mode}_exCode"] = ex_code
                        time.sleep(0.5)

                # Update average score
                update_average_score(entry, eval_mode)

                # Live save after every item
                save_results(data, output_path)
                logger.info(f"[Live Save] Saved after item {idx+1}")

                # Update progress counts
                cur_score = entry.get(f"average_{eval_mode}_score", -1)
                if cur_score != -1:
                    success_count += 1
                else:
                    fail_count += 1
                pbar.set_postfix_str(f"{success_count} ❌{fail_count}")
                pbar.update(1)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Saving progress...")
        print(f"\n\n⏸️  Interrupted. Progress saved.")
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
    finally:
        save_results(data, output_path)
        logger.info(f"Final save to {output_path}")
        print(f"\n💾 Saved to: {output_path}")
        print(f"📊 Results: {success_count} success / {fail_count} failed")

    # --- RETRY PHASE ---
    data = retry_failed_evaluations(data, output_path, eval_mode, active_judges, judge_clients, judge_providers)
    save_results(data, output_path)
    logger.info(f"Final output saved to {output_path}")


if __name__ == "__main__":
    main()