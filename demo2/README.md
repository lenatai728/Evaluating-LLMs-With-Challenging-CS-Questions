

create venv:
python3 -m venv venv

# 0a. Setup Module
chmod u+x setup.sh
./setup.sh

pip uninstall -y -r requirements.txt


# 0b. Terminal command to create folders
cd src
mkdir -p data/0_raw data/1_model_outputs data/2_processed data/3_scores data/4_final_results src logs


# Terminal command to Run programs
python3 1_get_model_responses.py --model openai/gpt-4o-mini   

# Bugs to Fix
- Modify Question Prompt: tell llm to remove math equation format, e.g. "Answer: \\( 2 \\log_2(n + 1) \\)" to "2 * log2(n + 1)"

- for MS type, make marks partial if only answer some correct. (done: 0-2 scoring)
    - done: if llm answers more choice than gold answer's choices -> score 0
    - done: if llm answers same # of choices & partially correct -> score 1

# TODO:
- 1_get_model_responses.py (do at later stage)
    - add arg "--run" that prompts the number of data entries to process 
    - count the number of data entries successfully responsed by llms and written to "answer_output"
    - get: failed at what question id


- 4_eval_llm_judge.py
    - For Non-MC/MS/TF Questions, Implement Semantic Similarity i.e. RoberTa-large on 'answer_output' and 'gold_answer'
        - if >0.9 -> evaluate by 'llms-as-judge'
        - if <0.9 -> score 2 (full score)


# procedure

## generate LLMs responses from input.json
python3 1_get_model_responses.py --model openai/gpt-4o-mini
+ openai_gpt-4o-mini_output.json

## separate LLMs response strings (Rationale:...Answer:...) to answer_output and rationale_output
python3 2_process_outputs.py
+ openai_gpt-4o-mini_processed.json

## evaluate MC/MS/TF only by exact matching
python3 3_eval_mc_ms_tf.py
+ openai_gpt-4o-mini_scored_partial.json

## In Answer Mode, evaluate answers from FB/OE questions by semantic similiarity & LLM judge
## In Rationale Mode, evaluate rationales from ALL questions by semantic similarity & LLM judge, producing evaluation scores each LLM evaluator, in v3, error taxonomy is not yet implemented. Other problems arise: payment required for API tokens if not set max_tokens, setting max_tokens is ok for testing only but not real evaluation
python3 4_eval_llm_judge_v4.py --mode answer --input_file openai_gpt-4o-mini_scored_partial.json
+ openai_gpt-4o-mini_scored_partial_final_answer.json
(Wait for expert score to calculate Final Answer Score manually…)

python3 4_eval_llm_judge_v4.py --mode rationale --input_file openai_gpt-4o-mini_scored_partial_final_answer.json
+ openai_gpt-4o-mini_scored_partial_final_answer_final_rationale.json
(actually only evaluation scores of LLMs, not yet final rationale score for LLMs which refers to the average eval score of LLMs)

python3 5_generate_report.py --input_file openai_gpt-4o-mini_scored_partial_final_answer_final_rationale.json
