# A. Setup Module
```
python3 -m venv venv

chmod u+x setup.sh

./setup.sh
```

# B. Demonstration of Procedures
```
cd demo2/src
```
## 1. Get into virtual environment
```
source venv/bin/activate
```
## 2. generate LLMs responses from input.json
```
python3 1_get_model_responses.py --model openai/gpt-4o-mini   
```
## 3. separate LLMs response strings (Rationale:...Answer:...) to answer & rationale outputs
```
python3 2_process_outputs.py
```
## 4. Evaluate answers from FB/OE questions by semantic similiarity & LLMs judge
```
python3 4_eval_llm_judge.py --mode answer --input_file openai_gpt-4o-mini_scored_partial.json
```
## 5. Evaluate rationales from ALL questions by semantic similarity & LLMs judge
```
python3 4_eval_llm_judge.py --mode rationale --input_file openai_gpt-4o-mini_scored_partial_final_answer.json
```
## 6. Generate reports for specific model
```
python3 5_generate_report.py --input_file openai_gpt-4o-mini_scored_partial_final_answer_final_rationale.json
```

# C. Note
## Terminal command to uninstall modules from requirements.txt
```
pip uninstall -y -r requirements.txt
```

## Terminal command to create folders
```
cd src

mkdir -p data/0_raw data/1_model_outputs data/2_processed data/3_scores data/4_final_results src logs

```