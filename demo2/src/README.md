# A. Setup Virtual Environment + Required Modules (MacOS Terminal only)
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
## 2. Generate LLMs responses from input.json
```
python3 1_get_model_responses.py --model openai/gpt-4o-mini   
```
- One Args: 
    - "--model <model-name>"
## 3. Separate LLMs response strings (Rationale:...Answer:...) to Answer & Rationale outputs
```
python3 2_process_outputs.py
```
## 4. Evaluate on answers from MC/MS/TF questions ONLY by exact matching
```
python3 3_eval_mc_ms_tf.py
```
## 5. Evaluate on answers from FB/OE questions by semantic similiarity & LLMs judge
```
python3 4_eval_llm_judge.py --mode answer --input_file openai_gpt-4o-mini_scored_partial.json
```
- Two Args:
    - "--mode answer" : Evaluate on answers / "--mode rationale": Evaluate on rationales
    - "--input_file <filename>"
## 6. Evaluate on rationales from ALL questions by semantic similarity & LLMs judge
```
python3 4_eval_llm_judge.py --mode rationale --input_file openai_gpt-4o-mini_scored_partial_final_answer.json
```
- Two Args:
    - "--mode answer" : Evaluate on answers / "--mode rationale": Evaluate on rationales
    - "--input_file <filename>"
## 7. Generate reports for specific model
```
python3 5_generate_report.py --input_file openai_gpt-4o-mini_scored_partial_final_answer_final_rationale.json
```
- One Arg:
    - "--input_file <filename>"
# C. Note
## Terminal command to uninstall modules from requirements.txt
```
pip uninstall -y -r requirements.txt
```
## Terminal command to create required folders initially (if not included)
```
cd src
mkdir -p data/0_raw data/1_model_outputs data/2_processed data/3_scores data/4_final_results src logs
```