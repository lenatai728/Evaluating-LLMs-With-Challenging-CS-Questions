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
### Concurrent
```
python3 1_get_model_responses.py --model [model-name] --provider [provider-name] --concurrent [number]
```
### Sequential (by default)
```
python3 1_get_model_responses.py --model [model-name] --provider [provider-name]
```
### With HuggingFace fallback
```
python3 1_get_model_responses.py --model [model-name] --provider [provider-name] --fallback-hf
```
- Four Args: 
    - "--model [model-name]"
    - "--provider [provider-name]" (default: openrouter)
    - "--concurrent [number]" : Number of concurrent requests 
    - "--fallback-hf" : Use HuggingFace fallback
## 3. Separate LLMs response strings (Rationale:...Answer:...) to Answer & Rationale outputs
```
python3 2_process_outputs.py
```
## 4. Evaluate on answers from MC/MS/TF questions ONLY by exact matching
```
python3 3_eval_mc_ms_tf.py
```
## 5. Evaluate on answers from FB/OE questions by semantic similiarity & LLMs judge
### Concurrent 
```
python3 4_eval_llm_judge.py --mode answer --input_file [filename] --judge_id [judge-id] --concurrent [number]
```
### Sequential (by default)
```
python3 4_eval_llm_judge.py --mode answer --input_file [filename] --judge_id [judge-id]
```
- Three Args:
    - "--mode answer" : Evaluate on answers / "--mode rationale": Evaluate on rationales
    - "--input_file [filename]"
    - "--judge_id [judge-id]" : Judge model to use (default: deepseek-reasoner & gpt-4o-mini)
## 6. Evaluate on rationales from ALL questions by semantic similarity & LLMs judge
### Concurrent
```
python 4_eval_llm_judge.py --mode rationale --input_file [filename] --judge_id [judge-id] --concurrent [number]
```
### Sequential (by default)
```
python3 4_eval_llm_judge.py --mode rationale --input_file [filename] --judge_id [judge-id]
```
- Four Args:
    - "--mode answer" : Evaluate on answers / "--mode rationale": Evaluate on rationales
    - "--input_file [filename]"
    - "--judge_id [judge-id]" : Judge model to use (default: deepseek-reasoner)
    - "--concurrent [number]" : Number of concurrent requests
## 7. Generate reports for specific model
```
python3 5_generate_report.py --input_file [filename]
```
- One Arg: "--input_file [filename]"
## 8. Generate graphs from reports
```
python3 6_visualize.py --input_files data/4_final_results/*_result_by_model.csv
```
- One Arg: "--input_files [filename]"
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
### Read Logs Live
```
tail -f logs/xxx.log
```