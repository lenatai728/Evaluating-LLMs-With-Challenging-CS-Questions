
# List of models to evaluate (matches your Proposal Table II)
CANDIDATE_MODELS = [
    "openai/gpt-4o-mini",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-2b-it",
    "microsoft/Phi-3-mini-128k-instruct"
]

# The model used to JUDGE the candidates (usually the strongest one)
JUDGE_MODEL = "openai/gpt-4-turbo"