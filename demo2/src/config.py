
# List of models to evaluate
CANDIDATE_MODELS = [
    "openai/gpt-4o-mini",
    "mistralai/mixtral-8x7b-instruct",
    "openai/gpt-oss-120b:free",
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "deepseek-chat",
    "deepseek-reasoner",
]

# OpenRouter model name → HuggingFace model name (for --fallback-hf)
# Add mappings for models that exist on both platforms
HF_MODEL_MAP = {
    "openai/gpt-oss-120b:free": "openai/gpt-oss-120b",
}
