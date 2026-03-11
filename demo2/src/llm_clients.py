"""
Shared LLM client module for OpenRouter, HuggingFace, Google AI Studio, and Ollama.
Provides reusable client creation and API call functions with retry/backoff.
"""

import os
import time
import logging
from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import InferenceClient
from google import genai
from google.genai import types
from config import HF_MODEL_MAP

load_dotenv()
logger = logging.getLogger(__name__)

# --- RETRY CONFIGURATION ---
INLINE_RETRIES = 3       # Retries per provider for transient errors
BACKOFF_BASE = 2         # Exponential backoff base (seconds)
RPM_COOLDOWN = 4.1       # Essential to stay under 15 RPM (for Google API only)

# =============================================================================
# CLIENT CREATION
# =============================================================================

def create_openrouter_client(model_name):
    """Create and return a reusable OpenAI client for OpenRouter."""
    key_str = f"{model_name.split('/')[-1].upper()}-OPENROUTER-API-KEY"
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv(key_str)
    )

def create_deepseek_client(model_name):
    """Create and return a reusable OpenAI client for DeepSeek."""
    key_str = f"{model_name.split('/')[-1].upper()}-API-KEY"
    return OpenAI(
        base_url="https://api.deepseek.com",
        api_key=os.getenv(key_str)
    )
    
def create_huggingface_client(model_name):
    """Create and return a reusable HuggingFace InferenceClient.
    
    Uses HF_MODEL_MAP from config.py to map model names to HuggingFace names.
    Returns None if no mapping exists for the given model.
    """
    hf_model = HF_MODEL_MAP.get(model_name)
    if not hf_model:
        logger.warning(f"No HuggingFace mapping found for '{model_name}' in HF_MODEL_MAP. HF fallback disabled.")
        return None
    
    hf_token = os.getenv("HF-TOKEN")
    if not hf_token:
        logger.warning("HF-TOKEN not found in .env. HF fallback disabled.")
        return None
    
    return InferenceClient(model=hf_model, token=hf_token)


def create_google_client():
    """Create and return a reusable Google AI Studio client."""
    api_key = os.getenv("GOOGLE-API-KEY")
    if not api_key:
        logger.warning("GOOGLE-API-KEY not found in .env. Google provider unavailable.")
        return None
    
    return genai.Client(api_key=api_key)

def create_google_clients():
    """Create and return reusable Google AI Studio clients for all available keys.
    
    Returns a list of (key_name, client) tuples, or None if no keys found.
    """
    key_names = ["GOOGLE-API-KEY", "GOOGLE-API-KEY-2"]

    clients = []
    for key_name in key_names:
        api_key = os.getenv(key_name)
        if api_key:
            clients.append((key_name, genai.Client(api_key=api_key)))
            logger.info(f"Loaded Google API key: {key_name}")
        else:
            logger.warning(f"{key_name} not found in .env. Skipping.")

    if not clients:
        logger.warning("No Google API keys found. Google provider unavailable.")
        return None

    logger.info(f"Google client pool: {len(clients)} key(s) available for rotation.")
    return clients

# def call_google(clients, model_name, system_prompt, user_prompt):
#     """Call Google AI Studio API with retry, exponential backoff, and key rotation.
    
#     On each failure (API error OR empty response), the next attempt switches
#     to the other API key:
#         attempt 1 → key 1
#         attempt 2 → key 2  (switched after error)
#         attempt 3 → key 1  (switched back)
#         ...
    
#     If only one key is available, all attempts use that same key.
    
#     Returns the response content string, or "ERROR_RESPONSE" if all retries fail.
#     """
#     config = types.GenerateContentConfig(
#         temperature=0.0,
#     )

#     num_clients = len(clients)

#     for attempt in range(1, INLINE_RETRIES + 1):
#         # ── Rotate key each attempt: 0, 1, 0, 1, ... ──
#         client_idx = (attempt - 1) % num_clients
#         key_name, client = clients[client_idx]

#         try:
#             logger.debug(
#                 f"[Google] {model_name} attempt {attempt}/{INLINE_RETRIES} "
#                 f"using {key_name}"
#             )

#             response = client.models.generate_content(
#                 model=model_name,
#                 contents=f"{system_prompt}\n\n{user_prompt}",
#                 config=config,
#             )

#             content = response.text

#             # Handle None/empty responses
#             if content is None or content.strip() == "":
#                 logger.warning(
#                     f"[Google] Empty/None response for {model_name} "
#                     f"using {key_name} (attempt {attempt}/{INLINE_RETRIES})"
#                 )
#                 if attempt < INLINE_RETRIES:
#                     wait_time = BACKOFF_BASE ** attempt
#                     next_key = clients[attempt % num_clients][0]
#                     logger.info(
#                         f"Switching to {next_key}, retrying in {wait_time}s..."
#                     )
#                     time.sleep(wait_time)
#                     continue
#                 return "ERROR_RESPONSE"

#             # MANDATORY: Sleep after success to maintain RPM for the next call
#             time.sleep(RPM_COOLDOWN)
#             return content

#         except Exception as e:
#             logger.error(
#                 f"[Google] API Error for {model_name} "
#                 f"using {key_name} (attempt {attempt}/{INLINE_RETRIES}): {e}"
#             )
#             if attempt < INLINE_RETRIES:
#                 wait_time = BACKOFF_BASE ** attempt
#                 next_key = clients[attempt % num_clients][0]
#                 logger.info(
#                     f"Switching to {next_key}, retrying in {wait_time}s..."
#                 )
#                 time.sleep(wait_time)
#             else:
#                 return "ERROR_RESPONSE"
#     return "ERROR_RESPONSE"

# =============================================================================
# API CALL FUNCTIONS (with retry + exponential backoff)
# =============================================================================

def call_openrouter(client, model_name, system_prompt, user_prompt):
    """Call OpenRouter API with retry and exponential backoff.
    
    Returns the response content string, or "ERROR_RESPONSE" if all retries fail.
    """
    for attempt in range(1, INLINE_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            }
                        ]
                    }
                ],
                temperature=0.0,
            )
            
            content = response.choices[0].message.content
            
            # Handle None/empty responses
            if content is None or content.strip() == "":
                logger.warning(f"[OpenRouter] Empty/None response for {model_name} (attempt {attempt}/{INLINE_RETRIES})")
                if attempt < INLINE_RETRIES:
                    wait_time = BACKOFF_BASE ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return "ERROR_RESPONSE"
            
            return content

        except Exception as e:
            logger.error(f"[OpenRouter] API Error for {model_name} (attempt {attempt}/{INLINE_RETRIES}): {e}")
            if attempt < INLINE_RETRIES:
                wait_time = BACKOFF_BASE ** attempt
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return "ERROR_RESPONSE"
    
    return "ERROR_RESPONSE"


def call_huggingface(client, system_prompt, user_prompt):
    """Call HuggingFace Inference API with retry and exponential backoff.
    
    Returns the response content string, or "ERROR_RESPONSE" if all retries fail.
    """
    for attempt in range(1, INLINE_RETRIES + 1):
        try:
            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            
            content = response.choices[0].message.content
            
            # Handle None/empty responses
            if content is None or content.strip() == "":
                logger.warning(f"[HuggingFace] Empty/None response (attempt {attempt}/{INLINE_RETRIES})")
                if attempt < INLINE_RETRIES:
                    wait_time = BACKOFF_BASE ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return "ERROR_RESPONSE"
            
            return content

        except Exception as e:
            logger.error(f"[HuggingFace] API Error (attempt {attempt}/{INLINE_RETRIES}): {e}")
            if attempt < INLINE_RETRIES:
                wait_time = BACKOFF_BASE ** attempt
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return "ERROR_RESPONSE"
    
    return "ERROR_RESPONSE"


def call_google(client, model_name, system_prompt, user_prompt):
    """Call Google AI Studio API with retry and exponential backoff.
    
    Returns the response content string, or "ERROR_RESPONSE" if all retries fail.
    """
    config = types.GenerateContentConfig(
        # system_instruction=system_prompt,
        temperature=0.0,
    )
    for attempt in range(1, INLINE_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=f"{system_prompt}\n\n{user_prompt}",
                config=config,
            )
            
            content = response.text
            
            # Handle None/empty responses
            if content is None or content.strip() == "":
                logger.warning(f"[Google] Empty/None response for {model_name} (attempt {attempt}/{INLINE_RETRIES})")
                if attempt < INLINE_RETRIES:
                    wait_time = BACKOFF_BASE ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return "ERROR_RESPONSE"
            
            # MANDATORY: Sleep after success to maintain RPM for the next call
            time.sleep(RPM_COOLDOWN)
            
            return content

        except Exception as e:
            logger.error(f"[Google] API Error for {model_name} (attempt {attempt}/{INLINE_RETRIES}): {e}")
            if attempt < INLINE_RETRIES:
                wait_time = BACKOFF_BASE ** attempt
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return "ERROR_RESPONSE"
    
    return "ERROR_RESPONSE"

# def call_google(clients, model_name, system_prompt, user_prompt):
#     """Call Google AI Studio API with retry, exponential backoff, and key rotation.
    
#     On each failure (API error OR empty response), the next attempt switches
#     to the other API key:
#         attempt 1 → key 1
#         attempt 2 → key 2  (switched after error)
#         attempt 3 → key 1  (switched back)
#         ...
    
#     If only one key is available, all attempts use that same key.
    
#     Returns the response content string, or "ERROR_RESPONSE" if all retries fail.
#     """
#     config = types.GenerateContentConfig(
#         temperature=0.0,
#     )

#     num_clients = len(clients)

#     for attempt in range(1, INLINE_RETRIES + 1):
#         # ── Rotate key each attempt: 0, 1, 0, 1, ... ──
#         client_idx = (attempt - 1) % num_clients
#         key_name, client = clients[client_idx]

#         try:
#             logger.debug(
#                 f"[Google] {model_name} attempt {attempt}/{INLINE_RETRIES} "
#                 f"using {key_name}"
#             )

#             response = client.models.generate_content(
#                 model=model_name,
#                 contents=f"{system_prompt}\n\n{user_prompt}",
#                 config=config,
#             )

#             content = response.text

#             # Handle None/empty responses
#             if content is None or content.strip() == "":
#                 logger.warning(
#                     f"[Google] Empty/None response for {model_name} "
#                     f"using {key_name} (attempt {attempt}/{INLINE_RETRIES})"
#                 )
#                 if attempt < INLINE_RETRIES:
#                     wait_time = BACKOFF_BASE ** attempt
#                     next_key = clients[attempt % num_clients][0]
#                     logger.info(
#                         f"Switching to {next_key}, retrying in {wait_time}s..."
#                     )
#                     time.sleep(wait_time)
#                     continue
#                 return "ERROR_RESPONSE"

#             # MANDATORY: Sleep after success to maintain RPM for the next call
#             time.sleep(RPM_COOLDOWN)
#             return content

#         except Exception as e:
#             logger.error(
#                 f"[Google] API Error for {model_name} "
#                 f"using {key_name} (attempt {attempt}/{INLINE_RETRIES}): {e}"
#             )
#             if attempt < INLINE_RETRIES:
#                 wait_time = BACKOFF_BASE ** attempt
#                 next_key = clients[attempt % num_clients][0]
#                 logger.info(
#                     f"Switching to {next_key}, retrying in {wait_time}s..."
#                 )
#                 time.sleep(wait_time)
#             else:
#                 return "ERROR_RESPONSE"

#     return "ERROR_RESPONSE"

def call_deepseek(client, model_name, system_prompt, user_prompt):
    """Call DeepSeek API (OpenAI-compatible) with retry and exponential backoff.
    
    DeepSeek API uses the same chat completions format as OpenAI.
    Supported models: deepseek-chat, deepseek-reasoner, etc.
    
    Args:
        client: OpenAI client instance pointed at https://api.deepseek.com
        model_name: "deepseek-chat" / "deepseek-reasoner"
        system_prompt: System prompt string
        user_prompt: User prompt string
    
    Returns:
        Response content string, or "ERROR_RESPONSE" if all retries fail.
    """
    
    for attempt in range(1, INLINE_RETRIES + 1):
        try:
            logger.debug(
                f"[DeepSeek] {model_name} attempt {attempt}/{INLINE_RETRIES}"
            )

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                temperature=0.0,
            )

            content = response.choices[0].message.content

            # Handle None/empty responses
            if content is None or content.strip() == "":
                logger.warning(
                    f"[DeepSeek] Empty/None response for {model_name} "
                    f"(attempt {attempt}/{INLINE_RETRIES})"
                )
                if attempt < INLINE_RETRIES:
                    wait_time = BACKOFF_BASE ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return "ERROR_RESPONSE"

            return content

        except Exception as e:
            logger.error(
                f"[DeepSeek] API Error for {model_name} "
                f"(attempt {attempt}/{INLINE_RETRIES}): {e}"
            )
            if attempt < INLINE_RETRIES:
                wait_time = BACKOFF_BASE ** attempt
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return "ERROR_RESPONSE"

    return "ERROR_RESPONSE"

# =============================================================================
# UNIFIED FUNCTION (Primary provider → optional HuggingFace fallback)
# =============================================================================

def get_llm_response(system_prompt, user_prompt, model_name, primary_client, provider="openrouter", hf_client=None):
    """Get LLM response from the specified primary provider with optional HuggingFace fallback.
    
    Args:
        system_prompt: System prompt string
        user_prompt: User prompt string
        model_name: Model name for the API call
        primary_client: Client instance for the primary provider
        provider: Primary provider name ("openrouter", "hf", "google", or "ollama")
        hf_client: Optional HuggingFace InferenceClient for fallback
    
    Returns:
        Response content string, or "ERROR_RESPONSE" if all attempts fail.
    """
    # Try primary provider
    if provider == "openrouter":
        response = call_openrouter(primary_client, model_name, system_prompt, user_prompt)
    elif provider == "hf":
        response = call_huggingface(primary_client, system_prompt, user_prompt)
    elif provider == "google":
        response = call_google(primary_client, model_name, system_prompt, user_prompt)
    elif provider == "deepseek":
        response = call_deepseek(primary_client, model_name, system_prompt, user_prompt)
    else:
        logger.error(f"Unknown provider: {provider}")
        response = "ERROR_RESPONSE"
    
    if response != "ERROR_RESPONSE":
        return response
    
    # Fallback to HuggingFace if available (and not already the primary)
    if hf_client is not None and provider != "hf":
        logger.info(f"[Fallback] {provider} failed for {model_name}. Trying HuggingFace...")
        response = call_huggingface(hf_client, system_prompt, user_prompt)
        
        if response != "ERROR_RESPONSE":
            logger.info(f"[Fallback] HuggingFace succeeded for {model_name}.")
            return response
        else:
            logger.error(f"[Fallback] HuggingFace also failed for {model_name}.")
    
    return "ERROR_RESPONSE"