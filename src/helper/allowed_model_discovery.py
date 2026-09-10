import os
import re
import requests
from typing import Any
from dotenv import load_dotenv

load_dotenv()

OUTPUT_FILE = "free_tier_models.txt"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

def get_json(url: str, headers: dict[str, str] | None = None, params: dict[str, str] | None = None) -> dict[str, Any]:
    response = requests.get(url, headers=headers or {}, params=params or {}, timeout=30)
    response.raise_for_status()
    return response.json()

def get_openrouter_free_models() -> list[dict[str, Any]]:
    if not OPENROUTER_API_KEY:
        return []
    data = get_json(
        "https://openrouter.ai/api/v1/models",
        {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    )
    models = []
    for model in data.get("data", []):
        pricing = model.get("pricing", {})
        try:
            prompt = float(pricing.get("prompt", "1"))
            completion = float(pricing.get("completion", "1"))
        except (TypeError, ValueError):
            continue
        if prompt == 0 and completion == 0:
            models.append({
                "id": model.get("id"),
                "name": model.get("name", model.get("id")),
                "context": model.get("context_length"),
                "architecture": model.get("architecture", {}),
                "pricing": pricing
            })
    return sorted(models, key=lambda x: (x["id"] or "").lower())

def get_groq_free_models() -> list[str]:
    if not GROQ_API_KEY:
        return []
    data = get_json(
        "https://api.groq.com/openai/v1/models",
        {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
    )
    current_free_plan_ids = {
        "canopylabs/orpheus-arabic-saudi",
        "canopylabs/orpheus-v1-english",
        "groq/compound",
        "groq/compound-mini",
        "meta-llama/llama-prompt-guard-2-22m",
        "meta-llama/llama-prompt-guard-2-86m",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-safeguard-20b",
        "qwen/qwen3.6-27b",
        "qwen/qwen3.8-27b",
        "whisper-large-v3",
        "whisper-large-v3-turbo"
    }
    available_ids = {
        model.get("id")
        for model in data.get("data", [])
        if model.get("id")
    }
    return sorted(current_free_plan_ids.intersection(available_ids), key=str.lower)

def get_gemini_free_models() -> list[str]:
    if not GEMINI_API_KEY:
        return []
    get_json(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": GEMINI_API_KEY, "pageSize": "1000"}
    )
    current_free_text_models = {
        "gemini-3-flash-preview",
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.1-flash-lite",
        "gemini-2.5-pro",
        "gemini-3.6-flash",
        "gemini-3.6-flash-lite",
        "gemini-3.6-flash-preview-tts",
        "gemini-3.6-flash-native-audio-preview-12-2025"
    }
    available = set()
    page_token = None

    while True:
        params = {"key": GEMINI_API_KEY, "pageSize": "1000"}
        if page_token:
            params["pageToken"] = page_token

        data = get_json(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params=params
        )

        for model in data.get("models", []):
            name = model.get("name", "")
            model_id = name.removeprefix("models/")
            methods = model.get("supportedGenerationMethods", [])
            if "generateContent" in methods:
                available.add(model_id)

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return sorted(current_free_text_models.intersection(available), key=str.lower)

def get_huggingface_models() -> list[dict[str, Any]]:
    if not HF_TOKEN:
        return []
    data = get_json(
        "https://router.huggingface.co/v1/models",
        {"Authorization": f"Bearer {HF_TOKEN}"}
    )
    models = []

    for model in data.get("data", []):
        providers = model.get("providers") or []
        provider_names = []

        for provider in providers:
            if isinstance(provider, dict):
                provider_name = provider.get("provider")
                if provider_name:
                    provider_names.append(provider_name)
            elif isinstance(provider, str):
                provider_names.append(provider)

        if provider_names:
            models.append({
                "id": model.get("id"),
                "name": model.get("id"),
                "providers": sorted(set(provider_names)),
                "context": model.get("context_length"),
                "pricing": model.get("pricing")
            })

    return sorted(models, key=lambda x: (x["id"] or "").lower())

def write_model_section(file, title: str, models: list[dict[str, Any]]) -> None:
    file.write(f"{title}\n")
    file.write("=" * len(title) + "\n")
    file.write(f"Count: {len(models)}\n\n")

    for index, model in enumerate(models, 1):
        file.write(f"{index}. {model.get('id', 'UNKNOWN')}\n")
        if model.get("name") and model.get("name") != model.get("id"):
            file.write(f"   Name: {model['name']}\n")
        if model.get("context"):
            file.write(f"   Context: {model['context']}\n")
        if model.get("providers"):
            file.write(f"   Providers: {', '.join(model['providers'])}\n")
        if model.get("pricing"):
            file.write(f"   Pricing: {model['pricing']}\n")
        file.write("\n")

def main() -> None:
    results: dict[str, Any] = {
        "OpenRouter": [],
        "Groq": [],
        "Gemini": [],
        "Hugging Face": []
    }

    try:
        results["OpenRouter"] = get_openrouter_free_models()
    except Exception as error:
        results["OpenRouter"] = [{"id": f"ERROR: {error}"}]

    try:
        results["Groq"] = [{"id": model_id} for model_id in get_groq_free_models()]
    except Exception as error:
        results["Groq"] = [{"id": f"ERROR: {error}"}]

    try:
        results["Gemini"] = [{"id": model_id} for model_id in get_gemini_free_models()]
    except Exception as error:
        results["Gemini"] = [{"id": f"ERROR: {error}"}]

    try:
        results["Hugging Face"] = get_huggingface_models()
    except Exception as error:
        results["Hugging Face"] = [{"id": f"ERROR: {error}"}]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write("FREE-TIER / INCLUDED-CREDIT MODEL CATALOG\n")
        file.write("========================================\n\n")
        file.write("OpenRouter and Groq are filtered using their current provider model/free-plan information.\n")
        file.write("Gemini is filtered against the current free-tier model set and then intersected with models available to the supplied API key.\n")
        file.write("Hugging Face is listed separately because its free plan provides monthly inference credits rather than a fixed permanently-free model list.\n\n")

        write_model_section(file, "OPENROUTER — FREE MODELS", results["OpenRouter"])
        write_model_section(file, "GROQ — FREE PLAN MODELS", results["Groq"])
        write_model_section(file, "GEMINI — FREE-TIER MODELS", results["Gemini"])
        write_model_section(file, "HUGGING FACE — MODELS ACCESSIBLE THROUGH HF INFERENCE PROVIDERS", results["Hugging Face"])

    print(f"Saved {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
