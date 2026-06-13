from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any
import os
from pathlib import Path
import re

from defenders.obfuscation.stage8_custom_classifier import load_stage8_distilbert_classifier


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


DEFAULT_LLAMA_GUARD_MODEL_ID = os.getenv(
    "JGUARD_STAGE8_MODEL_ID", "meta-llama/Llama-Guard-3-8B"
)
DEFAULT_USE_DISTILBERT = _env_flag("JGUARD_STAGE8_USE_DISTILBERT", True)
_DEFAULT_DISTILBERT_ARTIFACTS_DIR = (
    Path(__file__).resolve().parent / "stage8_custom_classifier" / "models"
)
DEFAULT_DISTILBERT_ARTIFACTS_DIR = os.getenv(
    "JGUARD_STAGE8_DISTILBERT_ARTIFACTS_DIR", str(_DEFAULT_DISTILBERT_ARTIFACTS_DIR)
)

LLAMA_GUARD_CATEGORY_MAP: dict[str, str] = {
    # Violent crimes → harmful / violent toxic content
    "S1": "toxicity",

    # Non-violent crimes → scams, fraud, hacking, criminal activity
    "S2": "scams",

    # Sex-related crimes → abusive sexual content
    "S3": "toxicity",

    # Child sexual exploitation → severe abusive sexual content
    "S4": "toxicity",

    # Defamation → false or fabricated harmful claims
    "S5": "hallucination",

    # Specialized advice → dangerous misinformation / unsafe guidance
    "S6": "hallucination",

    # Privacy violations → leakage of sensitive information
    "S7": "data_exposure",

    # Intellectual property violations → unauthorized access/use of protected content
    "S8": "copyright_infringement",

    # Indiscriminate weapons → dangerous/extremist harmful content
    "S9": "toxicity",

    # Hate → hate speech / abusive content
    "S10": "toxicity",

    # Suicide & self-harm → dangerous self-harm content
    "S11": "toxicity",

    # Sexual content → explicit sexual/offensive content
    "S12": "toxicity",

    # Elections misinformation → factual hallucinations / misinformation
    "S13": "hallucination",

    # Code interpreter abuse → privilege escalation / sandbox escape
    "S14": "unauthorized_access",
}

Stage8Classifier = Callable[[str], dict[str, Any]]


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _extract_category_codes(response: str) -> list[str]:
    codes = re.findall(r"\bS(?:1[0-4]|[1-9])\b", response.upper())
    return _unique_in_order(codes)


def parse_llama_guard_response(response: str) -> dict[str, Any]:
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    is_safe = bool(lines) and lines[0].lower().startswith("safe")
    raw_categories = [] if is_safe else _extract_category_codes("\n".join(lines[1:]))
    mapped_categories = _unique_in_order(
        [LLAMA_GUARD_CATEGORY_MAP[code] for code in raw_categories if code in LLAMA_GUARD_CATEGORY_MAP]
    )

    return {
        "is_safe": is_safe,
        "raw_categories": raw_categories,
        "mapped_categories": mapped_categories,
        "raw_response": response,
    }


@lru_cache(maxsize=1)
def load_llama_guard_bundle(model_id: str = DEFAULT_LLAMA_GUARD_MODEL_ID) -> tuple[Any, Any]:
    import torch
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=4)
def load_distilbert_classifier(artifacts_dir: str = DEFAULT_DISTILBERT_ARTIFACTS_DIR) -> Stage8Classifier:
    artifacts = Path(artifacts_dir)
    if not artifacts.exists():
        raise FileNotFoundError(
            "DistilBERT artifacts directory was not found. Expected path: "
            f"{artifacts}"
        )
    return load_stage8_distilbert_classifier(artifacts)


def classify_llama_guard_input(
    text: str, model_id: str = DEFAULT_LLAMA_GUARD_MODEL_ID
) -> dict[str, Any]:
    import torch
    tokenizer, model = load_llama_guard_bundle(model_id)
    messages = [{"role": "user", "content": text}]

    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device) if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None else None

    with torch.no_grad():
        generate_kwargs = {"max_new_tokens": 100, "do_sample": False, "pad_token_id": tokenizer.eos_token_id}
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask
        output = model.generate(
            input_ids,
            **generate_kwargs,
        )

    response = tokenizer.decode(
        output[0][input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    return parse_llama_guard_response(response)


def classify_distilbert_input(
    text: str,
    artifacts_dir: str = DEFAULT_DISTILBERT_ARTIFACTS_DIR,
) -> dict[str, Any]:
    classifier = load_distilbert_classifier(artifacts_dir)
    return classifier(text)


def classify_input(
    text: str,
    model_id: str = DEFAULT_LLAMA_GUARD_MODEL_ID,
    *,
    use_distilbert: bool = DEFAULT_USE_DISTILBERT,
    distilbert_artifacts_dir: str = DEFAULT_DISTILBERT_ARTIFACTS_DIR,
) -> dict[str, Any]:
    if use_distilbert:
        return classify_distilbert_input(text, artifacts_dir=distilbert_artifacts_dir)
    return classify_llama_guard_input(text, model_id=model_id)


def _derive_obfuscation_depth(metadata_envelope: dict[str, Any]) -> int:
    explicit_depth = metadata_envelope.get("obfuscation_depth")
    if isinstance(explicit_depth, (int, float)):
        return max(0, int(explicit_depth))

    metadata = metadata_envelope.get("metadata")
    if isinstance(metadata, dict):
        depth = metadata.get("decode_layers")
        if isinstance(depth, (int, float)):
            return max(0, int(depth))

    if isinstance(metadata, dict):
        transformations_applied = metadata.get("transformations_applied", 0)
        if isinstance(transformations_applied, (int, float)):
            return max(0, int(transformations_applied))

    return 0


def classify_stage8(
    *,
    canonical_text: str,
    metadata_envelope: dict[str, Any],
    classifier: Stage8Classifier | None = None,
    model_id: str | None = None,
    use_distilbert: bool = DEFAULT_USE_DISTILBERT,
    distilbert_artifacts_dir: str | None = None,
) -> dict[str, Any]:
    selected_distilbert_dir = distilbert_artifacts_dir or DEFAULT_DISTILBERT_ARTIFACTS_DIR
    selected_llama_model_id = model_id or DEFAULT_LLAMA_GUARD_MODEL_ID

    if classifier is not None:
        result = classifier(canonical_text)
        selected_backend = "custom"
        selected_model_ref = "custom_callable"
    elif use_distilbert:
        result = classify_distilbert_input(canonical_text, artifacts_dir=selected_distilbert_dir)
        selected_backend = "distilbert_finetuned"
        selected_model_ref = str(result.get("model_name", selected_distilbert_dir))
    else:
        result = classify_llama_guard_input(canonical_text, model_id=selected_llama_model_id)
        selected_backend = "llama_guard"
        selected_model_ref = selected_llama_model_id

    obfuscation_depth = _derive_obfuscation_depth(metadata_envelope)

    severity = round(1.0 + (0.2 * obfuscation_depth), 2) if not result["is_safe"] else 0.0
    action = "pass"
    if not result["is_safe"]:
        action = "reject" if severity >= 1.2 else "flag"

    final_envelope = dict(metadata_envelope)
    final_envelope["obfuscation_depth"] = obfuscation_depth
    final_envelope["stage8"] = {
        **result,
        "severity": severity,
        "action": action,
        "obfuscation_depth": obfuscation_depth,
        "model_id": selected_model_ref,
        "classifier_backend": selected_backend,
    }
    return final_envelope