from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any
import os
import re


DEFAULT_MODEL_ID = os.getenv("JGUARD_STAGE8_MODEL_ID", "meta-llama/Llama-Guard-3-8B")

CATEGORY_MAP: dict[str, str] = {
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
        [CATEGORY_MAP[code] for code in raw_categories if code in CATEGORY_MAP]
    )

    return {
        "is_safe": is_safe,
        "raw_categories": raw_categories,
        "mapped_categories": mapped_categories,
        "raw_response": response,
    }


@lru_cache(maxsize=1)
def load_classifier_bundle(model_id: str = DEFAULT_MODEL_ID) -> tuple[Any, Any]:
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


def classify_input(text: str, model_id: str = DEFAULT_MODEL_ID) -> dict[str, Any]:
    import torch
    tokenizer, model = load_classifier_bundle(model_id)
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
) -> dict[str, Any]:
    result = (
        classifier(canonical_text)
        if classifier is not None
        else classify_input(canonical_text, model_id=(model_id or DEFAULT_MODEL_ID))
    )
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
        "model_id": model_id or DEFAULT_MODEL_ID,
    }
    return final_envelope