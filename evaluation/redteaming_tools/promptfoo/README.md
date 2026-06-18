# promptfoo red-teaming pipeline

A drop-in counterpart to the `garak` integration: it drives
[promptfoo](https://www.promptfoo.dev/)'s red-team engine and produces the
**same attack-prompt dataset** so the two tools are interchangeable.

## Interface parity with `garak`

| garak                              | promptfoo (this package)                |
| ---------------------------------- | --------------------------------------- |
| `ATTACK_TYPES` (category → probes) | `ATTACK_TYPES` (category → plugins + strategies) |
| `build_dataset_from_garak(...)`    | `build_dataset_from_promptfoo(...)`     |
| `run_with_config_defaults()`       | `run_with_config_defaults()`            |
| `GarakRunResult`                   | `PromptfooRunResult`                     |
| `PromptRecord` (7 fields)          | `PromptRecord` (identical 7 fields)     |

Each output record is a `PromptRecord`:

```json
{"prompt": "...", "probe": "harmful:hate/prompt-injection",
 "attack_types": ["context_manipulation"], "detector": "promptfoo:redteam:harmful:hate",
 "status": "...", "is_hit": true, "source_log": "context_manipulation.results.json"}
```

### Concept mapping

* **plugin** → garak *probe* (generates adversarial prompts for a harm category).
  Stored in `probe`.
* **strategy** → prompt transform / delivery (jailbreak, base64, crescendo, ...).
  Appended to `probe` as `plugin/strategy` when not `basic`.
* **assertion/grader** → garak *detector*. Stored in `detector`.
* `is_hit` = the target **failed** the red-team assertion (attack succeeded) —
  the inverse of a passing promptfoo test.

## How it runs

For each selected attack type the pipeline:

1. writes a promptfoo red-team config (JSON),
2. `promptfoo redteam generate` → adversarial test cases,
3. `promptfoo eval` → results JSON,
4. parses results into `PromptRecord`s, then dedupes and writes
   `outputs/attack_prompts.jsonl`.

Running per attack type keeps every prompt unambiguously labelled.

## Usage

```bash
# default target (config.py): ollama:chat:qwen2.5:3b-instruct
python -m evaluation.redteaming_tools.promptfoo.promptfoo_pipeline \
    --attack-types context_manipulation obfuscation \
    --num-tests 5 --max-prompts 20
```

```python
from evaluation.redteaming_tools.promptfoo import run_with_config_defaults
result = run_with_config_defaults()
print(result.prompt_count, "prompts ->", result.output_path)
```

## Requirements

* `promptfoo` on PATH (`npm install -g promptfoo`) — falls back to `npx promptfoo`.
* A reachable target (default expects Ollama serving `qwen2.5:3b-instruct`).
* Network access for promptfoo's hosted generation/grading service.

Available plugins/strategies: `promptfoo redteam generate --help`.
