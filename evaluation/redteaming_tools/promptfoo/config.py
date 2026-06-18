TARGET_SAMPLES = 20

# Command used to invoke promptfoo. Resolved against PATH; falls back to
# "npx promptfoo" when the global binary is not found.
PROMPTFOO_BIN = "promptfoo"

# Number of base test cases generated per plugin (before strategy expansion).
NUM_TESTS_PER_PLUGIN = 5

# 0 disables the per-phase (generate / eval) subprocess timeout.
PROMPTFOO_TIMEOUT_SECONDS = 0

# promptfoo provider id for the target under test.
DEFAULT_TARGET_ID = "ollama:chat:qwen2.5:3b-instruct"
DEFAULT_TARGET_LABEL = "qwen2.5-3b-instruct"

# System purpose handed to promptfoo so it can tailor adversarial probes.
DEFAULT_PURPOSE = "A general-purpose AI assistant"

REPORTS_DIR = "evaluation/redteaming_tools/promptfoo/outputs/promptfoo_reports"
DATASET_PATH = "evaluation/redteaming_tools/promptfoo/outputs/attack_prompts.jsonl"

MIN_PROMPT_CHARS = 12
