"""Attack-type catalogue for the promptfoo red-teaming pipeline.

Mirrors ``garak/attacks.py``: each attack type groups a family of probes. In
garak a probe both *generates* prompts and *transforms* them. promptfoo splits
those two jobs:

* a **plugin** generates adversarial prompts for a vulnerability category, and
* a **strategy** transforms / delivers those prompts (encoding, jailbreak,
  multi-turn, ...).

So every attack type maps to a ``{"plugins": [...], "strategies": [...]}`` pair.
Plugins are kept disjoint across attack types so a generated prompt maps back to
exactly one attack type. Run ``promptfoo redteam generate --help`` for the full
catalogue of available plugins and strategies.
"""

from __future__ import annotations

ATTACK_TYPES: dict[str, dict[str, list[str]]] = {
    "context_manipulation": {
        # Persona / instruction-override / context-injection attacks.
        "plugins": [
            "harmful:harassment-bullying",
            "harmful:hate",
            "hijacking",
        ],
        "strategies": [
            "basic",
            "prompt-injection",
            # "jailbreak",            # iterative single-turn jailbreak (slow)
            # "jailbreak:composite",
        ],
    },
    "obfuscation": {
        # Encoding / smuggling attacks that hide intent from safety filters.
        "plugins": [
            "harmful:cybercrime",
            "ascii-smuggling",
        ],
        "strategies": [
            "basic",
            "base64",
            "rot13",
            "leetspeak",
            # "hex",
            # "homoglyph",
            # "morse",
        ],
    },
    "multi_turn": {
        # Multi-turn / escalation attacks built up over a conversation.
        "plugins": [
            "harmful:illegal-drugs",
            "harmful:violent-crime",
        ],
        "strategies": [
            "basic",
            "crescendo",
            # "goat",
            # "mischievous-user",
        ],
    },
}


def get_selected_plugins(selected_attack_types: list[str]) -> list[str]:
    selected: list[str] = []
    for attack_type in selected_attack_types:
        selected.extend(ATTACK_TYPES.get(attack_type, {}).get("plugins", []))
    return sorted(set(selected))


def get_selected_strategies(selected_attack_types: list[str]) -> list[str]:
    selected: list[str] = []
    for attack_type in selected_attack_types:
        selected.extend(ATTACK_TYPES.get(attack_type, {}).get("strategies", []))
    return sorted(set(selected))


def plugin_to_attack_type_map() -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = {}
    for attack_type, spec in ATTACK_TYPES.items():
        for plugin in spec.get("plugins", []):
            reverse.setdefault(plugin, []).append(attack_type)
    return reverse


def strategy_to_attack_type_map() -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = {}
    for attack_type, spec in ATTACK_TYPES.items():
        for strategy in spec.get("strategies", []):
            reverse.setdefault(strategy, []).append(attack_type)
    return reverse
