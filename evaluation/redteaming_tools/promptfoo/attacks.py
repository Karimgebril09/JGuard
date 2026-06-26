from __future__ import annotations

ATTACK_TYPES: dict[str, dict[str, list[str]]] = {
    "context_manipulation": {
        "plugins": [
            "harmful:harassment-bullying",
            "harmful:hate",
            "hijacking",
        ],
        "strategies": [
            "basic",
            "prompt-injection",
            # "jailbreak",
            # "jailbreak:composite",
        ],
    },
    "obfuscation": {
        "plugins": [
            "harmful:cybercrime",
            "ascii-smuggling",
        ],
        "strategies": [
            "basic",
            "base64",
            "rot13",
            "leetspeak",
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
