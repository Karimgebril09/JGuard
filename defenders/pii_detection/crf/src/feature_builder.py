import re
from defenders.pii_detection.crf.src.features import FeaturesManager

def looks_like_email(word):
    result = re.fullmatch(r".+@.+\..+", word)
    return bool(result)


def looks_like_ipv4(word):
    result = re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", word)
    return bool(result)


def looks_like_ipv6(word):
    result = re.fullmatch(r"([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}", word)
    return bool(result)


def looks_like_mac(word):
    result = re.fullmatch(r"([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}", word)
    return bool(result)

def looks_like_hex(word):
    result = re.fullmatch(r"[0-9a-fA-F]{8,}", word)
    return bool(result)


def is_four_digits(word):
    result = re.fullmatch(r"\d{4}", word)
    return bool(result)

def build_features():
    manager = FeaturesManager()
    manager.add_feature("is_upper", lambda X, prev, curr, i: X[i].isupper() and curr != "O")
    manager.add_feature("is_title", lambda X, prev, curr, i: X[i].istitle() and curr != "O")
    manager.add_feature("is_lower", lambda X, prev, curr, i: X[i].islower() and curr != "O")
    manager.add_feature("has_digit", lambda X, prev, curr, i: any(c.isdigit() for c in X[i]) and curr != "O")
    manager.add_feature("has_upper", lambda X, prev, curr, i: any(c.isupper() for c in X[i]) and curr != "O")
    manager.add_feature("has_lower", lambda X, prev, curr, i: any(c.islower() for c in X[i]) and curr != "O")
    manager.add_feature("has_special", lambda X, prev, curr, i: any(not c.isalnum() for c in X[i]) and curr != "O")
    manager.add_feature("has_dot", lambda X, prev, curr, i: "." in X[i] and curr in ["B-EMAIL", "I-EMAIL", "B-IPV4", "I-IPV4"])
    manager.add_feature("has_colon", lambda X, prev, curr, i: ":" in X[i] and curr in ["B-IPV6", "I-IPV6", "B-MAC", "I-MAC"])
    manager.add_feature("has_dash", lambda X, prev, curr, i: "-" in X[i] and curr != "O")
    manager.add_feature("has_slash", lambda X, prev, curr, i: "/" in X[i] and curr != "O")
    manager.add_feature("has_at", lambda X, prev, curr, i: "@" in X[i] and curr in ["B-EMAIL", "I-EMAIL"])
    manager.add_feature("has_underscore", lambda X, prev, curr, i: "_" in X[i] and curr in ["B-USERNAME", "I-USERNAME"])
    manager.add_feature("has_numbers_and_digits", lambda X, prev, curr, i: any(c.isdigit() for c in X[i]) and any(c.isalpha() for c in X[i]) and curr != "O")
    manager.add_feature("len_greater_than_5", lambda X, prev, curr, i: len(X[i]) > 5 and curr != "O")
    manager.add_feature("len_greater_than_10", lambda X, prev, curr, i: len(X[i]) > 10 and curr != "O")
    manager.add_feature("len_greater_than_15", lambda X, prev, curr, i: len(X[i]) > 15 and curr != "O")
    manager.add_feature("email", lambda X, prev, curr, i: looks_like_email(X[i]) and curr in ["B-EMAIL", "I-EMAIL"])
    manager.add_feature("ipv6", lambda X, prev, curr, i: looks_like_ipv6(X[i]) and curr in ["B-IPV6", "I-IPV6"])
    manager.add_feature("ipv4", lambda X, prev, curr, i: looks_like_ipv4(X[i]) and curr in ["B-IPV4", "I-IPV4"])
    manager.add_feature("mac", lambda X, prev, curr, i: looks_like_mac(X[i]) and curr in ["B-MAC", "I-MAC"])
    manager.add_feature("hex", lambda X, prev, curr, i: looks_like_hex(X[i]) and curr != "O")
    manager.add_feature("is_four_digits", lambda X, prev, curr, i: is_four_digits(X[i]) and curr != "O")
    manager.add_feature("o_to_b", lambda X, prev, curr, i: prev == "O" and curr.startswith("B-"))
    manager.add_feature("illegal_o_to_i", lambda X, prev, curr, i: prev == "O" and curr.startswith("I-"))
    manager.add_feature("same_entity_transition", lambda X, prev, curr, i: prev.startswith(("B-", "I-")) and curr.startswith("I-") and prev[2:] == curr[2:])
    manager.add_feature("BOS", lambda X, prev, curr, i: i == 0)
    manager.add_feature("EOS", lambda X, prev, curr, i: i == len(X) - 1)

    return manager