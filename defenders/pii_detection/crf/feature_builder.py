
from features import FeaturesManager
def build_features():
    manager = FeaturesManager()

    manager.add_feature("token_has_at_symbol", 
                        lambda X, prev, curr, i: '@' in X[i] and curr in ["B-EMAIL", "I-EMAIL"])
    
    manager.add_feature("token_has_dots", 
                        lambda X, prev, curr, i: '.' in X[i] and curr in ["B-IPV4", "I-IPV4"])
    manager.add_feature("token_has_colons", 
                        lambda X, prev, curr, i: ':' in X[i] and curr in ["B-IPV6", "I-IPV6", "B-MAC", "I-MAC"])
    
    manager.add_feature("token_is_pure_digits", 
                        lambda X, prev, curr, i: X[i].isdigit() and curr in ["B-SSN", "I-SSN", "B-CREDITCARDNUMBER", "I-CREDITCARDNUMBER", "B-ACCOUNTNUMBER", "I-ACCOUNTNUMBER"])
    
    manager.add_feature("token_len_3_or_4", 
                        lambda X, prev, curr, i: len(X[i]) in [3, 4] and curr in ["B-SSN", "I-SSN", "I-CREDITCARDNUMBER"])

    manager.add_feature("token_is_alphanumeric_mix", 
                        lambda X, prev, curr, i: X[i].isalnum() and not X[i].isalpha() and not X[i].isdigit() and curr in ["B-PASSWORD", "I-PASSWORD", "B-USERNAME", "I-USERNAME"])
    
    def get_prev_lower(X, i):
        if i == 0: return ""
        return X[i-1].lower().rstrip(":= ")

    manager.add_feature("prev_word_indicates_email", 
                        lambda X, prev, curr, i: get_prev_lower(X, i) in ["email", "e-mail", "to", "from"] and curr.startswith("B-"))
    
    manager.add_feature("prev_word_indicates_phone", 
                        lambda X, prev, curr, i: get_prev_lower(X, i) in ["phone", "tel", "mobile", "cell"] and curr.startswith("B-"))
    
    manager.add_feature("prev_word_indicates_security", 
                        lambda X, prev, curr, i: get_prev_lower(X, i) in ["ssn", "social", "security"] and curr.startswith("B-"))
    
    manager.add_feature("prev_word_indicates_finance", 
                        lambda X, prev, curr, i: get_prev_lower(X, i) in ["card", "credit", "account", "acc", "iban"] and curr.startswith("B-"))
    
    manager.add_feature("prev_word_indicates_auth", 
                        lambda X, prev, curr, i: get_prev_lower(X, i) in ["user", "username", "password", "pass", "pw"] and curr.startswith("B-"))

    manager.add_feature("prev_word_indicates_network", 
                        lambda X, prev, curr, i: get_prev_lower(X, i) in ["ip", "ipv4", "ipv6", "mac", "host"] and curr.startswith("B-"))

    manager.add_feature("clean_bio_transition", 
                        lambda X, prev, curr, i: (prev.startswith("B-") or prev.startswith("I-")) and curr.startswith("I-") and prev[2:] == curr[2:])
    
    manager.add_feature("transition_o_to_b", 
                        lambda X, prev, curr, i: prev == "O" and curr.startswith("B-"))
    
    manager.add_feature("illegal_o_to_i_transition", 
                        lambda X, prev, curr, i: prev == "O" and curr.startswith("I-"))

    manager.add_feature(
        "token_is_lowercase",
        lambda X, prev, curr, i:
            X[i].islower() and curr != "O"
    )

    manager.add_feature(
        "token_is_uppercase",
        lambda X, prev, curr, i:
            X[i].isupper() and curr != "O"
    )

    manager.add_feature(
        "token_is_titlecase",
        lambda X, prev, curr, i:
            X[i].istitle() and curr != "O"
    )

    manager.add_feature(
        "token_contains_digit",
        lambda X, prev, curr, i:
            any(c.isdigit() for c in X[i]) and curr != "O"
    )

    manager.add_feature(
        "token_contains_alpha",
        lambda X, prev, curr, i:
            any(c.isalpha() for c in X[i]) and curr != "O"
    )

    manager.add_feature(
        "token_contains_dash",
        lambda X, prev, curr, i:
            "-" in X[i] and curr != "O"
    )

    manager.add_feature(
        "token_contains_slash",
        lambda X, prev, curr, i:
            "/" in X[i] and curr != "O"
    )

    manager.add_feature(
        "token_contains_underscore",
        lambda X, prev, curr, i:
            "_" in X[i] and curr in ["B-USERNAME", "I-USERNAME"]
    )

    manager.add_feature(
        "token_length_gt_10",
        lambda X, prev, curr, i:
            len(X[i]) > 10 and curr != "O"
    )

    manager.add_feature(
        "token_length_gt_15",
        lambda X, prev, curr, i:
            len(X[i]) > 15 and curr != "O"
    )

    manager.add_feature(
        "token_length_eq_16",
        lambda X, prev, curr, i:
            len(X[i]) == 16 and curr in
            ["B-CREDITCARDNUMBER", "I-CREDITCARDNUMBER"]
    )

    manager.add_feature(
        "token_length_eq_19",
        lambda X, prev, curr, i:
            len(X[i]) == 19 and curr in
            ["B-CREDITCARDNUMBER", "I-CREDITCARDNUMBER"]
    )

    manager.add_feature(
        "prefix_http",
        lambda X, prev, curr, i:
            X[i].lower().startswith("http")
            and curr != "O"
    )

    manager.add_feature(
        "prefix_www",
        lambda X, prev, curr, i:
            X[i].lower().startswith("www")
            and curr != "O"
    )

    manager.add_feature(
        "prefix_plus",
        lambda X, prev, curr, i:
            X[i].startswith("+")
            and curr in ["B-PHONE", "I-PHONE"]
    )

 
    manager.add_feature(
        "suffix_com",
        lambda X, prev, curr, i:
            X[i].lower().endswith(".com")
            and curr in ["B-EMAIL", "I-EMAIL"]
    )

    manager.add_feature(
        "suffix_org",
        lambda X, prev, curr, i:
            X[i].lower().endswith(".org")
            and curr in ["B-EMAIL", "I-EMAIL"]
    )

    manager.add_feature(
        "suffix_net",
        lambda X, prev, curr, i:
            X[i].lower().endswith(".net")
            and curr in ["B-EMAIL", "I-EMAIL"]
    )

    def get_next_lower(X, i):
        if i == len(X) - 1:
            return ""
        return X[i + 1].lower().rstrip(":= ")

    manager.add_feature(
        "next_word_is_email",
        lambda X, prev, curr, i:
            get_next_lower(X, i) == "email"
            and curr == "O"
    )

    manager.add_feature(
        "next_word_is_password",
        lambda X, prev, curr, i:
            get_next_lower(X, i) == "password"
            and curr == "O"
    )

    manager.add_feature(
        "next_word_is_username",
        lambda X, prev, curr, i:
            get_next_lower(X, i) == "username"
            and curr == "O"
    )

    manager.add_feature(
        "b_followed_by_same_i",
        lambda X, prev, curr, i:
            prev.startswith("B-")
            and curr.startswith("I-")
            and prev[2:] == curr[2:]
    )

    manager.add_feature(
        "i_followed_by_same_i",
        lambda X, prev, curr, i:
            prev.startswith("I-")
            and curr.startswith("I-")
            and prev[2:] == curr[2:]
    )

    manager.add_feature(
        "illegal_entity_switch",
        lambda X, prev, curr, i:
            prev.startswith("I-")
            and curr.startswith("I-")
            and prev[2:] != curr[2:]
    )

    manager.add_feature(
        "entity_end",
        lambda X, prev, curr, i:
            prev.startswith(("B-", "I-"))
            and curr == "O"
    )


    manager.add_feature(
        "password_like_token",
        lambda X, prev, curr, i:
            any(c.isupper() for c in X[i])
            and any(c.islower() for c in X[i])
            and any(c.isdigit() for c in X[i])
            and curr in ["B-PASSWORD", "I-PASSWORD"]
    )

    manager.add_feature(
        "exactly_one_at",
        lambda X, prev, curr, i:
            X[i].count("@") == 1
            and curr in ["B-EMAIL", "I-EMAIL"]
    )

    manager.add_feature(
        "email_has_dot_after_at",
        lambda X, prev, curr, i:
            (
                "@" in X[i]
                and "." in X[i].split("@")[-1]
            )
            and curr in ["B-EMAIL", "I-EMAIL"]
    )


    manager.add_feature(
        "three_dots",
        lambda X, prev, curr, i:
            X[i].count(".") == 3
            and curr in ["B-IPV4", "I-IPV4"]
    )

    manager.add_feature(
        "five_colons",
        lambda X, prev, curr, i:
            X[i].count(":") == 5
            and curr in ["B-MAC", "I-MAC"]
    )


    return manager