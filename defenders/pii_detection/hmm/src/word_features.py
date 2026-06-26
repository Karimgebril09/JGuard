import re
def extract_word_features(word):
    feats =[]

    for n in range(1, 5):
        if len(word) >=n:
            feats.append(f"PRE{n}_{word[:n].lower()}")
            feats.append(f"SUF{n}_{word[-n:].lower()}")
    if word.isupper():
        feats.append("ALL_CAPS")
    elif word[0].isupper():
        feats.append("INIT_CAP")
    elif any(c.isupper() for c in word):
        feats.append("HAS_CAP")
    else:
        feats.append("NO_CAP")

    if word.isdigit():
        feats.append("ALL_DIGITS")
    elif any(c.isdigit() for c in word):
        feats.append("HAS_DIGIT")

    if re.fullmatch(r'[\w.+-]+@[\w-]+\.[a-zA-Z]+', word):
        feats.append("IS_EMAIL")
    if re.fullmatch(r'\d{1,3}(\.\d{1,3}){3}', word):
        feats.append("IS_IPV4")
    if re.fullmatch(r'([0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}', word):
        feats.append("IS_IPV6")
    if re.fullmatch(r'([0-9a-fA-F]{2}[:\-]){5}[0-9a-fA-F]{2}', word):
        feats.append("IS_MAC")
    if re.fullmatch(r'\d{3}-\d{2}-\d{4}', word):
        feats.append("IS_SSN")
    if re.fullmatch(r'[\d\-]{13,19}', word):
        feats.append("IS_CREDITCARD")
    if re.fullmatch(r'\+?[\d\s\-\(\)]{7,15}', word):
        feats.append("IS_PHONE")
    if re.fullmatch(r'[A-Za-z0-9@#$%^&+=!]{6,}', word):
        feats.append("LOOKS_PASSWORD")

    l =len(word)
    feats.append("LEN_SHORT" if l <=3 else "LEN_MED" if l <=7 else "LEN_LONG")

    return feats

NER_TAGS =[
    'B-ACCOUNTNAME', 'B-ACCOUNTNUMBER', 'B-CREDITCARDNUMBER', 'B-EMAIL',
    'B-IP', 'B-IPV4', 'B-IPV6', 'B-MAC', 'B-PASSWORD', 'B-PHONE_NUMBER',
    'B-SSN', 'B-USERNAME',
    'O',
]

NER_TAG_TO_INDEX ={tag: i for i, tag in enumerate(NER_TAGS)}
NER_TAG_TO_INDEX['UNK'] =len(NER_TAGS)
INDEX_TO_NER_TAG ={i: tag for tag, i in NER_TAG_TO_INDEX.items()}

