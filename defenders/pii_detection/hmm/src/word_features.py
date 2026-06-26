import re
def extract_word_features(word):
    f=[]#here my feature icnlude the word

    for n in range(1,5):
        if len(word) >=n: #append prefix and suffix 
            f.append(f"PRE{n}_{word[:n].lower()}")
            f.append(f"SUF{n}_{word[-n:].lower()}")
    
    # see if the word is all caps or initial with caps or has any caps  or no 
    if word.isupper():
        f.append("ALL_CAPS")
    elif word[0].isupper():
        f.append("INIT_CAP")
    elif any(c.isupper() for c in word):
        f.append("HAS_CAP")
    else:
        f.append("NO_CAP")

    #see if it all digit
    if word.isdigit():
        f.append("ALL_DIGITS")
    elif any(c.isdigit() for c in word):
        f.append("HAS_DIGIT")

    #see if mail
    if re.fullmatch(r'[\w.+-]+@[\w-]+\.[a-zA-Z]+',word):
        f.append("IS_EMAIL")
    #see if ip4
    if re.fullmatch(r'\d{1,3}(\.\d{1,3}){3}',word):
        f.append("IS_IPV4")
    #see if ip6
    if re.fullmatch(r'([0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}',word):
        f.append("IS_IPV6")
    #see if mac
    if re.fullmatch(r'([0-9a-fA-F]{2}[:\-]){5}[0-9a-fA-F]{2}',word):
        f.append("IS_MAC")
    #see if ssn
    if re.fullmatch(r'\d{3}-\d{2}-\d{4}',word):
        f.append("IS_SSN")
    #see if credit card
    if re.fullmatch(r'[\d\-]{13,19}',word):
        f.append("IS_CREDITCARD")
    #see if phone number
    if re.fullmatch(r'\+?[\d\s\-\(\)]{7,15}',word):
        f.append("IS_PHONE")
    if re.fullmatch(r'[A-Za-z0-9@#$%^&+=!]{6,}',word):
        f.append("LOOKS_PASSWORD")

    l=len(word)
    f.append("LEN_SHORT" if l <=3 else "LEN_MED" if l <=7 else "LEN_LONG")

    return f

NER_TAGS=[
    'B-ACCOUNTNAME','B-ACCOUNTNUMBER','B-CREDITCARDNUMBER','B-EMAIL',
    'B-IP','B-IPV4','B-IPV6','B-MAC','B-PASSWORD','B-PHONE_NUMBER',
    'B-SSN','B-USERNAME',
    'O',
]

NER_TAG_TO_INDEX={tag: i for i,tag in enumerate(NER_TAGS)}
NER_TAG_TO_INDEX['UNK']=len(NER_TAGS)
INDEX_TO_NER_TAG={i: tag for tag,i in NER_TAG_TO_INDEX.items()}

