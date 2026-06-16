import re
from joblib import load

def get_word_shape(text):
    shape=""
    for char in text:
        if char.isupper():
            shape+="X"
        elif char.islower():
            shape+="x"
        elif char.isdigit():
            shape+="d"
        else:
            shape+=char
    return shape


def word2features(sentence, i):
    word = sentence[i]
    
    has_any_digit =sum(c.isdigit() for c in word)>0
    has_any_lower =sum(c.islower() for c in word)>0
    has_any_upper = sum(c.isupper() for c in word)>0
    has_any_special_char = sum(not c.isalnum() for c in word)>0
    number_of_digits = sum(c.isdigit() for c in word)
    number_of_alphabetical_characters = sum(c.isalpha() for c in word)
    number_of_special_characters = sum(not c.isalnum() for c in word)
    have_any_dot= "." in word
    have_any_colon = ":" in word
    have_any_dash = "-" in word
    have_any_slash = "/" in word
    have_any_at = "@" in word
    is_email=bool(re.fullmatch(r".+@.+\..+", word)) 
    is_ipv4= bool(re.fullmatch(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", word))
    is_mac= bool(re.fullmatch(r"([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}", word))
    is_hex= bool(re.fullmatch(r"[0-9a-fA-F]{8,}", word))
    is_four_digit= bool(re.fullmatch(r"\d{4}", word))
    features = {"bias":1.0}
    
    if i > 0:
        prev_word = sentence[i - 1]
        has_any_digit = sum(c.isdigit() for c in prev_word)>0
        features.update({
            "-1:word.lower()": prev_word.lower(),
            "-1:word.shape":get_word_shape(prev_word),
            "-1:word.len":len(prev_word),
            "-1:word.istitle()": prev_word.istitle(),
            "-1:word.isupper()":prev_word.isupper(),
            "-1:has_digit": has_any_digit,
            "prev_current": prev_word.lower()+"_"+word.lower(),
        })

    else:
        features["BOS"] = True

    features = {
        "word.lower()": word.lower(),
        "word[:2]": word[:2],
        "word[:3]":word[:3],
        "word[-2:]":word[-2:],
        "word[-3:]": word[-3:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.islower()": word.islower(),
        "word.len": len(word),
        "word.shape": get_word_shape(word),
        "word.has_digit()":has_any_digit,
        "word.has_lower()":has_any_lower,
        "word.has_upper()":has_any_upper,
        "word.has_special()":has_any_special_char,
        "digit_count":number_of_digits,
        "alpha_count":number_of_alphabetical_characters,
        "special_count":number_of_special_characters,
        "word.has_dot()": have_any_dot,
        "word.has_colon()": have_any_colon,
        "word.has_dash()":have_any_dash,
        "word.has_slash()":have_any_slash,
        "word.has_at()":have_any_at,
        "looks_email":is_email,
        "looks_ipv4":is_ipv4,
        "looks_mac": is_mac,
        "looks_hex": is_hex,
        "four_digits": is_four_digit,
    }



    if i < len(sentence) - 1:
        next_word = sentence[i + 1]
        features.update({
            "+1:word.lower()":next_word.lower(),
            "+1:word.shape":get_word_shape(next_word),
            "+1:word.len":len(next_word),
            "+1:word.istitle()":next_word.istitle(),
            "+1:word.isupper()":next_word.isupper(),
            "+1:has_digit":has_any_digit,
            "+0:+1": word.lower()+"_"+next_word.lower(),
        })

    else:
        features["EOS"] = True

    return features


class CRFPiiDetector:
    def __init__(self):
        self.model = load("./defenders/pii_detection/models/crf_model3.joblib")

    def get_features(self, text):
        words=text.split()
        features=[]
        for i in range(len(words)):
            word_features=word2features(words, i)
            features.append(word_features)

        return features
    
    def predict(self, text):
        features=self.get_features(text)
        predictions=self.model.predict([features])
        return predictions[0]
    

if __name__ == "__main__":
    detector = CRFPiiDetector()
    text = "My email is johndoe@gmail.com"
    predictions = detector.predict(text)
    print(predictions)