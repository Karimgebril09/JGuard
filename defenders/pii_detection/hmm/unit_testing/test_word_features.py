


from defenders.pii_detection.hmm.src.word_features import extract_word_features


class TestWordFeatures:
    
    def test_prefix_suffix_features(self):
        features = extract_word_features("test")
        
        # Should have 1-4 gram prefixes and suffixes
        assert any(f.startswith("PRE") for f in features), "Missing prefix features"
        assert any(f.startswith("SUF") for f in features), "Missing suffix features"

    
    def test_capitalization_features(self):
        #check capital features
        assert "ALL_CAPS" in extract_word_features("HELLO")
        assert "INIT_CAP" in extract_word_features("Hello")
        assert "HAS_CAP" in extract_word_features("heLLo")
        assert "NO_CAP" in extract_word_features("hello")
    
    def test_digit_features(self):
        #check digit features
        assert "ALL_DIGITS" in extract_word_features("12345")
        assert "HAS_DIGIT" in extract_word_features("test123")
        assert not any(f.startswith("DIGIT") for f in extract_word_features("test"))
    
    def test_email_detection(self):
        #check email
        features = extract_word_features("user@example.com")
        assert "IS_EMAIL" in features
    
    def test_phone_detection(self):
        #check phone number
        features = extract_word_features("+201001234567")
        assert "IS_PHONE" in features
    
    def test_ssn_detection(self):
        #check ssn
        features = extract_word_features("123-45-6789")
        assert "IS_SSN" in features
    
    def test_egyptian_national_id(self):
        #check national id
        features = extract_word_features("30001011234567")
        assert "ALL_DIGITS" in features
        assert "LEN_LONG" in features
    
    def test_length_categories(self):
        #check len
        assert "LEN_SHORT" in extract_word_features("ab")
        assert "LEN_MED" in extract_word_features("test")
        assert "LEN_LONG" in extract_word_features("verylongword")


if __name__ == "__main__":
    test = TestWordFeatures()
    test.test_prefix_suffix_features()
    test.test_capitalization_features()
    test.test_digit_features()
    test.test_email_detection()
    test.test_phone_detection()
    test.test_ssn_detection()
    test.test_egyptian_national_id()
    test.test_length_categories()