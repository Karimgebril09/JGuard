from defenders.pii_detection.src.pii_crf_predictor import get_word_shape, word2features, CRFPiiDetector

def test_word_shape():
    text1 = "karim"
    result1 = get_word_shape(text1)
    assert result1 == "xxxxx", f"Expected 'xxxxx' but got {result1}"

    text2 = "KARIM"
    result2 = get_word_shape(text2)
    assert result2 == "XXXXX", f"Expected 'XXXXX' but got {result2}"

    text3 = "01277883387"
    result3 = get_word_shape(text3)
    assert result3 == "ddddddddddd", f"Expected 'dddddddddd' but got {result3}"

    text4 = "karim123"
    result4 = get_word_shape(text4)
    assert result4 == "xxxxxddd", f"Expected 'xxxxxddd' but got {result4}"

    email_text = "karim@gmail.com"
    result5 = get_word_shape(email_text)
    assert result5 == "xxxxx@xxxxx.xxx", f"Expected 'xxxxx@xxxxx.com' but got {result5}"

    email_text2 = "karim_mahmoud@gmail.com"
    result6 = get_word_shape(email_text2)
    assert result6 == "xxxxx_xxxxxxx@xxxxx.xxx", f"Expected 'xxxxx_xxxxxxx@xxxxx.xxx' but got {result6}"



def test_word2features():
    sentence = ["johndoe@email.com"]
    features = word2features(sentence, 0)
    assert features['email'] == True

    sentence2 = ["123.456.789.012"]
    features2 = word2features(sentence2, 0)
    assert features2['ipv4'] == True

    sentence3 = ["00:1A:2B:3C:4D:5E"]
    features3 = word2features(sentence3, 0)
    assert features3['mac'] == True

    sentence4 = ["1A2B3C4D5E6F"]
    features4 = word2features(sentence4, 0)
    assert features4['hex'] == True

    sentence5 = ["1234"]
    features5 = word2features(sentence5, 0)
    assert features5['is_four_digits'] == True

    sentence7=["karim8"]
    features7=word2features(sentence7,0)
    assert features7['word has any digit']==True

    sentence8 = ["KaRIM"]
    features8 = word2features(sentence8, 0)
    assert features8["word has any lower"] == True

    sentence9 = ["KaRIM"]
    features9 = word2features(sentence9, 0)
    assert features9["word has any upper"] == True

    sentence10 = ["karim$123"]
    features10 = word2features(sentence10, 0)
    assert features10["word has any special"] == True


def run_tests():
    test_word_shape()
    test_word2features()
    print("All tests passed!")


if __name__ == "__main__":
    run_tests()