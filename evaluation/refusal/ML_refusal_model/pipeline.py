import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import Any, cast
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def load_data():
    """
    Load training and testing datasets from CSV files.
    Returns:
        train_df (DataFrame): Training dataset
        test_df (DataFrame): Testing dataset
    """
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    train_df = train_df.dropna(subset=["label"])
    test_df = test_df.dropna(subset=["label"])

    return train_df, test_df


# PREPROCESSING PIPELINE
REFUSAL_KEYWORDS = [
    'i cannot', 'i can\'t', 'i apologize', 'i\'m sorry', 'i cannot provide',
    'unable to', 'i cannot help', 'i don\'t', 'unfortunately', 'unable', 
    'i cannot assist', 'cannot assist', 'i cannot help', 'not able to',
    'i cannot answer', 'cannot answer', 'i cannot say', 'cannot say',
    'against my values', 'against my guidelines', 'not appropriate',
    'not allowed', 'not permitted', 'cannot provide', 'will not',
    'won\'t provide', 'won\'t help', 'refused', 'refuse', 'declining',
    'decline', 'don\'t think', 'don\'t feel', 'not comfortable',
    # LLM-specific refusal patterns
    'as an ai', 'as a language model', 'as an assistant', 'i must inform',
    'i should note', 'i need to clarify', 'it would not be appropriate',
    'not in a position to', 'beyond my capabilities', 'outside my scope',
    'i am not able', 'i am unable', "i'm not able", 'i must decline',
    'against my programming', 'violates my guidelines', 'cannot fulfill',
    'not something i can', 'i have to decline', 'ethically', 'harmful',
    'dangerous', 'illegal', 'unethical', 'inappropriate request',
    'cannot recommend', 'would advise against', 'strongly discourage',
    'i would not', 'i could not', 'i should not', 'i must not'
]

def expand_contractions(text):
    """
    Expand contractions to preserve meaning for refusal detection.
    E.g., "can't" -> "cannot", "won't" -> "will not"
    """
    contractions = {
        "can't": "cannot", "won't": "will not", "don't": "do not",
        "i'm": "i am", "i've": "i have", "i'll": "i will",
        "i'd": "i would", "shouldn't": "should not", "couldn't": "could not",
        "wouldn't": "would not", "isn't": "is not", "aren't": "are not",
        "doesn't": "does not", "didn't": "did not", "haven't": "have not",
        "hasn't": "has not", "wasn't": "was not", "weren't": "were not",
        "it's": "it is", "that's": "that is", "there's": "there is"
    }
    text_lower = text.lower()
    for contraction, expanded in contractions.items():
        text_lower = text_lower.replace(contraction, expanded)
    return text_lower


def preprocess_text(text):
    """
    Apply comprehensive preprocessing to text:
    1. Lowercasing
    2. Contraction expansion
    3. Noise removal and special character handling
    4. Extra whitespace removal
    5. Tokenization
    6. Lemmatization
    7. Emoji removal
    """
    text = text.lower()
    
    # Expand contractions to preserve meaning
    text = expand_contractions(text)

    # Remove emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "\U00002600-\U000026FF"  # misc symbols
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters but keep apostrophes and hyphens for contractions
    text = re.sub(r'[^a-zA-Z0-9\s\'\-]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    processed_text = ' '.join(tokens)
    
    return processed_text, tokens

# FEATURE EXTRACTION 
# 1. Response Length Features
def extract_length_features(response_text):
    words = response_text.split()
    word_count = len(words)
    avg_word_length = np.mean([len(w) for w in words]) if word_count > 0 else 0
    return {
        'response_length': len(response_text),
        'word_count': word_count,
        'avg_word_length': avg_word_length,
        'char_per_word': len(response_text) / max(word_count, 1)
    }

# 2. Refusal Keywords Detection
def detect_refusal_keywords(response_text):
    """Detect if refusal keywords appear in the response, especially at the beginning"""
    text_lower = response_text.lower()
    
    # Check if any refusal keyword appears in the first 100 characters (beginning)
    beginning = text_lower[:100]
    has_keyword_at_start = sum(1 for keyword in REFUSAL_KEYWORDS if keyword in beginning)
    
    # Overall keyword presence
    has_keyword_overall = sum(1 for keyword in REFUSAL_KEYWORDS if keyword in text_lower)
    
    return {
        'refusal_keyword_at_start': has_keyword_at_start,
        'refusal_keyword_overall': has_keyword_overall,
        'has_any_refusal_keyword': 1 if has_keyword_overall > 0 else 0
    }

# 3. Sentiment Analysis Features
def extract_sentiment_features(response_text):
    """Extract sentiment features using TextBlob"""
    try:
        blob = TextBlob(response_text)
        polarity = blob.sentiment.polarity  # -1 to 1 (negative to positive)
        subjectivity = blob.sentiment.subjectivity  # 0 to 1 (objective to subjective)
        
        return {
            'sentiment_polarity': polarity,
            'sentiment_subjectivity': subjectivity,
            'is_negative_sentiment': 1 if polarity < -0.1 else 0,
            'is_neutral_sentiment': 1 if -0.1 <= polarity <= 0.1 else 0,
            'is_positive_sentiment': 1 if polarity > 0.1 else 0
        }
    except:
        return {
            'sentiment_polarity': 0,
            'sentiment_subjectivity': 0,
            'is_negative_sentiment': 0,
            'is_neutral_sentiment': 1,
            'is_positive_sentiment': 0
        }

# 4. Response Structure Features
def extract_structure_features(response_text):
    """Extract structural features of the response"""
    sentences = [s for s in response_text.split('.') if s.strip()]
    sentence_count = len(sentences)
    avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentence_count > 0 else 0
    punctuation_count = sum(1 for c in response_text if c in string.punctuation)
    question_mark_count = response_text.count('?')
    exclamation_count = response_text.count('!')
    uppercase_ratio = sum(1 for c in response_text if c.isupper()) / max(len(response_text), 1)
    has_multiple_sentences = 1 if sentence_count > 1 else 0
    
    return {
        'sentence_count': sentence_count,
        'avg_sentence_length': avg_sentence_length,
        'punctuation_count': punctuation_count,
        'question_mark_count': question_mark_count,
        'exclamation_count': exclamation_count,
        'uppercase_ratio': uppercase_ratio,
        'has_multiple_sentences': has_multiple_sentences
    }

# 5. Apologetic Tone Features
def extract_apologetic_features(response_text):
    """Detect apologetic/formal tone indicators"""
    text_lower = response_text.lower()
    
    apologetic_words = ['sorry', 'apologize', 'apologies', 'regret', 'unfortunately']
    formal_indicators = ['unable', 'cannot', 'cannot help', 'assist', 'request', 'policy']
    
    has_apology = sum(1 for word in apologetic_words if word in text_lower)
    has_formal = sum(1 for word in formal_indicators if word in text_lower)
    
    return {
        'apology_word_count': has_apology,
        'formal_word_count': has_formal,
        'is_apologetic': 1 if has_apology > 0 else 0,
        'is_formal': 1 if has_formal > 0 else 0
    }

# 6. First-Person Pronoun Features
def extract_first_person_features(response_text):
    """Extract first-person pronoun density - refusals heavily use 'I' statements"""
    text_lower = response_text.lower()
    words = text_lower.split()
    first_person = ['i', "i'm", "i've", "i'll", "i'd", 'me', 'my', 'myself', 
                    'i am', 'i have', 'i will', 'i would']
    fp_count = sum(1 for w in words if w in first_person)
    
    return {
        'first_person_count': fp_count,
        'first_person_ratio': fp_count / max(len(words), 1)
    }

# 7. Hedging/Uncertainty Language Features
def extract_hedging_features(response_text):
    """Detect hedging and uncertainty language common in refusals"""
    text_lower = response_text.lower()
    hedging_words = ['however', 'but', 'although', 'while', 'instead', 
                     'alternatively', 'rather', 'perhaps', 'maybe', 'might',
                     'could', 'would suggest', 'recommend instead', 'consider']
    count = sum(1 for word in hedging_words if word in text_lower)
    
    return {
        'hedging_word_count': count,
        'has_hedging': 1 if count > 0 else 0
    }

# 8. Opening Pattern Detection
def extract_opening_pattern_features(response_text):
    """Detect refusal-specific opening patterns - very important for classification"""
    first_50 = response_text[:50].lower()
    
    refusal_openings = [
        r"^i('m| am) sorry", r"^i cannot", r"^i can't", r"^unfortunately",
        r"^i('m| am) not able", r"^i apologize", r"^as an ai",
        r"^i('m| am) unable", r"^i must", r"^i('m| am) afraid",
        r"^i('m| am) not in a position", r"^i regret", r"^i('m| am) sorry"
    ]
    
    starts_with_refusal = any(re.match(p, first_50) for p in refusal_openings)
    
    return {
        'starts_with_refusal_pattern': 1 if starts_with_refusal else 0
    }

# 9. Negation Density Features
def extract_negation_features(response_text):
    """Extract negation density - refusals have high negation counts"""
    text_lower = response_text.lower()
    negations = ["not", "no", "never", "don't", "won't", "can't", "cannot", 
                 "shouldn't", "wouldn't", "isn't", "aren't", "doesn't",
                 "do not", "will not", "should not", "would not", "is not",
                 "are not", "does not", "did not", "have not", "has not"]
    words = text_lower.split()
    
    # Count single-word negations
    neg_count = sum(1 for w in words if w in negations)
    # Also count multi-word negations
    for neg in negations:
        if ' ' in neg and neg in text_lower:
            neg_count += text_lower.count(neg)
    
    return {
        'negation_count': neg_count,
        'negation_ratio': neg_count / max(len(words), 1)
    }

# Extract all features
def extract_all_features(train_df, test_df):
    print("Extracting length features...")
    train_length_features = train_df['response'].apply(extract_length_features).apply(pd.Series)
    test_length_features = test_df['response'].apply(extract_length_features).apply(pd.Series)

    print("Extracting refusal keyword features...")
    train_keyword_features = train_df['response'].apply(detect_refusal_keywords).apply(pd.Series)
    test_keyword_features = test_df['response'].apply(detect_refusal_keywords).apply(pd.Series)

    print("Extracting sentiment features...")
    train_sentiment_features = train_df['response'].apply(extract_sentiment_features).apply(pd.Series)
    test_sentiment_features = test_df['response'].apply(extract_sentiment_features).apply(pd.Series)

    print("Extracting structure features...")
    train_structure_features = train_df['response'].apply(extract_structure_features).apply(pd.Series)
    test_structure_features = test_df['response'].apply(extract_structure_features).apply(pd.Series)

    print("Extracting apologetic tone features...")
    train_apology_features = train_df['response'].apply(extract_apologetic_features).apply(pd.Series)
    test_apology_features = test_df['response'].apply(extract_apologetic_features).apply(pd.Series)

    print("Extracting first-person pronoun features...")
    train_first_person_features = train_df['response'].apply(extract_first_person_features).apply(pd.Series)
    test_first_person_features = test_df['response'].apply(extract_first_person_features).apply(pd.Series)

    print("Extracting hedging language features...")
    train_hedging_features = train_df['response'].apply(extract_hedging_features).apply(pd.Series)
    test_hedging_features = test_df['response'].apply(extract_hedging_features).apply(pd.Series)

    print("Extracting opening pattern features...")
    train_opening_features = train_df['response'].apply(extract_opening_pattern_features).apply(pd.Series)
    test_opening_features = test_df['response'].apply(extract_opening_pattern_features).apply(pd.Series)

    print("Extracting negation features...")
    train_negation_features = train_df['response'].apply(extract_negation_features).apply(pd.Series)
    test_negation_features = test_df['response'].apply(extract_negation_features).apply(pd.Series)

    print("\nFeature extraction complete!")

    train_engineered_features = pd.concat([
        train_length_features,
        train_keyword_features,
        train_sentiment_features,
        train_structure_features,
        train_apology_features,
        train_first_person_features,
        train_hedging_features,
        train_opening_features,
        train_negation_features
    ], axis=1)

    test_engineered_features = pd.concat([
        test_length_features,
        test_keyword_features,
        test_sentiment_features,
        test_structure_features,
        test_apology_features,
        test_first_person_features,
        test_hedging_features,
        test_opening_features,
        test_negation_features
    ], axis=1)
    
    return train_engineered_features, test_engineered_features

def vectorize_tfidf(train_df, test_df, return_vectorizer=False):
    print("Generating TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=5, max_df=0.8)
    train_tfidf = tfidf_vectorizer.fit_transform(train_df['processed_response'])
    test_tfidf = tfidf_vectorizer.transform(test_df['processed_response'])

    print(f"TF-IDF shape - Train: {train_tfidf.shape}, Test: {test_tfidf.shape}")
    # Convert sparse matrices to dense for concatenation with other features
    train_tfidf_dense = cast(Any, train_tfidf).toarray()
    test_tfidf_dense = cast(Any, test_tfidf).toarray()

    # Create dataframes for vectorized features
    train_tfidf_df = pd.DataFrame(train_tfidf_dense, columns=[f'tfidf_{i}' for i in range(train_tfidf_dense.shape[1])])
    test_tfidf_df = pd.DataFrame(test_tfidf_dense, columns=[f'tfidf_{i}' for i in range(test_tfidf_dense.shape[1])])

    if return_vectorizer:
        return train_tfidf_df, test_tfidf_df, tfidf_vectorizer

    return train_tfidf_df, test_tfidf_df

def vectorize_count(train_df, test_df, return_vectorizer=False):
    print("\nGenerating Count Vectorizer features...")
    count_vectorizer = CountVectorizer(max_features=2000, ngram_range=(1, 2), min_df=5, max_df=0.8)
    train_count = count_vectorizer.fit_transform(train_df['processed_response'])
    test_count = count_vectorizer.transform(test_df['processed_response'])

    print(f"Count Vectorizer shape - Train: {train_count.shape}, Test: {test_count.shape}")

    # Convert sparse matrices to dense for concatenation with other features
    train_count_dense = cast(Any, train_count).toarray()
    test_count_dense = cast(Any, test_count).toarray()

    # Create dataframes for vectorized features
    train_count_df = pd.DataFrame(train_count_dense, columns=[f'count_{i}' for i in range(train_count_dense.shape[1])])
    test_count_df = pd.DataFrame(test_count_dense, columns=[f'count_{i}' for i in range(test_count_dense.shape[1])])

    if return_vectorizer:
        return train_count_df, test_count_df, count_vectorizer

    return train_count_df, test_count_df


