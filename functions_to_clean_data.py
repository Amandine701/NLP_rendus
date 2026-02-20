############# functions to clean data ############
import re
import spacy
from collections import Counter

# Load the French language model
nlp = spacy.load("fr_core_news_sm")

# Adding stop words specific to our corpus
custom_stops = ["candidat", "suppléant", "circonscription", "cest", "die", "der", "und", "das"]
for word in custom_stops:
    nlp.vocab[word].is_stop = True


def preprocess_text(text, use_lemma=True):
    """Full preprocessing: OCR cleaning + tokenization + stopwords removal + lemmatization.
    
    INPUT:
        - text: The raw string to process (e.g., OCR-extracted content).
        - use_lemma:  If True, returns tokens in their root form (lemma). 
                     If False, returns the original word.
    
    OUTPUT: A single string of cleaned, filtered, and space-separated tokens.
    """
    
    # --- OCR cleaning ---
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^\w\s\.,;!?]', '', text)
    text = re.sub(r'\b\d{1,2}\b', '', text)
    
    # --- Lowercase + spaCy processing ---
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha and len(token) > 2:
            tokens.append(token.lemma_ if use_lemma else token.text)
    
    return " ".join(tokens)


def filter_rare_words(corpus, min_docs=5):
    """
    Filter out words appearing in fewer than min_docs documents.
    
    Parameters:
        corpus : List of preprocessed text strings.
        min_docs : Minimum number of documents a word must appear in to be kept.
            
    Returns:
        filtered_corpus : Corpus with rare words removed.
    """
    # Compute document frequency
    doc_freq = Counter()
    for text in corpus:
        for word in set(text.split()):
            doc_freq[word] += 1

    # Keep words appearing in at least min_docs documents
    vocab = {word for word, freq in doc_freq.items() if freq >= min_docs}

    # Filter each document
    filtered_corpus = [" ".join([word for word in text.split() if word in vocab])
                       for text in corpus]
    
    return filtered_corpus