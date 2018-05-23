import re
import spacy

from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer


def is_valid_token(token):
    if len(token) < 3 or token in STOP_WORDS:
        return False

    return True


def clean_product(string, tokenizer):

    string = re.sub(r"[^A-Za-z(),!?\'\`]", " ", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()

    tokens = tokenizer(string)
    tokens = [str(x) for x in tokens if is_valid_token(x)]

    return " ".join(tokens)


def get_data(input_file_path):
    products, labels = list(), list()
    spacy_nlp = spacy.load("en")
    tokenizer = English().Defaults.create_tokenizer(spacy_nlp)

    with open(input_file_path) as input_file:
        for line in input_file:
            [product, category_string] = line.strip().split('\t')
            categories = category_string.strip().split('>')
            products.append(clean_product(product, tokenizer))
            labels.append(categories[-1])

    return products, labels


def get_tfidf_features(products):
    tfidf_vectorizer = TfidfVectorizer(input=products, strip_accents='unicode')
    features = tfidf_vectorizer.fit_transform(products)

    return features
