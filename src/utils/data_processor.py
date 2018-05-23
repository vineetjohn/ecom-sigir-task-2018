import logging
import re

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from spacy.lang.en import English

from src.config import global_config
from src.utils import stopword_aggregator

logger = logging.getLogger(global_config.logger_name)


def is_valid_token(token):
    return len(token) >= 3 and token not in stopword_aggregator.custom_stopwords


def clean_product(string, tokenizer):
    string = re.sub(r"[^A-Za-z(),!?\'`]", " ", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()

    tokens = tokenizer(string)
    tokens = [str(x) for x in tokens if is_valid_token(str(x))]

    return " ".join(tokens)


def get_data(input_file_path):
    products, labels = list(), list()
    spacy_nlp = spacy.load("en")
    tokenizer = English().Defaults.create_tokenizer(spacy_nlp)

    with open(input_file_path) as input_file:
        for line in input_file:
            [product, category_string] = line.strip().split('\t')
            categories = category_string.strip().split('>')
            cleaned_product = clean_product(product, tokenizer)
            logger.info("original: {}, cleaned: {}".format(product, cleaned_product))
            products.append(cleaned_product)
            labels.append(categories[-1])

    return products, labels


def get_tfidf_features(products):
    tfidf_vectorizer = TfidfVectorizer(input=products, strip_accents='unicode')
    features = tfidf_vectorizer.fit_transform(products)

    return features


def get_count_features(products):
    count_vectorizer = CountVectorizer(input=products, strip_accents='unicode')
    features = count_vectorizer.fit_transform(products)

    return features
