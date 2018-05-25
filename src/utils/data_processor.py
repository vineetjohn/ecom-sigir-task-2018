import logging
import re

import spacy
from spacy.lang.en import English

from src.config import global_config
from src.utils import stopword_aggregator

logger = logging.getLogger(global_config.logger_name)


def is_valid_token(token):
    return len(token) >= 3 and token not in stopword_aggregator.custom_stopwords


def clean_product(string, tokenizer):
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()

    tokens = tokenizer(string)
    tokens = [str(x) for x in tokens if is_valid_token(str(x))]

    return " ".join(tokens)


def get_training_data(input_file_path):
    products, labels = list(), list()
    spacy_nlp = spacy.load("en")
    tokenizer = English().Defaults.create_tokenizer(spacy_nlp)

    max_category_depth = 1
    with open(input_file_path) as input_file:
        i = 0
        for line in input_file:
            if not ((i + 1) % 10000):
                logger.info("{} lines processed".format(i + 1))
            i += 1
            [product, category_string] = line.strip().split('\t')
            categories = category_string.strip().split('>')
            max_category_depth = len(categories) if len(categories) > max_category_depth else max_category_depth
            cleaned_product = clean_product(product, tokenizer)
            if cleaned_product:
                logger.debug("original: {}, cleaned: {}".format(product, cleaned_product))
                products.append(cleaned_product)
                labels.append(category_string)
            else:
                logger.error("skipped product {}".format(product))
    logger.info("max_category_depth: {}".format(max_category_depth))

    return products, labels


def get_test_data(input_file_path):
    products = list()
    spacy_nlp = spacy.load("en")
    tokenizer = English().Defaults.create_tokenizer(spacy_nlp)

    with open(input_file_path) as input_file:
        for line in input_file:
            product = line.strip()
            cleaned_product = clean_product(product, tokenizer)
            products.append(cleaned_product)

    return products
