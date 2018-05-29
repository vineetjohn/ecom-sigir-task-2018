import re

from src_neural.utils import stopword_aggregator


def is_valid_token(token):
    return len(token) >= 3 and token not in stopword_aggregator.custom_stopwords


def clean_product(string, tokenizer):
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()

    tokens = tokenizer(string)
    tokens = [str(x) for x in tokens if is_valid_token(str(x))]

    return tokens
