import logging
import re

import torch

from src_neural.config.global_config import gconf
from src_neural.utils import stopword_aggregator

logger = logging.getLogger(gconf.logger_name)


def is_valid_token(token):
    return len(token) >= 3 and token not in stopword_aggregator.custom_stopwords


def clean_product(string, tokenizer):
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()

    tokens = tokenizer(string)
    tokens = [str(x) for x in tokens if is_valid_token(str(x))]

    return tokens


def get_bow_representation(text_sequences, vocab_size):
    bow_representations = list()
    for text_sequence in text_sequences:
        bow_representation = torch.zeros(vocab_size)
        for index in text_sequence:
            if index > 1:
                bow_representation[index] = 1
        bow_representations.append(bow_representation)

    return torch.stack(bow_representations).cuda()
