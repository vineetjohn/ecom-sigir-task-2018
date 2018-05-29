import argparse
import logging
import sys
from typing import Any

import spacy
from spacy.lang.en import English
from torchtext import data

from src_neural.config.global_config import gconf
from src_neural.utils import logging_inferface, data_processor

logger = logging.getLogger(gconf.logger_name)


class Options(argparse.Namespace):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.train_file_path = None
        self.model_save_path = None
        self.num_epochs = None


def train_model(train_file_path, model_save_path, epochs):
    spacy_nlp = spacy.load("en")
    spacy_tokenizer = English().Defaults.create_tokenizer(spacy_nlp)
    product_field = data.Field(
        sequential=True, lower=True, tokenize=(lambda x: data_processor.clean_product(x, spacy_tokenizer)))
    category_field = data.Field(sequential=False)
    dataset = data.TabularDataset(
        path=train_file_path, format="tsv",
        fields=[('product', product_field), ('category', category_field)])
    train_dataset, val_dataset = dataset.split(split_ratio=[0.9, 0.1])
    product_field.build_vocab(train_dataset)
    category_field.build_vocab(train_dataset)
    print(product_field.vocab.stoi)

    train_iterator = data.Iterator(
        dataset=train_dataset, batch_size=gconf.minibatch_size, repeat=False)
    for i, train_data in enumerate(train_iterator):
        # print(train_data)
        text, label = train_data.product, train_data.category
        # print(text)


def main(args):
    global logger
    logger = logging_inferface.setup_custom_logger(gconf.logger_name, gconf.log_level)

    options = Options()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file-path", type=str, required=True)
    parser.add_argument("--model-save-path", type=str, required=True)
    parser.add_argument("--vectorizer-save-path", type=str, required=True)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.parse_known_args(args=args, namespace=options)
    logger.info("options: {}".format(options.__dict__))
    logger.info("gconf: {}".format(gconf.__dict__))

    train_model(options.train_file_path, options.model_save_path, options.num_epochs)


if __name__ == '__main__':
    main(sys.argv[1:])
