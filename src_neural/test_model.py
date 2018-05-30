import argparse
import logging
import os
import sys
from typing import Any

import dill as pickle
import torch
from torchtext import data

from src_neural.config.global_config import gconf
from src_neural.config.model_config import mconf
from src_neural.models.classifier import NeuralClassifier
from src_neural.utils import logging_inferface, data_processor

logger = logging.getLogger(gconf.logger_name)


class Options(argparse.Namespace):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.model_save_path = None
        self.test_file_path = None
        self.predictions_save_path = None


def test_model(model_save_path, test_file_path, predictions_path):
    with open(os.path.join(model_save_path, gconf.vocab_filename), 'rb') as vocab_file:
        product_field = pickle.load(vocab_file)
    logger.info("loaded vocab")
    with open(os.path.join(model_save_path, gconf.labels_filename), 'rb') as labels_file:
        category_field = pickle.load(labels_file)
    logger.info("loaded labels")
    logger.info("vocab size: {}".format(len(product_field.vocab)))

    dataset = data.TabularDataset(
        path=test_file_path, format="tsv", fields=[('product', product_field)])
    test_iterator = data.Iterator(
        dataset=dataset, batch_size=mconf.minibatch_size, repeat=False)
    model = NeuralClassifier(len(product_field.vocab), len(category_field.vocab))
    model.cuda()
    model.eval()

    all_predictions = torch.LongTensor([]).cuda()
    for i, test_data in enumerate(test_iterator):
        text_sequences = test_data.product
        bow_representations = data_processor.get_bow_representation(
            text_sequences, len(product_field.vocab))
        logits = model(bow_representations)
        _, indices = torch.max(logits, 1)

        all_predictions = torch.cat([all_predictions, indices])

    with open(test_file_path, 'r') as test_file, open(predictions_path, 'w') as predictions_file:
        for product, prediction in zip(test_file, all_predictions):
            predictions_file.write("{}\t{}\n".format(product.strip(), category_field.vocab.itos[prediction]))
    logger.info("predictions written to file {}".format(predictions_path))


def main(args):
    global logger
    logger = logging_inferface.setup_custom_logger(gconf.logger_name, gconf.log_level)

    options = Options()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-save-path", type=str, required=True)
    parser.add_argument("--test-file-path", type=str, required=True)
    parser.add_argument("--predictions-save-path", type=str, required=True)
    parser.parse_known_args(args=args, namespace=options)

    logger.info("options: {}".format(options.__dict__))
    logger.info("global config: {}".format(gconf.__dict__))
    logger.info("model config: {}".format(mconf.__dict__))
    test_model(options.model_save_path, options.test_file_path, options.predictions_save_path)


if __name__ == '__main__':
    main(sys.argv[1:])
