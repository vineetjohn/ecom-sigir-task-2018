import argparse
import logging
import os
import sys
from typing import Any

import dill as pickle
import spacy
import torch
from sklearn.metrics import f1_score
from spacy.lang.en import English
from torchtext import data

from src_neural.config.global_config import gconf
from src_neural.config.model_config import mconf
from src_neural.models.classifier import NeuralClassifier
from src_neural.utils import logging_inferface, data_processor

logger = logging.getLogger(gconf.logger_name)


class Options(argparse.Namespace):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.train_file_path = None
        self.model_save_path = None
        self.num_epochs = None


def train_model(train_file_path, model_save_path, num_epochs):
    spacy_nlp = spacy.load("en")
    spacy_tokenizer = English().Defaults.create_tokenizer(spacy_nlp)
    product_field = data.Field(
        sequential=True, lower=True, tokenize=(lambda x: data_processor.clean_product(x, spacy_tokenizer)),
        batch_first=True)
    category_field = data.Field(sequential=False, pad_token=None, unk_token=None)
    dataset = data.TabularDataset(
        path=train_file_path, format="tsv",
        fields=[('product', product_field), ('category', category_field)])
    train_dataset, val_dataset = dataset.split(split_ratio=[0.9, 0.1])
    product_field.build_vocab(train_dataset)
    category_field.build_vocab(train_dataset)
    logger.info("vocab size: {}".format(len(product_field.vocab)))

    train_iterator = data.Iterator(
        dataset=train_dataset, batch_size=mconf.minibatch_size, repeat=False)
    val_iterator = data.Iterator(
        dataset=val_dataset, batch_size=mconf.minibatch_size, repeat=False)

    with open(os.path.join(model_save_path, gconf.vocab_filename), 'wb') as vocab_file:
        pickle.dump(product_field, vocab_file)
    logger.info("saved vocab to {}".format(model_save_path))

    model = NeuralClassifier(len(product_field.vocab), len(category_field.vocab))
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=mconf.learning_rate)
    model.train()
    for epoch in range(1, num_epochs + 1):
        train_iterator.init_epoch()
        for i, train_data in enumerate(train_iterator):
            text_sequences, labels = train_data.product, train_data.category
            bow_representations = data_processor.get_bow_representation(
                text_sequences, len(product_field.vocab))
            optimizer.zero_grad()
            logits = model(bow_representations)

            loss = torch.nn.functional.cross_entropy(input=logits, target=labels)
            logger.info("loss: {:.2f}, epoch: {}-{}".format(loss, epoch, i))
            loss.backward()
            optimizer.step()

        all_predictions = torch.LongTensor([]).cuda()
        all_labels = torch.LongTensor([]).cuda()
        for i, val_data in enumerate(val_iterator):
            text_sequences, labels = val_data.product, val_data.category
            bow_representations = data_processor.get_bow_representation(
                text_sequences, len(product_field.vocab))
            logits = model(bow_representations)
            _, indices = torch.max(logits, 1)

            all_predictions = torch.cat([all_predictions, indices])
            all_labels = torch.cat([all_labels, labels])

        validation_accuracy = f1_score(y_pred=all_predictions, y_true=all_labels,
                                       average='weighted')
        logger.info("validation_accuracy: {}".format(validation_accuracy))

        torch.save(model.state_dict(), os.path.join(model_save_path, gconf.model_filename))
        logger.info("saved model to {}".format(model_save_path))


def main(args):
    global logger
    logger = logging_inferface.setup_custom_logger(gconf.logger_name, gconf.log_level)

    options = Options()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file-path", type=str, required=True)
    parser.add_argument("--model-save-path", type=str, required=True)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.parse_known_args(args=args, namespace=options)

    logger.info("options: {}".format(options.__dict__))
    logger.info("global config: {}".format(gconf.__dict__))
    logger.info("model config: {}".format(mconf.__dict__))

    train_model(options.train_file_path, options.model_save_path, options.num_epochs)


if __name__ == '__main__':
    main(sys.argv[1:])
