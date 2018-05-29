import argparse
import logging
import pickle
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.config import global_config
from src.utils import logging_inferface, data_processor

logger = logging.getLogger(global_config.logger_name)


def train_model(train_file_path, model_save_path, vectorizer_save_path, epochs):
    products_train, labels = data_processor.get_training_data(train_file_path)
    num_train = len(products_train)

    vectorizer = TfidfVectorizer(strip_accents='unicode')
    vectorizer.fit_transform(products_train)
    logger.info("vectorized products")
    with open(vectorizer_save_path, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
        logger.info("saved vectorizer")

    logger.debug("products_train: {}".format(products_train))
    logger.info("product_count: {}".format(num_train))
    logger.info("labels: {}".format(len(labels)))

    features = vectorizer.transform(products_train)
    logger.info("features: {}".format(features.shape))
    del products_train, vectorizer

    label_set = list(set(labels))
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.01)

    classifier = SGDClassifier(n_jobs=8)
    for epoch in range(1, epochs + 1):
        logger.info("epoch: {}".format(epoch))
        start_index = 0

        num_train = features_train.shape[0]
        while start_index < num_train:
            end_index = min(start_index + global_config.minibatch_size, num_train)
            logger.info("start_index: {}".format(start_index))
            logger.info("end_index: {}".format(end_index))
            classifier.partial_fit(X=features_train[start_index:end_index],
                                   y=labels_train[start_index:end_index],
                                   classes=label_set)
            start_index = end_index

            logger.info("running validation")
            predictions = classifier.predict(features_test)
            logger.debug("predictions: {}".format(predictions))

            score = f1_score(y_pred=predictions, y_true=labels_test,
                             average='weighted')
            logger.info("validation f1-score: {}".format(score))

        logger.info("saving model ...")
        with open(model_save_path, 'wb') as model_save_file:
            pickle.dump(classifier, model_save_file)
        logger.info("model saved to {}".format(model_save_path))


def main(args):
    global logger
    logger = logging_inferface.setup_custom_logger(
        global_config.logger_name, global_config.log_level)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file-path", type=str, required=True)
    parser.add_argument("--model-save-path", type=str, required=True)
    parser.add_argument("--vectorizer-save-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    options = vars(parser.parse_args(args=args))
    logger.debug("arguments: {}".format(options))

    train_model(options['train_file_path'], options['model_save_path'], options['vectorizer_save_path'],
                options['epochs'])


if __name__ == '__main__':
    main(sys.argv[1:])
