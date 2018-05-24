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


def train_model(train_file_path, test_file_path, model_save_path, test_vectors_save_path):
    products_train, labels = data_processor.get_training_data(train_file_path)
    products_test = data_processor.get_test_data(test_file_path)
    all_products = products_train + products_test
    num_train = len(products_train)

    vectorizer = TfidfVectorizer(input=all_products, strip_accents='unicode')

    features = vectorizer.fit_transform(all_products)
    logger.info("vectorized products")
    with open(test_vectors_save_path, 'wb') as test_vectors_file:
        pickle.dump(features[num_train:], test_vectors_file)
        logger.info("saved test vectors")

    logger.debug("products_train: {}".format(products_train))
    logger.info("product_count: {}".format(num_train))
    logger.info("features: {}".format(features.shape))
    logger.info("labels: {}".format(len(labels)))
    del products_train, products_test, all_products

    label_set = list(set(labels))
    scores = list()
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features[:num_train], labels, test_size=0.01)

    num_train = features_train.shape[0]

    classifier = SGDClassifier(n_jobs=8)
    start_index = 0
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
        scores.append(score)

    # logger.info("aggregate validation f1-score: {}".format(statistics.mean(scores)))
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
    parser.add_argument("--test-file-path", type=str, required=True)
    parser.add_argument("--model-save-path", type=str, required=True)
    parser.add_argument("--test-vectors-save-path", type=str, required=True)
    options = vars(parser.parse_args(args=args))
    logger.debug("arguments: {}".format(options))
    train_model(options['train_file_path'], options['test_file_path'], options['model_save_path'],
                options['test_vectors_save_path'])


if __name__ == '__main__':
    main(sys.argv[1:])
