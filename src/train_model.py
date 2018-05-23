import argparse
import logging
import sys
import statistics

from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from src.config import global_config
from src.utils import logging_inferface, data_processor

logger = logging.getLogger(global_config.logger_name)


def train_model(input_file_path):
    products, labels = data_processor.get_data(input_file_path)
    features = data_processor.get_tfidf_features(products)
    logger.debug("products: {}".format(products))
    logger.debug("product_count: {}".format(len(products)))
    logger.debug("features: {}".format(features.shape))

    scores = list()
    for i in range(5):
        logger.info("running cross-validation #{}".format(i + 1))
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.1, random_state=0)

        classifier = MultinomialNB(fit_prior='weighted')
        classifier.fit(X=features_train, y=labels_train)
        predictions = classifier.predict(X=features_test)
        logger.debug("predictions: {}".format(predictions))

        score = f1_score(y_pred=predictions,
                         y_true=labels_test, average='micro')
        scores.append(score)

    logger.info("f1-score: {}".format(statistics.mean(scores)))


def main(args):
    global logger
    logger = logging_inferface.setup_custom_logger(
        global_config.logger_name, global_config.log_level)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file-path", type=str, required=True)
    options = vars(parser.parse_args(args=args))
    logger.debug("arguments: {}".format(options))
    train_model(options['input_file_path'])


if __name__ == '__main__':
    main(sys.argv[1:])
