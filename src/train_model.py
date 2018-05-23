import argparse
import logging
import sys

from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB

from src.config import global_config
from src.utils import logging_inferface, data_processor

logger = logging.getLogger(global_config.logger_name)


def train_model(input_file_path):
    products, labels = data_processor.get_data(input_file_path)
    features = data_processor.get_product_features(products)
    logger.debug(len(products))
    logger.debug(features.shape)

    clf = MultinomialNB(fit_prior='weighted')
    clf.fit(X=features, y=labels)
    predictions = clf.predict(X=features)
    logger.debug(predictions)

    score = f1_score(y_pred=predictions, y_true=labels, average='weighted')

    logger.info("F1-Score: {}".format(score))


def main(args):
    global logger
    logger = logging_inferface.setup_custom_logger(global_config.logger_name, global_config.log_level)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file-path", type=str, required=True)
    options = vars(parser.parse_args(args=args))
    logger.info("Arguments: {}".format(options))
    train_model(options['input_file_path'])


if __name__ == '__main__':
    main(sys.argv[1:])
