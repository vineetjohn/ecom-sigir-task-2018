import argparse
import logging
import pickle
import sys

from src.config import global_config
from src.utils import logging_inferface, data_processor

logger = logging.getLogger(global_config.logger_name)


def convert_prediction_to_hierarchy(prediction, taxonomy):
    hierarchy = prediction.copy()
    prediction_node = taxonomy.nodes[prediction]
    while prediction_node.parent.id != -1:
        prediction_node = prediction_node.parent
        hierarchy = "{}>{}".format(prediction_node.id, hierarchy)
    return hierarchy


def test_model(model_path, test_file_path, vectorizer_save_path, predictions_path):
    with open(vectorizer_save_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
        logger.info("loaded vectorizer into memory")

    products_test = data_processor.get_test_data(test_file_path)
    num_test = len(products_test)
    features = vectorizer.transform(products_test)
    del products_test, vectorizer

    with open(model_path, 'rb') as model_file:
        classifier = pickle.load(model_file)
        logger.info("loaded model into memory")

    start_index = 0
    all_predictions = list()
    while start_index < num_test:
        logger.info("start_index: {}".format(start_index))
        end_index = min(start_index + global_config.minibatch_size, num_test)
        predictions = classifier.predict(features[start_index:end_index])
        all_predictions.extend(predictions)
        start_index = end_index

    with open(test_file_path, 'r') as test_file, open(predictions_path, 'w') as predictions_file:
        for product, prediction in zip(test_file, all_predictions):
            predictions_file.write("{}\t{}\n".format(product.strip(), prediction))

    logger.info("predictions written to file {}".format(predictions_path))


def main(args):
    global logger
    logger = logging_inferface.setup_custom_logger(
        global_config.logger_name, global_config.log_level)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-save-path", type=str, required=True)
    parser.add_argument("--test-file-path", type=str, required=True)
    parser.add_argument("--vectorizer-save-path", type=str, required=True)
    parser.add_argument("--predictions-save-path", type=str, required=True)
    options = vars(parser.parse_args(args=args))
    logger.debug("arguments: {}".format(options))
    test_model(options['model_save_path'], options['test_file_path'], options['vectorizer_save_path'],
               options['predictions_save_path'])


if __name__ == '__main__':
    main(sys.argv[1:])
