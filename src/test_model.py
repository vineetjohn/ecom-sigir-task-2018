import argparse
import logging
import pickle
import sys

from src.config import global_config
from src.utils import logging_inferface

logger = logging.getLogger(global_config.logger_name)


def convert_prediction_to_hierarchy(prediction, taxonomy):
    hierarchy = prediction.copy()
    prediction_node = taxonomy.nodes[prediction]
    while prediction_node.parent.id != -1:
        prediction_node = prediction_node.parent
        hierarchy = "{}>{}".format(prediction_node.id, hierarchy)
    return hierarchy


def test_model(model_path, test_file_path, test_vectors_path, predictions_path, taxonomy_file_path):
    with open(model_path, 'rb') as model_file:
        classifier = pickle.load(model_file)
        logger.info("loaded model into memory")

    with open(test_vectors_path, 'rb') as test_vectors_file:
        features = pickle.load(test_vectors_file)
        logger.info("loaded test vectors into memory")

    num_test = features.shape[0]
    start_index = 0
    all_predictions = list()
    while start_index < num_test:
        logger.info("start_index: {}".format(start_index))
        end_index = min(start_index + global_config.minibatch_size, num_test)
        predictions = classifier.predict(features[start_index:end_index])
        all_predictions.extend(predictions)
        start_index = end_index

    category_paths = list()
    with open(taxonomy_file_path, 'rb') as taxonomy_file:
        taxonomy = pickle.load(taxonomy_file)
        logger.info("Loaded taxonomy")
        for prediction in all_predictions:
            category_paths.append(convert_prediction_to_hierarchy(prediction, taxonomy))

    with open(test_file_path, 'r') as test_file, open(predictions_path, 'w') as predictions_file:
        for product, category_path in zip(test_file, category_paths):
            predictions_file.write("{}\t{}\n".format(product.strip(), category_path))

    logger.info("predictions written to file {}".format(predictions_path))


def main(args):
    global logger
    logger = logging_inferface.setup_custom_logger(
        global_config.logger_name, global_config.log_level)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-save-path", type=str, required=True)
    parser.add_argument("--test-file-path", type=str, required=True)
    parser.add_argument("--test-vectors-save-path", type=str, required=True)
    parser.add_argument("--predictions-save-path", type=str, required=True)
    parser.add_argument("--taxonomy-file-path", type=str, required=True)
    options = vars(parser.parse_args(args=args))
    logger.debug("arguments: {}".format(options))
    test_model(options['model_save_path'], options['test_file_path'], options['test_vectors_save_path'],
               options['predictions_save_path'], options['taxonomy_file_path'])


if __name__ == '__main__':
    main(sys.argv[1:])
