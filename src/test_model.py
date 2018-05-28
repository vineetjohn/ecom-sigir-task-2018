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


def test_model(model_path, test_file_path, vectorizer_save_path, taxonomy_file_path, predictions_path):
    with open(vectorizer_save_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
        logger.info("loaded vectorizer into memory")

    products_test = data_processor.get_test_data(test_file_path)
    num_test = len(products_test)
    features = vectorizer.transform(products_test).todense()
    del products_test, vectorizer

    with open(taxonomy_file_path, 'rb') as taxonomy_file:
        taxonomy = pickle.load(taxonomy_file)
        logger.info("Loaded taxonomy")

    level_predictions = dict()

    for level in taxonomy.level_nodes:

        logger.info("Predicting level {}".format(level))

        with open(model_path + ".{}".format(level), 'rb') as model_file:
            classifier = pickle.load(model_file)
            logger.info("loaded model into memory")

        start_index = 0
        all_log_prob_dist = list()
        while start_index < num_test:
            logger.info("start_index: {}".format(start_index))
            end_index = min(start_index + global_config.minibatch_size, num_test)

            log_prob_dist = classifier.predict_log_proba(features[start_index:end_index])
            all_log_prob_dist.extend(log_prob_dist)

            start_index = end_index

        predictions = list()
        for log_prob_dist in all_log_prob_dist:
            class_probs = zip(log_prob_dist, classifier.classes_)
            class_probs = sorted(class_probs, key=lambda x: x[0], reverse=True)
            prediction = [x[1] for x in class_probs]
            predictions.append(prediction)

        if not level:
            level_predictions[level] = [x[0] for x in predictions]
        else:
            parent_predictions = level_predictions[level - 1]
            level_predictions_list = list()
            for index, prediction in enumerate(predictions):
                parent_class_pred = parent_predictions[index]
                # print("parent_class_pred: {}".format(parent_class_pred))
                current_class_pred = 0

                if parent_class_pred:
                    for class_pred in prediction:
                        if class_pred in taxonomy.nodes[parent_class_pred].children:
                            current_class_pred = class_pred
                            break
                level_predictions_list.append(current_class_pred)
            level_predictions[level] = level_predictions_list

    full_predictions = list()
    for i in range(num_test):
        full_prediction = ""
        for level in taxonomy.level_nodes:
            current_pred = level_predictions[level][i]
            if not current_pred:
                break
            full_prediction += ">{}".format(current_pred)
            if not taxonomy.nodes[current_pred].children:
                break
        full_predictions.append(full_prediction)

    with open(test_file_path, 'r') as test_file, open(predictions_path, 'w') as predictions_file:
        for product, prediction in zip(test_file, full_predictions):
            predictions_file.write("{}\t{}\n".format(product.strip(), prediction[1:]))

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
    parser.add_argument("--taxonomy-file-path", type=str, required=True)
    options = vars(parser.parse_args(args=args))
    logger.debug("arguments: {}".format(options))

    test_model(options['model_save_path'], options['test_file_path'], options['vectorizer_save_path'],
               options['taxonomy_file_path'], options['predictions_save_path'])


if __name__ == '__main__':
    main(sys.argv[1:])
