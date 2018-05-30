import argparse
import logging
import sys
from typing import Any

from src_neural.config.global_config import gconf
from src_neural.config.model_config import mconf
from src_neural.utils import logging_inferface

logger = logging.getLogger(gconf.logger_name)


class Options(argparse.Namespace):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.model_save_path = None
        self.test_file_path = None
        self.predictions_save_path = None


def test_model(model_save_path, test_file_path, predictions_path):


    # with open(test_file_path, 'r') as test_file, open(predictions_path, 'w') as predictions_file:
    #     for product, prediction in zip(test_file, all_predictions):
    #         predictions_file.write("{}\t{}\n".format(product.strip(), prediction))

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
