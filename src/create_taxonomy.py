import argparse
import logging
import sys

from src.config import global_config
from src.utils import log_initializer

logger = logging.getLogger(global_config.logger_name)


def generate_taxonomy(input_file_path, taxonomy_save_path):
    with open(input_file_path) as input_file:
        for line in input_file:
            [_, category_string] = line.strip().split('\t')
            categories = category_string.strip().split('>')
            logger.debug(categories)


def main(args):
    global logger
    logger = log_initializer.setup_custom_logger(global_config.logger_name, global_config.log_level)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file-path", type=str, required=True)
    parser.add_argument("--taxonomy-file-path", type=str)
    options = vars(parser.parse_args(args=args))
    logger.info("Arguments: {}".format(options))

    logger.info("Creating taxonomy")
    generate_taxonomy(options['input_file_path'], options['taxonomy_file_path'])


if __name__ == '__main__':
    main(sys.argv[1:])
