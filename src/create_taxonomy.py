import argparse
import logging
import pickle
import sys

from src.config import global_config
from src.utils import logging_inferface, taxonomy_interface

logger = logging.getLogger(global_config.logger_name)


def generate_taxonomy(input_file_path, taxonomy_save_path):
    taxonomy = taxonomy_interface.Taxonomy()
    with open(input_file_path) as input_file:
        for line in input_file:
            [_, category_string] = line.strip().split('\t')
            categories = category_string.strip().split('>')
            taxonomy.add_categories(categories)

    logger.debug(taxonomy)
    with open(taxonomy_save_path, 'wb') as taxonomy_file:
        pickle.dump(taxonomy, taxonomy_file)
        logger.info("Saved taxonomy to {}".format(taxonomy_save_path))


def main(args):
    global logger
    logger = logging_inferface.setup_custom_logger(global_config.logger_name, global_config.log_level)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file-path", type=str, required=True)
    parser.add_argument("--taxonomy-file-path", type=str, required=True)
    options = vars(parser.parse_args(args=args))
    logger.info("Arguments: {}".format(options))

    logger.info("Creating taxonomy")
    generate_taxonomy(options['input_file_path'], options['taxonomy_file_path'])


if __name__ == '__main__':
    main(sys.argv[1:])
