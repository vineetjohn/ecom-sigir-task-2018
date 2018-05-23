#!/usr/bin/env bash

PROJECT_DIR_PATH="$PWD/$(dirname $0)/../"
cd ${PROJECT_DIR_PATH}

PYTHONPATH=${PROJECT_DIR_PATH} \
python -u src/train_model.py "$@"
