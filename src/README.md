# SIGIR ECom Task 2018


## Build taxonomy and pickle it

```
./src/scripts/create_taxonomy.sh \
--input-file-path ${DATA_FILE_PATH} \
--taxonomy-file-path ${TAXONOMY_FILE_PATH}
```


## Train classifier model

```
./src/scripts/train_classifier.sh \
--train-file-path ${TRAIN_FILE_PATH} \
--test-file-path ${TEST_FILE_PATH} \
--model-save-path ${MODEL_SAVE_PATH} \
--vectorizer-save-path ${VECTORIZER_SAVE_PATH}
```


## Test classifier model

```
./src/scripts/test_classifier.sh \
--model-save-path ${MODEL_SAVE_PATH} \
--vectorizer-save-path ${VECTORIZER_SAVE_PATH} \
--predictions-save-path ${PREDICTIONS_SAVE_PATH}
```
