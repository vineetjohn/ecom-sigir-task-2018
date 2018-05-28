# SIGIR ECom Task 2018


## Build taxonomy and pickle it

```
./scripts/create_taxonomy.sh \
--input-file-path ${DATA_FILE_PATH} \
--taxonomy-file-path ${TAXONOMY_FILE_PATH}
```


## Train classifier model

```
./scripts/train_classifier.sh \
--train-file-path ${TRAIN_FILE_PATH} \
--model-save-path ${MODEL_SAVE_PATH} \
--taxonomy-file-path ${TAXONOMY_FILE_PATH} \
--vectorizer-save-path ${VECTORIZER_SAVE_PATH}
```


## Test classifier model

```
./scripts/test_classifier.sh \
--model-save-path ${MODEL_SAVE_PATH} \
--taxonomy-file-path ${TAXONOMY_FILE_PATH} \
--test-file-path ${TEST_FILE_PATH} \
--vectorizer-save-path ${VECTORIZER_SAVE_PATH} \
--predictions-save-path ${PREDICTIONS_SAVE_PATH}
```
