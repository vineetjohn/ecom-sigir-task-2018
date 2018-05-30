# SIGIR ECom Task 2018


## Train classifier model

```bash
./src_neural/scripts/train_classifier.sh \
--train-file-path ${TRAIN_FILE_PATH} \
--model-save-path ${MODEL_SAVE_PATH} \
--num_epochs ${NUM_EPOCHS}
```

## Test classifier model
```bash
./src_neural/scripts/test_classifier.sh \
--model-save-path ${MODEL_SAVE_PATH} \
--test-file-path ${TEST_FILE_PATH} \
--predictions-save-path ${PREDICTIONS_FILE_PATH}
```
