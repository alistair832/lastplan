# Three-Member Algorithm Comparison

This folder exists to satisfy the group requirement that each member develops a different image-classification algorithm and that the results are compared using the same fruit dataset.

## Member / algorithm structure

| Member | Algorithm | Category | Main file |
|---|---|---|---|
| Member 1 | MobileNetV3-Small Transfer Learning | Deep learning / pretrained CNN | `../train.py` + `../model.py` |
| Member 2 | Custom CNN from Scratch | Deep learning / CNN | `custom_cnn.py` |
| Member 3 | HOG + Linear SVM | Traditional machine learning | `hog_svm.py` |

Replace `Member 1`, `Member 2`, and `Member 3` with the actual group member names in the report or presentation. The repository structure does not claim who personally wrote which code; the group should assign and explain each member's experiment honestly.

## Common dataset

All three methods use the same Kaggle Fruits Classification dataset and the same supplied folder split:

- `train`: 9,700 images
- `valid`: 200 images
- `test`: 100 images
- classes: Apple, Banana, Grape, Mango, Strawberry

This keeps the test set consistent for the final comparison.

## Method 1 — MobileNetV3-Small Transfer Learning

The deployed FruitScan model is the existing MobileNetV3-Small transfer-learning system. It uses 224×224 RGB images, ImageNet normalization, augmentation during training, and a centroid-based verification layer for the deployed Unknown Gateway.

The current recorded test results are stored in `../artifacts/metrics.json`.

## Method 2 — Custom CNN from Scratch

`custom_cnn.py` trains a small convolutional neural network without pretrained weights. It contains four convolutional feature blocks followed by a compact fully connected classifier.

Pre-processing:

- resize/crop to 128×128 RGB
- ImageNet-style normalization
- random crop
- horizontal flip
- random rotation
- colour jitter

The validation split is used for early stopping, while the test split is kept for final evaluation.

Run:

```bash
python -m experiments.custom_cnn --data "PATH_TO_DATASET"
```

## Method 3 — HOG + Linear SVM

`hog_svm.py` is a traditional machine-learning baseline. Images are resized to 96×96, converted to grayscale, transformed into Histogram of Oriented Gradients (HOG) feature vectors, and classified with a Linear Support Vector Machine.

The validation split selects the SVM `C` value from the configured candidates. The selected model is then fitted using train + validation data and evaluated once on the test set.

Run:

```bash
python -m experiments.hog_svm --data "PATH_TO_DATASET"
```

## Evaluation metrics

Both new algorithms automatically produce:

- accuracy
- macro precision
- macro recall
- macro F1-score
- weighted precision / recall / F1-score
- per-class classification report
- confusion matrix
- training time for the new experiment

Then run:

```bash
python -m experiments.compare_results
```

This generates:

```text
experiments/results/
├── custom_cnn.json
├── hog_svm.json
├── model_comparison.csv
├── model_comparison.json
└── RESULTS.md
```

`RESULTS.md` is designed to make the final report comparison easier. It includes the overall table, per-class metrics, confusion matrices, and a concise method-characteristics table.

## Fair-comparison note

The algorithms deliberately use the same class labels and the same train/validation/test folder split. Input representation differs where appropriate to the algorithm: MobileNet uses 224×224 RGB, the custom CNN uses 128×128 RGB for efficient from-scratch training, and HOG + SVM uses 96×96 grayscale HOG descriptors. These representation differences should be stated in the methodology section rather than hidden.

## Final deployed system

The child-facing Streamlit application should continue to use only the selected final model. The three algorithms are experimental comparisons for the assignment; children do not need to select or see them in the app.
