# Three-Member Fruit Classification Experiments

This folder satisfies the group requirement that each of the three members develops a different image-classification algorithm and compares performance using the same fruit dataset.

## Shared problem and dataset

Problem: five-class fruit image classification for Apple, Banana, Grape, Mango, and Strawberry.

Dataset: Kaggle `utkarshsaxenadn/fruits-classification`.

All methods use the same official train/test class folders. The deep-learning methods additionally use the provided validation split for model selection.

## Member 1 — MobileNetV3-Small Transfer Learning

Implementation already exists in the project root:

- `model.py`
- `train.py`
- `verifier.py`
- `artifacts/metrics.json`

Method characteristics:

- pretrained MobileNetV3-Small visual backbone
- transfer learning / fine-tuning
- 224×224 input
- ImageNet normalization
- random crop, flip, rotation and colour augmentation
- centroid-based verification layer used by the final FruitScan Kids application

Current saved test result: 92% accuracy.

## Member 2 — Custom CNN from Scratch

Implementation: `member2_custom_cnn.py`

Architecture:

`Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → Adaptive Pool → Dense → 5 classes`

This model is intentionally trained from scratch so the group can compare a basic CNN against transfer learning.

Run:

```bash
python experiments/member2_custom_cnn.py --data "PATH_TO_DATASET" --epochs 8
```

Output:

`experiments/results/custom_cnn.json`

## Member 3 — HOG + Linear SVM

Implementation: `member3_hog_svm.py`

Pipeline:

`Resize → HOG handcrafted features → Linear SVM → 5 classes`

This provides a traditional machine-learning baseline that is meaningfully different from the two CNN-based approaches.

Run:

```bash
python experiments/member3_hog_svm.py --data "PATH_TO_DATASET"
```

Output:

`experiments/results/hog_svm.json`

## Comparison metrics

All methods are compared using:

- test accuracy
- macro precision
- macro recall
- macro F1-score
- per-class precision, recall and F1-score
- confusion matrix

After the Member 2 and Member 3 result files exist, run:

```bash
python experiments/compare_results.py
```

This creates:

- `experiments/results/model_comparison.csv`
- `experiments/results/model_comparison.json`
- `experiments/results/COMPARISON.md`

## Fair comparison rule

Do not compare results from different test datasets. All three algorithms must be evaluated on the same provided `test` split containing the same five classes.

The final Streamlit application does not expose all three algorithms to children. The experiments are an academic comparison; the deployed application continues to use the selected MobileNetV3-Small model and its existing verification gateway.
