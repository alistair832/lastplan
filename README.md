# FruitScan AI — Fruit Classification + Dataset Verification

FruitScan AI classifies uploaded images as **Apple, Banana, Grape, Mango, or Strawberry** and then performs an independent dataset-reference check before marking the result as verified.

Dataset: https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification

## Why this version is better

A normal five-class classifier always chooses one of its five labels, even when someone uploads an unrelated picture. FruitScan AI therefore uses three verification gates:

1. **Classifier confidence** — MobileNetV3-Small must be confident enough.
2. **Dataset similarity** — the uploaded image embedding must resemble the learned profile of the predicted fruit.
3. **Class separation** — the predicted fruit must be clearly more similar than the other fruit classes.

If any gate fails, Streamlit displays **UNKNOWN / NOT VERIFIED** instead of forcing an incorrect fruit label.

## Dataset verified from the supplied ZIP

The supplied Kaggle ZIP was scanned successfully:

| Split | Apple | Banana | Grape | Mango | Strawberry | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1,940 | 1,940 | 1,940 | 1,940 | 1,940 | 9,700 |
| Validation | 40 | 40 | 40 | 40 | 40 | 200 |
| Test | 20 | 20 | 20 | 20 | 20 | 100 |
| **Total** | **2,000** | **2,000** | **2,000** | **2,000** | **2,000** | **10,000** |

All **10,000 images were readable and 0 corrupt images were found** during the full integrity scan.

## Project files

```text
lastplan/
├── app.py                 # Streamlit upload and result UI
├── prepare_dataset.py     # Extract ZIP + verify dataset images
├── dataset_utils.py       # Dataset discovery and scan helpers
├── model.py               # MobileNetV3-Small classifier
├── train.py               # Training + verification calibration
├── verifier.py            # Confidence/similarity/margin verification
├── predict.py             # Command-line verification
├── requirements.txt
├── .gitignore
├── data/                  # generated locally, not committed
└── artifacts/             # generated training outputs
```

## 1. Install

Python 3.11+ is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Extract and scan the ZIP you already downloaded

Put the Kaggle ZIP in the project folder and run:

```bash
python prepare_dataset.py --zip "archive (1)(1).zip"
```

The script automatically locates the nested dataset, copies the train/valid/test folders into `data/`, checks the images, and writes a dataset summary to `artifacts/dataset_summary.json`.

## 3. Train the classifier and verifier

```bash
python train.py --data data --epochs 12
```

The training process uses MobileNetV3-Small transfer learning. It first trains the classification head and then fine-tunes the last feature blocks. After training, it scans the learned embeddings and calibrates verification thresholds using the validation set.

The main generated file is:

```text
artifacts/fruit_classifier.pt
```

This single checkpoint stores:

- trained classifier weights
- class names
- dataset centroid embeddings
- per-class similarity thresholds
- classifier confidence threshold
- class-separation margin threshold
- validation/test metrics

## 4. Run Streamlit

```bash
streamlit run app.py
```

Upload a JPG, PNG, or WebP image. The page displays:

- **VERIFIED fruit** or **UNKNOWN / NOT VERIFIED**
- classifier confidence
- dataset similarity
- verification score
- class-separation margin
- reason for rejection when verification fails
- probabilities for all five fruits
- dataset scan summary and test accuracy

## 5. Verify one image from the terminal

```bash
python predict.py "path/to/fruit.jpg"
```

## Verification logic

```text
Uploaded Image
      |
      v
MobileNetV3 Classifier
      |
      +---- Confidence threshold ---- fail ---> UNKNOWN
      |
      v
Feature Embedding
      |
      v
Compare with 5 dataset centroids
      |
      +---- Similarity threshold ---- fail ---> UNKNOWN
      |
      +---- Separation margin ------- fail ---> UNKNOWN
      |
      v
VERIFIED FRUIT
```

This approach is more reliable than simply displaying the highest softmax prediction because uncertain and out-of-distribution images can be rejected.

## Important limitation

The verifier substantially reduces forced misclassification, but it cannot mathematically prove that every possible uploaded image is or is not a fruit. It is still a machine-learning system trained around five known classes. The confidence, similarity, and class-margin gates provide conservative **unknown-image rejection**.

## GitHub note

The 10,000 dataset images and downloaded ZIP are intentionally excluded from GitHub. They remain local. The trained `.pt` file is also ignored by default because model artifacts may be large. For deployment, train locally and then either store the checkpoint in an approved model-storage location or remove the model ignore rule and commit the checkpoint if its size fits your repository policy.
