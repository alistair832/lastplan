# Fruit Classification Project

A complete image-classification project built around the Kaggle **Fruits Classification** dataset.

Dataset: https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification

## Dataset used

The provided dataset contains **10,000 fruit images** in five classes:

- Apple
- Banana
- Grape
- Mango
- Strawberry

The downloaded ZIP is already split into:

| Split | Images | Per class |
|---|---:|---:|
| Train | 9,700 | 1,940 |
| Validation | 200 | 40 |
| Test | 100 | 20 |

The dataset itself is intentionally not committed to GitHub. Download it from Kaggle and keep it locally.

## What this project includes

- `prepare_dataset.py` — extracts the already-downloaded Kaggle ZIP and finds the dataset root automatically.
- `dataset_utils.py` — validates the dataset structure and counts images.
- `train.py` — trains a MobileNetV2 transfer-learning classifier.
- `predict.py` — predicts one local image from the command line.
- `app.py` — Streamlit web interface for uploading and classifying fruit images.
- `requirements.txt` — Python dependencies.

## 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Extract the ZIP you already downloaded

Example:

```bash
python prepare_dataset.py --zip "archive (1)(1).zip"
```

By default the files are extracted under `data/`. The script automatically finds the nested `Fruits Classification` directory and confirms the train/valid/test counts.

If your dataset is already extracted, you can skip this step.

## 3. Train the model

```bash
python train.py --data data --epochs 8
```

The best model is saved as:

```text
models/fruit_classifier.keras
```

The detected class order is saved as:

```text
models/class_names.json
```

## 4. Predict one image

```bash
python predict.py "path/to/fruit-image.jpg"
```

## 5. Run the web application

```bash
streamlit run app.py
```

Open the local Streamlit address shown in the terminal, upload a fruit image, and the app will display the predicted class and confidence scores.

## Model design

The project uses **MobileNetV2 transfer learning** with ImageNet weights. The pretrained feature extractor is frozen while a new classification head learns the five fruit classes. Basic image augmentation is applied during training to improve generalization.

## Project structure

```text
lastplan/
├── app.py
├── dataset_utils.py
├── prepare_dataset.py
├── predict.py
├── train.py
├── requirements.txt
├── .gitignore
├── data/                  # local only; ignored by Git
│   └── Fruits Classification/
│       ├── train/
│       ├── valid/
│       └── test/
└── models/
    ├── fruit_classifier.keras   # generated after training
    └── class_names.json
```

## Notes

- Do not upload the 10,000 dataset images to this repository.
- Training requires TensorFlow and may be much faster with a supported GPU.
- Run `train.py` before using `predict.py` or `app.py`.
