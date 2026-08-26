# FruitScan Kids — Early-Years Fruit Learning with AI

FruitScan Kids is an educational Streamlit project for early-years learners. Children can **upload a fruit picture or use the live camera**, let the AI recognise the fruit, then continue into a simple lesson about pronunciation, seeds, plant growth, weather, soil, water, sunlight, photosynthesis, and easy fruit food activities.

Supported fruits:

- 🍎 Apple
- 🍌 Banana
- 🍇 Grape
- 🥭 Mango
- 🍓 Strawberry

Dataset: https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification

## Educational goal

The project turns fruit recognition into a learning journey:

1. **See it** — take or upload a fruit picture.
2. **Name it** — AI identifies and verifies the fruit.
3. **Say it** — use the browser pronunciation button and repeat the word.
4. **Explore it** — learn simple fruit facts and what the seed looks like.
5. **Grow it** — learn about the plant, weather, soil, water, sunlight, flowers, and fruit development.
6. **Understand leaves** — learn a simple early-years explanation of photosynthesis.
7. **Make something** — follow a short adult-supervised snack, drink, or dessert activity.

The app also includes a separate **🎓 Learn Fruits** tab, so children can browse lessons without scanning a picture first.

## Child-facing Streamlit sections

### 📷 Scan & Learn

Choose either:

- **📁 Upload Image**
- **🎥 Live Front Camera**

When a fruit is verified, the app automatically opens its learning lesson.

### 🎓 Learn Fruits

Select any supported fruit and explore:

- 🔊 pronunciation
- 🌟 general information
- 🌰 seed appearance
- 🌱 plant type and growth stages
- 🌤️ weather
- 🪴 soil / dirt
- 💧 water
- ☀️ sunlight
- 🌿 photosynthesis
- 🥪 simple food activity
- 🥤 simple drink activity
- 🍨 simple dessert activity
- 🧑‍🧑‍🧒 adult safety note

### 🛠️ Model Setup

This section is intended for teachers, parents, or project setup rather than children.

## AI verification design

A normal five-class classifier always chooses one of its known labels. FruitScan Kids keeps the existing FruitScan verification system so uncertain pictures can be rejected instead of teaching the wrong fruit.

Before a result is accepted, the system checks:

1. **Classifier confidence**
2. **Dataset similarity**
3. **Class separation**

If the checks fail, the child-facing interface simply says that it is not sure and asks for another picture. Technical values remain available inside a teacher/model-details expander.

## Dataset scan

The supplied Kaggle ZIP was verified successfully:

| Split | Apple | Banana | Grape | Mango | Strawberry | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1,940 | 1,940 | 1,940 | 1,940 | 1,940 | 9,700 |
| Validation | 40 | 40 | 40 | 40 | 40 | 200 |
| Test | 20 | 20 | 20 | 20 | 20 | 100 |
| **Total** | **2,000** | **2,000** | **2,000** | **2,000** | **2,000** | **10,000** |

All **10,000 images were readable and 0 corrupt images were found** during the integrity scan.

## Project files

```text
lastplan/
├── app.py                 # Main Streamlit application
├── education_ui.py        # Child-friendly learning interface
├── fruit_education.py     # Fruit facts, growing information, recipes and safety notes
├── prepare_dataset.py     # Extract ZIP + verify dataset images
├── dataset_utils.py       # Dataset discovery and scan helpers
├── model.py               # MobileNetV3-Small classifier
├── train.py               # Training + verifier calibration
├── verifier.py            # Confidence/similarity/margin verification
├── predict.py             # Command-line prediction
├── requirements.txt
├── .streamlit/config.toml
├── .gitignore
├── data/                  # generated locally
└── artifacts/             # model + metrics
```

## Run locally

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Use:

```text
Repository: alistair832/lastplan
Branch: main
Main file: app.py
```

Then deploy the app from Streamlit Community Cloud.

## Safety for early-years food activities

The kitchen activities are educational examples for **adult-supervised use**. Adults should manage knives, blenders, heat, food allergies, and age-appropriate choking safety. Whole grapes must not be served to young children; the app specifically reminds adults to cut grapes lengthwise into quarters.

## Important AI limitation

FruitScan Kids is a learning aid, not a perfect identification system. It currently understands five fruit classes. The verification layer reduces forced misclassification, but uncertain or unfamiliar images should still be tried again with a clearer picture.
