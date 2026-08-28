# FruitScan Kids — Early-Years Fruit Learning with AI

FruitScan Kids is a child-friendly Streamlit learning app. A learner can take a fruit photo or choose a picture, think and predict before the AI reveals an accepted result, then continue into visual learning activities, recipes, quizzes, and thinking games.

Supported fruits:

- 🍎 Apple
- 🍌 Banana
- 🍇 Grape
- 🥭 Mango
- 🍓 Strawberry

Dataset: https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification

## Child learning flow

1. **📷 Scan & Think** — take or choose a clear fruit picture.
2. **🤔 Predict** — make a guess and explain a clue before the answer is revealed.
3. **❓ Unknown Gateway** — uncertain or out-of-profile images are rejected and the learner is asked to retry.
4. **🌟 Activities** — choose one large activity card at a time.
5. **🧠 Think & Discover** — observe, predict, compare, sequence, and reason.
6. **👩‍🍳 Fruit Kitchen** — explore food, drink, and dessert ideas with adult help.
7. **🎮 Quiz & Games** — use picture quizzes, thinking games, and adaptive challenge levels.
8. **🎓 Learn Fruits** — browse lessons and the fruit collection.
9. **⚙️ Adult** — view a simple parent / teacher dashboard and project information.

## AI verification design

The recognition model is a MobileNetV3-Small five-class classifier with a conservative verification layer. Before a fruit result is accepted, FruitScan checks model confidence, dataset-centroid similarity, class separation, top-two probability separation, classifier/centroid agreement, and basic image quality.

If the image does not pass the gateway, the child-facing app shows **Unknown / Not Recognised** and asks the learner to retake the photo or choose another picture.

The trained model is loaded automatically from:

```text
artifacts/fruit_classifier.pt
```

Training controls are intentionally not shown in the deployed child-facing interface.

## Dataset and model results

The project dataset contains 10,000 images across five balanced fruit classes:

| Split | Apple | Banana | Grape | Mango | Strawberry | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1,940 | 1,940 | 1,940 | 1,940 | 1,940 | 9,700 |
| Validation | 40 | 40 | 40 | 40 | 40 | 200 |
| Test | 20 | 20 | 20 | 20 | 20 | 100 |
| **Total** | **2,000** | **2,000** | **2,000** | **2,000** | **2,000** | **10,000** |

Current saved MobileNetV3 metrics are available in `artifacts/metrics.json`.

## Three-member algorithm comparison

The assignment comparison is separated from the child-facing app. The group uses three different approaches:

| Member | Method | Type |
|---|---|---|
| Member 1 | MobileNetV3-Small Transfer Learning | Pretrained CNN / deep learning |
| Member 2 | Custom CNN from Scratch | CNN / deep learning |
| Member 3 | HOG + Linear SVM | Traditional machine learning |

The experiment code is stored under `experiments/`. All three methods use the same five classes and the same provided train/validation/test folder split. The automated benchmark creates accuracy, macro precision, macro recall, macro F1-score, per-class metrics, confusion matrices, and a final comparison table.

The child-facing Streamlit application continues to use only the selected deployed model; children are not asked to choose between algorithms.

## Project structure

```text
lastplan/
├── app.py                     # Main Streamlit application
├── about_ui.py                # About page
├── activity_cards.py          # Large child-friendly activity navigation
├── adaptive_learning.py       # Six-level adaptive learning challenges
├── adult_dashboard.py         # Parent / teacher session summary
├── camera_guidance.py         # Child-friendly photo guidance
├── camera_history.py          # Temporary session camera history
├── child_experience.py        # Rewards, collection, pronunciation, visual stories, recipes
├── child_quiz.py              # Picture quiz with progress tracking
├── education_ui.py            # Browseable fruit learning interface
├── fruit_education.py         # Fruit facts, growth, recipes, and safety information
├── fruit_learning_extras.py   # Extra child-learning facts and quiz content
├── progress_store.py          # Current-session learning progress
├── thinking_mode.py           # Prediction and higher-order thinking activities
├── model.py                   # MobileNetV3-Small model definition
├── verifier.py                # Unknown/verification gateway
├── train.py                   # Member 1 / developer MobileNet training
├── dataset_utils.py           # Shared dataset helpers
├── predict.py                 # Optional command-line model diagnostic
├── experiments/
│   ├── README.md              # Three-member methodology
│   ├── common.py              # Shared evaluation utilities
│   ├── custom_cnn.py          # Member 2 experiment
│   ├── hog_svm.py             # Member 3 experiment
│   ├── compare_results.py     # Builds final comparison tables
│   └── results/               # Generated benchmark outputs
├── requirements.txt
├── .streamlit/config.toml
├── .github/workflows/train-fruit-model.yml
├── .github/workflows/compare-fruit-algorithms.yml
└── artifacts/
    ├── fruit_classifier.pt    # Trained checkpoint used by the app
    ├── metrics.json           # Saved MobileNetV3 evaluation metrics
    └── dataset_summary.json   # Saved dataset summary
```

## Developer retraining

Retraining is kept outside the child-facing interface. The GitHub Actions workflow can download the public Kaggle dataset and run `train.py`, or a developer can train locally with the expected `train`, `valid`, and `test` folder structure.

The training pipeline depends on:

```text
train.py
model.py
dataset_utils.py
```

The generated checkpoint is then loaded by `verifier.py` and the Streamlit app.

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

```text
Repository: alistair832/lastplan
Branch: main
Main file: app.py
```

## Safety

Fruit Kitchen activities are educational examples for adult-supervised use. Adults should manage knives, blenders, heat, food allergies, and age-appropriate choking safety.

## AI limitation

FruitScan Kids currently understands five trained fruit classes. The Unknown Gateway reduces forced misclassification but cannot guarantee perfect rejection of every unseen object or fruit. Clear, centered photos with good lighting give the best results.
