from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from dataset_utils import find_dataset_root, scan_dataset
from model_utils import build_model, cosine_scores, extract_embedding, image_to_tensor

MODEL_DIR = Path('models')
MODEL_PATH = MODEL_DIR / 'fruit_classifier.pt'
REFERENCE_PATH = MODEL_DIR / 'reference_index.npz'
METRICS_PATH = MODEL_DIR / 'metrics.json'

st.set_page_config(page_title='FruitScan AI', page_icon='🍓', layout='wide')
st.title('🍓 FruitScan AI')
st.caption('Fruit classification with independent dataset-reference verification')


@st.cache_resource
def load_assets():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    names = saved['class_names']
    model = build_model(len(names), freeze_backbone=True).to(device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    ref = np.load(REFERENCE_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding='utf-8'))
    return device, model, names, ref['prototypes'], ref['thresholds'], metrics


required = [MODEL_PATH, REFERENCE_PATH, METRICS_PATH]
if not all(path.exists() for path in required):
    st.error('Trained verification assets are missing.')
    st.code('python prepare_dataset.py --zip "archive.zip"\npython train.py --data data --epochs 6\nstreamlit run app.py')
    st.info('The ZIP and extracted dataset are intentionally not stored in GitHub. Train once to create the small model/reference files.')
    st.stop()

device, model, class_names, prototypes, similarity_thresholds, metrics = load_assets()

with st.sidebar:
    st.header('System status')
    st.success('Classifier loaded')
    st.success('Dataset reference index loaded')
    st.write(f"**Classes:** {', '.join(class_names)}")
    st.write(f"**Test accuracy:** {metrics.get('test_accuracy', 0)*100:.2f}%")
    st.write(f"**Images scanned during training:** {metrics.get('dataset_images', 'N/A')}")
    st.write(f"**Confidence gate:** {metrics.get('confidence_threshold', 0.65)*100:.1f}%")

    if Path('data').exists():
        try:
            root = find_dataset_root('data')
            if st.button('Re-scan local dataset'):
                with st.spinner('Checking every dataset image...'):
                    report = scan_dataset(root, verify_images=True)
                if report['valid']:
                    st.success(f"Dataset verified: {report['total_images']} images, 0 corrupt")
                else:
                    st.error(f"Found {len(report['corrupt_images'])} corrupt images")
        except Exception:
            pass

uploaded = st.file_uploader('Upload a fruit image', type=['jpg', 'jpeg', 'png', 'webp'])
if uploaded is None:
    st.info('Supported trained classes: Apple, Banana, Grape, Mango, and Strawberry. Upload an image to begin.')
    st.stop()

try:
    image = Image.open(uploaded).convert('RGB')
except Exception:
    st.error('The uploaded file could not be read as an image.')
    st.stop()

left, right = st.columns([1, 1.25])
with left:
    st.image(image, caption='Uploaded image', use_container_width=True)

with torch.no_grad():
    tensor = image_to_tensor(image, device)
    probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    embedding = extract_embedding(model, tensor)[0].cpu().numpy()

similarities = cosine_scores(embedding, prototypes)
pred_idx = int(probabilities.argmax())
ref_idx = int(similarities.argmax())
pred_name = class_names[pred_idx]
confidence = float(probabilities[pred_idx])
similarity = float(similarities[pred_idx])
confidence_threshold = float(metrics.get('confidence_threshold', 0.65))
similarity_threshold = float(similarity_thresholds[pred_idx])

agreement = pred_idx == ref_idx
confidence_ok = confidence >= confidence_threshold
similarity_ok = similarity >= similarity_threshold
verified = agreement and confidence_ok and similarity_ok

normalized_similarity = float(np.clip((similarity + 1.0) / 2.0, 0.0, 1.0))
verification_score = 0.70 * confidence + 0.30 * normalized_similarity

with right:
    if verified:
        st.success(f'✅ VERIFIED: {pred_name}')
        st.subheader(pred_name)
    else:
        st.warning('⚠️ NOT VERIFIED / UNSURE')
        st.subheader(f'Classifier guess: {pred_name}')
        st.write('The system will not claim a verified fruit unless the classifier and dataset-reference check both pass.')

    c1, c2, c3 = st.columns(3)
    c1.metric('Classifier confidence', f'{confidence*100:.2f}%')
    c2.metric('Dataset similarity', f'{similarity*100:.2f}%')
    c3.metric('Verification score', f'{verification_score*100:.2f}%')

    st.write(f'**Nearest dataset class:** {class_names[ref_idx]}')
    checks = pd.DataFrame({
        'Check': ['Classifier confidence', 'Dataset similarity', 'Classifier/reference agreement'],
        'Required': [f'≥ {confidence_threshold*100:.1f}%', f'≥ {similarity_threshold*100:.1f}%', 'Same class'],
        'Actual': [f'{confidence*100:.1f}%', f'{similarity*100:.1f}%', 'Yes' if agreement else 'No'],
        'Pass': ['✅' if confidence_ok else '❌', '✅' if similarity_ok else '❌', '✅' if agreement else '❌'],
    })
    st.dataframe(checks, use_container_width=True, hide_index=True)

st.divider()
st.subheader('All class results')
results = pd.DataFrame({
    'Fruit': class_names,
    'Classifier probability': probabilities,
    'Dataset similarity': similarities,
}).sort_values('Classifier probability', ascending=False)
results['Classifier %'] = results['Classifier probability'].map(lambda x: f'{x*100:.2f}%')
results['Similarity %'] = results['Dataset similarity'].map(lambda x: f'{x*100:.2f}%')
st.dataframe(results[['Fruit', 'Classifier %', 'Similarity %']], use_container_width=True, hide_index=True)
st.bar_chart(results.set_index('Fruit')['Classifier probability'])

with st.expander('How verification works'):
    st.markdown('''
1. **Classifier:** MobileNetV3 predicts one of the five trained fruit classes.
2. **Reference verifier:** the same image is converted to an embedding and compared with class prototypes built from the training dataset.
3. **Confidence gate:** low-confidence predictions are rejected.
4. **Similarity gate:** uploads that do not resemble the training distribution are rejected.
5. **Agreement gate:** the classifier and reference verifier must select the same fruit.

This reduces false confidence when someone uploads an unrelated object, although no closed-set classifier can guarantee perfect out-of-distribution detection.
''')
