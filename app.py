from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from verifier import FruitVerifier

CHECKPOINT = Path('artifacts/fruit_verifier.pt')
SUMMARY = Path('artifacts/dataset_summary.json')

st.set_page_config(page_title='FruitScan AI', page_icon='🍓', layout='wide')
st.title('🍓 FruitScan AI')
st.caption('Classify an uploaded fruit, then independently verify it against the learned dataset profile.')

@st.cache_resource
def load_verifier():
    return FruitVerifier(CHECKPOINT)

if not CHECKPOINT.exists():
    st.error('The trained verifier is not available yet.')
    st.code('python prepare_dataset.py --zip "archive (1)(1).zip"\npython train.py --data data --epochs 6\nstreamlit run app.py')
    st.stop()

verifier = load_verifier()
with st.sidebar:
    st.header('System status')
    st.success('Classifier ready')
    st.success('Dataset verifier ready')
    meta = verifier.metadata
    st.write(f"**Classes:** {', '.join(verifier.class_names)}")
    st.write(f"**Test accuracy:** {float(meta.get('test_accuracy', 0))*100:.2f}%")
    st.write(f"**Dataset images:** {meta.get('dataset_images', 'N/A')}")
    st.write(f"**Confidence threshold:** {verifier.confidence_threshold*100:.1f}%")
    if SUMMARY.exists():
        summary = json.loads(SUMMARY.read_text(encoding='utf-8'))
        with st.expander('Dataset scan summary'):
            st.write(f"Total scanned: **{summary.get('total_images', 0)}**")
            rows = []
            for split, counts in summary.get('splits', {}).items():
                for fruit, count in counts.items():
                    rows.append({'Split': split, 'Fruit': fruit, 'Images': count})
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

uploaded = st.file_uploader('Upload a fruit image', type=['jpg', 'jpeg', 'png', 'webp'])
if uploaded is None:
    st.info('Supported classes: Apple, Banana, Grape, Mango, Strawberry.')
    st.stop()

try:
    image = Image.open(uploaded).convert('RGB')
except Exception:
    st.error('Could not read this file as an image.')
    st.stop()

result = verifier.predict(image)
left, right = st.columns([1, 1.25])
with left:
    st.image(image, caption='Uploaded image', use_container_width=True)
with right:
    if result.verified:
        st.success(f'✅ VERIFIED: {result.label}')
        st.subheader(result.label)
    else:
        st.warning('⚠️ UNKNOWN / NOT VERIFIED')
        st.subheader('The system rejected this image')
    c1, c2, c3 = st.columns(3)
    c1.metric('Classifier confidence', f'{result.confidence*100:.2f}%')
    c2.metric('Dataset similarity', f'{result.dataset_similarity*100:.2f}%')
    c3.metric('Verification score', f'{result.verification_score:.1f}%')
    st.metric('Class separation margin', f'{result.class_margin*100:.2f}%')
    if result.reasons:
        st.write('**Why it was rejected:**')
        for reason in result.reasons:
            st.write(f'- {reason}')

st.divider()
st.subheader('All classifier probabilities')
prob_df = pd.DataFrame([{'Fruit': name, 'Probability': value} for name, value in result.probabilities.items()]).sort_values('Probability', ascending=False)
prob_df['Probability %'] = prob_df['Probability'].map(lambda x: f'{x*100:.2f}%')
st.dataframe(prob_df[['Fruit', 'Probability %']], hide_index=True, use_container_width=True)
st.bar_chart(prob_df.set_index('Fruit')['Probability'])

with st.expander('How the verification works'):
    st.markdown('''
The app does **not** trust a single classification score. It uses a second dataset-reference check:

1. MobileNetV3-Small predicts one of the five fruit classes.
2. The upload is converted into a feature embedding.
3. That embedding is compared with class centroids built by scanning the training dataset.
4. The image must pass a calibrated confidence threshold.
5. It must be sufficiently similar to the predicted fruit's dataset centroid.
6. It must also be clearly separated from the other fruit centroids.

If any gate fails, the app returns **Unknown / Not Verified** instead of forcing a fruit label.
''')
