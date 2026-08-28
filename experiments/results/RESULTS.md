# Three-Algorithm Fruit Classification Results

All methods use the same dataset classes and the provided train/validation/test folder split.

## Overall comparison

| Rank | Algorithm | Accuracy | Macro Precision | Macro Recall | Macro F1 | Training Time |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MobileNetV3-Small Transfer Learning | 92.00% | 91.97% | 92.00% | 91.92% | Not recorded |
| 2 | Custom CNN from Scratch | 63.00% | 68.06% | 63.00% | 63.73% | 579.6 s |
| 3 | HOG + Linear SVM | 37.00% | 37.25% | 37.00% | 36.61% | 118.4 s |

## MobileNetV3-Small Transfer Learning

### Per-class metrics

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Apple | 89.47% | 85.00% | 87.18% | 20 |
| Banana | 90.91% | 100.00% | 95.24% | 20 |
| Grape | 95.00% | 95.00% | 95.00% | 20 |
| Mango | 89.47% | 85.00% | 87.18% | 20 |
| Strawberry | 95.00% | 95.00% | 95.00% | 20 |

### Confusion matrix

Rows are actual classes and columns are predicted classes.

| Actual \ Predicted | Apple | Banana | Grape | Mango | Strawberry |
|---|---:|---:|---:|---:|---:|
| Apple | 17 | 0 | 0 | 2 | 1 |
| Banana | 0 | 20 | 0 | 0 | 0 |
| Grape | 0 | 1 | 19 | 0 | 0 |
| Mango | 2 | 1 | 0 | 17 | 0 |
| Strawberry | 0 | 0 | 1 | 0 | 19 |

## Custom CNN from Scratch

### Per-class metrics

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Apple | 50.00% | 65.00% | 56.52% | 20 |
| Banana | 59.09% | 65.00% | 61.90% | 20 |
| Grape | 85.71% | 60.00% | 70.59% | 20 |
| Mango | 53.85% | 70.00% | 60.87% | 20 |
| Strawberry | 91.67% | 55.00% | 68.75% | 20 |

### Confusion matrix

Rows are actual classes and columns are predicted classes.

| Actual \ Predicted | Apple | Banana | Grape | Mango | Strawberry |
|---|---:|---:|---:|---:|---:|
| Apple | 13 | 2 | 0 | 4 | 1 |
| Banana | 1 | 13 | 0 | 6 | 0 |
| Grape | 3 | 3 | 12 | 2 | 0 |
| Mango | 1 | 4 | 1 | 14 | 0 |
| Strawberry | 8 | 0 | 1 | 0 | 11 |

## HOG + Linear SVM

### Per-class metrics

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Apple | 35.29% | 30.00% | 32.43% | 20 |
| Banana | 37.50% | 30.00% | 33.33% | 20 |
| Grape | 39.29% | 55.00% | 45.83% | 20 |
| Mango | 30.43% | 35.00% | 32.56% | 20 |
| Strawberry | 43.75% | 35.00% | 38.89% | 20 |

### Confusion matrix

Rows are actual classes and columns are predicted classes.

| Actual \ Predicted | Apple | Banana | Grape | Mango | Strawberry |
|---|---:|---:|---:|---:|---:|
| Apple | 6 | 5 | 2 | 3 | 4 |
| Banana | 2 | 6 | 3 | 8 | 1 |
| Grape | 1 | 3 | 11 | 3 | 2 |
| Mango | 4 | 2 | 5 | 7 | 2 |
| Strawberry | 4 | 0 | 7 | 2 | 7 |

## Method characteristics for discussion

| Method | Main advantages | Main disadvantages / characteristics |
|---|---|---|
| MobileNetV3-Small Transfer Learning | Lightweight pretrained visual features; strong deployment suitability; usually learns well with limited task-specific training | Depends on pretrained representations; more complex than traditional ML |
| Custom CNN from Scratch | Learns task-specific features; architecture is easy to explain and modify | Starts with no pretrained knowledge; may need more training and may overfit |
| HOG + Linear SVM | Traditional ML baseline; interpretable feature pipeline; comparatively simple classifier | Relies on handcrafted HOG features and may struggle with colour/texture or complex visual variation |

> Select the final model using the measured results together with deployment suitability. Do not claim a model is best until all experiments have completed.
