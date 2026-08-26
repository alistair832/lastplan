from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from dataset_utils import find_dataset_root, save_class_names

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def load_split(path: Path, shuffle: bool) -> tf.data.Dataset:
    return keras.utils.image_dataset_from_directory(
        path,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED if shuffle else None,
    )


def build_model(num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))

    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)

    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="fruit_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 5-class fruit image classifier.")
    parser.add_argument("--data", default="data", help="Dataset directory or parent directory")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    root = find_dataset_root(args.data)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_ds = load_split(root / "train", shuffle=True)
    valid_ds = load_split(root / "valid", shuffle=False)
    test_ds = load_split(root / "test", shuffle=False)

    names = train_ds.class_names
    save_class_names(names, model_dir / "class_names.json")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    valid_ds = valid_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    model = build_model(len(names))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            model_dir / "fruit_classifier.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    model.fit(train_ds, validation_data=valid_ds, epochs=args.epochs, callbacks=callbacks)

    best_model = keras.models.load_model(model_dir / "fruit_classifier.keras")
    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Saved model: {(model_dir / 'fruit_classifier.keras').resolve()}")


if __name__ == "__main__":
    main()
