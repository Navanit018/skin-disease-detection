import os
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

print("TensorFlow:", tf.__version__)

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 10

PROJECT_DIR = Path("C:/Users/ACER/OneDrive/Desktop/Skin Disease Detection")

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
CLASS_TO_INDEX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

DISEASE_INFO = {
    'akiec': {'name': 'Actinic Keratoses', 'severity': 'Medium'},
    'bcc': {'name': 'Basal Cell Carcinoma', 'severity': 'Low-Medium'},
    'bkl': {'name': 'Benign Keratosis', 'severity': 'Very Low'},
    'df': {'name': 'Dermatofibroma', 'severity': 'Very Low'},
    'nv': {'name': 'Melanocytic Nevi (Moles)', 'severity': 'Very Low'},
    'vasc': {'name': 'Vascular Lesions', 'severity': 'Very Low'},
    'mel': {'name': 'Melanoma', 'severity': 'HIGH'}
}

MEDICINES = {
    'akiec': ['5-Fluorouracil cream', 'Imiquimod cream', 'Cryotherapy'],
    'bcc': ['Surgery', 'Mohs surgery', 'Radiation therapy'],
    'bkl': ['No treatment needed', 'Cryotherapy if cosmetic'],
    'df': ['No treatment needed', 'Surgical removal if symptomatic'],
    'nv': ['No treatment needed', 'Monitor for changes'],
    'vasc': ['Laser therapy', 'Corticosteroids'],
    'mel': ['Surgery', 'Immunotherapy', 'Chemotherapy']
}


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1.0), self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * tf.pow(1. - pt, self.gamma)
        loss = focal_weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha, 'from_logits': self.from_logits})
        return config


class MixupGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size, alpha=0.2, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.12,
            height_shift_range=0.12,
            shear_range=0.1,
            zoom_range=0.15,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.85, 1.15],
            fill_mode='reflect',
        )
        self.indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[batch_idx].copy()
        y_batch = self.y[batch_idx].copy()

        aug_iter = self.datagen.flow(X_batch, y_batch, batch_size=len(X_batch), shuffle=False)
        X_aug = next(aug_iter)[0]

        lam = np.random.beta(self.alpha, self.alpha)
        perm = np.random.permutation(len(X_batch))
        X_mix = lam * X_aug + (1.0 - lam) * X_aug[perm]
        y_mix = lam * y_batch + (1.0 - lam) * y_batch[perm]

        return X_mix, y_mix

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_data():
    print("Loading metadata...")
    df = pd.read_csv(PROJECT_DIR / "HAM10000_metadata.csv")
    print(f"Total samples: {len(df)}")

    images = []
    labels = []

    for idx, row in df.iterrows():
        image_id = row['image_id']
        dx = row['dx']
        if dx not in CLASS_NAMES:
            continue
        for folder in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
            img_path = PROJECT_DIR / folder / f"{image_id}.jpg"
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                    img_array = np.array(img, dtype=np.float32)
                    images.append(img_array)
                    labels.append(CLASS_TO_INDEX[dx])
                    break
                except Exception as e:
                    pass
        if idx % 2000 == 0:
            print(f"  Processed {idx} images...")

    print(f"Loaded {len(images)} images")
    return np.array(images), np.array(labels)


def build_model():
    base = EfficientNetV2S(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=out)
    return model, base


def train():
    tf.random.set_seed(42)
    np.random.seed(42)

    X, y = load_data()

    X = X / 127.5 - 1.0

    y_cat = to_categorical(y, num_classes=NUM_CLASSES)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )

    y_int_train = np.argmax(y_train, axis=1)
    class_weights_arr = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_int_train), y=y_int_train
    )
    class_weight_dict = dict(enumerate(class_weights_arr))
    print(f"\nClass weights: {class_weight_dict}")

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")

    model, base_model = build_model()

    model.summary()

    train_gen = MixupGenerator(X_train, y_train, BATCH_SIZE)

    total_steps = (len(X_train) // BATCH_SIZE) * EPOCHS_PHASE1
    warmup_steps = len(X_train) // BATCH_SIZE * 1

    lr_schedule1 = CosineDecay(
        initial_learning_rate=0.0005,
        decay_steps=total_steps - warmup_steps,
        warmup_target=0.0005,
        warmup_steps=warmup_steps,
        alpha=1e-4
    )

    model.compile(
        optimizer=AdamW(learning_rate=lr_schedule1, weight_decay=1e-4, global_clipnorm=1.0),
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )

    callbacks1 = [
        EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
        ModelCheckpoint(PROJECT_DIR / 'best_model_advanced.keras', monitor='val_accuracy', save_best_only=True)
    ]

    print("\n=== Phase 1: Training Classifier Head ===")
    model.fit(
        train_gen,
        epochs=EPOCHS_PHASE1,
        validation_data=(X_val, y_val),
        callbacks=callbacks1,
        class_weight=class_weight_dict,
        verbose=1
    )

    print("\n--- Unfreezing for Phase 2 ---")
    base_model.trainable = True
    for layer in base_model.layers[:150]:
        layer.trainable = False

    total_steps2 = (len(X_train) // BATCH_SIZE) * EPOCHS_PHASE2

    lr_schedule2 = CosineDecay(
        initial_learning_rate=5e-5,
        decay_steps=total_steps2,
        warmup_target=5e-5,
        warmup_steps=len(X_train) // BATCH_SIZE,
        alpha=1e-6
    )

    model.compile(
        optimizer=AdamW(learning_rate=lr_schedule2, weight_decay=1e-5, global_clipnorm=1.0),
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )

    train_gen2 = MixupGenerator(X_train, y_train, BATCH_SIZE)

    callbacks2 = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7),
        ModelCheckpoint(PROJECT_DIR / 'best_model_advanced.keras', monitor='val_accuracy', save_best_only=True)
    ]

    print("\n=== Phase 2: Fine-tuning ===")
    model.fit(
        train_gen2,
        epochs=EPOCHS_PHASE2,
        validation_data=(X_val, y_val),
        callbacks=callbacks2,
        class_weight=class_weight_dict,
        verbose=1
    )

    print("\n=== Evaluating final model ===")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Final Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    model.save(PROJECT_DIR / 'skin_disease_model_v2.keras')
    print(f"Saved: {PROJECT_DIR / 'skin_disease_model_v2.keras'}")

    artifact = {
        'class_names': CLASS_NAMES,
        'disease_info': DISEASE_INFO,
        'medicines': MEDICINES,
        'img_size': IMG_SIZE
    }
    with open(PROJECT_DIR / 'model_artifact.pkl', 'wb') as f:
        pickle.dump(artifact, f)
    print(f"Artifact saved: {PROJECT_DIR / 'model_artifact.pkl'}")
    print("\n Training complete!")


if __name__ == '__main__':
    train()
