import os
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from PIL import Image
import random

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

print("TensorFlow version:", tf.__version__)


class F2Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes=7, name='f2_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.fp = self.add_weight(name='fp', shape=(num_classes,), initializer='zeros')
        self.fn = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_class = tf.argmax(y_pred, axis=1)
        y_true_class = tf.argmax(y_true, axis=1)
        pred_one_hot = tf.one_hot(y_pred_class, self.num_classes)
        true_one_hot = y_true
        tp = tf.reduce_sum(pred_one_hot * true_one_hot, axis=0)
        fp = tf.reduce_sum(pred_one_hot * (1 - true_one_hot), axis=0)
        fn = tf.reduce_sum((1 - pred_one_hot) * true_one_hot, axis=0)
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        f2 = (5.0 * precision * recall) / (4.0 * precision + recall + tf.keras.backend.epsilon())
        return tf.reduce_mean(f2)

    def reset_state(self):
        self.tp.assign(tf.zeros_like(self.tp))
        self.fp.assign(tf.zeros_like(self.fp))
        self.fn.assign(tf.zeros_like(self.fn))


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

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
        config.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config


IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15

PROJECT_DIR = Path(__file__).resolve().parent

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
CLASS_TO_INDEX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_INDEX.items()}

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

def load_data():
    print("Loading metadata...")
    df = pd.read_csv(PROJECT_DIR / "HAM10000_metadata.csv")
    print(f"Total samples: {len(df)}")

    images = []
    labels = []
    missing = 0

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
                    img = img.resize((IMG_SIZE, IMG_SIZE))
                    img_array = np.array(img) / 255.0
                    images.append(img_array)
                    labels.append(CLASS_TO_INDEX[dx])
                    break
                except Exception as e:
                    missing += 1

        if idx % 1000 == 0:
            print(f"Processed {idx} images...")

    print(f"Loaded {len(images)} images, missing: {missing}")
    return np.array(images), np.array(labels)

def build_model(num_classes):
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=[F2Score(num_classes=len(CLASS_NAMES))]
    )

    return model

def train():
    X, y = load_data()

    if len(X) == 0:
        print("No data loaded!")
        return

    y_cat = to_categorical(y, num_classes=len(CLASS_NAMES))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training: {len(X_train)}, Validation: {len(X_val)}")

    model = build_model(len(CLASS_NAMES))

    callbacks = [
        EarlyStopping(monitor='val_f2_score', patience=5, restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    ]

    print("\n--- Training Phase 1: Frozen Base ---")
    history1 = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    print(f"\nPhase 1 Best Val F2 Score: {max(history1.history['val_f2_score']):.4f}")

    print("\n--- Training Phase 2: Fine-tuning ---")
    base_model = model.layers[0]
    base_model.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=[F2Score(num_classes=len(CLASS_NAMES))]
    )

    history2 = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    print(f"\nPhase 2 Best Val F2 Score: {max(history2.history['val_f2_score']):.4f}")

    model.save(PROJECT_DIR / 'skin_disease_model.keras')

    artifact = {
        'model': model,
        'class_names': CLASS_NAMES,
        'disease_info': DISEASE_INFO,
        'medicines': MEDICINES,
        'img_size': IMG_SIZE
    }

    with open(PROJECT_DIR / 'model_artifact.pkl', 'wb') as f:
        pickle.dump(artifact, f)

    print("\n✅ Model trained and saved!")
    print(f"Model: {PROJECT_DIR / 'skin_disease_model.keras'}")
    print(f"Artifact: {PROJECT_DIR / 'model_artifact.pkl'}")

    val_loss, val_f2 = model.evaluate(X_val, y_val)
    print(f"\nFinal Validation F2 Score: {val_f2:.4f}")

if __name__ == '__main__':
    train()