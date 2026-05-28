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

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15

PROJECT_DIR = Path("C:/Users/ACER/OneDrive/Desktop/Skin Disease Detection")

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
        loss='categorical_crossentropy',
        metrics=['accuracy']
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
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
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

    print(f"\nPhase 1 Best Val Accuracy: {max(history1.history['val_accuracy']):.4f}")

    print("\n--- Training Phase 2: Fine-tuning ---")
    base_model = model.layers[0]
    base_model.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    print(f"\nPhase 2 Best Val Accuracy: {max(history2.history['val_accuracy']):.4f}")

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

    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"\nFinal Validation Accuracy: {val_acc:.4f}")

if __name__ == '__main__':
    train()