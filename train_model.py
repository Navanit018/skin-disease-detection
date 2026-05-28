import os
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import image_dataset_from_directory

from sklearn.model_selection import train_test_split

print("TensorFlow version:", tf.__version__)

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 7

CLASS_NAMES = [
    'akiec',    # Actinic Keratoses
    'bcc',      # Basal Cell Carcinoma
    'bkl',      # Benign Keratosis
    'df',       # Dermatofibroma
    'nv',       # Melanocytic Nevi
    'vasc',     # Vascular Lesions
    'mel'       # Melanoma
]

DISEASE_INFO = {
    'akiec': {
        'name': 'Actinic Keratoses',
        'description': 'Precancerous skin growths that appear as rough, scaly patches.',
        'medicines': ['5-Fluorouracil cream', 'Imiquimod cream', 'Cryotherapy'],
        'prevention': 'Use sunscreen daily, avoid excessive sun exposure'
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'description': 'Most common type of skin cancer, rarely spreads but damages skin.',
        'medicines': ['Surgery', 'Mohs surgery', 'Radiation therapy'],
        'prevention': 'Sun protection, regular skin checks'
    },
    'bkl': {
        'name': 'Benign Keratosis',
        'description': 'Non-cancerous raised spots on the skin.',
        'medicines': ['No treatment needed', 'Cryotherapy if cosmetic concern'],
        'prevention': 'None specific'
    },
    'df': {
        'name': 'Dermatofibroma',
        'description': 'Benign skin nodules, usually on legs.',
        'medicines': ['Usually no treatment', 'Surgical removal if symptomatic'],
        'prevention': 'None specific'
    },
    'nv': {
        'name': 'Melanocytic Nevi (Moles)',
        'description': 'Common moles, usually harmless.',
        'medicines': ['No treatment needed', 'Removal if concerned about changes'],
        'prevention': 'Monitor for changes in size/color/shape'
    },
    'vasc': {
        'name': 'Vascular Lesions',
        'description': 'Birthmarks or hemangiomas from blood vessels.',
        'medicines': ['Laser therapy', 'Corticosteroids'],
        'prevention': 'None specific'
    },
    'mel': {
        'name': 'Melanoma',
        'description': 'Most dangerous skin cancer, can spread to other organs.',
        'medicines': ['Surgery', 'Immunotherapy', 'Chemotherapy', 'Targeted therapy'],
        'prevention': 'Sun protection, avoid tanning beds, regular checks'
    }
}

def build_model():
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def fine_tune_model(model):
    base_model = model.layers[0]
    base_model.trainable = True

    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def prepare_data():
    data_dir = Path('skin_data')

    if not data_dir.exists():
        print("Dataset not found. Downloading HAM10000...")
        return None

    train_ds = image_dataset_from_directory(
        data_dir / 'train',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    val_ds = image_dataset_from_directory(
        data_dir / 'validation',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds

def train():
    train_ds, val_ds = prepare_data()

    if train_ds is None:
        print("Please download and prepare the HAM10000 dataset first.")
        print("Visit: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        return

    model = build_model()

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]

    print("Training Phase 1: Training classifier head...")
    history1 = model.fit(
        train_ds,
        epochs=15,
        validation_data=val_ds,
        callbacks=callbacks
    )

    model = fine_tune_model(model)

    print("Training Phase 2: Fine-tuning...")
    history2 = model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        callbacks=callbacks
    )

    model.save('skin_disease_model.h5')

    artifact = {
        'model': model,
        'class_names': CLASS_NAMES,
        'disease_info': DISEASE_INFO,
        'img_size': IMG_SIZE
    }

    with open('model_artifact.pkl', 'wb') as f:
        pickle.dump(artifact, f)

    print("Model trained and saved successfully!")
    print(f"Classes: {CLASS_NAMES}")

if __name__ == '__main__':
    train()