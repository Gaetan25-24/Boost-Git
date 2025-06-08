import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import random

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

emotions = ['joie', 'tristesse', 'colère', 'peur', 'calme', 'étonnement']
num_classes = len(emotions)

base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def create_emotion_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = base_model.predict(x)
    return features[0]

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

fake_img_path = "image/image.jpg"
if not os.path.exists(fake_img_path):
    img = Image.new('RGB', (224, 224), color='blue')
    img.save(fake_img_path)

features_base = extract_features(fake_img_path)

X_data = []
y_data = []
for i in range(20):
    X_data.append(features_base)  # On simplifie la demo
    y_data.append(np.random.randint(0, num_classes))

X_data = np.array(X_data)
y_data = np.array(y_data)

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=seed)

model = create_emotion_model(input_dim=features_base.shape[0])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_emotion_model.h5", save_best_only=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=4,
    callbacks=[early_stop, checkpoint],
    verbose=2
)

test_features = extract_features(fake_img_path).reshape(1, -1)
pred = model.predict(test_features)
predicted_class = emotions[np.argmax(pred)]

print(f"Émotion prédite : {predicted_class}")
