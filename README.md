# 🎭 Modèle de Classification d'Émotions avec TensorFlow et MobileNetV2

Ce projet utilise un modèle de deep learning basé sur MobileNetV2 pour classer des images faciales selon des émotions humaines : **joie, tristesse, colère, peur, calme, étonnement**.

## 📂 Fonctionnalités

- Extraction des features avec `MobileNetV2` pré-entraîné sur ImageNet.
- Entraînement d’un modèle personnalisé sur les embeddings extraits.
- Augmentation des données avec `ImageDataGenerator`.
- Sauvegarde du meilleur modèle via `ModelCheckpoint`.
- Prédiction d’émotion sur une image.

## 📦 Technologies utilisées

- Python
- TensorFlow / Keras
- NumPy
- PIL (Python Imaging Library)
- scikit-learn

## 🏗️ Architecture du modèle

- Entrée : vecteurs d'embeddings (1280 dimensions)
- Denses : 512 → 256 → 128 → 64 → 32
- Sortie : couche Softmax avec 6 classes
- Fonctions d'activation : ReLU
- Optimiseur : Adam
- Perte : `sparse_categorical_crossentropy`

## 📁 Émotions ciblées

- 😄 joie  
- 😢 tristesse  
- 😠 colère  
- 😨 peur  
- 😌 calme  
- 😲 étonnement  

## 🧪 Instructions d'utilisation

1. **Cloner le projet :**

```bash
git clone https://github.com/ton-compte/emotion-classifier.git
cd emotion-classifier
