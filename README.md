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

## Précision sur la validation (val_accuracy) : 
Généralement plus basse (par exemple, 20-50 %) en raison du faible nombre d'échantillons (4 pour la validation) 
et des labels aléatoires, ce qui rend la validation instable et peu fiable.En raison des données fictives, les performances
du modèle (par exemple, accuracy ~80-100 %, val_accuracy ~20-50 %) ne sont pas représentatives. Avec un jeu de données réel et diversifié, on pourrait s'attendre à une précision de validation d'environ 60-75 % pour ce type de modèle, mais cela nécessite un entraînement sur des images variées et correctement étiquetées.

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
