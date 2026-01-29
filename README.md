# 🐱 Système de Reconnaissance de Chat avec IA

Projet de distributeur automatique de croquettes utilisant la reconnaissance faciale pour identifier les chats individuels.

## 📋 Vue d'ensemble

Ce système utilise le **deep learning** et le **transfer learning** pour reconnaître un chat spécifique parmi d'autres. Le modèle est optimisé pour fonctionner sur **Raspberry Pi** grâce à **TensorFlow Lite**.

### Caractéristiques principales

- 🧠 **Transfer Learning** avec MobileNetV2 (optimisé pour appareils embarqués)
- 📸 **Extraction automatique** d'images depuis des vidéos
- 🔄 **Data augmentation** pour améliorer la robustesse
- 📊 **Métriques détaillées** pendant l'entraînement
- 💾 **Conversion TFLite** avec quantification pour Raspberry Pi
- ⚡ **Inférence rapide** (~50ms sur Raspberry Pi 4)

---

## 🛠️ Installation

### 1. Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de packages Python)
- (Optionnel) GPU NVIDIA avec CUDA pour accélérer l'entraînement

### 2. Cloner ou créer le projet

```bash
# Si vous avez déjà le dossier
cd cat-recognition

# Sinon, créer la structure
mkdir cat-recognition
cd cat-recognition
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

⏱️ Temps d'installation : environ 5-10 minutes

### 4. Créer la structure des dossiers

```bash
python config.py
```

Cela créera automatiquement tous les dossiers nécessaires :
```
cat-recognition/
├── data/
│   ├── raw/
│   │   ├── target_cat/    # ← Mettez vos photos du chat cible ici
│   │   └── other/         # ← Mettez les photos d'autres chats/objets ici
│   ├── processed/
│   └── test/
└── models/
```

---

## 📸 Collecte des données

### Option 1 : À partir d'une vidéo (RECOMMANDÉ)

Filmez le chat pendant 2-3 minutes dans différentes conditions :
- Différents angles (face, profil, 3/4)
- Différentes distances
- Différents éclairages

Ensuite, extrayez les images automatiquement :

```bash
# Pour le chat cible
python scripts/1_collect_from_video.py --video chemin/vers/video_chat.mp4 --class target_cat

# Pour les autres (chiens, autres chats, humains, objets)
python scripts/1_collect_from_video.py --video chemin/vers/video_autres.mp4 --class other
```

**Paramètres ajustables :**
- `--skip 10` : Extraire 1 frame toutes les 10 frames (ajustez selon la durée de la vidéo)
- `--blur-threshold 100` : Seuil de netteté (plus élevé = images plus nettes)

### Option 2 : Photos manuelles

Prenez 300-500 photos du chat cible et placez-les dans :
```
data/raw/target_cat/
```

Prenez 200-300 photos d'autres chats/objets et placez-les dans :
```
data/raw/other/
```

### Recommandations pour les photos

✅ **Bonnes pratiques :**
- Variété d'angles et de positions
- Différents éclairages (jour, soir, artificiel)
- Chat en mouvement et statique
- Photos nettes (éviter le flou)

❌ **À éviter :**
- Photos floues ou sombres
- Toujours le même angle
- Chat trop loin ou trop près

---

## 🚀 Entraînement du modèle

### Étape 1 : Préparer le dataset

```bash
python scripts/2_prepare_dataset.py
```

Ce script :
- Organise les images en ensembles train/validation (80/20)
- Vérifie la qualité du dataset
- Affiche les statistiques

### Étape 2 : Lancer l'entraînement

**Entraînement basique :**
```bash
python scripts/3_train_model.py
```

**Avec fine-tuning (recommandé pour meilleure précision) :**
```bash
python scripts/3_train_model.py --fine-tune
```

**Personnaliser le nombre d'epochs :**
```bash
python scripts/3_train_model.py --epochs 30 --fine-tune
```

⏱️ **Temps d'entraînement :**
- CPU : 30-60 minutes (selon le dataset)
- GPU : 5-15 minutes

📊 **Résultats attendus :**
- Accuracy > 95% : Excellent
- Accuracy 85-95% : Bon
- Accuracy < 85% : Besoin de plus de données ou d'ajustements

### Étape 3 : Convertir en TensorFlow Lite

```bash
python scripts/4_convert_to_tflite.py
```

Cela crée un fichier `.tflite` optimisé pour Raspberry Pi avec :
- Quantification int8 (réduction de 4x de la taille)
- Optimisations spécifiques pour les appareils embarqués

---

## 🧪 Test du modèle

### Tester une image

```bash
# Avec le modèle Keras
python scripts/5_test_inference.py --image chemin/vers/image.jpg

# Avec le modèle TFLite (pour simuler la Raspberry Pi)
python scripts/5_test_inference.py --image chemin/vers/image.jpg --tflite
```

### Tester un dossier entier

```bash
python scripts/5_test_inference.py --folder data/test/ --tflite
```

### Interpréter les résultats

```
📸 Image : mon_chat.jpg
   🎯 Prédiction : target_cat
   📊 Confiance : 98.5%
   ⏱️  Temps d'inférence : 45.2 ms
   ✅ C'EST LE BON CHAT !
```

- **Confiance > 70%** : Prédiction fiable
- **Confiance 50-70%** : Incertain
- **Confiance < 50%** : Peu fiable

---

## 🔧 Configuration avancée

Tous les paramètres sont dans `config.py` :

```python
# Dimensions des images
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Hyperparamètres d'entraînement
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

# Seuil de confiance pour l'inférence
CONFIDENCE_THRESHOLD = 0.7  # 70%

# Data augmentation
USE_DATA_AUGMENTATION = True
```

---

## 📦 Déploiement sur Raspberry Pi

### 1. Transférer le modèle

Copiez le fichier `models/final/cat_recognition_model.tflite` sur votre Raspberry Pi.

### 2. Installer TensorFlow Lite sur Raspberry Pi

```bash
pip install tflite-runtime
pip install opencv-python
pip install numpy
```

### 3. Script d'inférence pour Raspberry Pi

Utilisez `scripts/5_test_inference.py` directement sur la Raspberry Pi avec l'option `--tflite`.

### 4. Performances attendues

- **Raspberry Pi 4 (4GB)** : ~50ms par image (20 FPS)
- **Raspberry Pi 3** : ~150ms par image (6-7 FPS)

---

## 📊 Structure des fichiers

```
cat-recognition/
│
├── config.py                      # Configuration centralisée
├── requirements.txt               # Dépendances Python
├── README.md                      # Documentation
│
├── data/
│   ├── raw/                       # Photos brutes
│   ├── processed/                 # Données préparées (train/val)
│   └── test/                      # Images de test
│
├── models/
│   ├── checkpoints/               # Sauvegardes pendant l'entraînement
│   └── final/                     # Modèles finaux (.h5 et .tflite)
│
├── scripts/
│   ├── 1_collect_from_video.py    # Extraction depuis vidéo
│   ├── 2_prepare_dataset.py       # Préparation des données
│   ├── 3_train_model.py           # Entraînement
│   ├── 4_convert_to_tflite.py     # Conversion TFLite
│   └── 5_test_inference.py        # Test d'inférence
│
└── utils/
    ├── data_loader.py             # Chargement des données
    └── model_builder.py           # Construction du modèle
```

---

## 🎯 Workflow complet

```
1. 📸 Collecter des données
   → python scripts/1_collect_from_video.py

2. 🗂️  Préparer le dataset
   → python scripts/2_prepare_dataset.py

3. 🧠 Entraîner le modèle
   → python scripts/3_train_model.py --fine-tune

4. 💾 Convertir en TFLite
   → python scripts/4_convert_to_tflite.py

5. 🧪 Tester le modèle
   → python scripts/5_test_inference.py --tflite

6. 🚀 Déployer sur Raspberry Pi
   → Transférer le fichier .tflite
```

---

## 🐛 Résolution de problèmes

### Erreur : "No module named 'tensorflow'"
```bash
pip install tensorflow==2.15.0
```

### Erreur : Dataset vide
- Vérifiez que vous avez mis des images dans `data/raw/target_cat/` et `data/raw/other/`
- Lancez `python scripts/2_prepare_dataset.py`

### Accuracy trop faible (<85%)
- Collectez plus de données (au moins 300 images par classe)
- Augmentez le nombre d'epochs : `--epochs 100`
- Activez le fine-tuning : `--fine-tune`
- Vérifiez la qualité des images (netteté, variété)

### Temps d'inférence trop long sur Raspberry Pi
- Assurez-vous d'utiliser le modèle TFLite (pas le .h5)
- Vérifiez que la quantification est activée
- Utilisez une Raspberry Pi 4 (plus rapide)

---

## 📚 Ressources et références

### Technologies utilisées
- **TensorFlow** : Framework de deep learning
- **MobileNetV2** : Architecture optimisée pour mobile/embarqué
- **OpenCV** : Traitement d'images
- **TensorFlow Lite** : Déploiement sur appareils embarqués

### Documentation utile
- [TensorFlow Transfer Learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)

---

## 🤝 Contribution et support

Pour toute question ou problème, n'hésitez pas à :
- Consulter la documentation
- Vérifier les logs d'erreur
- Tester avec des données plus simples d'abord

---

## 📝 Licence

Ce projet est développé dans un cadre éducatif pour un distributeur automatique de croquettes pour chats.

---

## 🎓 À propos

Projet de reconnaissance faciale de chats utilisant le deep learning et le transfer learning, optimisé pour Raspberry Pi.

**Objectif** : Identifier un chat spécifique pour personnaliser la distribution de nourriture.

**Technologies** : TensorFlow, MobileNetV2, OpenCV, TensorFlow Lite

---

**Bon courage pour votre projet ! 🐱🚀**
