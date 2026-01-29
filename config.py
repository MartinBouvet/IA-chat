"""
Configuration centralisée pour le projet de reconnaissance de chat
"""
import os

# ============================================================================
# CHEMINS DES DOSSIERS
# ============================================================================

# Racine du projet
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = PROJECT_ROOT

# Dossiers de données
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')

# Dossiers de modèles
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, 'checkpoints')
FINAL_MODEL_DIR = os.path.join(MODELS_DIR, 'final')

# ============================================================================
# PARAMÈTRES DU MODÈLE
# ============================================================================

# Classes à reconnaître
CLASSES = ['target_cat', 'other']
NUM_CLASSES = len(CLASSES)

# Dimensions des images
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Modèle de base (transfer learning)
BASE_MODEL = 'MobileNetV2'  # Optimisé pour Raspberry Pi
WEIGHTS = 'imagenet'

# ============================================================================
# PARAMÈTRES D'ENTRAÎNEMENT
# ============================================================================

# Split des données
TRAIN_SPLIT = 0.8  # 80% pour l'entraînement
VAL_SPLIT = 0.2    # 20% pour la validation

# Hyperparamètres
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

# Early stopping
PATIENCE = 10  # Arrêt si pas d'amélioration après 10 epochs

# Data augmentation
USE_DATA_AUGMENTATION = True
AUGMENTATION_PARAMS = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.2,
    'shear_range': 0.15,
    'fill_mode': 'nearest'
}

# ============================================================================
# PARAMÈTRES DE CONVERSION TFLITE
# ============================================================================

# Optimisations pour Raspberry Pi
TFLITE_OPTIMIZATIONS = True
QUANTIZE_MODEL = True  # Réduction de la taille du modèle

# ============================================================================
# PARAMÈTRES D'INFÉRENCE
# ============================================================================

# Seuil de confiance pour accepter une prédiction
CONFIDENCE_THRESHOLD = 0.7  # 70% de confiance minimum

# ============================================================================
# PARAMÈTRES D'EXTRACTION VIDÉO
# ============================================================================

# Extraction d'images depuis une vidéo
FRAME_SKIP = 10  # Extraire 1 frame toutes les 10 frames
MIN_BLUR_THRESHOLD = 100  # Seuil pour éviter les images floues

# ============================================================================
# AUTRES PARAMÈTRES
# ============================================================================

# Seed pour la reproductibilité
RANDOM_SEED = 42

# Verbose
VERBOSE = 1  # 0: silent, 1: progress bar, 2: one line per epoch


def create_directories():
    """
    Crée tous les dossiers nécessaires s'ils n'existent pas
    """
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        os.path.join(RAW_DATA_DIR, 'target_cat'),
        os.path.join(RAW_DATA_DIR, 'other'),
        PROCESSED_DATA_DIR,
        os.path.join(PROCESSED_DATA_DIR, 'train', 'target_cat'),
        os.path.join(PROCESSED_DATA_DIR, 'train', 'other'),
        os.path.join(PROCESSED_DATA_DIR, 'validation', 'target_cat'),
        os.path.join(PROCESSED_DATA_DIR, 'validation', 'other'),
        TEST_DATA_DIR,
        MODELS_DIR,
        CHECKPOINTS_DIR,
        FINAL_MODEL_DIR,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Tous les dossiers ont été créés avec succès !")


if __name__ == "__main__":
    # Test de la configuration
    create_directories()
    print(f"\n📁 Racine du projet : {PROJECT_ROOT}")
    print(f"🖼️  Taille des images : {IMG_SIZE}")
    print(f"🎯 Classes : {CLASSES}")
    print(f"🔢 Nombre d'epochs : {EPOCHS}")
    print(f"📊 Batch size : {BATCH_SIZE}")
