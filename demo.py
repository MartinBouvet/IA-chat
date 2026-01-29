"""
Script de démonstration et vérification
========================================

Ce script vérifie que tout est correctement installé et configuré.

Usage:
    python demo.py
"""

import sys
import os

print("\n" + "="*70)
print("🐱 VÉRIFICATION DU SYSTÈME DE RECONNAISSANCE DE CHAT")
print("="*70)

# ============================================================================
# 1. Vérifier les imports
# ============================================================================

print("\n📦 Vérification des packages...")

packages = {
    'tensorflow': 'TensorFlow',
    'cv2': 'OpenCV',
    'numpy': 'NumPy',
    'PIL': 'Pillow',
    'sklearn': 'scikit-learn',
    'matplotlib': 'Matplotlib',
}

missing_packages = []

for package, name in packages.items():
    try:
        __import__(package)
        print(f"   ✅ {name}")
    except ImportError:
        print(f"   ❌ {name} - MANQUANT")
        missing_packages.append(name)

if missing_packages:
    print(f"\n⚠️  Packages manquants : {', '.join(missing_packages)}")
    print(f"   Installez-les avec : pip install -r requirements.txt")
else:
    print(f"\n✅ Tous les packages sont installés !")

# ============================================================================
# 2. Vérifier TensorFlow et GPU
# ============================================================================

print("\n🧠 Vérification de TensorFlow...")

try:
    import tensorflow as tf
    print(f"   ✅ Version : {tf.__version__}")
    
    # Vérifier le GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   ✅ GPU détecté : {len(gpus)} GPU(s) disponible(s)")
        for gpu in gpus:
            print(f"      • {gpu.name}")
    else:
        print(f"   ℹ️  Pas de GPU détecté (entraînement sur CPU)")
except Exception as e:
    print(f"   ❌ Erreur : {e}")

# ============================================================================
# 3. Vérifier la structure des dossiers
# ============================================================================

print("\n📁 Vérification de la structure des dossiers...")

import config

config.create_directories()

required_dirs = [
    config.RAW_DATA_DIR,
    config.PROCESSED_DATA_DIR,
    config.MODELS_DIR,
]

for directory in required_dirs:
    if os.path.exists(directory):
        print(f"   ✅ {directory}")
    else:
        print(f"   ❌ {directory} - MANQUANT")

# ============================================================================
# 4. Vérifier les données
# ============================================================================

print("\n📊 Vérification des données...")

target_cat_dir = os.path.join(config.RAW_DATA_DIR, 'target_cat')
other_dir = os.path.join(config.RAW_DATA_DIR, 'other')

target_cat_count = 0
other_count = 0

if os.path.exists(target_cat_dir):
    target_cat_count = len([f for f in os.listdir(target_cat_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

if os.path.exists(other_dir):
    other_count = len([f for f in os.listdir(other_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

print(f"   📸 Images 'target_cat' : {target_cat_count}")
print(f"   📸 Images 'other' : {other_count}")

if target_cat_count == 0 or other_count == 0:
    print(f"\n   ⚠️  Vous devez collecter des données avant de continuer !")
    print(f"   Utilisez : python scripts/1_collect_from_video.py")
else:
    total = target_cat_count + other_count
    print(f"\n   ✅ Total : {total} images prêtes pour l'entraînement")
    
    if total < 200:
        print(f"   ⚠️  Recommandation : Collectez au moins 200 images par classe")
    elif total < 500:
        print(f"   ℹ️  Bon début ! Plus de données = meilleure précision")
    else:
        print(f"   🎉 Excellent ! Vous avez assez de données")

# ============================================================================
# 5. Tester le modèle builder
# ============================================================================

print("\n🏗️  Test du ModelBuilder...")

try:
    from utils.model_builder import ModelBuilder
    builder = ModelBuilder()
    print(f"   ✅ ModelBuilder initialisé")
    
    # Construire un modèle de test
    print(f"   🔨 Construction d'un modèle de test...")
    model = builder.build_model()
    print(f"   ✅ Modèle construit avec succès")
    
    # Compter les paramètres
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"   📊 Paramètres entraînables : {trainable_params:,}")
    
except Exception as e:
    print(f"   ❌ Erreur : {e}")

# ============================================================================
# 6. Récapitulatif
# ============================================================================

print("\n" + "="*70)
print("📋 RÉCAPITULATIF")
print("="*70)

if missing_packages:
    print("\n❌ Configuration incomplète")
    print("   Installez les packages manquants et relancez ce script")
else:
    print("\n✅ Configuration vérifiée !")
    
    if target_cat_count > 0 and other_count > 0:
        print("\n🚀 Prêt à entraîner le modèle !")
        print("\nProchaines étapes :")
        print("   1. python scripts/2_prepare_dataset.py")
        print("   2. python scripts/3_train_model.py --fine-tune")
        print("   3. python scripts/4_convert_to_tflite.py")
        print("   4. python scripts/5_test_inference.py --tflite")
    else:
        print("\n📸 Prochaine étape : Collectez des données")
        print("\nOptions :")
        print("   • Depuis une vidéo : python scripts/1_collect_from_video.py --video chemin/video.mp4 --class target_cat")
        print("   • Ou placez manuellement vos photos dans data/raw/target_cat/ et data/raw/other/")

print("\n💡 Consultez le README.md pour plus de détails")
print("="*70 + "\n")
