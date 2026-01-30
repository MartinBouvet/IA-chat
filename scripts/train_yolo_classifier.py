"""
Script d'entraînement YOLOv8 Classification
===========================================
Ce script prépare les données (Train/Val/Test) et lance l'entraînement.
"""

import os
import shutil
import random
from ultralytics import YOLO

# Configuration
RAW_DATA_DIR = os.path.join('data', 'raw')
PROCESSED_DATA_DIR = os.path.join('data', 'processed')

def prepare_data(target_class, other_class='other', split_ratio=(0.7, 0.2, 0.1)):
    """
    Prépare les données pour un entraînement binaire : Target vs Other.
    On ignore toutes les autres classes présentes dans 'raw'.
    """
    print(f"🔄 Préparation des données pour : {target_class} vs {other_class}")
    
    if os.path.exists(PROCESSED_DATA_DIR):
        shutil.rmtree(PROCESSED_DATA_DIR)
    
    # On ne sélectionne QUE les deux classes qui nous intéressent
    classes_to_use = [target_class, other_class]

    for cls in classes_to_use:
        src_dir = os.path.join(RAW_DATA_DIR, cls)
        if not os.path.exists(src_dir):
            print(f"⚠️ Attention: Dossier introuvable {src_dir}")
            continue

        images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        
        splits = {
            'train': images[:int(len(images)*split_ratio[0])],
            'val': images[int(len(images)*split_ratio[0]):int(len(images)*(split_ratio[0]+split_ratio[1]))],
            'test': images[int(len(images)*(split_ratio[0]+split_ratio[1])):]
        }
        
        for split, split_images in splits.items():
            dest_dir = os.path.join(PROCESSED_DATA_DIR, split, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for img in split_images:
                shutil.copy(os.path.join(src_dir, img), os.path.join(dest_dir, img))
                
    print("✅ Données filtrées et prêtes !")

def train_model(target_class, project_name):
    # 1. Prépa
    prepare_data(target_class)

    # 2. Modèle base
    model = YOLO('yolov8n-cls.pt') 

    # 3. Train
    print(f"🚀 Lancement entraînement pour {project_name}...")
    model.train(
        data=PROCESSED_DATA_DIR,
        epochs=10,
        imgsz=224,
        project='runs/classify',
        name=project_name,
        exist_ok=True
    )
    print(f"✅ Modèle sauvegardé : runs/classify/{project_name}/weights/best.pt")

import sys

def main():
    # MENU D'ENTRAINEMENT (Via arguments pour automation)
    # Usage: python script.py [1|2]
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("1. Entraîner TinTin (target_cat vs other)")
        print("2. Entraîner Chat 2 (chat2 vs other)")
        choice = input("Choix (1/2) : ")

    if choice == '1':
        train_model('target_cat', 'yolo_tintin')
    elif choice == '2':
        train_model('chat2', 'yolo_chat2')
    else:
        print("Choix invalide.")

if __name__ == '__main__':
    main()
