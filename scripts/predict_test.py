from ultralytics import YOLO
import os
import glob
# Charger le modèle entraîné
model = YOLO('runs/classify/yolo_cat_classifier/weights/best.pt')
# Prendre une image de chat au hasard
img_path = glob.glob('data/processed/val/target_cat/*.jpg')[0]
# Prédire
results = model(img_path)
# Afficher le résultat
print(f"\n📸 Image: {os.path.basename(img_path)}")
print(f"🧠 Prédiction: {results[0].names[results[0].probs.top1]}")
print(f"📊 Confiance: {results[0].probs.top1conf.item():.2%}")
