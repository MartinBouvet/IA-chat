"""
Script de reconnaissance YOLOv8 en temps réel
============================================
Ce script utilise le modèle YOLOv8 fraîchement entraîné pour détecter
si le chat est le "bon" chat ou non.
"""

import cv2
import time
from ultralytics import YOLO
import os

# Chemin direct vers le modèle entraîné
MODEL_PATH = 'runs/classify/yolo_cat_classifier/weights/best.pt'

def main():
    print("🎥 Initialisation de la reconnaissance YOLO...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERREUR: Modèle introuvable à {MODEL_PATH}")
        return

    # Charger le modèle
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Modèle chargé avec succès !")
    except Exception as e:
        print(f"❌ Erreur de chargement du modèle: {e}")
        return

    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam.")
        return

    # Config webcam (HD)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n🚀 Démarrage ! Appuyez sur 'q' pour quitter.")
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Inférence YOLO
        # verbose=False pour ne pas spammer le terminal
        results = model(frame, verbose=False)
        
        # Récupération du résultat
        result = results[0]
        top1_index = result.probs.top1
        class_name = result.names[top1_index]
        confidence = result.probs.top1conf.item()

        # Calcul FPS
        inference_ms = (time.time() - start_time) * 1000
        fps = 1000 / inference_ms if inference_ms > 0 else 0

        # Logique d'affichage
        is_target = (class_name == 'target_cat')
        
        if is_target and confidence > 0.6:
            color = (0, 255, 0) # Vert
            text = f"BON CHAT ({confidence:.0%})"
        else:
            color = (0, 0, 255) # Rouge
            if is_target: # C'est lui mais faible confiance
                text = f"ARTHUR ? ({confidence:.0%})"
                color = (0, 165, 255) # Orange
            else:
                text = f"AUTRE CHAT ({confidence:.0%})"

        # Dessin Interface
        h, w = frame.shape[:2]
        
        # Bandeau de fond
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        
        # Texte Principal
        cv2.putText(frame, text, (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # Infos Techniques
        info_text = f"YOLOv8-Nano | FPS: {fps:.1f} | {inference_ms:.1f}ms"
        cv2.putText(frame, info_text, (w - 450, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Affichage
        cv2.imshow('YOLO Cat Recognition', frame)

        # Quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()