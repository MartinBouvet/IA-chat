import cv2
import sys
import time
from ultralytics import YOLO
import os
MODEL_PATH = 'runs/classify/yolo_cat_classifier/weights/best.pt'
def main():
    print("🔍 DIAGNOSTIC WEBCAM...")
    
    # 1. Test Permissions / Accès
    cap = cv2.VideoCapture(0)
    time.sleep(1) # Laisser le temps à la cam de s'allumer
    
    if not cap.isOpened():
        print("❌ ERREUR FATALE: Impossible d'ouvrir la caméra (Index 0).")
        print("💡 Vérifiez qu'aucune autre app n'utilise la caméra.")
        return
    print("✅ Caméra ouverte (Index 0). Tentative de lecture...")
    ret, frame = cap.read()
    
    if not ret:
        print("❌ ERREUR: Caméra ouverte mais impossible de lire une image !")
        print("💡 C'est souvent un problème de permissions macOS.")
        print("👉 Essayez de lancer le terminal via 'Réglages Système > Confidentialité > Caméra'")
        return
    else:
        print(f"✅ Image capturée avec succès ! Taille: {frame.shape}")
    # 2. Si ça marche, on lance YOLO
    print("\n🚀 Lancement du modèle...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Erreur modèle: {e}")
        return
    print("🎥 Fenêtre vidéo active. Appuyez sur 'q' pour quitter.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Perte du flux vidéo !")
            break
        results = model(frame, verbose=False)
        res = results[0]
        name = res.names[res.probs.top1]
        conf = res.probs.top1conf.item()
        # Affichage
        color = (0, 255, 0) if name == 'target_cat' and conf > 0.6 else (0, 0, 255)
        text = f"{name} ({conf:.0%})"
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Debug Cam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
