"""
Script de reconnaissance YOLOv8 sur flux réseau (Raspberry Pi)
=============================================================
Ce script se connecte à une caméra IP (Raspberry Pi) et applique
le modèle YOLOv8 pour la reconnaissance de chat.
"""

import cv2
import time
from ultralytics import YOLO
import os

# Chemin direct vers le modèle entraîné (sur le Mac)
MODEL_PATH = 'runs/classify/yolo_cat_classifier/weights/best.pt'

# Authentification et URL de la caméra Raspberry Pi
# Utilisation du protocole RTSP (Port 8554) pour MediaMTX
STREAM_URL = 'rtsp://192.168.137.54:8554/cam1'

# Force l'utilisation de TCP pour éviter les timeouts UDP/RTSP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

def main():
    print("🎥 Initialisation de la reconnaissance YOLO sur flux réseau...")
    print(f"📡 Connexion à: {STREAM_URL}")
    
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

    # Ouvrir le flux vidéo réseau
    cap = cv2.VideoCapture(STREAM_URL)
    
    # Laisser un peu de temps pour la connexion
    time.sleep(1.0)

    if not cap.isOpened():
        print("❌ Impossible de se connecter au flux vidéo.")
        print("   Vérifiez que la Raspberry Pi est allumée et sur le même réseau.")
        print("   Vérifiez l'URL: " + STREAM_URL)
        return

    print("✅ Flux vidéo connecté !")
    print("\n🚀 Démarrage ! Appuyez sur 'q' pour quitter.")
    
    # Compteurs pour stats basiques
    frame_count = 0
    start_time_glob = time.time()

    while True:
        loop_start = time.time()
        
        # Lecture de la frame
        ret, frame = cap.read()
        
        if not ret:
            print("⚠️ Perte du signal vidéo ou fin du flux.")
            # On essaie de reconnecter ? Pour l'instant on break
            break

        # Inférence YOLO
        results = model(frame, verbose=False)
        
        # Récupération du résultat
        result = results[0]
        # Vérification qu'il y a des résultats (classification retourne toujours qqch mais bon)
        if hasattr(result, 'probs'):
            top1_index = result.probs.top1
            class_name = result.names[top1_index]
            confidence = result.probs.top1conf.item()
        else:
            class_name = "Inconnu"
            confidence = 0.0

        # Calcul FPS instantané
        inference_ms = (time.time() - loop_start) * 1000
        fps = 1000 / inference_ms if inference_ms > 0 else 0

        # Logique d'affichage (Couleurs & Textes)
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
        
        # Bandeau de fond (translucide si possible, ici opaque pour simplicité)
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        
        # Texte Principal
        cv2.putText(frame, text, (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # Infos Techniques
        info_text = f"Reseau | FPS: {fps:.1f}"
        cv2.putText(frame, info_text, (w - 300, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Affichage
        cv2.imshow('YOLO Cat Recognition - Reseau', frame)

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
