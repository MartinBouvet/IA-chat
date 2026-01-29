"""
Script DRI : Détection, Reconnaissance, Identification (Final)
==============================================================
Logique :
1. YOLOv8 Nano détecte tout le monde.
2. Si HUMAIN -> On affiche juste (Pas d'identification).
3. Si CHAT / CHIEN / PELUCHE -> On envoie au modèle Custom pour Identification.
"""

import cv2
import time
import os
import threading
from ultralytics import YOLO

# --- CONFIGURATION ---
DETECTOR_MODEL_NAME = 'yolov8n.pt' 
IDENTIFIER_MODEL_PATH = 'runs/classify/yolo_cat_classifier/weights/best.pt'

# Flux Vidéo (RTSP avec TCP forcé)
STREAM_URL = 'rtsp://192.168.137.54:8554/cam1'
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Classes d'intérêt
TARGET_CLASSES_FOR_ID = [15, 16, 77] # 15:Cat, 16:Dog, 77:Teddy Bear
HUMAN_CLASS = 0 # 0:Person

class ThreadedCamera:
    """Lecture vidéo optimisée sans latence"""
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.status = False
        self.frame = None
        if self.capture.isOpened():
            self.status = True
            _, self.frame = self.capture.read()
            self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                if status:
                    self.frame = frame
                    self.status = True
                else:
                    self.status = False
                time.sleep(0.005)
            else:
                break

    def get_frame(self):
        return self.status, self.frame

def main():
    print("🤖 Lancement DRI (Detection -> Identification)...")
    
    # 1. Chargement
    try:
        detector = YOLO(DETECTOR_MODEL_NAME)
        identifier = YOLO(IDENTIFIER_MODEL_PATH)
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return

    # 2. Connexion
    print(f"📡 Connexion: {STREAM_URL}")
    cam = ThreadedCamera(STREAM_URL)
    time.sleep(2.0)

    if not cam.status:
        print("❌ Echec connexion vidéo.")
        return

    print("✅ Prêt ! 'q' pour quitter.")

    while True:
        ret, frame = cam.get_frame()
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        # --- ETAPE 1 : DÉTECTION (YOLOv8n) ---
        # On détecte tout avec un seuil assez bas
        results = detector(frame, verbose=False, conf=0.4)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf_det = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Gestion des HUMAINS
            if cls_id == HUMAN_CLASS:
                label = f"Humain {conf_det:.0%}"
                color = (255, 100, 0) # Bleu
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Gestion des ANIMAUX / PELUCHES
            elif cls_id in TARGET_CLASSES_FOR_ID:
                # C'est un candidat -> On passe à l'étape 2 (Identification)
                
                # Crop (Découpage)
                h, w = frame.shape[:2]
                pad = 10
                crop = frame[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]

                if crop.size > 0:
                    try:
                        # --- ETAPE 2 : IDENTIFICATION (Custom Model) ---
                        id_res = identifier(crop, verbose=False)[0]
                        top1_idx = id_res.probs.top1
                        id_name = id_res.names[top1_idx]
                        id_conf = id_res.probs.top1conf.item()

                        # Décision Finale
                        if id_name == 'target_cat' and id_conf > 0.6:
                            label = f"CIBLE ({id_conf:.0%})"
                            color = (0, 255, 0) # Vert
                        elif id_name == 'target_cat':
                            label = f"Cible ? ({id_conf:.0%})"
                            color = (0, 165, 255) # Orange
                        else:
                            label = f"Non ({id_conf:.0%})"
                            color = (0, 0, 255) # Rouge

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                    except Exception:
                        pass

        cv2.imshow('DRI Final', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
