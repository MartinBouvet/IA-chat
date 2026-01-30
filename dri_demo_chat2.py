"""
Script DRI VISUEL pour CHAT 2 (La nouvelle peluche)
===================================================
Similaire à dri_demo.py mais configuré pour détecter 'chat2'.
Affiche une fenêtre vidéo pour vérifier que ça marche.
"""

import cv2
import time
import os
import threading
from ultralytics import YOLO

# --- CONFIGURATION ---
DETECTOR_MODEL_NAME = 'yolov8n.pt'
IDENTIFIER_MODEL_PATH = 'runs/models/chat2.pt'

# Flux Vidéo
STREAM_URL = 'rtsp://192.168.137.54:8554/cam1'
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Paramètres
TARGET_CLASSES_FOR_ID = [15, 16, 77] # Chat, Chien, Teddy
HUMAN_CLASS = 0

class ThreadedCamera:
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
    print("🤖 [VISUEL - CHAT 2] Démarrage...")
    
    try:
        detector = YOLO(DETECTOR_MODEL_NAME)
        identifier = YOLO(IDENTIFIER_MODEL_PATH)
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return

    print(f"📡 Connexion: {STREAM_URL}")
    cam = ThreadedCamera(STREAM_URL)
    time.sleep(2.0)

    if not cam.status:
        print("❌ Echec vidéo.")
        return

    print("✅ Fenêtre ouverte ! Appuyez sur 'q' pour quitter.")

    while True:
        ret, frame = cam.get_frame()
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        results = detector(frame, verbose=False, conf=0.4)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # --- HUMAIN ---
            if cls_id == HUMAN_CLASS:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(frame, "Humain", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

            # --- ANIMAL / PELUCHE ---
            elif cls_id in TARGET_CLASSES_FOR_ID:
                h, w = frame.shape[:2]
                crop = frame[max(0,y1-10):min(h,y2+10), max(0,x1-10):min(w,x2+10)]

                if crop.size > 0:
                    try:
                        id_res = identifier(crop, verbose=False)[0]
                        top1_idx = id_res.probs.top1
                        id_name = id_res.names[top1_idx]
                        id_conf = id_res.probs.top1conf.item()

                        # Gestion des couleurs et labels
                        if id_name == 'chat2' and id_conf > 0.6:
                            label = f"NOUVEAU CHAT ({id_conf:.0%})"
                            color = (0, 255, 0) # Vert (C'est lui !)
                        elif id_name == 'target_cat' and id_conf > 0.6:
                            label = f"TinTin ({id_conf:.0%})"
                            color = (0, 255, 255) # Jaune (C'est l'ancien)
                        else:
                            label = f"Inconnu ({id_conf:.0%})"
                            color = (0, 0, 255) # Rouge

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                    except Exception:
                        pass

        cv2.imshow('Detection Chat 2', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
