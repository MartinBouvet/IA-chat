"""
Script DRI Headless pour CHAT 2 (La nouvelle peluche)
=====================================================
Ce script utilise le modèle multi-classes pour détecter spécifiquement 'chat2'.
"""

import cv2
import time
import os
import threading
import sys
from ultralytics import YOLO

# --- CONFIGURATION ---
DETECTOR_MODEL_NAME = 'yolov8n.pt'
# On utilise le nouveau modèle renommé
IDENTIFIER_MODEL_PATH = 'runs/models/chat2.pt'

# Flux Vidéo
STREAM_URL = 'rtsp://192.168.137.54:8554/cam1'
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Paramètres
TIMEOUT_SESSION = 60
SEUIL_CONF_ID = 0.6
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
    print("🤖 [IA - CHAT 2] Démarrage analyse...")
    
    try:
        detector = YOLO(DETECTOR_MODEL_NAME)
        identifier = YOLO(IDENTIFIER_MODEL_PATH)
    except Exception as e:
        print(f"❌ Erreur modèle: {e}")
        return

    print(f"📡 Connexion: {STREAM_URL}")
    cam = ThreadedCamera(STREAM_URL)
    time.sleep(2.0)

    if not cam.status:
        print("❌ Echec vidéo.")
        return

    print(f"✅ En attente de la peluche n°2 ({TIMEOUT_SESSION}s)...")
    start_time = time.time()
    last_log_time = 0

    while (time.time() - start_time) < TIMEOUT_SESSION:
        ret, frame = cam.get_frame()
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        results = detector(frame, verbose=False, conf=0.4)[0]
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_id in TARGET_CLASSES_FOR_ID:
                h, w = frame.shape[:2]
                crop = frame[max(0,y1-10):min(h,y2+10), max(0,x1-10):min(w,x2+10)]

                if crop.size > 0:
                    try:
                        id_res = identifier(crop, verbose=False)[0]
                        top1_idx = id_res.probs.top1
                        id_name = id_res.names[top1_idx]
                        id_conf = id_res.probs.top1conf.item()

                        # --- LOGIQUE IDENTIFICATION CHAT 2 ---
                        if id_name == 'chat2' and id_conf > SEUIL_CONF_ID:
                            print(f"✅ CIBLE IDENTIFIÉE : C'est la NOUVELLE PELUCHE (Chat 2) ! ({id_conf:.1%})")
                            # Action spécifique pour le chat 2 ici
                        
                        elif id_name == 'target_cat' and id_conf > SEUIL_CONF_ID:
                            print(f"ℹ️ (J'ai vu TinTin, mais je cherche l'autre aujourd'hui...)")

                    except Exception:
                        pass
        
        time.sleep(0.05)

    print("🛑 Fin.")
    sys.exit(0)

if __name__ == '__main__':
    main()
