import cv2
import time
import os
import threading
import sys
from ultralytics import YOLO

# --- CONFIGURATION ---
DETECTOR_MODEL_NAME = 'yolov8n.pt'
IDENTIFIER_MODEL_PATH = 'runs/classify/yolo_cat_classifier/weights/best.pt' # Vérifie ce chemin !

# Flux Vidéo (Ton URL RTSP via MediaMTX)
STREAM_URL = 'rtsp://192.168.137.54:8554/cam1'
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Paramètres de fonctionnement
TIMEOUT_SESSION = 60  # Le script s'arrête tout seul après 60 secondes d'analyse
SEUIL_CONF_ID = 0.6   # Seuil pour dire "C'est la bonne peluche"

# Classes
TARGET_CLASSES_FOR_ID = [15, 16, 77] # Chat, Chien, Teddy Bear
HUMAN_CLASS = 0

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
    print("🤖 [IA] Démarrage de l'analyse vidéo...")
    
    # 1. Chargement des modèles
    try:
        # verbose=False pour éviter que YOLO spamme la console
        detector = YOLO(DETECTOR_MODEL_NAME)
        identifier = YOLO(IDENTIFIER_MODEL_PATH)
    except Exception as e:
        print(f"❌ [IA] Erreur chargement modèles: {e}")
        return

    # 2. Connexion Caméra
    print(f"📡 [IA] Connexion au flux: {STREAM_URL}")
    cam = ThreadedCamera(STREAM_URL)
    time.sleep(2.0) # Chauffe du buffer

    if not cam.status:
        print("❌ [IA] Impossible de lire le flux vidéo.")
        return

    print(f"✅ [IA] Analyse en cours pour {TIMEOUT_SESSION} secondes...")
    start_time = time.time()

    # Variables pour éviter de spammer les logs (un log toutes les X secondes)
    last_log_time = 0
    log_interval = 2.0 

    while (time.time() - start_time) < TIMEOUT_SESSION:
        ret, frame = cam.get_frame()
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        # --- DÉTECTION ---
        results = detector(frame, verbose=False, conf=0.4)[0]
        
        current_time = time.time()
        should_log = (current_time - last_log_time) > log_interval

        human_detected = False
        animal_detected = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 1. Cas HUMAIN
            if cls_id == HUMAN_CLASS:
                human_detected = True
                if should_log:
                    print(f"🙋 [DETECT] Humain repéré ! (Conf: {conf:.2f})")

            # 2. Cas ANIMAL / PELUCHE -> On lance l'Identification
            elif cls_id in TARGET_CLASSES_FOR_ID:
                animal_detected = True
                
                # Extraction (Crop)
                h, w = frame.shape[:2]
                crop = frame[max(0,y1-10):min(h,y2+10), max(0,x1-10):min(w,x2+10)]

                if crop.size > 0:
                    try:
                        # Classification
                        id_res = identifier(crop, verbose=False)[0]
                        top1_idx = id_res.probs.top1
                        id_name = id_res.names[top1_idx]
                        id_conf = id_res.probs.top1conf.item()

                        if should_log:
                            print(f"🐾 [DETECT] Animal/Objet détecté. Analyse...")

                        # Résultat Identification
                        if id_name == 'target_cat' and id_conf > SEUIL_CONF_ID:
                            print(f"✅ CIBLE IDENTIFIÉE : C'est bien TinTin le chat ! ({id_conf:.1%})")
                            # ICI : Tu pourras ajouter plus tard l'ouverture de la trappe
                            # Ex: requests.get('http://ip-raspberry/ouvrir-trappe')
                        
                        elif should_log:
                             print(f"⚠️ [IDENTIFICATION] Ce n'est pas la cible ({id_name}, {id_conf:.1%})")

                    except Exception as e:
                        pass
        
        if (human_detected or animal_detected) and should_log:
            last_log_time = current_time
        
        # Pause légère pour économiser le CPU
        time.sleep(0.05)

    print("🛑 [IA] Fin de la session d'analyse (Timeout).")
    sys.exit(0)

if __name__ == '__main__':
    main()