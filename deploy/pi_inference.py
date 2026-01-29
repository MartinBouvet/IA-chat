from ultralytics import YOLO
import cv2
import time
# Chargement du modèle
print("Chargement du modèle...")
model = YOLO('model.pt')
print("Modèle OK.")
# Capture pour Raspberry Pi
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
print("Démarrage de la boucle...")
while True:
    ret, frame = cap.read()
    if not ret: break
    # Inférence
    results = model(frame, verbose=False)
    
    # Logique
    top1 = results[0].probs.top1
    conf = results[0].probs.top1conf.item()
    name = results[0].names[top1]
    # Affichage Console (pour debug)
    if name == 'target_cat' and conf > 0.6:
        print(f"✅ CHAT CIBLE DÉTECTÉ ! (Conf: {conf:.0%}) -> Action Servo ?")
        # Ici: code pour activer le servo
    else:
        print(f"❌ Autre / Rien ({name}, {conf:.0%})")
    time.sleep(0.1) # Pause pour ne pas surchauffer le Pi
