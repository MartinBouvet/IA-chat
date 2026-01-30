"""
Script de capture de données pour l'entraînement
================================================
Ce script permet de constituer un dataset pour un nouvel objet (ex: peluche 2).
Commandes :
- 'ESPACE' : Sauvegarder l'image actuelle
- 'q' : Quitter
"""

import cv2
import os
import time

def main():
    print("📸 Initialisation de la capture...")
    
    # Demander le nom de la classe (ex: 'target_plushie_2')
    class_name = input("Entrez le nom de l'objet (ex: 'tinytin_le_chat') : ").strip().replace(" ", "_")
    if not class_name:
        print("❌ Nom invalide.")
        return

    # Dossier de sauvegarde
    save_dir = os.path.join("data", "raw", class_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 Les images seront sauvegardées dans : {save_dir}")

    # Ouverture de la Webcam (locale par défaut pour la capture)
    # Si vous voulez utiliser la caméra IP, remplacez 0 par l'URL RTSP
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam.")
        return

    print("\n🚀 C'est parti !")
    print("👉 Appuyez sur [ESPACE] pour prendre une photo.")
    print("👉 Appuyez sur [q] pour quitter.\n")

    count = len(os.listdir(save_dir))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Affichage
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Images : {count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Capture Dataset - [ESPACE] pour sauver', display_frame)

        key = cv2.waitKey(1) & 0xFF

        # Sauvegarde
        if key == ord(' '):
            filename = f"{class_name}_{int(time.time()*1000)}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, frame)
            count += 1
            print(f"✅ Image sauvegardée : {filename} (Total: {count})")
            
            # Petit flash visuel (optionnel)
            cv2.rectangle(display_frame, (0,0), (frame.shape[1], frame.shape[0]), (255,255,255), 10)
            cv2.imshow('Capture Dataset - [ESPACE] pour sauver', display_frame)
            cv2.waitKey(50)

        # Quitter
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✨ Terminé ! {count} images au total dans {save_dir}")

if __name__ == '__main__':
    main()
