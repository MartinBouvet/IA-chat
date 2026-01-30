from ultralytics import YOLO
import cv2
import os

MODEL_PATH = 'runs/classify/yolo_cat_classifier/weights/multi_cat_model.pt'
TEST_IMAGE_DIR = 'data/raw/chat2'

def main():
    print(f"🧐 Inspection du modèle : {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print(f"✅ Modèle chargé.")
        print(f"📋 Classes connues : {model.names}")
    except Exception as e:
        print(f"❌ Erreur chargement : {e}")
        return

    # Test sur une image capturée
    images = [f for f in os.listdir(TEST_IMAGE_DIR) if f.endswith('.jpg')]
    if not images:
        print("❌ Aucune image trouvée pour tester.")
        return

    test_img_path = os.path.join(TEST_IMAGE_DIR, images[0])
    print(f"\n🧪 Test sur l'image : {test_img_path}")
    
    results = model(test_img_path)
    res = results[0]
    
    print("\n📊 Probabilités :")
    for i, conf in enumerate(res.probs.data):
        class_name = res.names[i]
        print(f"  - {class_name} : {conf:.4f} ({conf:.1%})")

    top1 = res.probs.top1
    print(f"\n🏆 Résultat final : {res.names[top1]} ({res.probs.top1conf:.1%})")

if __name__ == '__main__':
    main()
