import os
import sys
# Configuration anti-plantage Mac
os.environ['ULTRALYTICS_NO_AUTO_INSTALL'] = '1'
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
try:
    import config
    if not hasattr(config, 'BASE_DIR'):
        config.BASE_DIR = config.PROJECT_ROOT
except ImportError:
    pass
from ultralytics import YOLO
def train():
    project_dir = os.path.join(os.getcwd(), 'runs/classify')
    data_dir = os.path.join(os.getcwd(), 'data/processed')
    
    # Force CPU + Single Thread pour stabilité maximale
    model = YOLO('yolov8n-cls.pt')
    model.train(
        data=data_dir,
        epochs=20,
        imgsz=224,
        project=project_dir,
        name='yolo_cat_classifier',
        exist_ok=True,
        workers=0,
        device='cpu'
    )
    print("✅ Training Complete!")
    model.save(os.path.join(os.getcwd(), 'models/yolo_cat_classifier'))
if __name__ == '__main__':
    train()
