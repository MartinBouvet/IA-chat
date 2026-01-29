import os
import shutil
import random
ROOT = os.getcwd()
RAW = os.path.join(ROOT, 'data', 'raw')
PROC = os.path.join(ROOT, 'data', 'processed')
def repopulate():
    print(f"🧹 Clearing {PROC}...")
    if os.path.exists(PROC): shutil.rmtree(PROC)
    
    for split in ['train', 'val']:
        for cls in ['target_cat', 'other']:
            os.makedirs(os.path.join(PROC, split, cls), exist_ok=True)
    for cls in ['target_cat', 'other']:
        src = os.path.join(RAW, cls)
        if not os.path.exists(src): continue
        files = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(files)
        split = int(len(files) * 0.8)
        
        print(f"📦 {cls}: {len(files)} items -> Train:{split} Val:{len(files)-split}")
        for f in files[:split]:
            shutil.copy2(os.path.join(src, f), os.path.join(PROC, 'train', cls, f))
        for f in files[split:]:
            shutil.copy2(os.path.join(src, f), os.path.join(PROC, 'val', cls, f))
if __name__ == '__main__':
    repopulate()
