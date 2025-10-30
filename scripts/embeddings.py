import os, glob, csv
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1

# Carpetas de recortes
DATA_DIRS = [("data/cropped/me", 1), ("data/cropped/not_me", 0)]

# Salidas
OUT_DIR = "reports"
OUT_NPY = os.path.join(OUT_DIR, "embeddings.npy")
OUT_CSV = os.path.join(OUT_DIR, "labels.csv")

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def load_image_as_tensor(p):
    # Carga PNG 160x160 (ya recortado), lo normaliza a [0,1] y lo pasa a CHW
    img = Image.open(p).convert("RGB")
    arr = np.array(img).astype("float32") / 255.0
    t = torch.from_numpy(arr).permute(2,0,1)  # [3,160,160]
    return t

all_files, all_labels = [], []
for d, y in DATA_DIRS:
    files = []
    for ext in ("*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG"):
        files.extend(glob.glob(os.path.join(d, ext)))
    files.sort()
    all_files.extend(files)
    all_labels.extend([y]*len(files))

print(f"[INFO] Total imágenes: {len(all_files)} (me={all_labels.count(1)}, not_me={all_labels.count(0)})")

batch_size = 64
embeddings = []

with torch.no_grad():
    for i in range(0, len(all_files), batch_size):
        batch_paths = all_files[i:i+batch_size]
        batch_tensors = [load_image_as_tensor(p) for p in batch_paths]
        batch = torch.stack(batch_tensors).to(device)  # [B,3,160,160]
        embs = resnet(batch).cpu().numpy()            # [B,512]
        embeddings.append(embs)
        if (i//batch_size+1) % 5 == 0 or i+batch_size >= len(all_files):
            print(f"[INFO] Procesado {min(i+batch_size, len(all_files))}/{len(all_files)}")

X = np.vstack(embeddings) if embeddings else np.zeros((0,512), dtype="float32")
y = np.array(all_labels, dtype=np.int64)

np.save(OUT_NPY, X)

with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file","label"])
    for p, lab in zip(all_files, all_labels):
        w.writerow([p, lab])

print("[RES] embeddings:", X.shape, " labels:", y.shape, "->", OUT_NPY, OUT_CSV)
