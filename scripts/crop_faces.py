import os
import glob
from PIL import Image, ImageOps
import torch
from facenet_pytorch import MTCNN

SRC = {"me": "data/me", "not_me": "data/not_me"}
DST = {"me": "data/cropped/me", "not_me": "data/cropped/not_me"}

os.makedirs(DST["me"], exist_ok=True)
os.makedirs(DST["not_me"], exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Ajustes: EXIF fix, caras pequeñas y detector más permisivo en la 1a etapa
mtcnn = MTCNN(
    image_size=160,
    margin=40,                # más contexto alrededor
    min_face_size=12,         # aceptar caras más chicas
    thresholds=[0.5, 0.7, 0.7],  # un poco más permisivo al principio
    post_process=True,
    keep_all=False,           # 1 rostro (principal)
    device=device
)

def process_dir(src_dir, dst_dir):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(src_dir, e)))
    print(f"[INFO] {src_dir}: {len(paths)} imágenes")
    ok = 0
    for i, p in enumerate(paths, 1):
        try:
            img = Image.open(p).convert("RGB")
            # Corrige orientación EXIF (muy importante para fotos móviles)
            img = ImageOps.exif_transpose(img)

            # (opcional) reducir imágenes gigantes para acelerar sin perder detección
            # si el lado mayor > 2000 px, escala a 2000 manteniendo aspecto
            W, H = img.size
            max_side = max(W, H)
            if max_side > 2000:
                scale = 2000 / max_side
                img = img.resize((int(W*scale), int(H*scale)))

            face = mtcnn(img)
            if face is None:
                print(f"[WARN] Sin rostro: {p}")
                continue

            out = os.path.join(dst_dir, os.path.splitext(os.path.basename(p))[0] + ".png")
            Image.fromarray(
                (face.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
            ).save(out)
            ok += 1

            if i % 20 == 0:
                print(f"[INFO] Procesadas {i} en {src_dir} (ok={ok})")
        except Exception as e:
            print(f"[ERR] {p}: {e}")
    print(f"[RES] {src_dir}: {ok}/{len(paths)} con rostro")

if __name__ == "__main__":
    print(f"[INFO] device={device}")
    for k in SRC:
        process_dir(SRC[k], DST[k])
    print("[INFO] done.")
