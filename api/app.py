import io, os, time, logging, json
from datetime import datetime
from PIL import Image, ImageOps
import numpy as np
from flask import Flask, request, jsonify, g, render_template
from dotenv import load_dotenv
from joblib import load
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# --- Cargar .env
load_dotenv()

MODEL_PATH    = os.getenv("MODEL_PATH", "models/model.joblib")
SCALER_PATH   = os.getenv("SCALER_PATH", "models/scaler.joblib")
THRESHOLD     = float(os.getenv("THRESHOLD", "0.75"))
PORT          = int(os.getenv("PORT", "5000"))
MAX_MB        = int(os.getenv("MAX_MB", "5"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "me-verifier-v1")

app = Flask(__name__)

# --- Logging JSON a stdout
logger = logging.getLogger("me-verifier")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()  # stdout
    h.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(h)

# --- Modelos FaceNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(
    image_size=160,
    margin=40,
    min_face_size=12,
    thresholds=[0.5, 0.7, 0.7],
    post_process=True,
    keep_all=False,
    device=device
)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# --- Clasificador entrenado + scaler
clf = load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
scaler = load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

def allowed_mime(m: str) -> bool:
    return m in ("image/jpeg", "image/png")

# --- Middleware de timing + logging por request
@app.before_request
def _start_timer():
    g._t0 = time.time()

@app.after_request
def _log_after(response):
    try:
        t_ms = (time.time() - getattr(g, "_t0", time.time())) * 1000.0
        size = 0
        if "image" in request.files:
            f = request.files["image"]
            pos = f.stream.tell()
            f.stream.seek(0, io.SEEK_END)
            size = f.stream.tell()
            f.stream.seek(pos)
        log = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "path": request.path,
            "method": request.method,
            "status": response.status_code,
            "latency_ms": round(t_ms, 2),
            "bytes": int(size),
            "remote": request.remote_addr,
        }
        # si la respuesta tiene JSON con campos clave, añádelos
        data = None
        try:
            data = response.get_json(silent=True) or {}
        except Exception:
            data = {}
        for k in ("is_me", "score", "threshold", "model_version", "error"):
            if k in (data or {}):
                log[k] = data[k]
        logger.info(json.dumps(log, ensure_ascii=False))
    except Exception:
        pass
    return response

# --- Manejador global de errores (fallback)
@app.errorhandler(Exception)
def _handle_err(e):
    logger.info(json.dumps({
        "ts": datetime.utcnow().isoformat() + "Z",
        "path": request.path,
        "method": request.method,
        "status": 500,
        "error": str(e),
    }, ensure_ascii=False))
    return jsonify(error=str(e)), 500

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/healthz")
def healthz():
    return jsonify(
        status="ok",
        device=str(device),
        model_loaded=(clf is not None and scaler is not None),
        model_version=MODEL_VERSION,
        threshold=THRESHOLD
    ), 200

@app.post("/verify")
def verify():
    t0 = time.time()

    if "image" not in request.files:
        return jsonify(error="falta campo form-data 'image'"), 400

    f = request.files["image"]
    if not f or not f.mimetype or not allowed_mime(f.mimetype):
        return jsonify(error="solo image/jpeg o image/png"), 415

    # tamaño
    f.seek(0, io.SEEK_END)
    size_mb = f.tell() / (1024*1024)
    f.seek(0)
    if size_mb > MAX_MB:
        return jsonify(error=f"archivo supera {MAX_MB} MB"), 413

    if clf is None or scaler is None:
        return jsonify(error="modelo no entrenado. Falta models/model.joblib y scaler.joblib"), 503

    try:
        # leer imagen + corregir orientación EXIF
        img = Image.open(f.stream).convert("RGB")
        img = ImageOps.exif_transpose(img)

        # detectar rostro y recortar a 160x160
        face = mtcnn(img)
        if face is None:
            return jsonify(error="no se detectó rostro en la imagen"), 422

        with torch.no_grad():
            emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()[0]  # (512,)

        x = emb.reshape(1, -1)
        x = scaler.transform(x)

        # prob de clase "me" (1)
        if hasattr(clf, "predict_proba"):
            score = float(clf.predict_proba(x)[0, 1])
        else:
            from math import exp
            d = float(clf.decision_function(x)[0])
            score = 1.0/(1.0+exp(-d))

        is_me = bool(score >= THRESHOLD)
        timing_ms = (time.time() - t0) * 1000.0

        return jsonify(
            model_version=MODEL_VERSION,
            is_me=is_me,
            score=round(score, 4),
            threshold=THRESHOLD,
            timing_ms=round(timing_ms, 2)
        ), 200

    except Exception as e:
        # además del handler global, devolvemos JSON consistente aquí
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
