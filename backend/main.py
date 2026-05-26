from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.algorithms.classical import detect_copy_move
from backend.algorithms.deep_learning import predict_deepfake

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(
    title="Görüntü Sahteciliği Tespiti",
    description="Görüntü dosyalarının yüklenmesi ve sahtecilik tespiti için FastAPI tabanlı başlangıç uygulaması.",
)
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


def read_image_bytes(file_bytes: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Yüklenen dosya görüntü olarak açılmadı.")
    return image


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = FRONTEND_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse(content="<h1>Arayüz bulunamadı</h1>", status_code=404)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), threshold: Optional[float] = 0.5):
    allowed_extensions = {"jpg", "jpeg", "png", "gif"}
    name = file.filename.lower()
    if not any(name.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail="Sadece JPG, JPEG, PNG ve GIF formatları desteklenir.")

    content = await file.read()
    try:
        image = read_image_bytes(content)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    classical_results = detect_copy_move(image)
    ai_results = predict_deepfake(image, threshold=threshold)

    response = {
        "filename": file.filename,
        "classical_algorithms": classical_results,
        "ai_algorithms": ai_results,
        "message": "Görüntü gönderildi. Sahtecilik tespiti sonuçları hazır."
    }
    return JSONResponse(content=response)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "message": "API çalışıyor"}
