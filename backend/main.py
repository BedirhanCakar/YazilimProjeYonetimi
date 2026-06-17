from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
import traceback
import logging
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.algorithms.classical import detect_copy_move
from backend.algorithms.deep_learning import predict_deepfake
import numpy as _np
import torch as _torch


def _to_python_native(obj):
    """Recursively convert numpy/torch types to native Python types for JSON."""
    # scalars
    if isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, (_torch.Tensor,)):
        try:
            return obj.detach().cpu().numpy().tolist()
        except Exception:
            return str(obj)
    if isinstance(obj, (bytes, bytearray)):
        return None
    # arrays
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    # dict
    if isinstance(obj, dict):
        return {str(k): _to_python_native(v) for k, v in obj.items()}
    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [_to_python_native(v) for v in obj]
    # bool, int, float, str, None
    return obj

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(
    title="Görüntü Sahteciliği Tespiti",
    description="Görüntü dosyalarının yüklenmesi ve sahtecilik tespiti için FastAPI tabanlı Ar-Ge uygulaması.",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend statik dosyaları
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


def read_image_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Bytes'tan görüntü oku.
    
    Args:
        file_bytes: Dosya içeriği (bytes)
    
    Returns:
        BGR formatında numpy array
    
    Raises:
        ValueError: Dosya görüntü olarak açılamadıysa
    """
    image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Yüklenen dosya görüntü olarak açılmadı.")
    return image


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Ana arayüzü döndür."""
    html_path = FRONTEND_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            content="<h1>Hata: Arayüz dosyası bulunamadı</h1>",
            status_code=404
        )
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)


@app.post("/upload/")
async def upload_image(
    file: UploadFile = File(...),
    threshold: Optional[float] = 0.5
) -> JSONResponse:
    """
    Görüntü yükle ve analiz et.
    
    Args:
        file: Yüklenecek resim dosyası
        threshold: AI tespiti için güven eşiği (0-1)
    
    Returns:
        JSON formatında analiz sonuçları
    """
    try:
        # Dosya formatı kontrolü
        allowed_extensions = {"jpg", "jpeg", "png", "gif"}
        filename_lower = file.filename.lower()

        if not any(filename_lower.endswith(f".{ext}") for ext in allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail="Sadece JPG, JPEG, PNG ve GIF formatları desteklenir."
            )

        # Dosya içeriğini oku
        try:
            content = await file.read()
            image = read_image_bytes(content)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))

        # Analiz yap
        try:
            classical_results = detect_copy_move(image)
            ai_results = predict_deepfake(image, threshold=threshold)
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=f"Analiz sırasında hata: {str(error)}"
            )

        # Sonuçları döndür
        response = {
            "filename": file.filename,
            "image_shape": {
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
                "channels": int(image.shape[2]) if len(image.shape) > 2 else 1
            },
            "classical_algorithms": classical_results,
            "ai_algorithms": ai_results,
            "timestamp": None,  # Frontend'de istenirse eklenebilir
            "message": "Görüntü analizi tamamlandı"
        }

        safe = _to_python_native(response)
        return JSONResponse(content=jsonable_encoder(safe), status_code=200)
    except HTTPException:
        # Rethrow HTTPExceptions so FastAPI handles them normally
        raise
    except Exception as e:
        # Log full traceback to server console for debugging
        tb = traceback.format_exc()
        logging.error("Unhandled exception in /upload/: %s", tb)
        raise HTTPException(status_code=500, detail=f"Sunucu hatası: {str(e)}")


@app.get("/health")
def health() -> dict:
    """API sağlık kontrolü."""
    return {
        "status": "ok",
        "message": "API çalışıyor",
        "service": "Görüntü Sahteciliği Tespiti"
    }


@app.get("/info")
def info() -> dict:
    """Uygulama bilgileri."""
    return {
        "name": "Görüntü Sahteciliği Tespiti",
        "version": "1.0.0",
        "description": "Klasik CV ve Derin Öğrenme tabanlı görüntü sahtecilik tespiti",
        "algorithms": {
            "classical": ["SIFT", "SURF", "AKAZE", "ORB"],
            "deep_learning": ["CNN", "LSTM"],
            "analysis": ["ELA", "FFT"]
        }
    }

