"""
Görüntü Sahtecilik Tespiti Algoritmaları Modülü

Bu paket, klasik bilgisayar görüşü algoritmaları ve derin öğrenme modelleri
aracılığıyla görüntü sahteciliğini tespit etmek için tasarlanmıştır.

Sağlanan Fonksiyonlar:
    - detect_copy_move: Klasik algoritmalar ile copy-move tespiti
    - predict_deepfake: AI modelleri ile deepfake tespiti

Örnek Kullanım::

    from backend.algorithms import detect_copy_move, predict_deepfake
    import cv2
    
    image = cv2.imread('test.jpg')
    classical_results = detect_copy_move(image)
    ai_results = predict_deepfake(image, threshold=0.5)

.. versionadded:: 1.0.0
"""

from .classical import detect_copy_move
from .deep_learning import predict_deepfake

__all__ = [
    "detect_copy_move",
    "predict_deepfake",
]

__version__ = "1.0.0"
__author__ = "Ar-Ge Ekibi"
