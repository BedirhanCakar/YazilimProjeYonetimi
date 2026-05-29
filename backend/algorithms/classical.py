"""
Klasik Bilgisayar Görüşü Algoritmaları ile Copy-Move Sahtecilik Tespiti

Bu modül, SIFT, SURF, AKAZE ve ORB özellik çıkarma algoritmaları kullanarak
görüntülerdeki kopya-hareket manipülasyonlarını tespit eder.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional


def _detect_and_compute_sift(gray: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
    """SIFT detektörü ile özellik çıkarma."""
    try:
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        return keypoints, descriptors
    except Exception:
        return [], None


def _detect_and_compute_surf(gray: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
    """SURF detektörü ile özellik çıkarma."""
    try:
        surf = cv2.xfeatures2d.SURF_create(400)
        keypoints, descriptors = surf.detectAndCompute(gray, None)
        return keypoints, descriptors
    except Exception:
        return [], None


def _detect_and_compute_akaze(gray: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
    """AKAZE detektörü ile özellik çıkarma."""
    try:
        akaze = cv2.AKAZE_create()
        keypoints, descriptors = akaze.detectAndCompute(gray, None)
        return keypoints, descriptors
    except Exception:
        return [], None


def _detect_and_compute_orb(gray: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
    """ORB detektörü ile özellik çıkarma."""
    try:
        orb = cv2.ORB_create(5000)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        return keypoints, descriptors
    except Exception:
        return [], None


def _match_features(descriptors1: Optional[np.ndarray], 
                   descriptors2: Optional[np.ndarray],
                   norm_type: int) -> List:
    """
    İki tanımlayıcı seti arasında eşleştirme yapma.
    
    Args:
        descriptors1: Birinci tanımlayıcı seti
        descriptors2: İkinci tanımlayıcı seti
        norm_type: Mesafe normu türü
    
    Returns:
        Eşleştirme listesi
    """
    if descriptors1 is None or descriptors2 is None:
        return []
    
    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    try:
        matches = matcher.knnMatch(descriptors1, descriptors2, k=3)
        good_matches = []
        for match_triplet in matches:
            # Aynı anahtar nokta ile eşleştirmeyi engelle
            candidates = [m for m in match_triplet if m.queryIdx != m.trainIdx]
            if len(candidates) < 2:
                continue
            m, n = candidates[0], candidates[1]
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        return good_matches
    except cv2.error:
        return []


def _detect_copy_move_region(image: np.ndarray,
                            keypoints: List,
                            matches: List,
                            threshold: float = 0.05) -> Dict[str, Any]:
    """
    Copy-move bölgelerini algılama ve işaretleme.
    
    Args:
        image: Giriş görüntüsü
        keypoints: Çıkartılan özellik noktaları
        matches: Eşleştirilen noktalar
        threshold: Şüphe yüksekliği eşiği
    
    Returns:
        Bölge tespit sonuçları
    """
    height, width = image.shape[:2]
    
    if len(matches) == 0:
        return {
            "detected": False,
            "confidence": 0.0,
            "region_count": 0,
            "average_distance": 0.0
        }
    
    # Eşleştirme vektörlerinin uzunluğunu hesapla
    distances = []
    displacement_vectors = []
    for match in matches:
        kp_q = keypoints[match.queryIdx]
        kp_t = keypoints[match.trainIdx]
        dx = kp_t.pt[0] - kp_q.pt[0]
        dy = kp_t.pt[1] - kp_q.pt[1]
        dist = np.hypot(dx, dy)
        distances.append(dist)
        displacement_vectors.append((dx, dy))
    
    avg_distance = np.mean(distances) if distances else 0.0
    max_distance = np.max(distances) if distances else 0.0
    
    # Hareket tahmini
    consistency_score = 1.0 - min(np.std(distances) / (max_distance + 1e-6), 1.0)
    
    # Aynı yönde hareket eden eşleşmeleri bul
    translation_consistency = 0.0
    if displacement_vectors:
        disp_array = np.array(displacement_vectors, dtype=np.float32)
        quantized = np.round(disp_array / 20.0).astype(int)
        unique, counts = np.unique(quantized, axis=0, return_counts=True)
        best_cluster = int(np.max(counts)) if counts.size > 0 else 0
        translation_consistency = best_cluster / len(matches)
    
    # İstatistiksel analiz
    match_density = len(matches) / max((height * width / 10000), 1)
    confidence_score = float(np.clip(
        consistency_score * min(len(matches) / 250.0, 1.0) * (0.5 + translation_consistency * 0.5),
        0.0,
        1.0
    ))
    detected = (
        len(matches) > 80 and
        consistency_score > 0.85 and
        avg_distance > 18.0 and
        max_distance > 25.0 and
        match_density > 1.2 and
        translation_consistency > 0.35
    )
    
    return {
        "detected": detected,
        "confidence": confidence_score,
        "region_count": len(matches),
        "average_distance": float(avg_distance),
        "consistency_score": float(consistency_score),
        "translation_consistency": float(translation_consistency)
    }


def detect_copy_move(image: np.ndarray) -> dict:
    """
    Görüntü üzerinde copy-move sahteciliğini SIFT, SURF, AKAZE ve ORB 
    kullanarak tespit eder.
    
    Args:
        image: Analiz edilecek BGR formatında görüntü
    
    Returns:
        Her algoritmanın tespit sonuçları içeren sözlük
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    results = {}
    
    # SIFT
    sift_kp, sift_desc = _detect_and_compute_sift(gray)
    if sift_desc is not None:
        sift_matches = _match_features(sift_desc, sift_desc, cv2.NORM_L2)
        sift_region = _detect_copy_move_region(image, sift_kp, sift_matches)
        results["sift"] = {
            "available": True,
            "keypoints_count": len(sift_kp),
            "matches_count": len(sift_matches),
            "region_detection": sift_region
        }
    else:
        results["sift"] = {"available": False}
    
    # SURF
    surf_kp, surf_desc = _detect_and_compute_surf(gray)
    if surf_desc is not None:
        surf_matches = _match_features(surf_desc, surf_desc, cv2.NORM_L2)
        surf_region = _detect_copy_move_region(image, surf_kp, surf_matches)
        results["surf"] = {
            "available": True,
            "keypoints_count": len(surf_kp),
            "matches_count": len(surf_matches),
            "region_detection": surf_region
        }
    else:
        results["surf"] = {"available": False}
    
    # AKAZE
    akaze_kp, akaze_desc = _detect_and_compute_akaze(gray)
    if akaze_desc is not None:
        akaze_matches = _match_features(akaze_desc, akaze_desc, cv2.NORM_HAMMING)
        akaze_region = _detect_copy_move_region(image, akaze_kp, akaze_matches)
        results["akaze"] = {
            "available": True,
            "keypoints_count": len(akaze_kp),
            "matches_count": len(akaze_matches),
            "region_detection": akaze_region
        }
    else:
        results["akaze"] = {"available": False}
    
    # ORB
    orb_kp, orb_desc = _detect_and_compute_orb(gray)
    if orb_desc is not None:
        orb_matches = _match_features(orb_desc, orb_desc, cv2.NORM_HAMMING)
        orb_region = _detect_copy_move_region(image, orb_kp, orb_matches)
        results["orb"] = {
            "available": True,
            "keypoints_count": len(orb_kp),
            "matches_count": len(orb_matches),
            "region_detection": orb_region
        }
    else:
        results["orb"] = {"available": False}
    
    # Genel özet
    available_methods = [m for m, r in results.items() if r.get("available", False)]
    confidences = [results[m]["region_detection"]["confidence"] 
                  for m in available_methods]
    
    detected_methods = [
        results[m]["region_detection"]["detected"]
        for m in available_methods if m in results
    ]
    results["summary"] = {
        "available_methods": available_methods,
        "overall_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "method_count": len(available_methods),
        "consensus_detected": sum(detected_methods) >= 2,
        "detected_methods": sum(detected_methods)
    }
    
    return results
