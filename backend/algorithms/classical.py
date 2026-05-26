import cv2
import numpy as np


def _safe_create_sift():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create()
    if hasattr(cv2.xfeatures2d, "SIFT_create"):
        return cv2.xfeatures2d.SIFT_create()
    return None


def _safe_create_surf():
    if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SURF_create"):
        return cv2.xfeatures2d.SURF_create(400)
    return None


def _match_keypoints(desc1, desc2, use_l2=True):
    if desc1 is None or desc2 is None:
        return 0
    norm_type = cv2.NORM_L2 if use_l2 else cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(norm_type, crossCheck=True)
    try:
        matches = matcher.match(desc1, desc2)
        return len(matches)
    except cv2.error:
        return 0


def _detect_features(image_gray, detector, use_l2=True):
    if detector is None:
        return 0, 0
    keypoints, descriptors = detector.detectAndCompute(image_gray, None)
    return len(keypoints), descriptors


def detect_copy_move(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    results = {
        "sift": {"available": False, "keypoints": 0, "matches": 0},
        "surf": {"available": False, "keypoints": 0, "matches": 0},
        "akaze": {"available": False, "keypoints": 0, "matches": 0},
        "orb": {"available": False, "keypoints": 0, "matches": 0},
    }

    sift = _safe_create_sift()
    if sift is not None:
        keypoints, descriptors = _detect_features(gray, sift, use_l2=True)
        matches = _match_keypoints(descriptors, descriptors, use_l2=True)
        results["sift"] = {
            "available": True,
            "keypoints": keypoints,
            "matches": matches,
            "score": float(matches) / max(keypoints, 1),
        }

    surf = _safe_create_surf()
    if surf is not None:
        keypoints, descriptors = _detect_features(gray, surf, use_l2=True)
        matches = _match_keypoints(descriptors, descriptors, use_l2=True)
        results["surf"] = {
            "available": True,
            "keypoints": keypoints,
            "matches": matches,
            "score": float(matches) / max(keypoints, 1),
        }

    akaze = cv2.AKAZE_create()
    keypoints, descriptors = _detect_features(gray, akaze, use_l2=False)
    matches = _match_keypoints(descriptors, descriptors, use_l2=False)
    results["akaze"] = {
        "available": True,
        "keypoints": keypoints,
        "matches": matches,
        "score": float(matches) / max(keypoints, 1),
    }

    orb = cv2.ORB_create(5000)
    keypoints, descriptors = _detect_features(gray, orb, use_l2=False)
    matches = _match_keypoints(descriptors, descriptors, use_l2=False)
    results["orb"] = {
        "available": True,
        "keypoints": keypoints,
        "matches": matches,
        "score": float(matches) / max(keypoints, 1),
    }

    results["summary"] = {
        "feature_counts": {
            "sift": results["sift"]["keypoints"],
            "surf": results["surf"]["keypoints"],
            "akaze": results["akaze"]["keypoints"],
            "orb": results["orb"]["keypoints"],
        },
        "notes": "Klasik algoritma sonuçları, sahtecilik tespiti için özet skorlar içerir."
    }
    return results
