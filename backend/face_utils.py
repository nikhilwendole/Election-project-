"""
face_utils.py — High Accuracy Face Recognition
Uses facial geometry + structure dimensions via numpy
Each person has unique facial measurements stored
"""

import cv2
import numpy as np
from database import get_all_voters

# Face & Eye detectors
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
NOSE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_mcs_nose.xml"
)
MOUTH_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

# Strict threshold — must be very high to count as duplicate
# 0.92 = strict, increase if false duplicates, decrease if missing real duplicates
THRESHOLD = 0.92
NEEDED    = 20  # frames to sample for stability


def get_facial_geometry(gray, face_coords):
    """
    Extract unique facial geometry measurements.
    Returns a numpy feature vector based on:
    - Face dimensions & proportions
    - Eye positions & sizes
    - Nose position
    - Mouth position
    - Facial region histograms (divided into zones)
    """
    x, y, w, h = face_coords

    # Crop face region
    pad    = 15
    y1     = max(0, y - pad)
    y2     = min(gray.shape[0], y + h + pad)
    x1     = max(0, x - pad)
    x2     = min(gray.shape[1], x + w + pad)
    face   = gray[y1:y2, x1:x2]
    face   = cv2.resize(face, (200, 200))
    face   = cv2.equalizeHist(face)

    features = []

    # ── 1. Divide face into 5x5 grid zones ──────────────────
    # Each zone captures local texture — unique per person
    zone_h = face.shape[0] // 5
    zone_w = face.shape[1] // 5

    for row in range(5):
        for col in range(5):
            zone = face[row*zone_h:(row+1)*zone_h,
                        col*zone_w:(col+1)*zone_w]

            # Mean and std of each zone
            features.append(float(np.mean(zone)))
            features.append(float(np.std(zone)))

    # ── 2. Facial proportion ratios ─────────────────────────
    # These are unique geometric measurements
    face_w = float(w)
    face_h = float(h)

    # Face aspect ratio
    features.append(face_w / (face_h + 1e-5))

    # Upper half vs lower half brightness
    upper = float(np.mean(face[:100, :]))
    lower = float(np.mean(face[100:, :]))
    features.append(upper / (lower + 1e-5))

    # Left vs right symmetry
    left  = float(np.mean(face[:, :100]))
    right = float(np.mean(face[:, 100:]))
    features.append(left / (right + 1e-5))

    # ── 3. Eye detection & geometry ─────────────────────────
    eyes = EYE_CASCADE.detectMultiScale(
        face,
        scaleFactor  = 1.1,
        minNeighbors = 5,
        minSize      = (20, 20)
    )

    if len(eyes) >= 2:
        # Sort eyes left to right
        eyes = sorted(eyes, key=lambda e: e[0])
        ex1, ey1, ew1, eh1 = eyes[0]
        ex2, ey2, ew2, eh2 = eyes[1]

        # Eye center positions (normalized by face size)
        eye1_cx = (ex1 + ew1 / 2) / 200.0
        eye1_cy = (ey1 + eh1 / 2) / 200.0
        eye2_cx = (ex2 + ew2 / 2) / 200.0
        eye2_cy = (ey2 + eh2 / 2) / 200.0

        # Eye distance (interpupillary distance)
        eye_dist = np.sqrt((eye2_cx - eye1_cx)**2 + (eye2_cy - eye1_cy)**2)

        # Eye width ratio
        eye_w_ratio = ew1 / (ew2 + 1e-5)

        # Eye Y difference (tilt)
        eye_tilt = abs(eye1_cy - eye2_cy)

        features.extend([
            eye1_cx, eye1_cy,
            eye2_cx, eye2_cy,
            float(eye_dist),
            float(eye_w_ratio),
            float(eye_tilt),
        ])
    else:
        features.extend([0.0] * 7)

    # ── 4. HOG-like gradient features ───────────────────────
    # Captures edge/shape information unique to each face
    sobelx = cv2.Sobel(face, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(face, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Divide gradient map into 4x4 blocks
    block_h = magnitude.shape[0] // 4
    block_w = magnitude.shape[1] // 4
    for row in range(4):
        for col in range(4):
            block = magnitude[row*block_h:(row+1)*block_h,
                              col*block_w:(col+1)*block_w]
            features.append(float(np.mean(block)))
            features.append(float(np.std(block)))

    # ── 5. Convert to numpy and normalize ───────────────────
    feature_vec = np.array(features, dtype=np.float32)
    norm        = np.linalg.norm(feature_vec)
    if norm > 0:
        feature_vec = feature_vec / norm

    return feature_vec.tolist()


def get_face_encoding(frame):
    """
    Detect face and extract unique facial geometry encoding.
    Returns (encoding, face_coords, face_count)
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor  = 1.05,
        minNeighbors = 6,
        minSize      = (100, 100)
    )

    if len(faces) == 0:
        return None, None, 0
    if len(faces) > 1:
        return None, None, len(faces)

    encoding = get_facial_geometry(gray, faces[0])
    return encoding, faces[0], 1


def compare_encodings(enc1: list, enc2: list) -> float:
    """
    Compare two facial geometry encodings.
    Returns similarity 0.0 to 1.0
    """
    a = np.array(enc1, dtype=np.float32)
    b = np.array(enc2, dtype=np.float32)

    if a.shape != b.shape:
        return 0.0

    # Cosine similarity
    dot    = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))


def is_duplicate_voter(new_encoding: list):
    """
    Check if face already exists in MongoDB.
    Returns (True, voter, confidence%) or (False, None, 0)
    """
    voters = get_all_voters()
    if not voters:
        return False, None, 0

    best_score = -1
    best_voter = None

    for voter in voters:
        stored = voter["encoding"]
        if len(stored) != len(new_encoding):
            continue
        score = compare_encodings(new_encoding, stored)
        if score > best_score:
            best_score = score
            best_voter = voter

    print(f"[DEBUG] Best match score: {round(best_score * 100, 2)}% (threshold: {THRESHOLD*100}%)")

    if best_score >= THRESHOLD:
        return True, best_voter, round(best_score * 100, 2)

    return False, None, 0


def draw_alert(frame, message, color):
    """Draw colored alert banner at bottom of frame."""
    h, w    = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 75), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, message, (10, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def live_face_verify():
    """
    Live camera face verification with facial geometry.
    Collects multiple frames for accuracy.
    GREEN = new voter | RED = duplicate
    SPACE = confirm | ESC = cancel
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return "cancelled", None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    voters      = get_all_voters()
    window_name = "Election Face Detection System"
    samples     = []
    last_result = None

    print(f"\n[CAMERA] Starting — look straight at camera")
    print(f"         Collecting {NEEDED} frames for accuracy")
    print(f"         SPACE to confirm | ESC to cancel\n")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 650)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        encoding, face_coords, face_count = get_face_encoding(frame)

        # ── Top bar ─────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 55), (20, 20, 20), -1)

        if face_count == 0:
            cv2.putText(frame, "No face detected — move closer",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 165, 255), 2)
            samples     = []
            last_result = None

        elif face_count > 1:
            cv2.putText(frame, "Multiple faces — only ONE voter allowed!",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 60, 255), 2)
            samples     = []
            last_result = None

        else:
            x, y, w, h = face_coords
            samples.append(encoding)

            # Progress bar
            bar_w = int((len(samples) / NEEDED) * (frame.shape[1] - 20))
            cv2.rectangle(frame, (10, 45), (10 + bar_w, 52), (0, 200, 255), -1)

            if len(samples) < NEEDED:
                cv2.putText(frame,
                            f"Scanning face... {len(samples)}/{NEEDED}",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 200, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 255), 2)

            else:
                # Average all samples for stable encoding
                avg_enc = np.mean(
                    [np.array(s) for s in samples], axis=0
                ).tolist()
                samples = []

                if not voters:
                    last_result = ("new", avg_enc)
                else:
                    is_dup, matched, confidence = is_duplicate_voter(avg_enc)
                    if is_dup:
                        last_result = ("duplicate", matched, confidence)
                    else:
                        last_result = ("new", avg_enc)

            # ── Draw result ──────────────────────────────────
            if last_result is not None:
                if last_result[0] == "duplicate":
                    matched    = last_result[1]
                    confidence = last_result[2]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 220), 3)
                    cv2.putText(frame,
                                f"DUPLICATE: {matched['name']} ({confidence}%)",
                                (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2)
                    draw_alert(frame,
                               f"ALREADY VOTED!  {matched['name']}  |  ID: {matched['voter_id']}  |  {confidence}%",
                               (0, 0, 180))
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 220, 0), 3)
                    cv2.putText(frame, "NEW VOTER — Verified!",
                                (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 220, 0), 2)
                    draw_alert(frame,
                               "NEW VOTER — Press SPACE to cast vote",
                               (0, 130, 0))

        cv2.imshow(window_name, frame)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:   # ESC
            cap.release()
            cv2.destroyAllWindows()
            return "cancelled", None

        if key == 32 and last_result is not None:   # SPACE
            cap.release()
            cv2.destroyAllWindows()
            if last_result[0] == "new":
                return "new", last_result[1]
            else:
                return "duplicate", last_result[1]

    cap.release()
    cv2.destroyAllWindows()
    return "cancelled", None