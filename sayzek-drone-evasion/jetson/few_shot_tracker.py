"""
few_shot_tracker.py
===================
Genel drone sınıfı için kimlik takibi.

İlk birkaç frame'de görsel imza (renk histogramı) oluşturur.
Drone kaybolup tekrar görününce:
  1. Önce bbox pozisyon + boyut benzerliğine bakar (hızlı)
  2. Görsel imza benzerliğine bakar (güvenilir)
  3. İkisi de tutarsa aynı drone der

Ekstra kütüphane gerekmez — sadece OpenCV.
"""

import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Eşikler ───────────────────────────────────────────────
POS_TOLERANCE_PX   = 250   # merkez kayması (piksel)
SIZE_TOLERANCE_PCT = 0.70  # boyut farkı oranı
LOST_TIMEOUT_S     = 8.0   # kayıp sayılma süresi
MIN_FRAMES_FOR_SIG  = 3      # görsel imza için min frame sayısı
HIST_MATCH_THRESH  = 0.35   # histogram benzerlik eşiği (0-1, yüksek=benzer)


def _extract_signature(frame, bbox_xyxy) -> Optional[np.ndarray]:
    """
    Bbox bölgesinden renk histogramı çıkarır.
    HSV uzayında H ve S kanalı — ışık değişimine daha dayanıklı.
    """
    if frame is None or bbox_xyxy is None:
        return None

    x1, y1, x2, y2 = bbox_xyxy
    h, w = frame.shape[:2]

    # Sınır kontrolü
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # H: 0-180, S: 0-256 — 32 bin yeterli
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])

    hist = np.concatenate([hist_h, hist_s], axis=0)
    cv2.normalize(hist, hist)
    return hist.flatten()


def _compare_signatures(sig1: np.ndarray,
                         sig2: np.ndarray) -> float:
    """Korelasyon benzerliği — 1.0 = tam eşleşme."""
    return cv2.compareHist(
        sig1.reshape(-1, 1).astype(np.float32),
        sig2.reshape(-1, 1).astype(np.float32),
        cv2.HISTCMP_CORREL
    )


@dataclass
class TrackedDrone:
    drone_id:    str
    center:      tuple
    diameter:    float
    last_seen:   float        = field(default_factory=time.time)
    seen_count:  int          = 0
    signatures:  list         = field(default_factory=list)  # biriken histogramlar
    mean_sig:    Optional[np.ndarray] = None                 # ortalama imza


class FewShotTracker:

    def __init__(self):
        self._drones:  list[TrackedDrone] = []
        self._next_id: int = 1

    def update(self, center: tuple, diameter: float,
               frame=None, bbox_xyxy=None) -> str:
        """
        Tespit edilen drone'un ID'sini döndürür.
        frame + bbox_xyxy verilirse görsel imza da güncellenir.
        """
        now = time.time()

        # Kaybolmuş drone'ları temizle
        self._drones = [d for d in self._drones
                        if now - d.last_seen < LOST_TIMEOUT_S]

        match = self._find_match(center, diameter, frame, bbox_xyxy)

        if match is not None:
            # Aynı drone — güncelle
            match.center     = center
            match.diameter   = diameter
            match.last_seen  = now
            match.seen_count += 1

            # Görsel imza biriktir
            if frame is not None and bbox_xyxy is not None:
                sig = _extract_signature(frame, bbox_xyxy)
                if sig is not None and len(match.signatures) < 10:
                    match.signatures.append(sig)
                    # Ortalama imzayı güncelle
                    match.mean_sig = np.mean(match.signatures, axis=0)

            return match.drone_id

        else:
            # Yeni drone
            drone_id = f"drone_{self._next_id}"
            self._next_id += 1

            drone = TrackedDrone(
                drone_id   = drone_id,
                center     = center,
                diameter   = diameter,
                seen_count = 1
            )

            # İlk imzayı al
            if frame is not None and bbox_xyxy is not None:
                sig = _extract_signature(frame, bbox_xyxy)
                if sig is not None:
                    drone.signatures.append(sig)
                    drone.mean_sig = sig

            self._drones.append(drone)
            print(f"[FewShotTracker] Yeni drone: {drone_id}  "
                  f"center={center}  diameter={diameter:.0f}px")
            return drone_id

    def reset(self):
        self._drones  = []
        self._next_id = 1

    def status(self) -> str:
        lines = []
        for d in self._drones:
            age = time.time() - d.last_seen
            lines.append(
                f"  {d.drone_id}: seen={d.seen_count} "
                f"sigs={len(d.signatures)} age={age:.1f}s"
            )
        return "\n".join(lines) if lines else "  (boş)"

    # ── Eşleştirme ────────────────────────────────────────
    def _find_match(self, center, diameter,
                    frame, bbox_xyxy) -> Optional[TrackedDrone]:
        cx, cy     = center
        best       = None
        best_score = float('inf')

        for drone in self._drones:
            dcx, dcy = drone.center

            pos_diff = ((cx - dcx) ** 2 + (cy - dcy) ** 2) ** 0.5
            if pos_diff > POS_TOLERANCE_PX:
                continue

            hist_score = 1.0
            if (drone.mean_sig is not None
                    and frame is not None
                    and bbox_xyxy is not None):
                sig = _extract_signature(frame, bbox_xyxy)
                if sig is not None:
                    hist_score = _compare_signatures(drone.mean_sig, sig)
                    if hist_score < HIST_MATCH_THRESH:
                        continue

            score = pos_diff + (1.0 - hist_score) * 50
            if score < best_score:
                best_score = score
                best       = drone

        return best
