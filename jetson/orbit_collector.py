"""
orbit_collector.py
==================
MESAFE_KOR'da stabil kalınınca:
  1. Sağa/sola sweep yaparak farklı açılardan frame toplar
  2. Yaw controller'da kalır — düşman merkezde tutulur
  3. Tespit yoksa sweep dondurulur
  4. Frame tamamlanınca RTL komutu gönderir

Klasör yapısı:
  orbit_frames/
    enemy_YYYYMMDD_HHMMSS/
      images/   ← temiz frame (.jpg)
      labels/   ← YOLO etiket (.txt)
"""

import os
import time
import cv2
from dataclasses import dataclass
from enum import Enum, auto

import yaml
with open("config.yaml", "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

_oc = _cfg.get('orbit', {})

FRAME_COUNT            = _oc.get('frame_count', 30)
SAFE_MIN_PX            = _oc.get('safe_min_px', 50)
SAFE_MAX_PX            = _oc.get('safe_max_px', 115)
SAFE_IDEAL_PX          = _oc.get('safe_ideal_px', 86)
RTL_DELAY_S            = _oc.get('rtl_delay_s', 2.0)
FRAME_INTERVAL_S       = _oc.get('frame_interval_s', 1.5)
STABLE_FRAMES_REQUIRED = _oc.get('stable_frames_required', 8)
SAVE_DIR               = _oc.get('save_dir', 'orbit_frames')

SWEEP_DURATION_S = 4.0


class OrbitState(Enum):
    IDLE        = auto()
    STABILIZING = auto()
    SWEEP_RIGHT = auto()
    SWEEP_LEFT  = auto()
    DONE        = auto()

_vy = _oc.get('sweep_vy', 0.8)
SWEEP_COMMANDS = {
    OrbitState.SWEEP_RIGHT:  _vy,
    OrbitState.SWEEP_LEFT:  -_vy,
}

SWEEP_SEQUENCE = [
    OrbitState.SWEEP_RIGHT,
    OrbitState.SWEEP_LEFT,
]


@dataclass
class OrbitOutput:
    active:       bool       = False
    vx:           float      = 0.0
    vy:           float      = 0.0
    vz:           float      = 0.0
    frames_done:  int        = 0
    frames_total: int        = FRAME_COUNT
    state:        OrbitState = OrbitState.IDLE
    rtl_now:      bool       = False


class OrbitCollector:

    def __init__(self):
        self._state        = OrbitState.IDLE
        self._stable_count = 0
        self._frames_saved = 0
        self._last_frame_t = 0.0
        self._sweep_t      = 0.0
        self._img_dir      = ""
        self._lbl_dir      = ""
        self._done_t = 0.0
        os.makedirs(SAVE_DIR, exist_ok=True)

    def update(self, action: str, diameter_px: int,
               frame, dt: float = 0.05,
               bbox_xyxy=None, class_id: int = 0,
               img_w: int = 640, img_h: int = 640,
               target_id: str = "enemy") -> OrbitOutput:

        out = OrbitOutput(frames_total=FRAME_COUNT,
                          frames_done=self._frames_saved,
                          state=self._state)

        if self._state == OrbitState.DONE:
            out.rtl_now = False
            if time.time() - self._done_t >= RTL_DELAY_S:
                out.rtl_now = True
            return out

        # Orbit aktifken KAC/YAKLAS gelirse orbit'i koru
        if self._state in SWEEP_COMMANDS and action in ("KAC", "YAKLAS"):
            action = "MESAFE_KOR"

        in_safe_zone = SAFE_MIN_PX <= diameter_px <= SAFE_MAX_PX

        # ── IDLE / STABILIZING ────────────────────────────
        if self._state in (OrbitState.IDLE, OrbitState.STABILIZING):
            if action in ("MESAFE_KOR", "KAC") and in_safe_zone:
                if self._stable_count == 0 and not self._img_dir:
                    self._create_dirs(target_id)
                self._stable_count += 1
                self._state = OrbitState.STABILIZING

                # Stabilize olurken de frame kaydet
                now = time.time()
                if (frame is not None
                        and bbox_xyxy is not None
                        and (now - self._last_frame_t) >= FRAME_INTERVAL_S
                        and self._frames_saved < FRAME_COUNT):
                    self._save_frame(frame, bbox_xyxy, class_id, img_w, img_h)
                    self._last_frame_t = now

            elif action == "BEKLE":
                # Sadece BEKLE'de sıfırla — hedef tamamen kayboldu
                self._stable_count = 0
                self._state        = OrbitState.IDLE

            else:
                # YAKLAS veya diğeri — sayacı yavaş düşür, titreme yüzünden sıfırlanmasın
                self._stable_count = max(0, self._stable_count - 1)
                if self._stable_count == 0:
                    self._state   = OrbitState.IDLE

            if self._stable_count >= STABLE_FRAMES_REQUIRED:
                self._start_sweep()

            out.frames_done = self._frames_saved
            return out

        # ── SWEEP ─────────────────────────────────────────
        if self._state in SWEEP_COMMANDS:
            out.active = True

            if diameter_px == 0:
                out.vy = 0.0
                out.frames_done = self._frames_saved
                self._sweep_t += dt
                return out

            out.vy = SWEEP_COMMANDS[self._state]

            now = time.time()
            if (self._frames_saved < FRAME_COUNT
                    and frame is not None
                    and bbox_xyxy is not None
                    and (now - self._last_frame_t) >= FRAME_INTERVAL_S):
                self._save_frame(frame, bbox_xyxy, class_id, img_w, img_h)
                self._last_frame_t = now

            out.frames_done = self._frames_saved

            if time.time() - self._sweep_t >= SWEEP_DURATION_S:
                self._next_sweep()

            if self._frames_saved >= FRAME_COUNT:
                print(f"[OrbitCollector] {FRAME_COUNT} frame tamamlandi.")
                self._state    = OrbitState.DONE
                self._done_t   = time.time()
                out.state      = OrbitState.DONE

        return out

    # ── Yardımcılar ───────────────────────────────────────
    def _create_dirs(self, target_id: str):
        ts = time.strftime("%Y%m%d_%H%M%S")
        session       = os.path.join(SAVE_DIR, f"{target_id}_{ts}")
        self._img_dir = os.path.join(session, "images")
        self._lbl_dir = os.path.join(session, "labels")
        os.makedirs(self._img_dir, exist_ok=True)
        os.makedirs(self._lbl_dir, exist_ok=True)
        print(f"[OrbitCollector] Klasör oluşturuldu → {session}")

    def _start_sweep(self):
        self._sweep_t = time.time()
        self._state   = OrbitState.SWEEP_RIGHT
        print("[OrbitCollector] Sweep başladı")

    def _next_sweep(self):
        cur_idx = SWEEP_SEQUENCE.index(self._state)
        if cur_idx + 1 < len(SWEEP_SEQUENCE):
            self._state   = SWEEP_SEQUENCE[cur_idx + 1]
            self._sweep_t = time.time()
            print(f"[OrbitCollector] → {self._state.name}")
        else:
            # Sequence bitti ama frame yetmediyse başa dön
            if self._frames_saved < FRAME_COUNT:
                self._state   = SWEEP_SEQUENCE[0]
                self._sweep_t = time.time()
                print(f"[OrbitCollector] Döngü tekrar → {self._state.name}")
            else:
                self._state = OrbitState.DONE
                print("[OrbitCollector] Sweep tamamlandı.")

    def _save_frame(self, frame, bbox_xyxy, class_id, img_w, img_h):
        self._frames_saved += 1
        name = f"frame_{self._frames_saved:03d}"

        cv2.imwrite(os.path.join(self._img_dir, name + ".jpg"), frame)

        x1, y1, x2, y2 = bbox_xyxy
        cx = ((x1 + x2) / 2.0) / img_w
        cy = ((y1 + y2) / 2.0) / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h
        with open(os.path.join(self._lbl_dir, name + ".txt"), "w") as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        print(f"[OrbitCollector] Frame {self._frames_saved}/{FRAME_COUNT} kaydedildi")

    @property
    def is_done(self) -> bool:
        return self._state == OrbitState.DONE

    def reset(self):
        self._state        = OrbitState.IDLE
        self._stable_count = 0
        self._frames_saved = 0
        self._sweep_t      = 0.0
        self._img_dir      = ""
        self._lbl_dir      = ""