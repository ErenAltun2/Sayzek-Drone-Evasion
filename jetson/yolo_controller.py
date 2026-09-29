"""
yolo_controller.py  —  Gazebo modu (Profesyonel Sürüm)
=========================================================
İyileştirmeler:
    1. State Machine  — histerezis ile tutarlı karar
    2. EMA büyüme hızı — gürültüye dayanıklı
    3. Kalman Filter  — hedef pozisyon tahmini
    4. Geometrik kaçış — düşman vektörüne dik yön
    5. Mesafe tahmini — piksel → metre
    6. Yaw / vy ayrımı — titreme önlendi
"""

from collections import deque
from yolo_detector import DetectionResult
import math

import yaml
import os

# ── Config Yükleme ─────────────────────────────────────
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# ── Görüntü boyutları ─────────────────────────────────
IMAGE_W = cfg['kamera']['width']
IMAGE_H = cfg['kamera']['height']

# ── Hareket limitleri ─────────────────────────────────
MAX_VX = cfg['limitler']['max_vx']
MAX_VY = cfg['limitler']['max_vy']
MAX_VZ = cfg['limitler']['max_vz']
MAX_YAW_RATE = cfg['limitler']['max_yaw_rate']

# ── Tehdit eşikleri (piksel) ──────────────────────────
THREAT_PX          = cfg['tehdit_esikleri']['threat_px']
TARGET_DIAMETER_PX = cfg['tehdit_esikleri']['target_diameter_px']
ENEMY_TOO_CLOSE_PX = cfg['tehdit_esikleri']['enemy_too_close_px']
ENEMY_TOO_FAR_PX   = cfg['tehdit_esikleri']['enemy_too_far_px']
GROWTH_RATE_THREAT = cfg['tehdit_esikleri']['growth_rate_threat']
GROWTH_RATE_PANIC  = cfg['tehdit_esikleri']['growth_rate_panic']
CENTER_TOLERANCE_PX = cfg['tehdit_esikleri']['center_tolerance_px']

# ── Kontrol kazançları ────────────────────────────────
EVADE_GAIN    = cfg['kazanclar']['evade_gain']
GROWTH_GAIN   = cfg['kazanclar']['growth_gain']
LATERAL_GAIN  = cfg['kazanclar']['lateral_gain']
VERTICAL_GAIN = cfg['kazanclar']['vertical_gain']
YAW_GAIN      = cfg['kazanclar']['yaw_gain']
YAW_PANIC_DAMP = cfg['kazanclar']['yaw_panic_damp']

# ── Filtre ve State Parametreleri ─────────────────────
EMA_ALPHA            = cfg['filtreler']['ema_alpha']
GROWTH_HISTORY       = cfg['filtreler']['growth_history']
KF_PROCESS_NOISE     = cfg['filtreler']['kf_process_noise']
KF_MEAS_NOISE        = cfg['filtreler']['kf_meas_noise']
MIN_FRAMES_TO_CHANGE = cfg['state_machine']['min_frames_to_change']
LOST_TIMEOUT_FRAMES  = cfg['state_machine']['lost_timeout_frames']

# ── Mesafe tahmini (pinhole kamera modeli) ────────────
FOCAL_LENGTH_PX = cfg['kamera']['focal_length_px']
DRONE_WIDTH_M   = cfg['drone']['target_width_m']

# ══════════════════════════════════════════════════════
#  SIMPLE KALMAN FILTER (1D, her eksen için)
# ══════════════════════════════════════════════════════

class KalmanFilter1D:
    """
    Tek eksen için basit Kalman filtresi.
    Durum: [pozisyon, hız]
    """

    def __init__(self, q=KF_PROCESS_NOISE, r=KF_MEAS_NOISE):
        self.x  = 0.0   # pozisyon tahmini
        self.v  = 0.0   # hız tahmini
        self.p  = 1.0   # tahmin kovaryansı
        self.q  = q     # süreç gürültüsü
        self.r  = r     # ölçüm gürültüsü
        self._initialized = False

    def update(self, measurement: float, dt: float = 0.033) -> tuple[float, float]:
        """Ölçümü güncelle, (pozisyon, hız) döndür."""
        if not self._initialized:
            self.x = measurement
            self._initialized = True
            return self.x, 0.0

        # Predict
        x_pred = self.x + self.v * dt
        p_pred = self.p + self.q

        # Update
        k      = p_pred / (p_pred + self.r)
        self.x = x_pred + k * (measurement - x_pred)
        self.v = self.v + k * ((measurement - x_pred) / dt)
        self.p = (1.0 - k) * p_pred

        return self.x, self.v

    def predict(self, dt: float = 0.1) -> float:
        """dt saniye sonraki pozisyon tahmini."""
        return self.x + self.v * dt

    def reset(self):
        self.x = 0.0
        self.v = 0.0
        self.p = 1.0
        self._initialized = False


# ══════════════════════════════════════════════════════
#  CONTROL OUTPUT
# ══════════════════════════════════════════════════════

class ControlOutput:
    def __init__(self):
        self.vx       = 0.0
        self.vy       = 0.0
        self.vz       = 0.0
        self.yaw_rate = 0.0
        self.action   = "BEKLE"
        self.reason   = ""
        self.latency_ms = 0.0 

    def __str__(self):
        return (f"  Karar: {self.action}  "
                f"vx={self.vx:+.2f}  vy={self.vy:+.2f}  "
                f"vz={self.vz:+.2f}  yaw={self.yaw_rate:+.2f}\n"
                f"  {self.reason}")


# ══════════════════════════════════════════════════════
#  DRONE CONTROLLER
# ══════════════════════════════════════════════════════

class DroneController:

    def __init__(self):
        # EMA büyüme hızı
        self._ema_rate    = 0.0
        self._prev_diam   = None

        # State machine
        self._current_state    = "BEKLE"
        self._state_candidate  = "BEKLE"
        self._state_counter    = 0
        self._lost_counter     = 0

        # Kalman filtreleri (x ve y eksen için)
        self._kf_x = KalmanFilter1D()
        self._kf_y = KalmanFilter1D()

        # Geometrik kaçış için son hız vektörü
        self._escape_side = 1.0   # +1 veya -1
        self._escape_flip_t = 0.0

        import time
        self._last_t = time.time()

        # Yeni Eklenecek: Son hız komutlarını hafızada tutmak için
        self._last_out = ControlOutput()

    def _update_ema(self, diameter: float) -> float:
        """EMA ile düzgünleştirilmiş büyüme hızı."""
        if self._prev_diam is None:
            self._prev_diam = diameter
            return 0.0
        instant_rate    = diameter - self._prev_diam
        self._prev_diam = diameter
        self._ema_rate  = EMA_ALPHA * instant_rate + (1.0 - EMA_ALPHA) * self._ema_rate
        return self._ema_rate

    def _estimate_distance(self, diameter_px: float) -> float:
        """Piksel çapından metre cinsinden mesafe tahmini."""
        if diameter_px < 1:
            return 999.0
        return (FOCAL_LENGTH_PX * DRONE_WIDTH_M) / diameter_px

    def _update_state(self, candidate: str) -> str:
        # KAC acil durum — histerezis bekleme
        if candidate == "KAC":
            self._current_state   = "KAC"
            self._state_candidate = "KAC"
            self._state_counter   = MIN_FRAMES_TO_CHANGE
            return "KAC"

        if candidate == self._state_candidate:
            self._state_counter += 1
        else:
            self._state_candidate = candidate
            self._state_counter   = 1

        if self._state_counter >= MIN_FRAMES_TO_CHANGE:
            self._current_state = self._state_candidate

        return self._current_state

    def _geometric_escape_vy(self, error_x: float, panic_level: float) -> float:
        """
        Geometrik kaçış: düşman hangi taraftan geliyorsa
        karşı tarafa yanal hız üret.
        Düşman tam önde ise _escape_side yönüne git.
        """
        if abs(error_x) > CENTER_TOLERANCE_PX:
            # Düşman sağda → sola kaç, solda → sağa kaç
            direction = -1.0 if error_x > 0 else 1.0
        else:
            direction = self._escape_side

        return direction * LATERAL_GAIN * panic_level * min(1.2, 1.0 + panic_level)

    def compute(self, result: DetectionResult) -> ControlOutput:
        import time
        now = time.time()
        dt  = max(now - self._last_t, 1e-3)
        self._last_t = now

        out = ControlOutput()

# ── Tespit yok ────────────────────────────────
        if not result.detected:
            self._lost_counter += 1
            if self._lost_counter >= LOST_TIMEOUT_FRAMES:
                self._ema_rate = 0.0
                self._prev_diam = None
                self._kf_x.reset()
                self._kf_y.reset()
                self._update_state("BEKLE")
                
                # Tamamen dur
                self._last_out = ControlOutput() 
                out = self._last_out
                out.reason = f"Drone gorunmuyor ({self._lost_counter} frame)"
            else:
                # Kısa kayıp: Aniden durmak yerine hızları %20 oranında sönümlendir (Decay)
                self._last_out.vx *= 0.8
                self._last_out.vy *= 0.8
                self._last_out.vz *= 0.8
                self._last_out.yaw_rate *= 0.8
                
                out.vx = self._last_out.vx
                out.vy = self._last_out.vy
                out.vz = self._last_out.vz
                out.yaw_rate = self._last_out.yaw_rate
                
                out.action = self._current_state
                out.reason = f"Kisa kayip sönümlendirme ({self._lost_counter}/{LOST_TIMEOUT_FRAMES})"
            return out

        self._lost_counter = 0

        # ── Dost drone ────────────────────────────────
        if result.is_friendly:
            self._prev_diam = None
            self._ema_rate  = 0.0
            self._kf_x.reset()
            self._kf_y.reset()
            self._update_state("BEKLE")
            out.reason = f"Dost ({result.class_name}) — tepki yok"
            return out

        # ── Düşman drone veya tanımlanamayan drone ıcın ─────────────────────────────
        if result.is_enemy or result.is_orbit:
            diameter = result.bbox_diameter
            error_x  = result.error_x
            error_y  = result.error_y

            # Kalman ile düzgünleştirilmiş pozisyon
            kf_ex, vx_kf = self._kf_x.update(error_x, dt)
            kf_ey, vy_kf = self._kf_y.update(error_y, dt)
            if abs(vx_kf) > 1.5:
                self._escape_side = -1.0 if vx_kf > 0 else 1.0
            else:
                # Düşman lateral hareket etmiyorsa her 2 saniyede bir taraf değiştir
                now = time.time()
                if now - self._escape_flip_t > 2.0:
                    self._escape_side  *= -1.0
                    self._escape_flip_t = now

            # Gelecek pozisyon tahmini (0.1s ilerisi)
            pred_ex = self._kf_x.predict(dt=0.1)
            pred_ey = self._kf_y.predict(dt=0.1)

            # EMA büyüme hızı
            rate = self._update_ema(diameter)

            # Mesafe tahmini
            dist_m = self._estimate_distance(diameter)

            # Panik seviyesi
            panic_level = min(max(rate, 0.0) / GROWTH_RATE_PANIC, 1.0)

            # Erken uyarı: growth_rate yüksek ama EMA henüz ısınmadıysa
            # anlık oranı da panic hesabına kat
            instant_rate = diameter - (self._prev_diam or diameter)
            instant_panic = min(max(instant_rate, 0.0) / GROWTH_RATE_PANIC, 1.0)
            panic_level = max(panic_level, instant_panic * 0.7)  # %70 ağırlıkla

            # ── Candidate state belirle ───────────────
            if rate > GROWTH_RATE_PANIC or diameter >= THREAT_PX:
                candidate = "KAC"
            elif result.too_close:
                candidate = "KAC"
            elif result.too_far:
                candidate = "YAKLAS"
            else:
                candidate = "MESAFE_KOR"

            # State machine geçişi
            stable_state = self._update_state(candidate)
            out.action   = stable_state

            # ── Vx: ileri-geri ───────────────────────
            # satır ~299, size_ratio hesabı
            if stable_state == "KAC":
                size_ratio = (diameter - TARGET_DIAMETER_PX) / max(TARGET_DIAMETER_PX, 1)
                # Çap küçükse (uzaktaysa) size_ratio negatif — ileri itebilir, engelle
                size_ratio = max(size_ratio, 0.0)
                size_vx    = -EVADE_GAIN * size_ratio
                growth_vx  = -GROWTH_GAIN * max(rate, 0.0)
                panic_vx   = -MAX_VX * panic_level * 0.5
                out.vx     = max(size_vx + growth_vx + panic_vx, -MAX_VX)

            elif stable_state == "YAKLAS":
                # Gözlem için yaklaş — yavaş ve kontrollü
                ratio  = (TARGET_DIAMETER_PX - diameter) / max(TARGET_DIAMETER_PX, 1)
                out.vx = min(EVADE_GAIN * 0.4 * ratio, MAX_VX * 0.3)  # max 2.7 m/s

            elif stable_state == "MESAFE_KOR":
                # Gözlem mesafesinde: tamamen durma, hafif geri drift
                # Böylece düşman yaklaşırsa biraz yer açılmış olur
                size_ratio = (diameter - TARGET_DIAMETER_PX) / max(TARGET_DIAMETER_PX, 1)
                out.vx = max(-EVADE_GAIN * 0.3 * size_ratio, -MAX_VX * 0.2)

            else:
                out.vx = 0.0

            # ── Yaw + Vy: yatay ──────────────────────
            use_ex = pred_ex

            if stable_state == "KAC":
                ratio_x = use_ex / (IMAGE_W / 2.0)

                if panic_level > 0.7:
                    out.yaw_rate = max(min(YAW_GAIN * 0.5 * ratio_x, MAX_YAW_RATE), -MAX_YAW_RATE)
                    # Lateral hız varsa o yönün tersine, yoksa escape_side'a tam gaz
                    escape_dir = self._escape_side
                    out.vy = max(min(escape_dir * MAX_VY, MAX_VY), -MAX_VY)
                    out.vx = max(out.vx * 0.4, -MAX_VX * 0.4)  # geriye freni azalt, yana öncelik

                elif panic_level > 0.3:
                    # Orta panik: yaw + orta yana
                    yaw_scale = 1.0 - panic_level * (1.0 - YAW_PANIC_DAMP)
                    out.yaw_rate = max(min(YAW_GAIN * yaw_scale * ratio_x, MAX_YAW_RATE), -MAX_YAW_RATE)
                    out.vy = max(min(-LATERAL_GAIN * panic_level * ratio_x, MAX_VY), -MAX_VY)

                else:
                    # Düşük panik: yaw ile ortala, hafif yana
                    out.yaw_rate = max(min(YAW_GAIN * ratio_x, MAX_YAW_RATE), -MAX_YAW_RATE)
                    escape_dir = -1.0 if error_x > 0 else (1.0 if error_x < 0 else self._escape_side)
                    out.vy = max(min(escape_dir * LATERAL_GAIN * 0.4, MAX_VY), -MAX_VY)

            elif abs(use_ex) > CENTER_TOLERANCE_PX:
                ratio_x = use_ex / (IMAGE_W / 2.0)
                out.yaw_rate = max(min(YAW_GAIN * ratio_x, MAX_YAW_RATE), -MAX_YAW_RATE)
                out.vy = 0.0
            else:
                out.yaw_rate = 0.0
                out.vy = 0.0

            # ── Vz: dikey ────────────────────────────
            # Log analizi: error_y küçükken panik -2.0 ekliyordu → drone yere iniyordu
            # Düzeltme: panik modunda dikey kaçış YUKARI (negatif vz = yukarı) ve
            # sadece error_y tolerans dışındaysa hizalama yap.
            use_ey = pred_ey   # Kalman tahmini

            if abs(use_ey) > CENTER_TOLERANCE_PX:
                # Hedef kamera merkezinden dikey sapmış → hizala
                ratio_y        = use_ey / (IMAGE_H / 2.0)
                vertical_boost = 1.0 + panic_level * 0.5   # önceki 0.8 çok agresifti
                out.vz = max(min(
                    VERTICAL_GAIN * vertical_boost * ratio_y,
                    MAX_VZ), -MAX_VZ)
            elif panic_level > 0.6 and stable_state == "KAC":
                # Hedef merkezde ama panik var → YUKARI kaç (irtifa koru)
                # Önceki hata: -MAX_VZ * panic = aşağı gidiyordu!
                out.vz = -MAX_VZ * panic_level * 0.4  # NED: negatif = yukarı
            else:
                out.vz = 0.0

            # ── Reason ───────────────────────────────
            if stable_state == "KAC":
                if rate > GROWTH_RATE_PANIC:
                    out.reason = (f"PANIK  buyume:{rate:+.1f}px/f  "
                                  f"cap:{diameter}px  dist:{dist_m:.1f}m  "
                                  f"panik:{panic_level:.2f}")
                else:
                    out.reason = (f"Tehdit  buyume:{rate:+.1f}px/f  "
                                  f"cap:{diameter}px  dist:{dist_m:.1f}m")
            elif stable_state == "YAKLAS":
                out.reason = f"Uzaklasti  cap:{diameter}px  dist:{dist_m:.1f}m"
            elif stable_state == "MESAFE_KOR":
                out.reason = (f"Uygun mesafe  cap:{diameter}px  "
                              f"dist:{dist_m:.1f}m  buyume:{rate:+.1f}px/f")
            else:
                out.reason = "Stabil"

            # State machine geçiş bilgisi
            if self._state_counter < MIN_FRAMES_TO_CHANGE:
                out.reason += f"  [gecis:{self._state_counter}/{MIN_FRAMES_TO_CHANGE}]"
            if hasattr(result, 't_capture') and result.t_capture > 0:
                out.latency_ms = (time.time() - result.t_capture) * 1000
        self._last_out = out  # Güncel komutu hafızaya al
        return out
