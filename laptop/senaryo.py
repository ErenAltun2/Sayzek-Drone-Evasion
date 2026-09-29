"""
senaryo.py  —  Gazebo modu (Geliştirilmiş Sürüm)
==================================================
Mimari:
    iris_1 (14550) → Kaçan drone  → thread_manager.py yönetir
    iris_2 (14560) → Takip eden   → YOLO + PID kovalar

Çalıştırma:
    Terminal 1: python senaryo.py
    Terminal 2 (iris_1 havadayken): python thread_manager.py --drone

İyileştirmeler:
    - CameraTracker ayrı thread'de YOLO + PID (loop tıkanmaz)
    - PID anti-windup
    - Hedef kaybolunca akıllı arama dönüşü
    - Daha zengin HUD (hız çubukları, durum geçmişi)
    - Exception handling her aşamada
    - Takeoff paralel ve hata toleranslı
"""

import time
import math
import threading
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
from ultralytics import YOLO

from dronekit import connect, VehicleMode
from pymavlink import mavutil

# ══════════════════════════════════════════════
#  AYARLAR
# ══════════════════════════════════════════════

IRIS1_ADDR  = "udp:127.0.0.1:14550"
IRIS2_ADDR  = "udp:127.0.0.1:14560"
TARGET_ALT  = 10           # metre
PHASE1_DIST = 5           # metre

CAM_TOPIC   = "/iris_2/camera/image"
YOLO_MODEL  = "Yolo26-640.engine"

FRAME_W = 640
FRAME_H = 480

# Alan eşikleri
TARGET_AREA_MIN = 5_000
TARGET_AREA_MAX = 45_000

# PID katsayıları
KP_YAW = 0.003;  KI_YAW = 0.0001;  KD_YAW = 0.002
KP_VZ  = 0.005;  KI_VZ  = 0.0001;  KD_VZ  = 0.002
KP_VX  = 0.0001

MAX_YAW_RATE   = 1.2     # rad/s
MAX_VZ         = 0.8     # m/s
MAX_VX         = 2.5    # m/s — biraz daha agresif kovalama

# Hedef kaybolunca arama dönüşü
SEARCH_YAW_RATE   = 0.4   # rad/s
SEARCH_TIMEOUT_S  = 3.0   # bu kadar kayıptan sonra aramaya başla
LOST_LAND_TIMEOUT = 15.0  # bu kadar sonra hover (kovalama bitti sayılır)


# ══════════════════════════════════════════════
#  GENEL YARDIMCILAR
# ══════════════════════════════════════════════

def arm_and_takeoff(vehicle, name, altitude, timeout_arm=40, timeout_takeoff=60):
    """
    Drone'u arm eder ve hedef irtifaya çıkarır.
    Hata durumunda False döner.
    """
    print(f"[{name}] Pre-arm bekleniyor...")
    t0 = time.time()
    while not vehicle.is_armable:
        if time.time() - t0 > timeout_arm:
            print(f"[{name}] HATA: Pre-arm zaman aşımı!")
            return False
        time.sleep(1)

    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    t0 = time.time()
    while not vehicle.armed:
        if time.time() - t0 > 10:
            print(f"[{name}] HATA: Arm zaman aşımı!")
            return False
        time.sleep(0.5)

    print(f"[{name}] Havalanıyor → {altitude}m")
    vehicle.simple_takeoff(altitude)

    t0 = time.time()
    while True:
        if time.time() - t0 > timeout_takeoff:
            print(f"[{name}] HATA: Takeoff zaman aşımı!")
            return False
        alt = vehicle.location.global_relative_frame.alt
        print(f"[{name}]  irtifa: {alt:.1f}m / {altitude}m")
        if alt >= altitude * 0.92:
            print(f"[{name}] Hedef irtifaya ulaşıldı ✓")
            break
        time.sleep(1)
    return True


def send_velocity_yaw_rate(vehicle, vx, vy, vz, yaw_rate=0.0):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000010111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, yaw_rate
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


def hover(vehicle, duration):
    t_end = time.time() + duration
    while time.time() < t_end:
        send_velocity_yaw_rate(vehicle, 0, 0, 0, 0)
        time.sleep(0.1)


def condition_yaw(vehicle, heading_deg, relative=True, timeout=8):
    is_relative = 1 if relative else 0
    direction   = 1 if heading_deg >= 0 else -1
    msg = vehicle.message_factory.command_long_encode(
        0, 0,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,
        abs(heading_deg), 30, direction, is_relative,
        0, 0, 0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()
    time.sleep(timeout)


def move_forward(vehicle, name, distance, speed=2.5, timeout=25):
    start    = vehicle.location.global_relative_frame
    lat0, lon0 = start.lat, start.lon
    print(f"[{name}] {distance}m ileri gidiyor ({speed} m/s)...")

    t_end = time.time() + timeout
    while time.time() < t_end:
        cur  = vehicle.location.global_relative_frame
        dlat = (cur.lat - lat0) * 111_320.0
        dlon = (cur.lon - lon0) * 111_320.0 * math.cos(math.radians(lat0))
        dist = math.sqrt(dlat**2 + dlon**2)

        if dist >= distance * 0.90:
            print(f"[{name}] {distance}m hedefine ulaşıldı ✓")
            hover(vehicle, 0.5)
            return True

        send_velocity_yaw_rate(vehicle, speed, 0, 0, 0)
        time.sleep(0.4)

    print(f"[{name}] UYARI: {distance}m'ye ulaşılamadı.")
    hover(vehicle, 0.5)
    return False


# ══════════════════════════════════════════════
#  PID
# ══════════════════════════════════════════════

class PID:
    def __init__(self, kp, ki, kd, out_min=-1e9, out_max=1e9,
                 windup_limit=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.windup_limit = windup_limit or (abs(out_max) / max(ki, 1e-9))
        self._integral = 0.0
        self._prev_err = 0.0
        self._prev_t   = None

    def update(self, error):
        now = time.time()
        dt  = max(now - self._prev_t, 1e-4) if self._prev_t else 0.05
        self._prev_t = now

        # Anti-windup: integral'i sınırla
        self._integral = max(
            -self.windup_limit,
            min(self.windup_limit, self._integral + error * dt)
        )

        deriv          = (error - self._prev_err) / dt
        self._prev_err = error

        out = self.kp * error + self.ki * self._integral + self.kd * deriv
        return max(self.out_min, min(self.out_max, out))

    def reset(self):
        self._integral = 0.0
        self._prev_err = 0.0
        self._prev_t   = None


# ══════════════════════════════════════════════
#  CAMERA TRACKER (ROS2 Node)
# ══════════════════════════════════════════════

class CameraTracker(Node):
    """
    iris_2'nin kamerasından YOLO ile iris_1'i tespit eder,
    PID kontrolü ile kovalar.
    """

    def __init__(self, vehicle):
        super().__init__('iris2_tracker_node')
        self.vehicle = vehicle
        self.bridge  = CvBridge()
        self.model   = YOLO(YOLO_MODEL, task='detect')

        self.pid_yaw = PID(KP_YAW, KI_YAW, KD_YAW,
                           out_min=-MAX_YAW_RATE, out_max=MAX_YAW_RATE,
                           windup_limit=MAX_YAW_RATE * 2)
        self.pid_vz  = PID(KP_VZ, KI_VZ, KD_VZ,
                           out_min=-MAX_VZ, out_max=MAX_VZ,
                           windup_limit=MAX_VZ * 2)

        self._latest_frame  = None
        self._display_frame = None
        self._lock          = threading.Lock()

        self._last_detect_t  = None
        self._detect_count   = 0    # başarılı tespit sayısı
        self._lost_count     = 0    # kayıp frame sayısı
        self._searching      = False

        self.cx0 = FRAME_W / 2.0
        self.cy0 = FRAME_H / 2.0

        # Durum geçmişi (HUD için)
        self._status_history = []

        self.subscription = self.create_subscription(
            Image, CAM_TOPIC, self._image_callback, 10
        )

    def _image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._lock:
                self._latest_frame = frame
        except Exception as e:
            self.get_logger().warn(f"Frame dönüşüm hatası: {e}")

    def get_display_frame(self):
        with self._lock:
            return self._display_frame.copy() if self._display_frame is not None else None

    def _detect(self, frame):
        """En iyi tespiti döndür: (cx, cy, area, x1, y1, x2, y2, conf)"""
        try:
            results   = self.model(frame, imgsz=640, device=0, verbose=False)[0]
            best      = None
            best_area = 0

            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < 0.35:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    best = (cx, cy, area, x1, y1, x2, y2, conf)

            return best
        except Exception as e:
            self.get_logger().warn(f"Tespit hatası: {e}")
            return None

    def _send_commands(self, cx, cy, area) -> tuple[float, float, float]:
        """PID ile MAVLink komutu üret, (vx, vz, yaw_rate) döndür."""
        err_x    = cx - self.cx0
        yaw_rate = self.pid_yaw.update(err_x)

        # Kademeli hız: alan küçük → hızlan, büyük → yavaşla
        if area < TARGET_AREA_MIN:
            vx = MAX_VX
        elif area < TARGET_AREA_MAX:
            ratio = 1.0 - ((area - TARGET_AREA_MIN) /
                           (TARGET_AREA_MAX - TARGET_AREA_MIN))
            vx = 0.5 + ratio * (MAX_VX - 0.5)
        else:
            vx = -0.5 if area > TARGET_AREA_MAX * 1.2 else 0.0

        # Merkezden çok saptıysa önce ortala
        if abs(err_x) > FRAME_W * 0.2:
            vx *= 0.5

        err_y = cy - self.cy0
        vz    = self.pid_vz.update(err_y)
        vz    = max(min(vz, MAX_VZ), -MAX_VZ * 0.3)

        send_velocity_yaw_rate(self.vehicle,
                               vx=vx, vy=0.0, vz=vz,
                               yaw_rate=yaw_rate)
        return vx, vz, yaw_rate

    def _search_rotation(self):
        """Hedef kaybolunca yavaşça dönerek ara."""
        send_velocity_yaw_rate(self.vehicle,
                               vx=0.0, vy=0.0, vz=0.0,
                               yaw_rate=SEARCH_YAW_RATE)

    def _draw_hud(self, frame, detection, vx, vz, yaw_rate):
        h, w = frame.shape[:2]

        # Merkez artı
        cv2.drawMarker(frame, (w // 2, h // 2),
                       (255, 255, 255), cv2.MARKER_CROSS, 20, 1)

        if detection:
            cx, cy, area, x1, y1, x2, y2, conf = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cx_i, cy_i     = int(cx), int(cy)

            # Tespit kutusu
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.circle(frame, (cx_i, cy_i), 5, (0, 165, 255), -1)
            cv2.putText(frame, f"HEDEF  {conf:.2f}  {area:.0f}px²",
                        (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 1)

            # Merkeze çizgi
            cv2.line(frame, (w // 2, h // 2),
                     (cx_i, cy_i), (0, 255, 255), 1)

            # Hata çubuğu — yatay
            bar_w = int((cx - self.cx0) / (w / 2) * (w // 4))
            cv2.rectangle(frame,
                          (w // 2, 10), (w // 2 + bar_w, 22),
                          (0, 200, 255), -1)
            cv2.putText(frame, f"ex={cx - self.cx0:+.0f}px",
                        (w // 2 + 5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        # Alt telemetri bandı
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        if detection:
            status = "KOVALAMA"
            color  = (0, 165, 255)
        elif self._searching:
            status = "ARAMA"
            color  = (0, 200, 255)
        else:
            status = "HEDEF YOK"
            color  = (160, 160, 160)

        cv2.putText(frame,
                    f"{status}   vx={vx:+.2f}  vz={vz:+.2f}  yaw={yaw_rate:+.2f} rad/s",
                    (10, h - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)

        cv2.putText(frame,
                    f"Tespit:{self._detect_count}  Kayip:{self._lost_count}  "
                    f"iris_2 | YOLO Takip",
                    (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 120), 1)

        return frame

    def _loop(self, stop_event, hz=20):
        """Ana kontrol döngüsü — ayrı thread'de çalışır."""
        interval = 1.0 / hz
        print("\n[CameraTracker] Döngü başladı. Hedef aranıyor...")

        while not stop_event.is_set():
            t0 = time.time()

            with self._lock:
                frame = self._latest_frame.copy() \
                    if self._latest_frame is not None else None

            if frame is None:
                time.sleep(interval)
                continue

            detection = self._detect(frame)
            vx = vz = yaw_rate = 0.0

            if detection:
                self._last_detect_t  = time.time()
                self._detect_count  += 1
                self._lost_count     = 0
                self._searching      = False
                self.pid_yaw.reset() if self._searching else None

                cx, cy, area = detection[0], detection[1], detection[2]
                vx, vz, yaw_rate = self._send_commands(cx, cy, area)

                print(f"\r[Tracker] cx={cx:4.0f} cy={cy:4.0f} "
                      f"alan={area:6.0f}px²  conf={detection[7]:.2f}  "
                      f"yaw={yaw_rate:+.2f}  vx={vx:+.2f}  vz={vz:+.2f}  "
                      f"tespit={self._detect_count}", end="")
            else:
                self._lost_count += 1
                lost_s = time.time() - (self._last_detect_t or time.time())

                if lost_s > LOST_LAND_TIMEOUT:
                    # Çok uzun süre kayıp → hover
                    print(f"\r[CameraTracker] Hedef {lost_s:.0f}s kayıp → hover", end="")
                    send_velocity_yaw_rate(self.vehicle, 0, 0, 0, 0)
                    self._searching = False

                elif lost_s > SEARCH_TIMEOUT_S:
                    # Arama dönüşü başlat
                    self._searching = True
                    self._search_rotation()
                    print(f"\r[CameraTracker] Hedef {lost_s:.1f}s kayıp → aranıyor...", end="")

                else:
                    # Kısa kayıp → bekle
                    send_velocity_yaw_rate(self.vehicle, 0, 0, 0, 0)

            # HUD
            hud = self._draw_hud(frame, detection, vx, vz, yaw_rate)
            with self._lock:
                self._display_frame = hud

            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))

    def start(self, stop_event):
        self._thread = threading.Thread(
            target=self._loop, args=(stop_event,),
            name="CameraTracker-Loop", daemon=True
        )
        self._thread.start()

    def join(self, timeout=3.0):
        if hasattr(self, '_thread'):
            self._thread.join(timeout=timeout)


# ══════════════════════════════════════════════
#  ANA AKIŞ
# ══════════════════════════════════════════════

def main():
    rclpy.init(args=sys.argv)

    print("=" * 56)
    print("  SENARYO: YOLO Kamera Takip (iris_2 kovalar)")
    print("=" * 56)

    # ── Bağlantı ──────────────────────────────
    print("\n[iris_1] Bağlanıyor...")
    try:
        v1 = connect(IRIS1_ADDR, wait_ready=True, timeout=60)
    except Exception as e:
        print(f"[iris_1] HATA: {e}")
        rclpy.shutdown()
        return

    print("[iris_2] Bağlanıyor...")
    try:
        v2 = connect(IRIS2_ADDR, wait_ready=True, timeout=60)
    except Exception as e:
        print(f"[iris_2] HATA: {e}")
        v1.close()
        rclpy.shutdown()
        return

    # ── Takeoff (paralel) ─────────────────────
    results = [False, False]

    def do_takeoff_1():
        results[0] = arm_and_takeoff(v1, "iris_1", TARGET_ALT)

    def do_takeoff_2():
        results[1] = arm_and_takeoff(v2, "iris_2", TARGET_ALT)

    th1 = threading.Thread(target=do_takeoff_1)
    th2 = threading.Thread(target=do_takeoff_2)
    th1.start(); th2.start()
    th1.join();  th2.join()

    if not all(results):
        print("Takeoff başarısız! Çıkılıyor.")
        v1.close(); v2.close()
        rclpy.shutdown()
        return

    print("\n✓ Her iki drone havada!")
    print("thread_manager.py'yi başlatmak için 10 saniye var...")
    print("  → Terminal 2: python thread_manager.py --drone\n")
    time.sleep(10)

    # ── Faz 1: iris_2 ileri git + 180° dön ───
    print("\n[iris_2] ── FAZ 1: İleri gidiyor ──")
    move_forward(v2, "iris_2", distance=PHASE1_DIST, speed=1.0)
    hover(v2, 1.0)

    print("[iris_2] 180° dönüyor (iris_1'e bakıyor)...")
    condition_yaw(v2, heading_deg=180, relative=True, timeout=8)
    print("[iris_2] Dönüş tamamlandı ✓")
    hover(v2, 1.0)

    # ── Faz 2: YOLO takip ─────────────────────
    tracker = CameraTracker(v2)
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(tracker,), daemon=True
    )
    spin_thread.start()

    print("\n[iris_2] ── FAZ 2: YOLO + PID takip başlıyor ──")
    print("Kamera penceresi açılıyor... ('q' ile çık)\n")

    stop_event = threading.Event()
    tracker.start(stop_event)

    cv2.namedWindow("iris_2 | YOLO Kamera Takip")
    cv2.resizeWindow("iris_2 | YOLO Kamera Takip", FRAME_W, FRAME_H)

    try:
        while not stop_event.is_set():
            frame = tracker.get_display_frame()
            if frame is not None:
                cv2.imshow("iris_2 | YOLO Kamera Takip", frame)

            key = cv2.waitKey(33) & 0xFF   # ~30 FPS display
            if key == ord('q'):
                print("\n[Senaryo] Kullanıcı durdurdu.")
                break

    except KeyboardInterrupt:
        print("\n[Senaryo] KeyboardInterrupt.")
    finally:
        stop_event.set()
        tracker.join(timeout=3.0)
        cv2.destroyAllWindows()

    # ── İniş ──────────────────────────────────
    print("\nDrone'lar indiriliyor...")
    try:
        v2.mode = VehicleMode("LAND")
    except Exception as e:
        print(f"[iris_2] İniş hatası: {e}")
    try:
        v1.mode = VehicleMode("LAND")
    except Exception as e:
        print(f"[iris_1] İniş hatası: {e}")

    time.sleep(3)

    try:
        tracker.destroy_node()
    except Exception:
        pass

    v1.close()
    v2.close()
    rclpy.shutdown()
    print("Tamamlandı.")


if __name__ == "__main__":
    main()
