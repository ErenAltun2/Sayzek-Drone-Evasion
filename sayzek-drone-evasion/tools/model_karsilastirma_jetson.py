"""
model-karsilastirma-jetson.py  —  Jetson AGX / Orin
=====================================================
Jetson, LAN kablosu üzerinden laptop'taki Gazebo'ya bağlanır.
Laptop IP : 10.42.0.1  (veya --laptop-ip ile değiştir)
Jetson IP : 10.42.0.67

Gereksinimler (laptop tarafında):
    MAVProxy forwarding açık olmalı:
        mavproxy.py --master=udp:127.0.0.1:14550 --out=udp:10.42.0.67:14550
        mavproxy.py --master=udp:127.0.0.1:14560 --out=udp:10.42.0.67:14560
    ROS2 DDS domain ID eşleşmeli (aynı ağda otomatik bulunur) VEYA
        export ROS_DOMAIN_ID=0  her iki tarafta da aynı olmalı

Thread yapısı (thread_manager.py stili):
    cam._spin_thread  → rclpy.spin  (CameraSource içinde başlar)
    T2  yolo_thread   → frame al → YOLO → annotate → SharedBuffer
    T3  drone_thread  → iris_2 hareket, GPS log
    T4  gpu_thread    → tegrastats log (Jetson'a özel)
    Ana → namedWindow + imshow + waitKey  (cv2 SADECE burada)

Çalıştırma:
    python model-karsilastirma-jetson.py --model Yolo26-640.engine
    python model-karsilastirma-jetson.py --model yolov8n.pt --max-dist 40
    python model-karsilastirma-jetson.py --model Yolo26-640.engine --no-display
    python model-karsilastirma-jetson.py --model yolov8n.pt --laptop-ip 10.42.0.1

Tuşlar:  q → çık  |  p → duraklat/devam  |  s → anlık ekran görüntüsü
"""

import argparse
import csv
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from dronekit import connect, VehicleMode
from pymavlink import mavutil
from ultralytics import YOLO


# ══════════════════════════════════════════════
#  AYARLAR  (--laptop-ip ile veya burada değiştir)
# ══════════════════════════════════════════════

DEFAULT_LAPTOP_IP  = "10.42.0.1"

# MAVLink adresleri — Jetson UDP portları
# Laptop'ta MAVProxy bu portlara forward etmeli
def build_addrs(laptop_ip: str):
    # Jetson kendi portlarını dinler (0.0.0.0)
    # Laptop MAVProxy bu portlara --out=udp:10.42.0.67:14550 ile gönderir
    return (
        "udp:0.0.0.0:14550",   # iris_1 — Jetson bu portu dinler
        "udp:0.0.0.0:14560",   # iris_2 — Jetson bu portu dinler
    )

# ROS2 kamera topic'i — laptop Gazebo'da aynı isim
CAM_TOPIC          = "/iris_1/camera/image/compressed"

YOLO_CONF          = 0.30
TARGET_ALT         = 10
SS_INTERVAL_M      = 2           # her kaç metrede otomatik screenshot

FRIENDLY_CLASSES   = {}
ENEMY_CLASSES      = {0, 1, 2, 3, 4, 5}
ENEMY_TOO_CLOSE_PX = 180
ENEMY_TOO_FAR_PX   = 80

# Tegrastats log aralığı (saniye) — Jetson güç/sıcaklık takibi
TEGRA_INTERVAL     = 0.5


# ══════════════════════════════════════════════
#  1. DRONE YARDIMCILARI
#  (drone_guard.py + model-karsilastirma.py'den)
# ══════════════════════════════════════════════

def arm_and_takeoff(vehicle, name, altitude):
    print(f"[{name}] Pre-arm bekleniyor...")
    t0 = time.time()
    while not vehicle.is_armable:
        if time.time() - t0 > 40:
            print(f"[{name}] HATA: Pre-arm zaman aşımı!")
            return False
        time.sleep(1)
    vehicle.mode  = VehicleMode("GUIDED")
    vehicle.armed = True
    t0 = time.time()
    while not vehicle.armed:
        if time.time() - t0 > 10:
            print(f"[{name}] HATA: Arm zaman aşımı!")
            return False
        time.sleep(0.5)
    print(f"[{name}] Havalânıyor → {altitude}m")
    vehicle.simple_takeoff(altitude)
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"[{name}]  irtifa: {alt:.1f}m / {altitude}m")
        if alt >= altitude * 0.88:
            print(f"[{name}] Hedef irtifaya ulaşıldı")
            break
        time.sleep(1)
    return True


def send_velocity_yaw_rate(vehicle, vx, vy, vz, yaw_rate=0.0):
    """drone_guard.py / DroneGuard._send_velocity ile aynı."""
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000010111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, yaw_rate,
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


def hover(vehicle, duration):
    t_end = time.time() + duration
    while time.time() < t_end:
        send_velocity_yaw_rate(vehicle, 0, 0, 0, 0)
        time.sleep(0.1)


def get_gps(vehicle):
    loc = vehicle.location.global_relative_frame
    return loc.lat, loc.lon, loc.alt


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ══════════════════════════════════════════════
#  2. KAMERA KAYNAĞI
#  yolo_detector.py / GazeboSource + model-karsilastirma.py / CameraSource
#  ile BİREBİR AYNI pattern:
#    - rclpy lazy import (constructor içinde)
#    - rclpy.init() zaten yapılmışsa tekrar yapmaz
#    - spin thread CONSTRUCTOR İÇİNDE başlar (daemon)
#    - dış kodun rclpy.spin() çağırması GEREKMEZ
#  ROS2 DDS otomatik keşif sayesinde laptop Gazebo topic'ini bulur.
# ══════════════════════════════════════════════

class CameraSource:
    """
    yolo_detector.py / GazeboSource ile birebir aynı:
    CompressedImage topic'i dinler, numpy ile decode eder.
    """
    def __init__(self, topic=CAM_TOPIC):
        import rclpy
        from sensor_msgs.msg import CompressedImage

        self._frame = None
        self._lock  = threading.Lock()

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node("model_kars_cam_node")
        self._node.create_subscription(CompressedImage, topic, self._cb, 10)

        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self._node,), daemon=True
        )
        self._spin_thread.start()
        print(f"[CameraSource] Topic: {topic}  |  spin thread başladı")

    def _cb(self, msg):
        import numpy as np
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                with self._lock:
                    self._frame = frame
        except Exception:
            pass

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def release(self):
        try:
            self._node.destroy_node()
        except Exception:
            pass


# ══════════════════════════════════════════════
#  3. SHARED BUFFER
#  thread_manager.py / SharedBuffer + model-karsilastirma.py stili
# ══════════════════════════════════════════════

class SharedBuffer:
    def __init__(self):
        self._lock    = threading.Lock()
        self.frame    = None
        self.detected = False
        self.conf     = 0.0
        self.cls_name = ""
        self.diameter = 0
        self.running  = True
        self.yolo_fps = 0.0   # gerçek YOLO inference FPS (yolo_thread'den gelir)

    def set(self, frame, detected, conf, cls_name, diameter):
        with self._lock:
            self.frame    = frame
            self.detected = detected
            self.conf     = conf
            self.cls_name = cls_name
            self.diameter = diameter

    def get_display(self):
        with self._lock:
            if self.frame is None:
                return None, False, 0.0, "", 0, 0.0
            return (self.frame.copy(),
                    self.detected, self.conf,
                    self.cls_name, self.diameter,
                    self.yolo_fps)

    def stop(self):
        with self._lock:
            self.running = False

    @property
    def is_running(self):
        with self._lock:
            return self.running


# ══════════════════════════════════════════════
#  4. YOLO THREAD
#  model-karsilastirma.py / yolo_thread ile aynı mantık
#  YOLO inference tamamen Jetson GPU'sunda çalışır
# ══════════════════════════════════════════════

def yolo_thread(buf: SharedBuffer, cam: CameraSource, model: YOLO, model_tag: str):
    print("[yolo_thread] Başladı — kamera bekleniyor...")
    no_frame_log  = 0
    _inf_count    = 0
    _inf_t_start  = time.time()

    while buf.is_running:
        frame = cam.read()

        if frame is None:
            no_frame_log += 1
            if no_frame_log % 100 == 0:
                print(f"\r[yolo_thread] Kamera frame'i yok ({no_frame_log * 0.01:.0f}s)...",
                      end="", flush=True)
            time.sleep(0.01)
            continue

        if no_frame_log > 0:
            print(f"\n[yolo_thread] İlk frame alındı!")
            no_frame_log = -1

        annotated      = frame.copy()
        h, w           = frame.shape[:2]
        cx_img, cy_img = w // 2, h // 2

        cv2.drawMarker(annotated, (cx_img, cy_img),
                       (255, 255, 255), cv2.MARKER_CROSS, 14, 1)

        preds = model(frame, conf=YOLO_CONF, imgsz=640, device='0', verbose=False)

        if not preds or len(preds[0].boxes) == 0:
            _draw_status_bar(annotated, "Tespit yok", (160, 160, 160), model_tag)
            # FPS güncelle
            _inf_count += 1
            _elapsed = time.time() - _inf_t_start
            if _elapsed >= 1.0:
                with buf._lock:
                    buf.yolo_fps = _inf_count / _elapsed
                _inf_count   = 0
                _inf_t_start = time.time()
            buf.set(annotated, False, 0.0, "", 0)
            time.sleep(0.005)
            continue

        boxes  = preds[0].boxes

        # yolo_detector.py stili: düşman varsa en büyük bbox, yoksa en yüksek conf
        enemy_indices  = [i for i, c in enumerate(boxes.cls.tolist())
                          if int(c) in ENEMY_CLASSES]
        friend_indices = [i for i, c in enumerate(boxes.cls.tolist())
                          if int(c) in FRIENDLY_CLASSES]

        if enemy_indices:
            areas    = [(boxes.xyxy[i][2] - boxes.xyxy[i][0]) *
                        (boxes.xyxy[i][3] - boxes.xyxy[i][1])
                        for i in enemy_indices]
            best_idx = enemy_indices[int(areas.index(max(areas)))]
        elif friend_indices:
            confs    = [float(boxes.conf[i]) for i in friend_indices]
            best_idx = friend_indices[int(confs.index(max(confs)))]
        else:
            best_idx = int(boxes.conf.argmax())

        best     = boxes[best_idx]
        cls_id   = int(best.cls[0])
        conf     = float(best.conf[0])
        cls_name = model.names[cls_id]

        x1, y1, x2, y2 = map(int, best.xyxy[0])
        cx       = (x1 + x2) // 2
        cy       = (y1 + y2) // 2
        diameter = max(x2 - x1, y2 - y1)

        if cls_id in FRIENDLY_CLASSES:
            color  = (0, 255, 0)
            status = f"DOST: {cls_name}  ({conf:.2f})"
        elif cls_id in ENEMY_CLASSES:
            if diameter > ENEMY_TOO_CLOSE_PX:
                color  = (0, 0, 255)
                status = f"DUSMAN YAKIN — KAC  ({cls_name}  {conf:.2f})"
            elif diameter < ENEMY_TOO_FAR_PX:
                color  = (255, 100, 0)
                status = f"DUSMAN UZAK — YAKLAS  ({cls_name}  {conf:.2f})"
            else:
                color  = (0, 165, 255)
                status = f"DUSMAN — MESAFE KOR  ({cls_name}  {conf:.2f})"
        else:
            color  = (128, 128, 128)
            status = f"BILINMEYEN: {cls_name}  ({conf:.2f})"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, (cx, cy), 4, color, -1)
        cv2.line(annotated, (cx_img, cy_img), (cx, cy), color, 1)
        cv2.putText(annotated, f"{cls_name}  {conf:.2f}  {diameter}px",
                    (x1, max(y1 - 8, 44)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        _draw_status_bar(annotated, status, color, model_tag)
        # FPS güncelle
        _inf_count += 1
        _elapsed = time.time() - _inf_t_start
        if _elapsed >= 1.0:
            with buf._lock:
                buf.yolo_fps = _inf_count / _elapsed
            _inf_count   = 0
            _inf_t_start = time.time()
        buf.set(annotated, True, conf, cls_name, diameter)
        time.sleep(0.005)


# ══════════════════════════════════════════════
#  5. DRONE / GPS THREAD
# ══════════════════════════════════════════════

class DroneState:
    def __init__(self):
        self._lock    = threading.Lock()
        self.distance = 0.0
        self.step     = 0

    def update(self, distance, step):
        with self._lock:
            self.distance = distance
            self.step     = step

    def read(self):
        with self._lock:
            return self.distance, self.step


def drone_thread(state: DroneState, buf: SharedBuffer,
                 v1, v2, dist_logger, args, save_dir: str, model_tag: str):
    step      = 0
    last_snap = -1   # son screenshot'ın çekildiği mesafe basamağı
    print("[drone_thread] Başladı.")
    while buf.is_running:
        lat1, lon1, alt1 = get_gps(v1)
        lat2, lon2, alt2 = get_gps(v2)
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        state.update(distance, step)

        if distance >= args.max_dist:
            print(f"\n[drone_thread] Hedef mesafe: {distance:.1f}m — Test bitti.")
            buf.stop()
            break

        with buf._lock:
            _det  = buf.detected
            _conf = buf.conf
            _cls  = buf.cls_name
            _dia  = buf.diameter

        dist_logger.log({
            "step":             step,
            "timestamp":        datetime.now().isoformat(),
            "distance_m":       round(distance, 3),
            "iris1_lat": lat1,  "iris1_lon": lon1,  "iris1_alt": round(alt1, 2),
            "iris2_lat": lat2,  "iris2_lon": lon2,  "iris2_alt": round(alt2, 2),
            "yolo_detected":    _det,
            "yolo_conf":        round(_conf, 4) if _det else "",
            "yolo_class":       _cls,
            "bbox_diameter_px": _dia,
        })

        print(f"\rAdım:{step:3d}  Mes:{distance:6.2f}m  "
              f"Conf:{'%.3f' % _conf if _det else '  ---'}  "
              f"Det:{'E' if _det else 'H'}  {_cls[:16]}",
              end="", flush=True)

        # Otomatik screenshot — GPS mesafesi güncellenir güncellenmez al
        # Ana thread'deki frame gecikmesi olmaz, mesafe frame ile senkronize
        cur_snap = int(distance) // SS_INTERVAL_M * SS_INTERVAL_M
        if cur_snap > last_snap and cur_snap >= SS_INTERVAL_M:
            last_snap = cur_snap
            with buf._lock:
                _snap_frame = buf.frame.copy() if buf.frame is not None else None
                _snap_fps   = buf.yolo_fps
            if _snap_frame is not None:
                os.makedirs(save_dir, exist_ok=True)
                # Mesafe ve tespit bilgisini frame üzerine yaz
                _draw_info_overlay(_snap_frame, step, distance,
                                   _conf, _det, _snap_fps, _dia)
                _draw_bottom_bar(_snap_frame, model_tag, distance)
                fname = os.path.join(
                    save_dir,
                    f"metre_{cur_snap:03d}m_{model_tag}_{datetime.now().strftime('%H%M%S%f')}.png"
                )
                cv2.imwrite(fname, _snap_frame)
                print(f"\n[SS] {cur_snap}m → {fname}")

        send_velocity_yaw_rate(v2, vx=+args.speed, vy=0, vz=0, yaw_rate=0)
        step += 1
        time.sleep(args.step_dur)


# ══════════════════════════════════════════════
#  6. ÇİZİM YARDIMCILARI
# ══════════════════════════════════════════════

def _draw_status_bar(frame, text, color, model_tag):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    label = f"[{model_tag}]"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, label,
                (frame.shape[1] - tw - 8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def _draw_info_overlay(frame, step, distance, conf, detected, fps, diameter=0):
    w       = frame.shape[1]
    panel_w = 255
    line_h  = 26
    pad     = 8

    det_label = "TESPIT: EVET" if detected else "TESPIT: YOK "
    det_color = (0, 255, 128) if detected else (80, 80, 255)

    lines = [
        ("FPS",      f"{fps:.1f}",                            (0, 220, 255)),
        (det_label,  "",                                       det_color),
        ("Dogr",     f"{conf:.3f}" if detected else "---",
                     (0, 255, 128) if detected else (120, 120, 120)),
        ("GPS Mes",  f"{distance:.2f} m",                     (255, 200, 0)),
        ("BBox",     f"{diameter} px" if detected else "---", (200, 160, 255)),
        ("Adim",     f"{step}",                               (200, 200, 200)),
    ]

    total_h = len(lines) * line_h + pad * 2
    x0 = w - panel_w - 6
    y0 = 42

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + total_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, (lbl, val, col) in enumerate(lines):
        y    = y0 + pad + (i + 1) * line_h - 4
        text = lbl if val == "" else f"{lbl:<10}: {val}"
        cv2.putText(frame, text, (x0 + pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)


def _draw_bottom_bar(frame, model_tag, distance):
    h, w  = frame.shape[:2]
    bar_h = 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame,
                f"Model: {model_tag}   GPS: {distance:.2f}m   q:cik  p:dur  s:SS",
                (10, h - bar_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1)


# ══════════════════════════════════════════════
#  7. CSV LOGGER'LAR
# ══════════════════════════════════════════════

DIST_COLUMNS = [
    "step", "timestamp", "distance_m",
    "iris1_lat", "iris1_lon", "iris1_alt",
    "iris2_lat", "iris2_lon", "iris2_alt",
    "yolo_detected", "yolo_conf", "yolo_class", "bbox_diameter_px",
]


class DistanceLogger:
    def __init__(self, path, model_name):
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._f.write(f"# Model: {model_name}\n")
        self._f.write(f"# Test tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._f.write(f"# Kamera topic: {CAM_TOPIC}\n")
        self._f.write(f"# YOLO confidence eşiği: {YOLO_CONF}\n")
        self._w = csv.DictWriter(self._f, fieldnames=DIST_COLUMNS)
        self._w.writeheader()
        print(f"[DistLogger] CSV: {path}")

    def log(self, row):
        self._w.writerow(row)
        self._f.flush()

    def close(self):
        self._f.close()


# ══════════════════════════════════════════════
#  8. TEGRASTATS LOGGER  (Jetson'a özgü)
#  nvidia-smi yerine tegrastats kullanılır —
#  Jetson'da GPU, CPU, güç ve sıcaklık bilgisi verir.
# ══════════════════════════════════════════════

TEGRA_COLUMNS = ["zaman_s", "gpu_kullanim_%", "cpu_kullanim_%",
                 "ram_mib", "sicaklik_c", "guc_w"]


class TegraLogger:
    """
    tegrastats çıktısını parse eder, CSV'ye yazar.
    Örnek tegrastats satırı:
        RAM 2048/7771MB ... CPU [45%@1420,30%@1420,...] ... GPU 78% ... PLL@42C ...
    """

    def __init__(self, path, model_name):
        self._path    = path
        self._stop    = threading.Event()
        self._start_t = time.time()
        self._f       = open(path, "w", newline="", encoding="utf-8")
        self._f.write(f"# Model: {model_name}\n")
        self._f.write(f"# Test tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._w = csv.writer(self._f)
        self._w.writerow(TEGRA_COLUMNS)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        print(f"[TegraLogger] CSV: {path}")

    def _parse(self, line: str):
        """tegrastats satırından ilgili değerleri çıkar."""
        gpu_pct  = 0.0
        cpu_pct  = 0.0
        ram_mib  = 0.0
        temp_c   = 0.0
        power_w  = 0.0
        try:
            # GPU: "GR3D_FREQ 78%"
            if "GR3D_FREQ" in line:
                idx = line.index("GR3D_FREQ") + len("GR3D_FREQ ")
                gpu_pct = float(line[idx:].split("%")[0])

            # RAM: "RAM 2048/7771MB"
            if "RAM " in line:
                ram_part = line.split("RAM ")[1].split("/")[0]
                ram_mib  = float(ram_part)

            # CPU: "CPU [45%@1420,30%@1420,...]" — ortalama al
            if "CPU [" in line:
                cpu_str = line.split("CPU [")[1].split("]")[0]
                percs   = [float(p.split("%")[0]) for p in cpu_str.split(",")
                           if "%" in p and p[0].isdigit()]
                if percs:
                    cpu_pct = sum(percs) / len(percs)

            # Sıcaklık: "CPU@42C" veya "SOC0@45C" ilkini al
            import re
            temps = re.findall(r'@(\d+)C', line)
            if temps:
                temp_c = float(temps[0])

            # Güç: "POM_5V_IN 3500/3500" mW → W
            if "POM_5V_IN" in line:
                pw_str  = line.split("POM_5V_IN")[1].strip().split()[0]
                power_w = float(pw_str.split("/")[0]) / 1000.0

        except Exception:
            pass
        return gpu_pct, cpu_pct, ram_mib, temp_c, power_w

    def _loop(self):
        try:
            proc = subprocess.Popen(
                ["tegrastats", f"--interval", str(int(TEGRA_INTERVAL * 1000))],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True
            )
            for line in proc.stdout:
                if self._stop.is_set():
                    proc.terminate()
                    break
                t = round(time.time() - self._start_t, 2)
                gpu, cpu, ram, tmp, pwr = self._parse(line.strip())
                self._w.writerow([t, gpu, cpu, ram, tmp, pwr])
                self._f.flush()
        except FileNotFoundError:
            # tegrastats yoksa nvidia-smi'ye düş (x86 test ortamı)
            print("[TegraLogger] tegrastats bulunamadı, nvidia-smi deneniyor...")
            self._loop_nvidiasmi()

    def _loop_nvidiasmi(self):
        """Fallback: nvidia-smi (geliştirme/test ortamı için)."""
        while not self._stop.is_set():
            t = round(time.time() - self._start_t, 2)
            try:
                out = subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits"
                ], text=True).strip()
                parts = [p.strip() for p in out.split(",")]
                gpu   = float(parts[0])
                ram   = float(parts[1])
                tmp   = float(parts[2])
                pwr   = float(parts[3])
                self._w.writerow([t, gpu, 0.0, ram, tmp, pwr])
                self._f.flush()
            except Exception:
                self._w.writerow([t, 0, 0, 0, 0, 0])
            time.sleep(TEGRA_INTERVAL)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)
        self._f.close()
        print(f"[TegraLogger] Kaydedildi: {self._path}")


# ══════════════════════════════════════════════
#  9. ANA AKIŞ
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Jetson YOLO model karşılaştırma")
    parser.add_argument("--model",      default="yolo-v8.pt",    help="Model dosyası")
    parser.add_argument("--max-dist",   default=35.0, type=float, help="Maksimum test mesafesi (m)")
    parser.add_argument("--speed",      default=0.4,  type=float, help="iris_2 hızı (m/s)")
    parser.add_argument("--step-dur",   default=1.5,  type=float, help="Adım süresi (s)")
    parser.add_argument("--output-dir", default="./test_output",  help="Çıktı klasörü")
    parser.add_argument("--no-display", action="store_true",      help="Görüntü gösterme")
    parser.add_argument("--laptop-ip",  default=DEFAULT_LAPTOP_IP,
                        help=f"Laptop IP adresi (varsayılan: {DEFAULT_LAPTOP_IP})")
    args = parser.parse_args()

    IRIS1_ADDR, IRIS2_ADDR = build_addrs(args.laptop_ip)
    SHOW      = not args.no_display
    model_tag = os.path.splitext(os.path.basename(args.model))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")

    dist_csv = os.path.join(args.output_dir, f"distance_conf_{model_tag}_{ts}.csv")
    tegra_csv= os.path.join(args.output_dir, f"tegra_log_{model_tag}_{ts}.csv")
    save_dir = os.path.join(args.output_dir, "frames")

    print("=" * 60)
    print("  Jetson YOLO Model Karşılaştırma")
    print("=" * 60)
    print(f"[Config] Model     : {args.model}")
    print(f"[Config] Laptop IP : {args.laptop_ip}")
    print(f"[Config] iris_1    : {IRIS1_ADDR}")
    print(f"[Config] iris_2    : {IRIS2_ADDR}")
    print(f"[Config] Kamera    : {CAM_TOPIC}")
    print(f"[Config] Maks mes  : {args.max_dist}m")
    print(f"[Config] Hız       : {args.speed}m/s")
    print()

    # ── Tegrastats logger ─────────────────────
    tegra_logger = TegraLogger(tegra_csv, args.model)
    tegra_logger.start()

    # ── DroneKit bağlantısı ───────────────────
    print(f"[iris_1] Bağlanıyor: {IRIS1_ADDR}")
    v1 = connect(IRIS1_ADDR, wait_ready=True, timeout=60)
    print(f"[iris_2] Bağlanıyor: {IRIS2_ADDR}")
    v2 = connect(IRIS2_ADDR, wait_ready=True, timeout=60)

    # Paralel kalkış (thread_manager.py stili)
    results = [False, False]
    def _t1(): results[0] = arm_and_takeoff(v1, "iris_1", TARGET_ALT)
    def _t2(): results[1] = arm_and_takeoff(v2, "iris_2", TARGET_ALT + 1)
    th1 = threading.Thread(target=_t1)
    th2 = threading.Thread(target=_t2)
    th1.start(); th2.start(); th1.join(); th2.join()

    if not all(results):
        print("Kalkış başarısız!")
        tegra_logger.stop(); v1.close(); v2.close()
        return

    print("\nHer iki drone havada! 3s stabilizasyon...")
    hover(v1, 1.0); hover(v2, 3.0)

    # ── ROS2 kamera — spin thread otomatik başlar ──
    cam = CameraSource(CAM_TOPIC)

    # ── YOLO modeli — Jetson GPU (device='0') ─────
    import torch.nn as nn
    import ultralytics.utils.loss as _loss_mod
    if not hasattr(_loss_mod, "E2ELoss"):
        class E2ELoss(nn.Module):
            def __init__(self, model): super().__init__()
            def forward(self, *a, **kw): pass
        _loss_mod.E2ELoss = E2ELoss

    print("[Model] Yükleniyor...")
    model = YOLO(args.model, task='detect')

    # TensorRT warmup (yolo_detector.py / _warmup ile aynı)
    print("[Model] TensorRT warmup başlıyor...")
    # imgsz model adından otomatik belirlenir (Yolo26-640.engine → 640)
    import re as _re; _m = _re.search(r"[-_](\d{3,4})\.engine", args.model)
    _imgsz = int(_m.group(1)) if _m else 640
    print(f"[Model] imgsz: {_imgsz}")
    dummy = np.zeros((_imgsz, _imgsz, 3), dtype=np.uint8)
    for i in range(3):
        model(dummy, imgsz=_imgsz, device="0", verbose=False)
    print("[Model] Warmup tamamlandı. Sistem hazır.\n")

    buf         = SharedBuffer()
    state       = DroneState()
    dist_logger = DistanceLogger(dist_csv, args.model)

    # ── Arka plan thread'leri ─────────────────────
    t_yolo = threading.Thread(
        target=yolo_thread,
        args=(buf, cam, model, model_tag),
        daemon=True
    )
    t_drone = threading.Thread(
        target=drone_thread,
        args=(state, buf, v1, v2, dist_logger, args, save_dir, model_tag),
        daemon=True
    )
    t_yolo.start()
    t_drone.start()

    print(f"[Test] model:{model_tag} | maks:{args.max_dist}m | hız:{args.speed}m/s")

    # ── Görüntü penceresi — ANA THREAD ────────────
    WIN = f"Jetson — {model_tag}"
    if SHOW:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 640, 360)
        placeholder = np.zeros((540, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Kamera bekleniyor...",
                    (320, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        cv2.imshow(WIN, placeholder)
        cv2.waitKey(1)
        print("  q:çık   p:duraklat/devam   s:screenshot\n")

    last_frame = None
    yolo_fps   = 0.0

    try:
        while buf.is_running:
            frame, detected, conf, cls_name, diameter, yolo_fps = buf.get_display()
            if frame is not None:
                last_frame = frame

            if last_frame is None:
                if SHOW:
                    cv2.imshow(WIN, placeholder)
                    if cv2.waitKey(50) & 0xFF == ord('q'):
                        buf.stop(); break
                else:
                    time.sleep(0.05)
                continue

            distance, step = state.read()

            display = last_frame.copy()
            _draw_info_overlay(display, step, distance, conf, detected, yolo_fps, diameter)
            _draw_bottom_bar(display, model_tag, distance)

            if SHOW:
                cv2.imshow(WIN, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[Ekran] Çıkılıyor...")
                    buf.stop(); break
                elif key == ord('p'):
                    print("\n[Ekran] Duraklatıldı. Devam için p...")
                    while True:
                        if cv2.waitKey(100) & 0xFF == ord('p'):
                            print("\n[Ekran] Devam ediliyor..."); break
                elif key == ord('s'):
                    os.makedirs(save_dir, exist_ok=True)
                    fname = os.path.join(
                        save_dir,
                        f"manuel_{model_tag}_step{step:04d}_{datetime.now().strftime('%H%M%S%f')}.png"
                    )
                    cv2.imwrite(fname, display)
                    print(f"\n[Ekran] Manuel SS: {fname}")
            else:
                time.sleep(0.033)

    except KeyboardInterrupt:
        print("\n[KeyboardInterrupt] Durduruldu.")

    finally:
        buf.stop()
        if SHOW:
            cv2.destroyAllWindows()

        dist_logger.close()
        tegra_logger.stop()
        cam.release()

        distance, step = state.read()
        print(f"\n[Bitti] {step} adım | son mesafe: {distance:.2f}m")
        print(f"  Mesafe CSV  : {dist_csv}")
        print(f"  Tegra CSV   : {tegra_csv}")
        print(f"  SS klasörü  : {save_dir}")

        print("\nDrone'lar indiriliyor...")
        v2.mode = VehicleMode("LAND")
        try:
            v1.mode = VehicleMode("LAND")
        except Exception:
            pass
        time.sleep(3)
        v1.close()
        v2.close()

        try:
            import rclpy
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
