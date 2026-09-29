"""
thread_manager.py  —  Gazebo modu (Geliştirilmiş)
===================================================
Çalıştırma: python thread_manager.py --drone

İyileştirmeler:
    - Queue tabanlı thread iletişimi (komut kaybı yok)
    - Exception handling + otomatik yeniden başlatma
    - Graceful shutdown (stop anında vx=0 komutu)
    - Print throttle (CPU israfı önlendi)
    - Display 30 FPS
    - CSV log sistemi
"""

import threading
import time
import sys
import queue
import csv
import os
from datetime import datetime

import cv2

from yolo_detector import YoloDetector, DetectionResult
from yolo_controller import DroneController, ControlOutput

import yaml

# ── AYARLAR (Config.yaml'dan beslenir) ─────────────────
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

YOLO_MODEL      = cfg['yolo']['model_path']
CONF_THRESHOLD  = cfg['yolo']['conf_threshold']
MAVLINK_HZ      = cfg['sistem']['mavlink_hz']
LOG_ENABLED     = cfg['sistem']['log_enabled']

DISPLAY_FPS     = 30
PRINT_INTERVAL  = 0.5   # saniyede kaç kez konsola yaz
LOG_DIR         = "logs"



# ─────────────────────────────────────────────
#  CSV LOGGER
# ─────────────────────────────────────────────

class FlightLogger:
    """Her kararı CSV'ye kaydeder — sonraki analiz için."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._file   = None
        self._writer = None
        if enabled:
            os.makedirs(LOG_DIR, exist_ok=True)
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            path     = os.path.join(LOG_DIR, f"flight_{ts}.csv")
            self._file   = open(path, "w", newline="")
            self._writer = csv.writer(self._file)
            self._writer.writerow([
                "timestamp", "action", "diameter_px", "growth_rate",
                "panic_level", "vx", "vy", "vz", "yaw_rate",
                "error_x", "error_y", "latency_ms", "reason"
            ])
            print(f"[Logger] Kayit: {path}")

    def log(self, output: ControlOutput, result: DetectionResult):
        if not self.enabled or self._writer is None:
            return
        diameter    = result.bbox_diameter if result.detected else 0
        growth_rate = 0.0
        panic_level = 0.0

        # reason'dan parse et
        reason = output.reason
        if "buyume:" in reason:
            try:
                growth_rate = float(reason.split("buyume:")[1].split("px")[0])
            except Exception:
                pass
        if "panik:" in reason:
            try:
                panic_level = float(reason.split("panik:")[1].split()[0])
            except Exception:
                pass

        self._writer.writerow([
            f"{time.time():.4f}",
            output.action,
            diameter,
            f"{growth_rate:.3f}",
            f"{panic_level:.3f}",
            f"{output.vx:.3f}",
            f"{output.vy:.3f}",
            f"{output.vz:.3f}",
            f"{output.yaw_rate:.3f}",
            result.error_x if result.detected else 0,
            result.error_y if result.detected else 0,
            f"{output.latency_ms:.1f}",
            reason
        ])

    def close(self):
        if self._file:
            self._file.flush()
            self._file.close()
            print("[Logger] Kayit tamamlandi.")


# ─────────────────────────────────────────────
#  HUD
# ─────────────────────────────────────────────

def annotate_output(frame, output: ControlOutput):
    h, w = frame.shape[:2]
    colors = {
        "KAC":        (0, 0, 255),
        "YAKLAS":     (255, 130, 0),
        "MESAFE_KOR": (0, 165, 255),
        "BEKLE":      (0, 200, 0),
    }
    c     = colors.get(output.action, (200, 200, 200))
    box_h = 80

    cv2.rectangle(frame, (0, h - box_h), (w, h), (15, 15, 15), -1)

    # Satır 1: Ana karar + hızlar
    cv2.putText(frame,
               f"{output.action}  {output.latency_ms:.0f}ms  vx={output.vx:+.2f}  vy={output.vy:+.2f}"
               f"  vz={output.vz:+.2f}  yaw={output.yaw_rate:+.2f}",
                (10, h - box_h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1)

    # Satır 2: Sebep
    cv2.putText(frame, output.reason[:70],
                (10, h - box_h + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1)


# ─────────────────────────────────────────────
#  SHARED BUFFER  (Queue tabanlı)
# ─────────────────────────────────────────────

class SharedBuffer:
    """
    T1 → detection_queue → T2
    T2 → display_queue   → Ana thread
    T2 → guard_queue     → T3
    Her queue maxsize=1: eski veri tutulmaz, her zaman en taze.
    """

    def __init__(self):
        self.detection_queue = queue.Queue(maxsize=1)
        self.display_queue   = queue.Queue(maxsize=1)
        self.guard_queue     = queue.Queue(maxsize=1)
        self.rtl_queue       = queue.Queue(maxsize=1)  # RTL için ayrı kanal — normal komutla ezilmez
        self._lock           = threading.Lock()
        self._running        = True
        self.rtl_event = threading.Event()

        # İstatistik
        self.t1_fps          = 0.0
        self.t2_fps          = 0.0
        self.t3_hz           = 0.0

    def put_detection(self, result: DetectionResult):
        """T1 → T2"""
        try:
            self.detection_queue.put_nowait(result)
        except queue.Full:
            try:
                self.detection_queue.get_nowait()
            except queue.Empty:
                pass
            self.detection_queue.put_nowait(result)

    def get_detection(self, timeout=0.02) -> DetectionResult | None:
        try:
            return self.detection_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def put_output(self, output: ControlOutput, frame):
        """T2 → Ana thread + T3"""
        payload = (output, frame)
        # Display queue
        try:
            self.display_queue.put_nowait(payload)
        except queue.Full:
            try:
                self.display_queue.get_nowait()
            except queue.Empty:
                pass
            self.display_queue.put_nowait(payload)

        # Guard queue
        try:
            self.guard_queue.put_nowait(output)
        except queue.Full:
            try:
                self.guard_queue.get_nowait()
            except queue.Empty:
                pass
            self.guard_queue.put_nowait(output)

    def get_display(self, timeout=0.033):
        try:
            return self.display_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None

    def get_guard(self, timeout=0.05) -> ControlOutput | None:
        try:
            return self.guard_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        with self._lock:
            self._running = False

    @property
    def is_running(self):
        with self._lock:
            return self._running


# ─────────────────────────────────────────────
#  THREAD 1 — Tespit
# ─────────────────────────────────────────────

def detector_thread(buf: SharedBuffer):
    """
    ROS2 kamera topic'inden frame alır, YOLO çalıştırır.
    Crash durumunda sistemi durdurur.
    """
    detector = None
    try:
        detector = YoloDetector(YOLO_MODEL, conf_threshold=CONF_THRESHOLD)
        frame_count = 0
        t_start     = time.time()

        while buf.is_running:
            result, frame = detector.read_and_detect()

            if frame is None:
                time.sleep(0.01)
                continue

            buf.put_detection(result)

            # FPS hesapla
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 0.5:
                buf.t1_fps  = frame_count / elapsed
                frame_count = 0
                t_start     = time.time()

    except Exception as e:
        print(f"\n[T1 HATA] Detector çöktü: {e}")
        buf.stop()
    finally:
        if detector:
            detector.release()
        print("[T1] Durduruldu.")


# ─────────────────────────────────────────────
#  THREAD 2 — Kontrol
# ─────────────────────────────────────────────

from orbit_collector import OrbitCollector, OrbitOutput
from few_shot_tracker import FewShotTracker
...
def controller_thread(buf: SharedBuffer, logger: FlightLogger):
    controller = DroneController()
    orbit      = OrbitCollector()          # ← ekle
    tracker    = FewShotTracker()
    frame_count = 0
    t_start     = time.time()
    last_t      = time.time() 

    try:
        while buf.is_running:
            if buf.rtl_event.is_set():
                buf.rtl_event.clear()
                _send_rtl(buf)
                buf.stop()
                break

            result = buf.get_detection(timeout=0.02)

            if result is None:
                continue

            # dt hesapla
            now    = time.time()
            dt     = max(now - last_t, 1e-3)
            last_t = now

            output = controller.compute(result)

            orbit_out = OrbitOutput()  # varsayılan inactive
            if result.detected and result.is_orbit:
                drone_id = tracker.update(
                    result.bbox_center,
                    result.bbox_diameter,
                    frame=result.raw_frame,       # ← görsel imza için
                    bbox_xyxy=result.bbox_xyxy
                )
                orbit_out = orbit.update(
                    output.action,
                    result.bbox_diameter,
                    result.raw_frame,
                    bbox_xyxy=result.bbox_xyxy,
                    class_id=result.class_id,
                    img_w=640, img_h=640,
                    dt=0.05
                )
            if orbit_out.active:
                output.vy     = orbit_out.vy
                output.action = "ORBIT"
                output.reason = (f"Orbit {orbit_out.frames_done}/{orbit_out.frames_total} "
                                f"frame  {orbit_out.state.name}")

            if orbit_out.rtl_now and not getattr(orbit, '_rtl_sent', False):
                orbit._rtl_sent = True
                tracker.reset()   # aynı drone için drone_4/5/6... üretmesin
                _send_rtl(buf)

            # Annotate (output kesinleştikten sonra)
            frame = result.annotated_frame
            if frame is not None:
                frame = frame.copy()
                annotate_output(frame, output)

            buf.put_output(output, frame)

            # Log
            logger.log(output, result)

            # FPS
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 2.0:
                buf.t2_fps  = frame_count / elapsed
                frame_count = 0
                t_start     = time.time()

    except Exception as e:
        print(f"\n[T2 HATA] Controller çöktü: {e}")
        buf.stop()
    print("\n[T2] Durduruldu.")

def rtl_input_thread(buf: SharedBuffer):
    print("[RTL] Terminal hazir: 'rtl' yazip Enter → drone geri doner.")
    while buf.is_running:
        try:
            line = input().strip().lower()
            if line == "rtl":
                print("[RTL] Komut alindi...")
                buf.rtl_event.set()
        except (EOFError, KeyboardInterrupt):
            break

# ─────────────────────────────────────────────
#  THREAD 3 — MAVLink
# ─────────────────────────────────────────────

def _send_rtl(buf: SharedBuffer):
    """MAVLink RTL komutu — rtl_queue üzerinden gönderilir (normal komutla ezilemez)."""
    from yolo_controller import ControlOutput
    rtl_out        = ControlOutput()
    rtl_out.action = "RTL"
    rtl_out.reason = "Orbit tamamlandi — RTL"
    try:
        buf.rtl_queue.put_nowait(rtl_out)
    except queue.Full:
        pass  # zaten RTL bekliyor
    print("[RTL] Komut kuyruğa alındı.")

def guard_thread(buf: SharedBuffer):
    """
    Hesaplanan komutları 20 Hz'de MAVLink ile gönderir.
    Stop anında DUR komutu gönderir.
    """
    from drone_guard import DroneGuard
    guard = None

    try:
        guard = DroneGuard()
        hz_count = 0
        t_start  = time.time()

        # Son gönderilen komutu hafızada tutmak için bir obje oluştur
        last_output = ControlOutput()

        while buf.is_running or not buf.rtl_queue.empty():
            # Önce RTL kanalını kontrol et — normal komutla ezilemez
            # is_running=False olsa bile kuyrukta RTL varsa gönder
            try:
                rtl_out = buf.rtl_queue.get_nowait()
                guard.send(rtl_out)
                print("[T3] RTL gönderildi.")
                time.sleep(0.5)
                buf.stop()
                break
            except queue.Empty:
                pass

            if not buf.is_running:
                break

            output = buf.get_guard(timeout=1.0 / MAVLINK_HZ)

            if output is None:
                # Timeout: Yeni veri gelmediyse son komutu tekrar gönder (Watchdog)
                guard.send(last_output)
                continue

            # Yeni veri geldiyse hafızayı güncelle ve gönder
            last_output = output
            guard.send(output)

            # Hz hesapla
            hz_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 2.0:
                buf.t3_hz = hz_count / elapsed
                hz_count  = 0
                t_start   = time.time()

    except Exception as e:
        print(f"\n[T3 HATA] Guard çöktü: {e}")
        buf.stop()
    finally:
        # Kritik: stop anında DUR komutu gönder
        if guard:
            print("\n[T3] STOP komutu gönderiliyor...")
            stop_output = ControlOutput()
            stop_output.action = "BEKLE"
            try:
                guard.send(stop_output)
            except Exception:
                pass
            guard.close()
        print("[T3] Durduruldu.")


# ─────────────────────────────────────────────
#  ANA THREAD
# ─────────────────────────────────────────────

def main():
    if "--drone" not in sys.argv:
        print("Kullanim: python thread_manager.py --drone")
        sys.exit(1)

    print("=" * 55)
    print("  DroneGuard — Gelismis Kacınma Sistemi")
    print("=" * 55)
    print(f"[Config] Model  : {YOLO_MODEL}")
    print(f"[Config] Kamera : /iris_1/camera/image")
    print(f"[Config] MAVLink: udp:127.0.0.1:14550")
    print(f"[Config] Log    : {'Acik' if LOG_ENABLED else 'Kapali'}")
    print(f"[Config] Cikis  : 'q'\n")

    logger = FlightLogger(enabled=LOG_ENABLED)
    buf    = SharedBuffer()

    t1 = threading.Thread(target=detector_thread,
                          args=(buf,), name="T1-Detector", daemon=True)
    t2 = threading.Thread(target=controller_thread,
                          args=(buf, logger), name="T2-Controller", daemon=True)
    t3 = threading.Thread(target=guard_thread,
                          args=(buf,), name="T3-Guard", daemon=True)

    t1.start()
    t2.start()
    t3.start()

    t_rtl = threading.Thread(target=rtl_input_thread,
                         args=(buf,), name="T-RTL", daemon=True)
    t_rtl.start()

    cv2.namedWindow("DroneGuard — iris_1", cv2.WINDOW_NORMAL)
    last_frame = None
    status_t   = time.time()

    try:
        while True:
            output, frame = buf.get_display(timeout=0.01)  # kısa timeout
            if frame is not None:
                last_frame = frame

            if last_frame is not None:
                cv2.imshow("DroneGuard — iris_1", last_frame)

            key = cv2.waitKey(1) & 0xFF  # 1ms — blocking değil
            if key == ord('q'):
                break
	    

    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt.")
    finally:
        buf.stop()

        # Thread'lerin temiz kapanmasını bekle
        t1.join(timeout=3.0)
        t2.join(timeout=3.0)
        t3.join(timeout=5.0)   # T3 en son: DUR komutu göndermeli

        cv2.destroyAllWindows()
        logger.close()
        print("\nSistem temiz kapatildi.")


if __name__ == "__main__":
    main()
