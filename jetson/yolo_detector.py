"""
yolo_detector.py  —  Gazebo modu
=================================
Sadece GazeboSource kullanır.
Çalıştırma: python thread_manager.py --drone
"""

import time

from ultralytics import YOLO
import cv2, threading
from abc import ABC, abstractmethod

# ── Sınıf tanımları ───────────────────────────
FRIENDLY_CLASSES   = {}
ENEMY_CLASSES      = {0, 1, 2, 3, 4, 5}
ORBIT_CLASSES    = {}
# Controller parametreleriyle eşleşmeli:
# 147px → 3.0m (çok yakın)  |  55px → 8.0m (çok uzak)
ENEMY_TOO_CLOSE_PX = 230
ENEMY_TOO_FAR_PX   = 57
MIN_STABLE_FRAMES  = 1

GAZEBO_CAMERA_TOPIC = "/iris_1/camera/image/compressed"


# ═══════════════════════════════════════════════
#  FRAME SOURCE
# ═══════════════════════════════════════════════

class FrameSource(ABC):
    @abstractmethod
    def read(self, path_or_index=None):
        pass

    def release(self):
        pass


class GazeboSource(FrameSource):
    """ROS2 /iris_1/camera/image topic'inden frame alır."""

    def __init__(self, topic=GAZEBO_CAMERA_TOPIC):
        import rclpy
        from sensor_msgs.msg import CompressedImage
        from cv_bridge import CvBridge
        import numpy as np
        import cv2

        self._bridge = CvBridge()
        self._frame  = None
        self._lock   = threading.Lock()

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node("yolo_frame_source")
        self._node.create_subscription(CompressedImage, topic, self._callback, 10)

        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self._node,), daemon=True
        )
        self._spin_thread.start()
        print(f"[GazeboSource] Topic: {topic}")

    def _callback(self, msg):
        import numpy as np
        import cv2
        np_arr=np.frombuffer(msg.data,np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        with self._lock:
            self._frame = frame

    def read(self, path_or_index=None):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def release(self):
        try:
            self._node.destroy_node()
        except Exception:
            pass


# ═══════════════════════════════════════════════
#  DETECTION RESULT
# ═══════════════════════════════════════════════

class DetectionResult:
    def __init__(self):
        self.detected        = False
        self.is_friendly     = False
        self.is_enemy        = False
        self.is_orbit        = False
        self.too_close       = False
        self.too_far         = False
        self.class_id        = -1
        self.class_name      = ""
        self.bbox_center     = (0, 0)
        self.bbox_diameter   = 0
        self.error_x         = 0
        self.error_y         = 0
        self.annotated_frame = None
        self.raw_frame       = None   # ← ekle — HUD'suz temiz frame
        self.bbox_xyxy       = None   # ← ekle — (x1,y1,x2,y2) normalize
        self.t_capture = 0.0 


# ═══════════════════════════════════════════════
#  YOLO DETECTOR
# ═══════════════════════════════════════════════

class YoloDetector:
    def __init__(self, model_path="Yolo26-640.engine", conf_threshold=0.5):
            # YOLO26 (v10 tabanlı) modelleri için gerekli yama
        import torch.nn as nn
        import ultralytics.utils.loss as _loss_mod
        if not hasattr(_loss_mod, "E2ELoss"):
            class E2ELoss(nn.Module):
                def __init__(self, model): super().__init__()
                def forward(self, *a, **kw): pass
            _loss_mod.E2ELoss = E2ELoss
            
        self.model     = YOLO(model_path, task='detect')
        self.conf      = conf_threshold
        self.source    = GazeboSource()
        self._counter  = 0
        self._last_cls = -1
        print(f"[YoloDetector] Model   : {model_path}")
        print(f"[YoloDetector] Siniflar: {self.model.names}")
        print(f"[YoloDetector] Dusman  : {ENEMY_CLASSES}\n")
        self._warmup()

    def _warmup(self):
        """TensorRT motoru isitir — ilk gercek cikarimda donma olmaz."""
        import numpy as np
        print("[YoloDetector] TensorRT warmup basliyor...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for i in range(3):
            self.model(dummy, imgsz=640, device='0', verbose=False)
        print("[YoloDetector] Warmup tamamlandi. Sistem hazir.\n")

    def read_and_detect(self):
        frame = self.source.read()
        if frame is None:
            return DetectionResult(), None
        result = self.detect(frame)
        result.t_capture = time.time()   # ← tek satır ekleme
        return result, frame

    def detect(self, frame) -> DetectionResult:
        r = DetectionResult()
        h, w = frame.shape[:2]
        cx_img, cy_img = w // 2, h // 2
        r.raw_frame = frame.copy()   # annotate'den önce — temiz
        r.annotated_frame = frame.copy()

        preds = self.model(frame, conf=self.conf, imgsz=640, device='0', verbose=False)
        if not preds or len(preds[0].boxes) == 0:
            self._counter = max(0, self._counter - 2)
            self._draw_status(r.annotated_frame, "Tespit yok", (160, 160, 160))
            return r

        boxes = preds[0].boxes

        # Öncelikli seçim: düşman sınıfı varsa en büyük bbox, yoksa en yüksek conf
        # Hata: argmax(conf) dost drone'u düşmana tercih edebiliyordu
        enemy_indices  = [i for i, c in enumerate(boxes.cls.tolist())
                          if int(c) in ENEMY_CLASSES]
        friend_indices = [i for i, c in enumerate(boxes.cls.tolist())
                          if int(c) in FRIENDLY_CLASSES]

        if enemy_indices:
            # En büyük bbox'a sahip düşmanı seç (en yakın = en tehlikeli)
            areas      = [(boxes.xyxy[i][2]-boxes.xyxy[i][0]) *
                          (boxes.xyxy[i][3]-boxes.xyxy[i][1])
                          for i in enemy_indices]
            best_idx   = enemy_indices[int(areas.index(max(areas)))]
        elif friend_indices:
            # Düşman yok, dost var → en yüksek conflu dostu göster
            confs      = [float(boxes.conf[i]) for i in friend_indices]
            best_idx   = friend_indices[int(confs.index(max(confs)))]
        else:
            # Ne düşman ne dost → en yüksek conflu bilinmeyen
            best_idx   = int(boxes.conf.argmax())

        best     = boxes[best_idx]
        cls_id   = int(best.cls[0])
        cls_name = self.model.names[cls_id]

        x1, y1, x2, y2 = map(int, best.xyxy[0])
        r.bbox_xyxy = (x1, y1, x2, y2)
        cx, cy   = (x1 + x2) // 2, (y1 + y2) // 2
        diameter = max(x2 - x1, y2 - y1)

        self._counter  = self._counter + 1 if cls_id == self._last_cls else 1
        self._last_cls = cls_id

        if self._counter < MIN_STABLE_FRAMES:
            self._draw_status(r.annotated_frame,
                              f"Dogrulanıyor {self._counter}/{MIN_STABLE_FRAMES}",
                              (0, 200, 255))
            return r

        r.detected      = True
        r.class_id      = cls_id
        r.class_name    = cls_name
        r.bbox_center   = (cx, cy)
        r.bbox_diameter = diameter
        r.error_x       = cx - cx_img
        r.error_y       = cy - cy_img

        if cls_id in FRIENDLY_CLASSES:
            r.is_friendly = True
            color  = (0, 255, 0)
            status = f"DOST: {cls_name}"
        elif cls_id in ENEMY_CLASSES:
            r.is_enemy = True
            if diameter > ENEMY_TOO_CLOSE_PX:
                r.too_close = True
                color  = (0, 0, 255)
                status = f"DUSMAN YAKIN — KAC  ({cls_name})"
            elif diameter < ENEMY_TOO_FAR_PX:
                r.too_far = True
                color  = (255, 100, 0)
                status = f"DUSMAN UZAK — YAKLAS  ({cls_name})"
            else:
                color  = (0, 165, 255)
                status = f"DUSMAN — MESAFE KOR  ({cls_name})"
        elif cls_id in ORBIT_CLASSES:
            r.is_orbit = True
            color  = (255, 0, 255)
            status = f"GENEL DRONE — KESIF  ({cls_name})"
        else:
            color  = (128, 128, 128)
            status = f"BILINMEYEN: {cls_name}"

        self._draw_status(r.annotated_frame, status, color)
        cv2.rectangle(r.annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(r.annotated_frame, (cx, cy), 4, color, -1)
        cv2.putText(r.annotated_frame, f"{cls_name}  {diameter}px",
                    (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.drawMarker(r.annotated_frame, (cx_img, cy_img),
                       (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
        return r

    def release(self):
        self.source.release()

    def _draw_status(self, frame, text, color):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
