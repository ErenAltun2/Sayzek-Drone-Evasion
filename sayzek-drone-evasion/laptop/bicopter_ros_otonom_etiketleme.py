#!/usr/bin/env python3

import subprocess
import random
import math
import time
import os
from datetime import datetime
import glob
import cv2
import numpy as np

print("otomatik etiketleme sistemi başlatılıyor...")
print("Durdurmak için Ctrl+C basın")
print()

# Screenshot ve label klasörleri oluştur (ABSOLUTE PATH kullan)
BASE_DIR = os.path.abspath("dataset")
screenshot_dir = os.path.join(BASE_DIR, "images")
label_dir = os.path.join(BASE_DIR, "labels")
visualized_dir = os.path.join(BASE_DIR, "visualized")  # Bbox çizili görseller için

os.makedirs(screenshot_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)
os.makedirs(visualized_dir, exist_ok=True)

print(f"✓ Dataset klasörleri oluşturuldu")
print(f"  - Images: {screenshot_dir}")
print(f"  - Labels: {label_dir}")
print(f"  - Visualized: {visualized_dir}")
print()

# Ekran çözünürlüğü
SCREEN_WIDTH = 1849
SCREEN_HEIGHT = 968
MAX_SAMPLES = 355  # Toplam oluşturulacak sample sayısı
# GELİŞTİRİLMİŞ PARAMETRELER
MIN_DISTANCE = 3.0   # Minimum kamera mesafesi
MAX_DISTANCE = 30.0  # Maksimum kamera mesafesi (uzak gözetleme)

# DRONE GÖRSEL BOYUTLARI (gerçek screenshot'lardan kalibre edildi)
# Fiziksel SDF boyutu değil, Gazebo'da kameraya yansıyan görsel boyut.
# İki referans görsel kullanılarak perspektif projeksiyondan ters hesaplandı:
#   - dist=3.05m elev=11.5° → ~190x120px hedef
#   - dist=6.40m elev=43.8° → ~70x75px hedef
DRONE_WIDTH  = 0.182  # görsel efektif genişlik (Y ekseni, kalibre)
DRONE_HEIGHT = 0.170  # görsel efektif yükseklik (Z ekseni, kalibre)

# Kamera FOV (Gazebo default)
FOV_HORIZONTAL = 60  # derece
FOV_VERTICAL = 60 * (SCREEN_HEIGHT / SCREEN_WIDTH)  # aspect ratio'ya göre


def quat_to_rot(q):
    """Quaternion (x,y,z,w) → 3x3 rotasyon matrisi"""
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)  ],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)  ],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]
    ])


def read_gz_pose(topic, name_filter=None):
    """Gazebo topic'ten pozisyon ve quaternion oku."""
    import re
    try:
        cmd = ["gz", "topic", "-e", "-n", "1", "-t", topic]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if result.returncode != 0:
            return None
        text = result.stdout

        if name_filter:
            idx = text.find(f'name: "{name_filter}"')
            if idx == -1:
                return None
            text = text[idx:]

        pos_idx = text.find('position {')
        ori_idx = text.find('orientation {')
        if pos_idx == -1 or ori_idx == -1:
            return None

        pos_block = text[pos_idx:pos_idx+120]
        ori_block = text[ori_idx:ori_idx+200]

        coords = re.findall(r'[xyz]:\s*([-\d.e+]+)', pos_block)
        px, py, pz = float(coords[0]), float(coords[1]), float(coords[2])

        qvals = re.findall(r'[xyzw]:\s*([-\d.e+]+)', ori_block)
        qx, qy, qz, qw = float(qvals[0]), float(qvals[1]), float(qvals[2]), float(qvals[3])

        return np.array([px, py, pz]), np.array([qx, qy, qz, qw])
    except Exception as e:
        print(f"  ⚠️ Pose okuma hatası ({topic}): {e}")
        return None


def calculate_bbox(distance, elevation_angle=0):
    """
    Gerçek 3D projeksiyon ile bbox hesapla.
    Gazebo'dan kamera ve drone pozisyonunu okur, perspektif projeksiyonla
    tam doğru center_x/center_y hesaplar.

    distance ve elevation_angle fallback için tutuldu.
    """
    # Kamera pozisyonu
    cam_result = read_gz_pose("/gui/camera/pose")
    drone_result = read_gz_pose("/world/baylands_01/pose/info", name_filter="bicopter_with_ardupilot")
    if cam_result is None or drone_result is None:
        print("  ⚠️ Pose okunamadı, tahmini değerler kullanılıyor")
        center_x, center_y = 0.5, 0.42
    else:
        cam_pos, cam_quat = cam_result
        drone_pos, _      = drone_result

        # Drone → kamera koordinat sistemi
        R          = quat_to_rot(cam_quat)
        drone_cam  = R.T @ (drone_pos - cam_pos)

        # Gazebo kamera eksenleri: [0]=ileri, [2]=sağ, [1]=yukarı
        depth = drone_cam[0]
        right = drone_cam[2]
        up    = drone_cam[1]

        if depth < 0.1:
            print(f"  ⚠️ Drone kameranın arkasında (depth={depth:.2f})")
            center_x, center_y = 0.5, 0.42
        else:
            half_fov_h = math.tan(math.radians(FOV_HORIZONTAL / 2))
            half_fov_v = half_fov_h * (SCREEN_HEIGHT / SCREEN_WIDTH)
            center_x = 0.5 + right / (depth * half_fov_h * 2) * 1.0
            center_y = 0.5 - up    / (depth * half_fov_v * 2) * 1.0
            center_x = 0.5 + (right / depth) / (2 * half_fov_h)
            center_y = 0.5 - (up    / depth) / (2 * half_fov_v)

            print(f"  📐 3D: depth={depth:.2f}m R={right:.3f} U={up:.3f} → cx={center_x:.3f} cy={center_y:.3f}")

        # Bbox boyutu için gerçek mesafeyi kullan
        distance = float(np.linalg.norm(drone_pos - cam_pos))
        elevation_angle = math.degrees(math.asin(
            max(-1.0, min(1.0, (drone_pos[2] - cam_pos[2]) / distance))
        ))

    # center_y: 3D projeksiyondan gelen değeri direkt kullan (correction yok)
    # Follow kamerası tüm açılarda drone'u yeterince ortaya alıyor

    # Bbox boyutu
    focal  = 1.0 / math.tan(math.radians(FOV_HORIZONTAL / 2))
    er     = math.radians(elevation_angle)
    ew     = DRONE_WIDTH  * abs(math.cos(er))
    eh     = DRONE_HEIGHT * abs(math.cos(er)) + DRONE_WIDTH * abs(math.sin(er))
    width  = (ew * focal) / distance
    height = (eh * focal) / distance

    # Orantılı padding: yakında daha fazla, uzakta daha az
    # Mesafe arttıkça drone zaten küçük, sabit padding orantısız büyür
    padding_scale = max(0.3, min(1.0, 5.0 / distance))  # 5m'de 1.0, 15m'de 0.33
    width  += padding_scale * 15 / SCREEN_WIDTH
    height += padding_scale * 18 / SCREEN_HEIGHT

    width  = min(max(width,  0.02), 0.90)
    height = min(max(height, 0.015), 0.90)

    # Küçük gürültü (±2 piksel)
    center_x = max(0.05, min(0.95, center_x + random.uniform(-2/SCREEN_WIDTH,  2/SCREEN_WIDTH)))
    center_y = max(0.05, min(0.95, center_y + random.uniform(-2/SCREEN_HEIGHT, 2/SCREEN_HEIGHT)))

    return center_x, center_y, width, height


def save_label(label_path, bbox, class_id=4):
    """
    YOLO formatında label dosyası kaydet
    Format: class_id center_x center_y width height
    """
    center_x, center_y, width, height = bbox
    with open(label_path, 'w') as f:
        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")


def get_latest_file(directory, extension="*.png"):
    """
    Belirtilen dizindeki en son oluşturulan dosyayı bul
    """
    files = glob.glob(os.path.join(directory, extension))
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def draw_bbox_on_image(image_path, bbox, output_path, distance, elevation):
    """
    Görsel üzerine bounding box çiz
    bbox: (center_x, center_y, width, height) normalized [0-1]
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  ⚠️ Görsel okunamadı: {image_path}")
        return False
    
    h, w = img.shape[:2]
    
    # YOLO formatını pixel koordinatlarına çevir
    center_x, center_y, box_width, box_height = bbox
    
    x_center_px = int(center_x * w)
    y_center_px = int(center_y * h)
    box_w_px = int(box_width * w)
    box_h_px = int(box_height * h)
    
    # Sol üst ve sağ alt köşe
    x1 = int(x_center_px - box_w_px / 2)
    y1 = int(y_center_px - box_h_px / 2)
    x2 = int(x_center_px + box_w_px / 2)
    y2 = int(y_center_px + box_h_px / 2)
    
    # Bounding box çiz (yeşil, kalınlık 3)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # Bilgi metni
    text1 = f"Dist: {distance:.2f}m | Elev: {elevation:.1f}deg"
    text2 = f"Box: {box_width:.3f} x {box_height:.3f} | Pixels: {box_w_px}x{box_h_px}"
    
    cv2.putText(img, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Class label
    cv2.putText(img, "bicopter", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imwrite(output_path, img)
    return True


# Follow'u aktif et
print("Follow servisi aktif ediliyor...")
follow_cmd = [
    "gz", "service",
    "-s", "/gui/follow",
    "--reqtype", "gz.msgs.StringMsg",
    "--reptype", "gz.msgs.Boolean",
    "--timeout", "2000",
    "--req", 'data: "bicopter_with_ardupilot"'
]
result = subprocess.run(follow_cmd, capture_output=True, text=True)
if result.returncode == 0:
    print("✓ Follow aktif edildi!\n")
else:
    print("✗ Follow aktif edilemedi!")
    print(f"Hata: {result.stderr}")
    exit(1)

sample_count = 0

# İSTATİSTİKLER
distance_samples = []
elevation_samples = []

try:
    while sample_count < MAX_SAMPLES:        # RASTGELE AÇILAR - DARALTILMIŞ (sadece ön yarım daire)
        # 0-360 yerine -90 ile +90 arası (önden ve yanlardan)
        angle = random.uniform(-90, 90)  # Yatay açı - sadece ön yarım
        
        # MESAFE - Uzak gözetleme için dağılım
        # %30 yakın (3-8m), %40 orta (8-18m), %30 uzak (18-30m)
        rand = random.random()
        if rand < 0.30:
            radius = random.uniform(3.0, 8.0)
        elif rand < 0.70:
            radius = random.uniform(8.0, 18.0)
        else:
            radius = random.uniform(18.0, 30.0)
        
        # YÜKSEKLİK AÇISI - Dengeli dağılım
        # -30° ile +60° arası ama eşit dağılım için bölümlere ayır
        elev_rand = random.random()
        if elev_rand < 0.25:
            elevation_angle = random.uniform(-30, 0)   # %25 aşağıdan bakış
        elif elev_rand < 0.55:
            elevation_angle = random.uniform(0, 20)    # %30 yatay/hafif yukarı
        elif elev_rand < 0.80:
            elevation_angle = random.uniform(20, 45)   # %25 orta açı
        else:
            elevation_angle = random.uniform(45, 60)   # %20 yüksek açı
        
        # Z yüksekliği hesapla (elevation açısından)
        z = radius * math.sin(math.radians(elevation_angle))
        
        # Yatay mesafe (x-y düzleminde)
        horizontal_distance = radius * math.cos(math.radians(elevation_angle))
        
        # Açıyı radyana çevir ve X, Y hesapla
        angle_rad = math.radians(angle)
        x = horizontal_distance * math.cos(angle_rad)
        y = horizontal_distance * math.sin(angle_rad)
        
        print(f"\n{'='*70}")
        print(f"Sample #{sample_count + 1}")
        print(f"Yatay açı: {angle:.1f}° | Yükseklik açısı: {elevation_angle:.1f}°")
        print(f"Mesafe: {radius:.2f}m | Yükseklik: {z:.2f}m")
        print(f"  -> Offset: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        
        # Offset servisini çağır
        cmd = [
            "gz", "service",
            "-s", "/gui/follow/offset",
            "--reqtype", "gz.msgs.Vector3d",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "2000",
            "--req", f"x: {x}, y: {y}, z: {z}"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  ⚠️ Offset hatası: {result.stderr.strip()}")
            continue
        
        print("  ✓ Offset uygulandı")
        
        # Kameranın hareket etmesi için bekle
        time.sleep(2.5)
        
        # Screenshot öncesi dosyaları kaydet
        files_before = set(glob.glob(os.path.join(screenshot_dir, "*.png")))
        
        # Screenshot al - absolute path kullan!
        screenshot_cmd = [
                "gz", "service",
                "-s", "/gui/screenshot",
                "--reqtype", "gz.msgs.StringMsg",
                "--reptype", "gz.msgs.Boolean",
                "--timeout", "2000",
                "--req", f'data: "{screenshot_dir}/"'
            ]
        
        print(f"  📷 Screenshot path: {screenshot_dir}/")
        screenshot_result = subprocess.run(screenshot_cmd, capture_output=True, text=True)
        print(f"     returncode={screenshot_result.returncode} | stdout={screenshot_result.stdout.strip()} | stderr={screenshot_result.stderr.strip()[:80]}")
        
        if screenshot_result.returncode == 0:
            # Screenshot oluşması için yeterince bekle
            time.sleep(2.0)
            
            # Yeni dosyayı bul
            files_after = set(glob.glob(os.path.join(screenshot_dir, "*.png")))
            new_files = files_after - files_before
            
            if new_files:
                latest_image = list(new_files)[0]
                image_name = os.path.basename(latest_image)
                
                label_name = image_name.replace('.png', '.txt')
                label_path = os.path.join(label_dir, label_name)
                
                viz_name = image_name.replace('.png', '_viz.png')
                viz_path = os.path.join(visualized_dir, viz_name)
                
                # GELİŞTİRİLMİŞ BBOX HESAPLAMA
                bbox = calculate_bbox(radius, elevation_angle)
                center_x, center_y, width, height = bbox
                
                print(f"  📸 Screenshot: {image_name}")
                print(f"  📦 Bounding Box:")
                print(f"     Normalized: ({center_x:.4f}, {center_y:.4f}) | {width:.4f} x {height:.4f}")
                print(f"     Pixels: {int(width*SCREEN_WIDTH)}px x {int(height*SCREEN_HEIGHT)}px")
                print(f"     Ekran kapsamı: %{(width*height*100):.1f}")
                
                # Label kaydet
                save_label(label_path, bbox, class_id=4)
                print(f"  🏷️  Label: {label_name}")
                
                # Visualize
                if draw_bbox_on_image(latest_image, bbox, viz_path, radius, elevation_angle):
                    print(f"  🎨 Visualized: {viz_name}")
                
                # İstatistik topla
                distance_samples.append(radius)
                elevation_samples.append(elevation_angle)
                
                sample_count += 1
                
                # Her 10 sample'da bir istatistik göster
                if sample_count % 10 == 0:
                    avg_dist = sum(distance_samples[-10:]) / min(10, len(distance_samples))
                    avg_elev = sum(elevation_samples[-10:]) / min(10, len(elevation_samples))
                    print(f"\n  📊 Son 10 sample ortalaması:")
                    print(f"     Mesafe: {avg_dist:.2f}m | Yükseklik açısı: {avg_elev:.1f}°")
                
                print(f"  ✅ Toplam sample: {sample_count}")
            else:
                print(f"  ⚠️ Yeni screenshot bulunamadı!")
        else:
            print(f"  ⚠️ Screenshot hatası: {screenshot_result.stderr.strip()}")
        
        # Bekleme süresi
        time.sleep(2)

except KeyboardInterrupt:
    print("\n\n" + "="*70)
    print("Otomatik etiketleme durduruldu.")
    print(f"✅ Toplam {sample_count} sample oluşturuldu")
    
    if distance_samples:
        print(f"\n📊 İSTATİSTİKLER:")
        print(f"   Ortalama mesafe: {sum(distance_samples)/len(distance_samples):.2f}m")
        print(f"   Min/Max mesafe: {min(distance_samples):.2f}m / {max(distance_samples):.2f}m")
        print(f"   Ortalama yükseklik açısı: {sum(elevation_samples)/len(elevation_samples):.1f}°")
        print(f"   Min/Max açı: {min(elevation_samples):.1f}° / {max(elevation_samples):.1f}°")
    
    print(f"\n📁 Dataset konumu:")
    print(f"   Images: {screenshot_dir}/")
    print(f"   Labels: {label_dir}/")
    print(f"   Visualized: {visualized_dir}/")
    print("="*70)