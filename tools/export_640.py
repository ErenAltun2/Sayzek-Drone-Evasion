import torch
import torch.nn as nn
import ultralytics.utils.loss as _loss_mod
from ultralytics import YOLO

# --- YOLO26 Yaması (Önceki sürümlerdeki hataları önlemek için) ---
if not hasattr(_loss_mod, "E2ELoss"):
    class E2ELoss(nn.Module):
        def __init__(self, model): super().__init__()
        def forward(self, *a, **kw): pass
    _loss_mod.E2ELoss = E2ELoss
# -----------------------------------------------------------------

# Kendi eğittiğin .pt dosyasının adını buraya yaz (örneğin 'best.pt')
MODEL_ADI = "Yolo26-640.pt" 

print(f"YOLO Modeli Yükleniyor: {MODEL_ADI}...")
model = YOLO(MODEL_ADI)

print("FP16 ve 640x640 TensorRT formatına dönüştürülüyor...")
model.export(
    format="engine",
    device="0",
    imgsz=640,         # Yeni eğitim çözünürlüğün
    half=True,         # Güvenilir FP16
    int8=False,        # INT8 çökmesini engelliyoruz
    workspace=4
)
print("Dönüştürme Tamamlandı! Yeni .engine motorun hazır.")
