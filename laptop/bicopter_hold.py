#!/usr/bin/env python3
"""
Bicopter - Havada Sabit Tut
Kullanım:
  1. MAVProxy'den manuel kaldır (STABILIZE + rc 3 1600)
  2. İstediğin yüksekliğe gelince bu scripti çalıştır
  3. Script QLOITER'a geçer ve uçak orada sabit kalır
"""

from pymavlink import mavutil
import time

CONNECT = "udp:localhost:14550"

MODE_QLOITER = 19
MODE_QRTL    = 20

def set_mode(mav, mode_id, name=""):
    mav.mav.set_mode_send(
        mav.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )
    time.sleep(0.8)
    print(f"  → Mod: {name}")

def get_alt(mav):
    msg = mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2)
    return (msg.relative_alt / 1000.0) if msg else 0.0

# ── BAĞLAN ──────────────────────────────────────────────────────────────────
print("=" * 50)
print("Bicopter Havada Tut")
print("=" * 50)
print()
print("ÖNCE MAVProxy'den uçağı kaldır:")
print("  mode stabilize")
print("  arm throttle")
print("  rc 3 1600   ← kalkınca...")
print("  rc 3 1200   ← motoru kapat (STABILIZE'da mid throttle)")
print()
input("Uçak havaya kalktıktan sonra ENTER'a bas...")

print(f"\nBağlanıyor → {CONNECT}")
mav = mavutil.mavlink_connection(CONNECT)
mav.wait_heartbeat()
print(f"✓ Heartbeat | system={mav.target_system}")

alt = get_alt(mav)
print(f"  Mevcut yükseklik: {alt:.1f}m\n")

# ── QLOITER → SABİT KAL ─────────────────────────────────────────────────────
print("QLOITER moduna geçiliyor - uçak sabit kalacak...")
set_mode(mav, MODE_QLOITER, "QLOITER")
time.sleep(1)

alt = get_alt(mav)
print(f"\n✅ Uçak {alt:.1f}m'de sabit tutuluyor!")
print("   Ctrl+C → QRTL ile otomatik iniş\n")

# ── TUTMA DÖNGÜSÜ ────────────────────────────────────────────────────────────
try:
    while True:
        alt = get_alt(mav)
        hb = mav.recv_match(type="HEARTBEAT", blocking=False)
        mode = hb.custom_mode if hb else "?"
        print(f"  Alt: {alt:.1f}m | Mode: {mode}        ", end="\r")
        time.sleep(1)

except KeyboardInterrupt:
    print("\n\nİniş → QRTL")
    set_mode(mav, MODE_QRTL, "QRTL")
    print("✓ Uçak otomatik inecek.")
