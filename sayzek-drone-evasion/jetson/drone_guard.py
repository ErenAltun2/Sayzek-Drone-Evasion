"""
drone_guard.py  —  Gazebo modu
================================
ControlOutput → MAVLink hız komutu (iris_1, port 14550)
"""
import os
import time
from dronekit import connect
from pymavlink import mavutil
from yolo_controller import ControlOutput
from dronekit import connect, VehicleMode


DRONEKIT_ADDRESS = os.getenv("MAVLINK_ADDR", "udp:0.0.0.0:14550")
STOP_ON_IDLE     = True


class DroneGuard:
    def __init__(self):
        print(f"[DroneGuard] Baglaniyor: {DRONEKIT_ADDRESS}")
        self.vehicle = connect(DRONEKIT_ADDRESS, wait_ready=True)
        print("[DroneGuard] Baglanti kuruldu.")

    def send(self, output: ControlOutput):
        if output.action == "RTL":
            self._send_rtl_command()
            return
        if output.action == "BEKLE" and STOP_ON_IDLE:
            self._send_velocity(0.0, 0.0, 0.0, 0.0)
            return
        self._send_velocity(output.vx, output.vy, output.vz, output.yaw_rate)

    def _send_rtl_command(self):
        self.vehicle.mode = VehicleMode("RTL")
        print("[DroneGuard] RTL modu aktif.")

    def _send_velocity(self, vx, vy, vz, yaw_rate=0.0):
        # type_mask: hız + yaw_rate aktif, pozisyon/ivme devre dışı
        # 0b0000010111000111 → vx,vy,vz + yaw_rate
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000010111000111,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, yaw_rate
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    def close(self):
        try:
            self._send_velocity(0.0, 0.0, 0.0)
            time.sleep(0.2)
            self.vehicle.close()
            print("[DroneGuard] Baglanti kapatildi.")
        except Exception as e:
            print(f"[DroneGuard] Kapatma hatasi: {e}")
