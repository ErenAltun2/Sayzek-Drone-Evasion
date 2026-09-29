# SAYZEK — Vision-Based UAV Threat Detection, Classification and Evasion

**🇬🇧 English | [🇹🇷 Türkçe](README.md)**

> **Advanced Image-Processing-Based Threat Detection, Classification and Evasion for Autonomous Aerial Systems**
> Eren Altun, Mustafa Kale — Software Engineering, Karabük University

An integrated system that combines YOLO26-based enemy UAV detection, pixel-derived distance estimation, a rule-based adaptive evasion controller, and autonomous in-field data collection for previously unseen UAV types (few-shot adaptation). All components were validated in a **Hardware-in-the-Loop (HIL)** setup on an **NVIDIA Jetson Orin Nano**.

📄 [Paper](docs/paper.pdf) (Turkish) · 🖼️ [Poster](docs/poster.pdf) · ▶️ [Presentation video](https://www.youtube.com/watch?v=yl1YAEBlThU&t=23s)

## Key results

| Result | Value |
|---|---|
| YOLO26-1280 detection range (Jetson Orin Nano, HIL) | **21.6 m** |
| YOLO26-960 / YOLO26-640 detection range | 15.4 m / 11.5 m |
| Evasion success rate in simulation | **86.7%** (13 of 16 attempts) |
| Autonomous data collection | 30 frames, no human intervention, YOLO-format labels |

## Architecture

![System architecture](docs/images/architecture.png)

The simulation (Gazebo + ArduPilot SITL) runs on a **laptop**. Model inference and motion-command generation run on the **Jetson**. The two sides communicate over a LAN (Ethernet) using ROS2 (Fast DDS) and MAVLink.

The Jetson side runs four parallel threads connected by `maxsize=1` queues, so each thread always works with the freshest data:

| Thread | Role |
|---|---|
| T1 – Detection | Reads frames from the ROS2 camera topic, runs YOLO26 (TensorRT) inference |
| T2 – Control | Computes Vx/Vy/Vz/Yaw; manages OrbitCollector and FewShotTracker |
| T3 – MAVLink | Sends velocity commands to ArduPilot at 20 Hz; RTL uses a separate queue |
| T-RTL | Immediate RTL when `rtl` is typed in the terminal |

### How it works

- **Threat estimation:** instantaneous bounding-box diameter and an EMA-smoothed growth rate are combined into a normalized panic coefficient in [0, 1].
- **Filtering:** independent 1-D Kalman filters per image axis; control laws use a 100 ms look-ahead position estimate.
- **Evasion:** a state machine with hysteresis (KAC / YAKLAS / MESAFE_KOR / BEKLE) and per-axis control laws for forward/back, lateral, vertical velocity and yaw rate. Parameters live in `jetson/config.yaml`.
- **Data collection:** once a "generic" (unknown) drone is stably tracked in the safe distance band, `OrbitCollector` performs left/right sweeps and saves clean frames with auto-generated YOLO labels; `FewShotTracker` keeps the target identity via HSV histogram signatures. After 30 frames the system triggers RTL.

## Repository structure

```
├── jetson/        # Main system running on the Jetson Orin Nano
│   ├── thread_manager.py     # entry point (T1–T3, HUD, CSV log)
│   ├── yolo_detector.py      # ROS2 camera + YOLO26 detection
│   ├── yolo_controller.py    # state machine, EMA, Kalman, evasion control laws
│   ├── drone_guard.py        # DroneKit / MAVLink velocity and RTL commands
│   ├── orbit_collector.py    # automatic sweep + labeled data collection
│   ├── few_shot_tracker.py   # target re-identification via HSV histograms
│   ├── config.yaml           # thresholds, gains, filters, orbit parameters
│   └── fastdds_jetson.xml
├── laptop/        # Simulation side
│   ├── senaryo.py                        # iris_2 (enemy) PID pursuit scenario
│   ├── bicopter_ros_otonom_etiketleme.py # auto-labeled dataset generation from Gazebo
│   ├── bicopter_hold.py                  # helper to hold the bicopter in the air
│   └── fastdds_laptop.xml
├── tools/
│   ├── export_640.py                     # .pt → TensorRT .engine (FP16)
│   └── model_karsilastirma_jetson.py     # model comparison test on Jetson
└── docs/          # paper and poster
```

## Installation

**Requirements:** Ubuntu 22.04, ROS2 Humble, Gazebo Harmonic, ArduPilot SITL, Python 3.10. On the Jetson: JetPack (CUDA/TensorRT).

```bash
pip install -r requirements.txt
```

Model weights (`*.pt`, `*.engine`) are **not** included in this repository. Convert your own trained `Yolo26-640.pt` to a TensorRT engine:

```bash
python tools/export_640.py     # set MODEL_ADI inside the script to your weights file
```

A `.engine` file is device-specific; build it on the Jetson itself.

## Usage

**1) Laptop:** start Gazebo + ArduPilot SITL, then forward the iris_1 and iris_2 MAVLink ports to the Jetson with MAVProxy:

```bash
mavproxy.py --master=udp:127.0.0.1:14550 --out=udp:<JETSON_IP>:14550
mavproxy.py --master=udp:127.0.0.1:14560 --out=udp:<JETSON_IP>:14560
python laptop/senaryo.py          # iris_2 (enemy) chases iris_1
```

**2) Jetson** (once iris_1 is airborne; `config.yaml` is read relative to the working directory, so run from inside `jetson/`):

```bash
cd jetson
python thread_manager.py --drone
```

Press `q` to quit; type `rtl` in the terminal for an immediate return-to-launch.

**Network setup:** the IPs in `fastdds_*.xml` (`10.42.0.1` laptop, `10.42.0.67` Jetson) are example values. Adjust them to your network and activate the profile with:

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/path/to/fastdds_jetson.xml   # fastdds_laptop.xml on the laptop
```

## Known notes

- `ENEMY_TOO_CLOSE_PX=230` / `ENEMY_TOO_FAR_PX=57` in `jetson/yolo_detector.py` differ from `config.yaml` and Table I of the paper (130 / 45); these constants set the `too_close` / `too_far` flags.
- `laptop/bicopter_hold.py` sets `MODE_QRTL = 20`; in ArduPlane, 20 is QLAND and QRTL is 21.
- All results come from simulation/HIL data; real flight tests are future work.

## License

This repository is licensed under **AGPL-3.0**, since it uses Ultralytics YOLO (AGPL-3.0). See [LICENSE](LICENSE).

## Citation

If you use this project in your academic or technical research, please cite it as follows:

```bibtex
@misc{altun_kale_sayzek,
  author       = {Altun, Eren and Kale, Mustafa},
  title        = {Advanced Image-Processing-Based Threat Detection, Classification and Evasion for Autonomous Aerial Systems},
  institution  = {Karabuk University},
  year         = {2026},
  howpublished = {\url{[https://github.com/ErenAltun2/Sayzek-Drone-Evasion](https://github.com/ErenAltun2/Sayzek-Drone-Evasion)}}
}
```
