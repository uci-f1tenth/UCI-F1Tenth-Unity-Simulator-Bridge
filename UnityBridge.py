import socket
import struct
import time
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Tuple, Optional

class VehicleStatus(IntEnum):
    OK = 0
    CRASHED = 1

@dataclass
class TelemetryData:
    sequenceId: int = 0
    timeSinceStart: float = 0.0
    fixedDeltaTime: float = 0.0
    speedMS: float = 0.0
    steering: float = 0.0
    steeringAngleRad: float = 0.0
    throttle: float = 0.0
    wheelTicks: int = 0
    currentWheelRPM: float = 0.0
    imuAngle: Tuple[float, float, float] = (0, 0, 0)
    linAccel: Tuple[float, float, float] = (0, 0, 0)
    angAccel: Tuple[float, float, float] = (0, 0, 0)
    reward: float = 0.0
    lapCount: int = 0
    isCrashed: VehicleStatus = VehicleStatus.OK
    lidarCount: int = 0
    lidarRanges: list[float] = field(default_factory=list)

class UnityTelemetryBridge:
    def __init__(self, ip="127.0.0.1", rx_port=5005, tx_port=5006, lidar_max_dist=20.0):
        self.ip = ip
        self.rx_port = rx_port
        self.tx_port = tx_port
        self.lidar_max_dist = lidar_max_dist
        
        # --- FORMAT STRING (88 Bytes) ---
        # < : Little Endian
        # I : Sequence ID (1 uint)
        # 6f: time, dt, speed, steer, steerRad, throttle (6 floats)
        # i : wheelTicks (1 int)
        # f : wheelRPM (1 float)
        # 9f: imuAngle(3), linAccel(3), angAccel(3) (9 floats)
        # f : reward (1 float)
        # 3i: lapCount, isCrashed, lidarCount (3 ints)
        self.header_format = '<I6fi f9f f3i' 
        self.header_size = struct.calcsize(self.header_format)
        
        # Networking
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind to 0.0.0.0 to ensure it listens on all interfaces
        self.sock.bind((self.ip, self.rx_port))
        
        self.unity_address = (self.ip, self.tx_port)
        
        # State Tracking
        self.latest_data = TelemetryData()
        self.last_sequence_id = -1
        self.running = False
        self._lock = threading.Lock()

    def start(self):
        """Starts the background listener thread."""
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        print(f"✅ Bridge started. Listening on :{self.rx_port}, Sending to {self.ip}:{self.tx_port}")

    def _listen_loop(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(65535)
                
                if len(data) < self.header_size:
                    # Ignore packets that don't meet minimum header size
                    continue

                # 1. Unpack fixed-size header
                h = struct.unpack(self.header_format, data[:self.header_size])
                
                # 2. Sequence Check (Drop old/duplicate packets)
                seq_id = h[0]
                if self.last_sequence_id != -1:
                    if seq_id <= self.last_sequence_id:
                        continue
                    if seq_id > self.last_sequence_id + 1:
                        lost = seq_id - self.last_sequence_id - 1
                        print(f"⚠️ Packet Loss Warning: Skipped {lost} frames. (Current Seq: {seq_id})")
                
                self.last_sequence_id = seq_id

                # 3. Extract Lidar (if points exist)
                lidar_n = h[21]
                ranges = []
                if lidar_n > 0:
                    payload_offset = self.header_size
                    total_len = payload_offset + (lidar_n * 2)
                    
                    if len(data) >= total_len:
                        raw_lidar = struct.unpack(f'<{lidar_n}H', data[payload_offset:total_len])
                        ranges = [(v / 65535.0) * self.lidar_max_dist for v in raw_lidar]
                    else:
                        print(f"❌ Lidar Payload Error: Expected {lidar_n} points, but packet too small.")

                # 4. Thread-safe Update
                with self._lock:
                    self.latest_data = TelemetryData(
                        sequenceId       = h[0],
                        timeSinceStart   = h[1],
                        fixedDeltaTime   = h[2],
                        speedMS          = h[3],
                        steering         = h[4],
                        steeringAngleRad = h[5],
                        throttle         = h[6],
                        wheelTicks       = h[7],
                        currentWheelRPM  = h[8],
                        imuAngle         = (h[9], h[10], h[11]),
                        linAccel         = (h[12], h[13], h[14]),
                        angAccel         = (h[15], h[16], h[17]),
                        reward           = h[18],
                        lapCount         = h[19],
                        isCrashed        = VehicleStatus(h[20]),
                        lidarCount       = lidar_n,
                        lidarRanges      = ranges
                    )

            except Exception as e:
                if self.running:
                    print(f"🚨 Bridge Error: {e}")

    def get_telemetry(self) -> TelemetryData:
        """Returns the most recent telemetry data received."""
        with self._lock:
            return self.latest_data

    def send_control(self, steering: float, throttle: float):
        """Sends steering and throttle commands to Unity."""
        try:
            pkt = struct.pack('<ff', float(steering), float(throttle))
            self.sock.sendto(pkt, self.unity_address)
        except Exception as e:
            print(f"❌ Send Error: {e}")

    def stop(self):
        """Cleanly shuts down the bridge."""
        self.running = False
        self.sock.close()
        print("🛑 Bridge stopped.")