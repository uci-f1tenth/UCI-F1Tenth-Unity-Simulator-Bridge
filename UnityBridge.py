import socket
import struct
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Tuple

class VehicleStatus(IntEnum):
    OK = 0
    CRASHED = 1

@dataclass
class TelemetryData:
    time: float = 0.0
    dt: float = 0.0
    speed: float = 0.0
    steer_cmd: float = 0.0
    steer_phys: float = 0.0
    throttle: float = 0.0
    ticks: int = 0
    rpm: float = 0.0
    euler: Tuple[float, float, float] = (0, 0, 0)
    lin_accel: Tuple[float, float, float] = (0, 0, 0)
    ang_accel: Tuple[float, float, float] = (0, 0, 0)
    reward: float = 0.0
    lap_count: int = 0
    status: VehicleStatus = VehicleStatus.OK
    lidar_count: int = 0
    lidar_ranges: list[float] = field(default_factory=list)

class UnityTelemetryBridge:
    def __init__(self, ip="127.0.0.1", rx_port=5005, tx_port=5006, lidar_max_dist=20.0):
        self.ip = ip
        self.rx_port = rx_port
        self.tx_port = tx_port
        self.lidar_max_dist = lidar_max_dist
        
        # 6f (system) + 1i (ticks) + 1f (rpm) + 9f (vectors) + 1f (reward) + 3i (lap, crash, lidar)
        self.header_format = '<6fif10f3i' 
        self.header_size = 84
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.ip, self.rx_port))
        
        self.unity_address = (self.ip, self.tx_port)
        self.latest_data = TelemetryData()
        self.running = False
        self._lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

    def _listen_loop(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(65535)
                if len(data) >= self.header_size:
                    # 1. Unpack fixed-size header
                    h = struct.unpack(self.header_format, data[:self.header_size])
                    
                    lidar_n = h[20]
                    ranges = []
                    
                    # 2. Extract Lidar array if points exist
                    if lidar_n > 0:
                        payload_offset = self.header_size
                        total_len = payload_offset + (lidar_n * 2)
                        
                        if len(data) >= total_len:
                            # Unpack N unsigned shorts (H) from the lidar payload section
                            # '<' Little Endian, '{lidar_n}H' repeat H for lidar_n times
                            raw_lidar = struct.unpack(f'<{lidar_n}H', data[payload_offset:total_len])
                            
                            # Convert quantized ushorts back to meters
                            # (val / 65535.0) scales to 0.0-1.0, then multiply by max distance
                            ranges = [(v / 65535.0) * self.lidar_max_dist for v in raw_lidar]

                    # 3. Thread-safe update of the data object
                    with self._lock:
                        self.latest_data = TelemetryData(
                            time=h[0], dt=h[1], speed=h[2],
                            steer_cmd=h[3], steer_phys=h[4], throttle=h[5],
                            ticks=h[6], rpm=h[7],
                            euler=(h[8], h[9], h[10]),
                            lin_accel=(h[11], h[12], h[13]),
                            ang_accel=(h[14], h[15], h[16]),
                            reward=h[17],
                            lap_count=h[18],
                            status=VehicleStatus(h[19]),
                            lidar_count=lidar_n,
                            lidar_ranges=ranges
                        )
            except Exception as e:
                if self.running: print(f"Bridge Error: {e}")

    def get_telemetry(self) -> TelemetryData:
        with self._lock:
            return self.latest_data

    def send_control(self, steering: float, throttle: float):
        try:
            pkt = struct.pack('<ff', float(steering), float(throttle))
            self.sock.sendto(pkt, self.unity_address)
        except Exception as e:
            print(f"Send Error: {e}")

    def stop(self):
        self.running = False
        self.sock.close()