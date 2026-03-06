import socket
import struct
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Tuple

class VehicleStatus(IntEnum):
    OK = 0
    CRASHED = 1

@dataclass
class TelemetryData:
    """
    Optimized Data Class.
    Vectors are now stored as numpy arrays for faster processing,
    but the names match your specific requested units.
    """
    sequenceId: int = 0
    timeSinceStart: float = 0.0
    fixedDeltaTime: float = 0.0
    speedMS: float = 0.0
    steering: float = 0.0
    steeringAngleDeg: float = 0.0  # MATCHED: Degrees
    throttle: float = 0.0
    wheelTicks: int = 0
    currentWheelRPM: float = 0.0
    
    # vectors stored as numpy arrays for speed, names matched to your request
    imuAngleDeg: np.ndarray = field(default_factory=lambda: np.zeros(3)) 
    linAccel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angAccelDeg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    reward: float = 0.0
    lapCount: int = 0
    isCrashed: VehicleStatus = VehicleStatus.OK
    lidarCount: int = 0
    lidarRanges: np.ndarray = field(default_factory=lambda: np.zeros(0))

class UnityTCPBridge:
    def __init__(self, ip="127.0.0.1", port=5005, lidar_max_dist=20.0):
        self.ip = ip
        self.port = port
        self.lidar_scale = lidar_max_dist / 65535.0
        
        # --- FORMAT STRING (88 Bytes) ---
        # Matches your specific C# struct layout
        self.header_format = '<I6fif9ff3i' 
        self.header_size = struct.calcsize(self.header_format)
        
        # Optimization: Pre-allocate reusable buffer
        self._header_buffer = bytearray(self.header_size)
        
        # State Tracking
        self.data = TelemetryData() # Reuse this single instance
        self.last_sequence_id = -1
        self.running = False
        self.connected = False
        self._lock = threading.Lock()
        self.sock = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._connection_loop, daemon=True)
        self.thread.start()
        print(f"🔄 Bridge starting... Connecting to {self.ip}:{self.port}")

    def _connection_loop(self):
        while self.running:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                print(f"⏳ Connecting to {self.ip}:{self.port}...")
                self.sock.connect((self.ip, self.port))
                
                self.connected = True
                self.last_sequence_id = -1
                print("✅ Connected to Unity!")
                
                self._listen_loop()
                
            except (ConnectionRefusedError, OSError):
                time.sleep(1.0)
            finally:
                self.connected = False
                if self.sock: self.sock.close()

    def _recv_into(self, buffer, n):
        """Zero-copy read directly into pre-allocated memory."""
        view = memoryview(buffer)[:n]
        while n > 0:
            n_read = self.sock.recv_into(view, n)
            if n_read == 0:
                raise ConnectionError("Socket closed")
            view = view[n_read:]
            n -= n_read

    def _listen_loop(self):
        # Cache lookups for speed
        unpack = struct.unpack
        fmt = self.header_format
        header_buf = self._header_buffer
        lidar_scale = self.lidar_scale
        
        while self.running and self.connected:
            try:
                # 1. Read Header (Zero Copy)
                self._recv_into(header_buf, self.header_size)
                h = unpack(fmt, header_buf)
                
                # 2. Sequence Check
                seq_id = h[0]
                if seq_id <= self.last_sequence_id and self.last_sequence_id != -1:
                    continue 
                self.last_sequence_id = seq_id

                # 3. Read Lidar Payload
                lidar_n = h[21]
                lidar_ranges = np.empty(0)

                if lidar_n > 0:
                    bytes_needed = lidar_n * 2
                    lidar_data = self.sock.recv(bytes_needed, socket.MSG_WAITALL)
                    
                    # Optimization: Numpy Vectorization (C-Speed)
                    raw = np.frombuffer(lidar_data, dtype=np.uint16)
                    lidar_ranges = raw.astype(np.float32) * lidar_scale

                # 4. Update State (In Place)
                with self._lock:
                    d = self.data
                    d.sequenceId       = h[0]
                    d.timeSinceStart   = h[1]
                    d.fixedDeltaTime   = h[2]
                    d.speedMS          = h[3]
                    d.steering         = h[4]
                    d.steeringAngleDeg = h[5]  # MATCHED
                    d.throttle         = h[6]
                    d.wheelTicks       = h[7]
                    d.currentWheelRPM  = h[8]
                    
                    # Direct slice assignment (Fastest way to update vectors)
                    d.imuAngleDeg[:]   = h[9:12] # MATCHED
                    d.linAccel[:]      = h[12:15]
                    d.angAccelDeg[:]   = h[15:18] # MATCHED
                    
                    d.reward           = h[18]
                    d.lapCount         = h[19]
                    d.isCrashed        = VehicleStatus(h[20])
                    d.lidarCount       = lidar_n
                    d.lidarRanges      = lidar_ranges

            except (ConnectionError, struct.error, OSError) as e:
                print(f"🔌 Connection Lost: {e}")
                break
            except Exception as e:
                print(f"🚨 Unexpected Error: {e}")
                break

    def get_telemetry(self) -> TelemetryData:
        return self.data

    def send_control(self, steering: float, throttle: float):
        if not self.connected or not self.sock: return
        try:
            self.sock.sendall(struct.pack('<ff', float(steering), float(throttle)))
        except OSError:
            self.connected = False

    def stop(self):
        self.running = False
        if self.sock: self.sock.close()