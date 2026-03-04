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

class UnityTCPBridge:
    def __init__(self, ip="127.0.0.1", port=5005, lidar_max_dist=20.0):
        self.ip = ip
        self.port = port
        self.lidar_max_dist = lidar_max_dist
        
        # --- FORMAT STRING (88 Bytes) ---
        # Same structure as before
        self.header_format = '<I6fi f9f f3i' 
        self.header_size = struct.calcsize(self.header_format)
        
        # State Tracking
        self.latest_data = TelemetryData()
        self.last_sequence_id = -1
        self.running = False
        self.connected = False
        self._lock = threading.Lock()
        
        self.sock = None

    def start(self):
        """Starts the connection and the listener thread."""
        self.running = True
        self.thread = threading.Thread(target=self._connection_loop, daemon=True)
        self.thread.start()
        print(f"🔄 Bridge starting... Connecting to {self.ip}:{self.port}")

    def _connection_loop(self):
        """
        Main Loop: Tries to connect, then runs the listen loop.
        If disconnected, it goes back to trying to connect.
        """
        while self.running:
            try:
                # 1. Create TCP Socket
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) # Critical for latency
                
                # 2. Try Connect
                print(f"⏳ Attempting to connect to Unity at {self.ip}:{self.port}...")
                self.sock.connect((self.ip, self.port))
                
                self.connected = True
                self.last_sequence_id = -1 # Reset sequence tracking on new connection
                print("✅ Connected to Unity!")
                
                # 3. Enter Receive Loop
                self._listen_loop()
                
            except ConnectionRefusedError:
                time.sleep(1) # Unity probably isn't running, retry
            except Exception as e:
                print(f"❌ Connection Error: {e}")
                time.sleep(1)
            finally:
                self.connected = False
                if self.sock:
                    self.sock.close()

    def _recv_exact(self, n):
        """Helper to read exactly n bytes from the TCP stream."""
        data = bytearray()
        while len(data) < n:
            packet = self.sock.recv(n - len(data))
            if not packet:
                raise ConnectionError("Socket closed")
            data.extend(packet)
        return data

    def _listen_loop(self):
        """Reads data from the open socket."""
        while self.running and self.connected:
            try:
                # 1. Read Fixed Header (88 bytes)
                # We use _recv_exact because TCP might give us partial packets
                header_data = self._recv_exact(self.header_size)
                
                # Unpack
                h = struct.unpack(self.header_format, header_data)
                
                # 2. Sequence Check
                seq_id = h[0]
                if self.last_sequence_id != -1:
                    if seq_id <= self.last_sequence_id:
                        continue
                    # Only warn if we've already received at least one packet this session
                    if seq_id > self.last_sequence_id + 1:
                        print(f"⚠️ Gap detected...")
                else:
                    print(f"📈 First packet received. Syncing Sequence at: {seq_id}")

                self.last_sequence_id = seq_id

                # 3. Read Lidar Payload
                lidar_n = h[21]
                ranges = []
                
                if lidar_n > 0:
                    lidar_bytes_needed = lidar_n * 2
                    lidar_data = self._recv_exact(lidar_bytes_needed)
                    
                    raw_lidar = struct.unpack(f'<{lidar_n}H', lidar_data)
                    ranges = [(v / 65535.0) * self.lidar_max_dist for v in raw_lidar]

                # 4. Update State
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

            except (ConnectionError, struct.error, OSError):
                print("🔌 Disconnected from Unity.")
                break # Exit to outer loop to reconnect
            except Exception as e:
                print(f"🚨 Parse Error: {e}")
                break

    def get_telemetry(self) -> TelemetryData:
        with self._lock:
            return self.latest_data

    def send_control(self, steering: float, throttle: float):
        if not self.connected or not self.sock:
            return
            
        try:
            # 8 bytes (2 floats)
            pkt = struct.pack('<ff', float(steering), float(throttle))
            self.sock.sendall(pkt) # sendall ensures all bytes go out
        except Exception as e:
            print(f"❌ Send Error: {e}")
            self.connected = False

    def step(self, steering: float, throttle: float, timeout=1.0) -> TelemetryData:
        """
        Sends command and blocks until the next physics frame arrives.
        """
        if not self.connected:
            # Wait a bit if not connected yet to avoid crashing the training loop
            time.sleep(0.1)
            return self.get_telemetry()

        prev_id = self.last_sequence_id
        self.send_control(steering, throttle)
        
        start_t = time.time()
        while self.last_sequence_id == prev_id:
            if time.time() - start_t > timeout:
                print("⏲️ Step Timeout: Unity didn't respond.")
                break
            time.sleep(0.001)
            
        return self.get_telemetry()

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()
        print("🛑 Bridge stopped.")