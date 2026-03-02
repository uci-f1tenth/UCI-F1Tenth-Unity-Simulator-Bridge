import socket
import struct
import time
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class UnityTelemetry:
    seq_id: int
    time: float
    dt: float
    speed: float
    steer_cmd: float
    steer_phys: float
    throttle: float
    wheel_ticks: int
    wheel_rpm: float
    euler: List[float]
    lin_accel: List[float]
    ang_accel: List[float]
    reward: float
    lap_count: int
    is_crashed: bool
    lidar_count: int
    lidar_ranges: List[float] = field(default_factory=list)

class UnityEnvironment:
    # ---------------------------------------------------------
    # EXPLICIT FORMAT STRING CONSTRUCTION
    # ---------------------------------------------------------
    # We use string multiplication to guarantee the count is correct.
    # Total Items: 1 + 6 + 1 + 1 + 9 + 1 + 3 = 22 items
    # Total Bytes: 22 * 4 = 88 bytes
    
    FMT_SEQ       = "I"      # 1 uint
    FMT_MOTION    = "f" * 6  # 6 floats (Time, DT, Spd, SteerC, SteerP, Thr)
    FMT_TICKS     = "i"      # 1 int (WheelTicks)
    FMT_RPM       = "f"      # 1 float (WheelRPM)
    FMT_IMU       = "f" * 9  # 9 floats (Euler*3, Lin*3, Ang*3)
    FMT_REWARD    = "f"      # 1 float (Reward)
    FMT_STATUS    = "i" * 3  # 3 ints (Lap, Crash, LidarCount)
    
    # Combined: "<Iffffffiffffffffffiii"
    HEADER_FMT = "<" + FMT_SEQ + FMT_MOTION + FMT_TICKS + FMT_RPM + FMT_IMU + FMT_REWARD + FMT_STATUS
    
    HEADER_SIZE = 88 
    MAX_LIDAR_DIST = 20.0

    def __init__(self, host="127.0.0.1", recv_port=5005, send_port=5006):
        self.host = host
        self.recv_port = recv_port
        self.send_port = send_port
        
        # 1. VERIFY SIZE AT STARTUP
        expected_size = struct.calcsize(self.HEADER_FMT)
        print(f"🔍 DEBUG: Format String expects {expected_size} bytes.")
        print(f"🔍 DEBUG: Header Size set to {self.HEADER_SIZE} bytes.")
        
        if expected_size != self.HEADER_SIZE:
            # This should never happen now that we use explicit math
            raise ValueError(f"CRITICAL ERROR: Format string size ({expected_size}) != HEADER_SIZE ({self.HEADER_SIZE})")

        # Setup Network
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.recv_sock.bind((self.host, self.recv_port))
        self.recv_sock.settimeout(2.0) 

        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_address = (self.host, self.send_port)
        self.last_seq = -1

    def step(self, steering: float, throttle: float) -> Optional[UnityTelemetry]:
        # 1. Send Command (2 floats)
        cmd_data = struct.pack("<ff", steering, throttle)
        self.send_sock.sendto(cmd_data, self.send_address)

        # 2. Receive Telemetry
        try:
            data, _ = self.recv_sock.recvfrom(8192)
            
            if len(data) < self.HEADER_SIZE:
                return None

            # 3. UNPACK HEADER (Guaranteed 88 bytes)
            h = struct.unpack(self.HEADER_FMT, data[:self.HEADER_SIZE])
            
            # Filter duplicates
            seq = h[0]
            if self.last_seq != -1 and seq <= self.last_seq:
                return None
            self.last_seq = seq

            # 4. MAP DATA (Indices aligned to the explicit blocks above)
            # 0: Seq
            # 1-6: Motion
            # 7: Ticks
            # 8: RPM
            # 9-17: IMU
            # 18: Reward
            # 19: Lap
            # 20: Crash
            # 21: LidarCount
            
            telemetry = UnityTelemetry(
                seq_id      = h[0],
                time        = h[1],
                dt          = h[2],
                speed       = h[3],
                steer_cmd   = h[4],
                steer_phys  = h[5],
                throttle    = h[6],
                wheel_ticks = h[7],      
                wheel_rpm   = h[8],      
                euler       = [h[9], h[10], h[11]],
                lin_accel   = [h[12], h[13], h[14]],
                ang_accel   = [h[15], h[16], h[17]],
                reward      = h[18],
                lap_count   = h[19],     
                is_crashed  = bool(h[20]), 
                lidar_count = h[21]      
            )

            # 5. DECODE LIDAR
            lidar_bytes = telemetry.lidar_count * 2
            if len(data) >= self.HEADER_SIZE + lidar_bytes:
                lidar_data = data[self.HEADER_SIZE : self.HEADER_SIZE + lidar_bytes]
                lidar_q = struct.unpack(f"<{telemetry.lidar_count}H", lidar_data)
                telemetry.lidar_ranges = [(x / 65535.0) * self.MAX_LIDAR_DIST for x in lidar_q]

            return telemetry

        except socket.timeout:
            return None
        except struct.error as e:
            print(f"❌ STRUCT ERROR: {e}")
            return None

# --- TEST LOOP ---
if __name__ == "__main__":
    env = UnityEnvironment()
    print("✅ Environment Ready.")
    
    try:
        for i in range(1, 1000):
            # Drive forward slowly
            obs = env.step(0.0, 0.5)
            
            if obs:
                status = "CRASHED" if obs.is_crashed else "OK"
                #print(f"Step {i:03} | Speed: {obs.speed:.2f} | Reward: {obs.reward:.4f} | Status: {status}")
            else:
                print(f"Step {i:03} | No Data")
            
            #time.sleep(1/200)
    except KeyboardInterrupt:
        print("Stopped.")