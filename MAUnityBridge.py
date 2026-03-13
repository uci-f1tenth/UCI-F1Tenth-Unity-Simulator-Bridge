import socket
import struct
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple

class MAVehicleStatus(IntEnum):
    OK = 0
    CRASHED = 1

@dataclass
class MATelemetryHub:
    """
    Highly optimized Data Class for Multi-Agent Reinforcement Learning.
    All data is stored as parallel numpy arrays of size [number_of_agents].
    This matches the C# MADataHub design and allows instant batch feeding to RL models.
    """
    sequenceId: int = 0
    timeSinceStart: float = 0.0
    fixedDeltaTime: float = 0.0
    numberOfAgents: int = 0
    
    # --- 1D Arrays (Shape: [numberOfAgents]) ---
    speedMS: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    throttle: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    steering: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    steeringAngleDeg: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    currentWheelRPM: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    appliedWheelTorque: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    
    isCrashed: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    reward: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    
    # --- 2D Arrays (Shape: [numberOfAgents, 3]) ---
    imuAngleDeg: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    linAccelMS: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    angAccelDeg: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    
    # --- 2D Lidar Array (Shape: [numberOfAgents, lidarResolution]) ---
    lidarRanges: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))

class MATCPTunnel:
    def __init__(self, ip="127.0.0.1", port=5005, lidar_max_dist=20.0):
        self.ip = ip
        self.port = port
        self.lidar_scale = lidar_max_dist / 65535.0
        
        # --- C# STRUCT FORMATS ---
        # Global Header: SeqID(uint), Time(float), DT(float), AgentCount(int) = 16 bytes
        self.global_header_fmt = '<Iffi'
        self.global_header_size = struct.calcsize(self.global_header_fmt)
        
        # Agent Metadata: 15 floats, 1 int(crash), 1 float(reward), 1 int(lidar count) = 72 bytes
        self.agent_meta_fmt = '<15fifi'
        self.agent_meta_size = struct.calcsize(self.agent_meta_fmt)
        
        # Memory Buffers
        self._global_buf = bytearray(self.global_header_size)
        self._agent_buf = bytearray(self.agent_meta_size)
        
        # State Tracking
        self.hub = MATelemetryHub()
        self.last_sequence_id = -1
        self.running = False
        self.connected = False
        self._lock = threading.Lock()
        self.sock = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._connection_loop, daemon=True)
        self.thread.start()
        print(f"🔄 MA Bridge starting... Connecting to {self.ip}:{self.port}")

    def _connection_loop(self):
        while self.running:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                print(f"⏳ Connecting to {self.ip}:{self.port}...")
                self.sock.connect((self.ip, self.port))
                
                self.connected = True
                self.last_sequence_id = -1
                print("✅ Connected to Multi-Agent Unity Environment!")
                
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

    def _resize_hub(self, num_agents: int, lidar_resolution: int = 0):
        """Dynamically resizes numpy arrays if the number of agents changes."""
        with self._lock:
            self.hub.numberOfAgents = num_agents
            
            self.hub.speedMS = np.zeros(num_agents, dtype=np.float32)
            self.hub.throttle = np.zeros(num_agents, dtype=np.float32)
            self.hub.steering = np.zeros(num_agents, dtype=np.float32)
            self.hub.steeringAngleDeg = np.zeros(num_agents, dtype=np.float32)
            self.hub.currentWheelRPM = np.zeros(num_agents, dtype=np.float32)
            self.hub.appliedWheelTorque = np.zeros(num_agents, dtype=np.float32)
            
            self.hub.isCrashed = np.zeros(num_agents, dtype=np.int32)
            self.hub.reward = np.zeros(num_agents, dtype=np.float32)
            
            self.hub.imuAngleDeg = np.zeros((num_agents, 3), dtype=np.float32)
            self.hub.linAccelMS = np.zeros((num_agents, 3), dtype=np.float32)
            self.hub.angAccelDeg = np.zeros((num_agents, 3), dtype=np.float32)
            
            if lidar_resolution > 0:
                self.hub.lidarRanges = np.zeros((num_agents, lidar_resolution), dtype=np.float32)

    def _listen_loop(self):
        unpack = struct.unpack
        global_fmt = self.global_header_fmt
        agent_fmt = self.agent_meta_fmt
        lidar_scale = self.lidar_scale
        
        while self.running and self.connected:
            try:
                # 1. Read Global Header
                self._recv_into(self._global_buf, self.global_header_size)
                seq_id, time_start, dt, num_agents = unpack(global_fmt, self._global_buf)
                
                # Sequence Check (Skip stale packets)
                if seq_id <= self.last_sequence_id and self.last_sequence_id != -1:
                    continue 
                self.last_sequence_id = seq_id

                # Ensure arrays match the Unity Agent Count
                if self.hub.numberOfAgents != num_agents:
                    self._resize_hub(num_agents)

                with self._lock:
                    h = self.hub
                    h.sequenceId = seq_id
                    h.timeSinceStart = time_start
                    h.fixedDeltaTime = dt
                
                    # 2. Iterate over each agent to parse payload
                    for i in range(num_agents):
                        # Read Agent Metadata (72 bytes)
                        self._recv_into(self._agent_buf, self.agent_meta_size)
                        meta = unpack(agent_fmt, self._agent_buf)
                        
                        # Apply Physics & Inputs
                        h.speedMS[i] = meta[0]
                        h.throttle[i] = meta[1]
                        h.steering[i] = meta[2]
                        h.steeringAngleDeg[i] = meta[3]
                        h.currentWheelRPM[i] = meta[4]
                        h.appliedWheelTorque[i] = meta[5]
                        
                        # Apply IMU Data
                        h.imuAngleDeg[i] = meta[6:9]
                        h.linAccelMS[i] = meta[9:12]
                        h.angAccelDeg[i] = meta[12:15]
                        
                        # Apply Status
                        h.isCrashed[i] = meta[15]
                        h.reward[i] = meta[16]
                        
                        # 3. Read Lidar Payload for this agent
                        lidar_n = meta[17]
                        if lidar_n > 0:
                            # Resize lidar array if we discover resolution dynamically
                            if h.lidarRanges.shape[1] != lidar_n:
                                h.lidarRanges = np.zeros((num_agents, lidar_n), dtype=np.float32)

                            bytes_needed = lidar_n * 2
                            lidar_data = self.sock.recv(bytes_needed, socket.MSG_WAITALL)
                            
                            # Fast C-Speed Numpy vectorization
                            raw_lidar = np.frombuffer(lidar_data, dtype=np.uint16)
                            h.lidarRanges[i] = raw_lidar.astype(np.float32) * lidar_scale

            except (ConnectionError, struct.error, OSError) as e:
                print(f"🔌 Connection Lost: {e}")
                break
            except Exception as e:
                print(f"🚨 Unexpected Error: {e}")
                break

    def get_telemetry(self) -> MATelemetryHub:
        """Returns the thread-safe telemetry hub."""
        return self.hub

    def send_controls(self, steers: np.ndarray, throttles: np.ndarray):
        """
        Takes parallel numpy arrays of steering and throttle values and sends them to Unity.
        Both arrays MUST have a length equal to the number of agents.
        """
        if not self.connected or not self.sock: return
        
        num_agents = len(steers)
        if num_agents == 0 or num_agents != len(throttles):
            return
            
        try:
            # Interleave Steer and Throttle: [S0, T0, S1, T1, S2, T2...]
            controls = np.empty((num_agents * 2,), dtype=np.float32)
            controls[0::2] = steers
            controls[1::2] = throttles
            
            # Pack array directly into bytes and send
            packet = struct.pack(f'<{num_agents * 2}f', *controls)
            self.sock.sendall(packet)
            
        except OSError:
            self.connected = False

    def stop(self):
        self.running = False
        if self.sock: self.sock.close()