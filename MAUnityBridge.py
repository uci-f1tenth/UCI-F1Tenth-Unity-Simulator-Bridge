import socket
import struct
import threading
import numpy as np
from dataclasses import dataclass, field
from copy import deepcopy

@dataclass
class MATelemetryHub:
    """
    Structured data container for all telemetry received from Unity.
    Designed to easily convert to PyTorch/TensorFlow tensors.
    """
    # --- GLOBAL HEADER (36 Bytes) ---
    sequenceId: int = 0
    timeSinceStart: float = 0.0
    fixedDeltaTime: float = 0.0
    numberOfAgents: int = 0
    
    # Global Sensor Config
    lidarMinAngleDeg: float = 0.0
    lidarMaxAngleDeg: float = 0.0
    lidarMinRange: float = 0.0
    lidarMaxRange: float = 0.0
    lidarScanDirection: int = 0  
    
    # --- 1D ARRAYS (Shape: [numberOfAgents]) ---
    speedMS: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    throttle: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    steering: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    steeringAngleDeg: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    currentWheelRPM: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    appliedWheelTorque: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    wheelTicks: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    
    isCrashed: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    reward: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    
    # --- 2D ARRAYS (Shape: [numberOfAgents, 3]) ---
    imuAngleDeg: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    linAccelMS: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    angAccelDeg: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    
    # --- HIGH DIMENSIONAL SENSORS ---
    # Shape: [numberOfAgents, lidarResolution]
    lidarRanges: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))


class MATCPTunnel:
    """
    High-performance TCP Bridge for Multi-Agent Reinforcement Learning.
    Handles batched telemetry ingestion and batched action broadcasting.
    """
    def __init__(self, ip="127.0.0.1", port=5005):
        self.ip = ip
        self.port = port
        self.hub = MATelemetryHub()
        
        # --- RECEIVE PACKET FORMATS (Unity -> Python) ---
        # Global Header (36 Bytes)
        self.global_header_fmt = '<Iffi4fi'
        self.global_header_size = struct.calcsize(self.global_header_fmt)
        
        # Agent Metadata (76 Bytes)
        self.agent_meta_fmt = '<6fi9fifi'
        self.agent_meta_size = struct.calcsize(self.agent_meta_fmt)
        
        # Pre-allocate bytearrays for zero-copy socket reads
        self._global_buf = bytearray(self.global_header_size)
        self._agent_buf = bytearray(self.agent_meta_size)
        self._global_view = memoryview(self._global_buf)
        self._agent_view = memoryview(self._agent_buf)
        
        # --- SEND PACKET FORMATS (Python -> Unity) ---
        # Unity expects: [Steer0, Throt0, Steer1, Throt1, ...]
        self.action_batch_fmt = ''
        self.action_batch_size = 0
        self.action_buffer = None
        
        # --- THREADING & STATE ---
        self.sock = None
        self.connected = False
        self.running = False
        self._thread = None
        self._lock = threading.Lock()
        self.last_sequence_id = -1
        
    # ==========================================
    # LIFECYCLE MANAGEMENT
    # ==========================================
    def start(self):
        """Starts the background TCP client thread."""
        self.running = True
        self._thread = threading.Thread(target=self._client_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Cleanly shuts down the server and socket."""
        self.running = False
        self.connected = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        if self._thread:
            self._thread.join(timeout=1.0)
            
    # ==========================================
    # DATA ACCESS & CONTROL (For RL Loop / Visualizer)
    # ==========================================
    def get_telemetry(self) -> MATelemetryHub:
        """Returns a thread-safe deepcopy of the latest environment state."""
        with self._lock:
            return deepcopy(self.hub)

    def send_action_batch(self, steering_array: np.ndarray, throttle_array: np.ndarray):
        """
        Packs and sends control inputs for ALL agents simultaneously.
        Matches Unity C# loop: [Steer0, Throt0, Steer1, Throt1, ...]
        """
        if not self.connected or not self.sock or self.hub.numberOfAgents == 0:
            return
            
        num_agents = self.hub.numberOfAgents
        
        # Validation
        if len(steering_array) != num_agents or len(throttle_array) != num_agents:
            print(f"Warning: Action arrays must match agent count ({num_agents}).")
            return
            
        try:
            # Flatten arrays: [s0, t0, s1, t1...]
            interleaved = np.empty(num_agents * 2, dtype=np.float32)
            interleaved[0::2] = steering_array
            interleaved[1::2] = throttle_array
            
            # Pack using struct.pack_into for maximum efficiency
            struct.pack_into(self.action_batch_fmt, self.action_buffer, 0, *interleaved)
            self.sock.sendall(self.action_buffer)
            
        except Exception as e:
            print(f"[TCP] Failed to send action batch: {e}")
            self.connected = False
            
    # ==========================================
    # INTERNAL OPTIMIZATION METHODS
    # ==========================================
    def _resize_hub(self, num_agents: int):
        """Pre-allocates Numpy arrays and Send buffers when agent count changes."""
        self.hub.numberOfAgents = num_agents
        self.hub.speedMS = np.zeros(num_agents, dtype=np.float32)
        self.hub.throttle = np.zeros(num_agents, dtype=np.float32)
        self.hub.steering = np.zeros(num_agents, dtype=np.float32)
        self.hub.steeringAngleDeg = np.zeros(num_agents, dtype=np.float32)
        self.hub.currentWheelRPM = np.zeros(num_agents, dtype=np.float32)
        self.hub.appliedWheelTorque = np.zeros(num_agents, dtype=np.float32)
        self.hub.wheelTicks = np.zeros(num_agents, dtype=np.int32)
        self.hub.isCrashed = np.zeros(num_agents, dtype=np.int32)
        self.hub.reward = np.zeros(num_agents, dtype=np.float32)
        self.hub.imuAngleDeg = np.zeros((num_agents, 3), dtype=np.float32)
        self.hub.linAccelMS = np.zeros((num_agents, 3), dtype=np.float32)
        self.hub.angAccelDeg = np.zeros((num_agents, 3), dtype=np.float32)
        self.hub.lidarRanges = np.zeros((num_agents, 0), dtype=np.float32)
        
        # Pre-compile the struct format for the Action Batch (e.g., '<20f' for 10 agents)
        self.action_batch_fmt = f'<{num_agents * 2}f' 
        self.action_batch_size = struct.calcsize(self.action_batch_fmt)
        self.action_buffer = bytearray(self.action_batch_size)

    def _recv_into_view(self, view, size):
        """Zero-copy receive loop using memoryviews."""
        bytes_recv = 0
        while bytes_recv < size:
            chunk = self.sock.recv_into(view[bytes_recv:], size - bytes_recv)
            if chunk == 0:
                raise ConnectionError("Socket connection broken")
            bytes_recv += chunk

    # ==========================================
    # NETWORK LOOPS
    # ==========================================
    def _client_loop(self):
        import time
        while self.running:
            try:
                # 1. Create socket
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                # 2. Python acts as the CLIENT and connects to Unity
                self.sock.connect((self.ip, self.port))
                
                self.connected = True
                self.last_sequence_id = -1
                print(f"\n[TCP] ✅ Successfully connected to Unity Server at {self.ip}:{self.port}!")
                
                # 3. Enter the data receive loop
                self._listen_loop()
                
            except ConnectionRefusedError:
                # Unity server isn't running yet. Wait and retry silently.
                time.sleep(1.0)
            except Exception as e:
                if self.running:
                    print(f"[TCP] ⚠️ Connection dropped: {e}. Retrying...")
            finally:
                self.connected = False
                if self.sock:
                    try:
                        self.sock.close()
                    except:
                        pass
                time.sleep(1.0) # Prevent CPU spam while reconnecting

    def _listen_loop(self):
        unpack = struct.unpack
        global_fmt = self.global_header_fmt
        agent_fmt = self.agent_meta_fmt
        
        while self.running and self.connected:
            try:
                # 1. READ GLOBAL HEADER
                self._recv_into_view(self._global_view, self.global_header_size)
                g_data = unpack(global_fmt, self._global_buf)
                
                seq_id = g_data[0]
                if seq_id <= self.last_sequence_id and self.last_sequence_id != -1: 
                    continue 
                self.last_sequence_id = seq_id

                num_agents = g_data[3]
                with self._lock:
                    if self.hub.numberOfAgents != num_agents:
                        self._resize_hub(num_agents)

                    h = self.hub
                    h.sequenceId = seq_id
                    h.timeSinceStart = g_data[1]
                    h.fixedDeltaTime = g_data[2]
                    
                    # Store Global Sensor Data
                    h.lidarMinAngleDeg = g_data[4]
                    h.lidarMaxAngleDeg = g_data[5]
                    h.lidarMinRange = g_data[6]
                    h.lidarMaxRange = g_data[7]
                    h.lidarScanDirection = g_data[8]
                
                    # 2. READ AGENT LOOP
                    for i in range(num_agents):
                        self._recv_into_view(self._agent_view, self.agent_meta_size)
                        meta = unpack(agent_fmt, self._agent_buf)
                        
                        # Populate physics arrays
                        h.speedMS[i] = meta[0]
                        h.throttle[i] = meta[1]
                        h.steering[i] = meta[2]
                        h.steeringAngleDeg[i] = meta[3]
                        h.currentWheelRPM[i] = meta[4]
                        h.appliedWheelTorque[i] = meta[5]
                        
                        h.wheelTicks[i] = meta[6]
                        
                        h.imuAngleDeg[i] = meta[7:10]
                        h.linAccelMS[i] = meta[10:13]
                        h.angAccelDeg[i] = meta[13:16]
                        
                        h.isCrashed[i] = meta[16]
                        h.reward[i] = meta[17]
                        
                        # 3. READ HIGH-DIMENSIONAL LIDAR
                        lidar_n = meta[18]
                        if lidar_n > 0:
                            if h.lidarRanges.shape[1] != lidar_n:
                                h.lidarRanges = np.zeros((num_agents, lidar_n), dtype=np.float32)

                            bytes_needed = lidar_n * 2
                            lidar_data = self.sock.recv(bytes_needed, socket.MSG_WAITALL)
                            
                            # Fast vectorized decompression using Numpy
                            raw_lidar = np.frombuffer(lidar_data, dtype=np.uint16)
                            if h.lidarMaxRange > 0:
                                h.lidarRanges[i] = (raw_lidar.astype(np.float32) / 65535.0) * h.lidarMaxRange
                                
            except Exception as e:
                print(f"[TCP] Connection broken in listen loop: {e}")
                self.connected = False
                break