import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from UnityBridge import UnityTCPBridge, VehicleStatus

class F110UnityEnv(gym.Env):
    def __init__(self, lidar_size=1080):
        super().__init__()

        self.lidar_size = lidar_size
        self.bridge = UnityTCPBridge(port=5005)
        
        # --- Action Space ---
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32), 
            high=np.array([1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )

        # --- Observation Space ---
        self.obs_size = self.lidar_size + 7 + 9
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.obs_size,), 
            dtype=np.float32
        )

        # --- OPTIMIZATION: Zero-Copy Buffer ---
        self.obs_buffer = np.zeros(self.obs_size, dtype=np.float32)
        self.lidar_view = self.obs_buffer[0 : self.lidar_size]
        self.scalar_view = self.obs_buffer[self.lidar_size : self.lidar_size + 7]
        self.vector_view = self.obs_buffer[self.lidar_size + 7 :]

        # --- UPS Tracking Variables ---
        self.ups_counter = 0
        self.last_status_time = time.time()

        self.bridge.start()

    def _update_obs_buffer(self, car):
        """Writes data directly into the pre-allocated numpy buffer."""
        # 1. Lidar
        if len(car.lidarRanges) == self.lidar_size:
            np.copyto(self.lidar_view, car.lidarRanges)
        else:
            self.lidar_view[:] = 0.0

        # 2. Scalars
        self.scalar_view[0] = car.fixedDeltaTime
        self.scalar_view[1] = car.speedMS
        self.scalar_view[2] = car.steering
        self.scalar_view[3] = car.steeringAngleDeg
        self.scalar_view[4] = car.throttle
        self.scalar_view[5] = (car.wheelTicks % 10000) / 10000.0
        self.scalar_view[6] = car.currentWheelRPM / 10000.0

        # 3. Vectors
        self.vector_view[0:3] = car.imuAngleDeg
        self.vector_view[3:6] = car.linAccel
        self.vector_view[6:9] = car.angAccelDeg

        return self.obs_buffer

    def step(self, action):
        # Wait for connection to start training
        while not self.bridge.connected:
            time.sleep(1.0)

        steering, throttle = action
        
        # 1. Capture Current Sequence ID
        start_seq_id = self.bridge.data.sequenceId

        # 2. Send Control
        self.bridge.send_control(steering, throttle)

        # 3. Sync Wait (Spin-lock)
        timeout = time.time() + 1.0
        while True:
            current_seq_id = self.bridge.data.sequenceId
            
            if current_seq_id > start_seq_id:
                break
            
            if time.time() > timeout:
                break
                
            time.sleep(0) # Yield CPU 

        # 4. Get Reference to updated data
        car = self.bridge.data
        
        # 5. Populate Buffer
        obs = self._update_obs_buffer(car)
        reward = float(car.reward)
        terminated = (car.isCrashed == VehicleStatus.CRASHED)
        
        # --- 6. UPS Tracking Logic ---
        self.ups_counter += 1
        now = time.time()
        if now - self.last_status_time >= 1.0:
            ups = self.ups_counter / (now - self.last_status_time)
            print(f"📊 [UPS: {ups:5.1f}] | Seq: {car.sequenceId:6d} | Speed: {car.speedMS:5.2f} m/s | Reward: {reward:6.2f}")
            self.ups_counter = 0
            self.last_status_time = now
        
        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        while not self.bridge.connected:
            time.sleep(0.5)

        self.bridge.send_control(100.0, 100.0)
        
        timeout = time.time() + 5.0
        while True:
            if self.bridge.data.lidarCount > 0:
                break
            if time.time() > timeout:
                print("⚠️ Reset Timeout: No Lidar data received.")
                break
            time.sleep(0.5)

        obs = self._update_obs_buffer(self.bridge.data)
        return obs, {}

    def close(self):
        self.bridge.stop()