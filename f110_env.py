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
        # Using dtype=np.float32 inside the array prevents the casting warning
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32), 
            high=np.array([1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )

        # --- Observation Space ---
        # For scalars like inf, np.float32() is the way to go
        #self.obs_size = self.lidar_size + 7 + 9
        self.obs_size = self.lidar_size + 5
        self.observation_space = spaces.Box(
            low=np.float32(-np.inf), 
            high=np.float32(np.inf), 
            shape=(self.obs_size,), 
            dtype=np.float32
        )

        # --- UPS Tracking Variables ---
        self.ups_counter = 0
        self.last_status_time = time.time()

        self.bridge.start()

    def _get_obs(self, car):
        # 1. Lidar Processing with Empty Data Guard
        raw_lidar = np.array(car.lidarRanges, dtype=np.float32)
        
        # If Unity hasn't sent data yet, return zeros so the NN doesn't crash
        if len(raw_lidar) == 0:
            lidar = np.zeros(self.lidar_size, dtype=np.float32)
        elif len(raw_lidar) != self.lidar_size:
            original_indices = np.linspace(0, len(raw_lidar) - 1, self.lidar_size)
            lidar = np.interp(original_indices, np.arange(len(raw_lidar)), raw_lidar).astype(np.float32)
        else:
            lidar = raw_lidar

        # 2. Scalars (7)
        scalars = np.array([
            car.fixedDeltaTime,
            car.speedMS,
            car.steering,
            car.steeringAngleRad,
            car.throttle
            #(car.wheelTicks % 10000) / 10000.0,
            #car.currentWheelRPM / 10000.0
        ], dtype=np.float32)

        # 3. Vectors (9)
        #vectors = np.array([
        #    *car.imuAngle,
        #    *car.linAccel,
        #    *car.angAccel
        #], dtype=np.float32)

        #return np.concatenate([lidar, scalars, vectors])
        return np.concatenate([lidar, scalars])

    def step(self, action):
        steering, throttle = action
        
        # 1. Capture ID before sending command
        # (Wait for connection first if needed)
        while not self.bridge.connected:
             time.sleep(1.0)

        last_telemetry = self.bridge.get_telemetry()
        
        # 2. Send control
        self.bridge.send_control(steering, throttle)

        # 3. Synchronous Wait (Wait for Unity to complete a physics frame)
        start_wait = time.time()
        while True:
            car = self.bridge.get_telemetry()
            if car.sequenceId > last_telemetry.sequenceId:
                break 
            
            if time.time() - start_wait > 1.0:
                print("⚠️ Sync Warning: Unity heartbeat lost.")
                break
            time.sleep(0.0001) # Tiny sleep to prevent 100% CPU core usage

        # 4. Process Data
        obs = self._get_obs(car)
        reward = float(car.reward)
        terminated = (car.isCrashed == VehicleStatus.CRASHED)
        
        # --- 5. UPS Tracking Logic ---
        self.ups_counter += 1
        now = time.time()
        if now - self.last_status_time >= 1.0:
            ups = self.ups_counter / (now - self.last_status_time)
            # Use raw_lidar length for debugging or just fixed 1080 concept
            print(f"📊 [UPS: {ups:5.1f}] | Seq: {car.sequenceId:6d} | Speed: {car.speedMS:5.2f} m/s | Reward: {reward:6.2f}")
            self.ups_counter = 0
            self.last_status_time = now

        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Ensure bridge is connected before requesting reset
        while not self.bridge.connected:
            print("⏳ Waiting for Unity connection...")
            time.sleep(1.0)

        # Send Reset signal
        self.bridge.send_control(100.0, 100.0)
        
        # CRITICAL: Wait until Unity actually sends a valid packet with Lidar data
        car = self.bridge.get_telemetry()
        timeout = time.time() + 5.0
        while len(car.lidarRanges) == 0:
            car = self.bridge.get_telemetry()
            if time.time() > timeout:
                print("⚠️ Reset Timeout: Unity is connected but not sending Lidar data.")
                break
            time.sleep(0.1)

        return self._get_obs(car), {}

    def close(self):
        self.bridge.stop()