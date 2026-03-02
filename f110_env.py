import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from UnityBridge import UnityTelemetryBridge, VehicleStatus

class F110UnityEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()

        # 1. Initialize Bridge (Using your exact ports)
        self.bridge = UnityTelemetryBridge(ip="127.0.0.1", rx_port=5005, tx_port=5006)
        
        # 2. Define Action Space: [Steering (-1 to 1), Throttle (0 to 1)]
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), 
                                       high=np.array( [1.0, 1.0]), dtype=np.float32)

        # 3. Define Observation Space: 
        # We'll use a fixed slice of Lidar (middle 2/3rds like your FTG example)
        # Assuming Unity sends 1080 points, middle 2/3 is ~720 points
        # Inside __init__
        self.obs_size = 1097
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.obs_size,), 
            dtype=np.float32
        )

        self.bridge.start()

    def _get_obs(self, car):
        # 1. Handle Lidar (Full 1080)
        lidar = np.array(car.lidarRanges, dtype=np.float32)
        if len(lidar) != 1080:
            # Pad or resize to exactly 1080 to prevent NN shape errors
            lidar = np.resize(lidar, (1080,))
        
        # 2. Extract and Normalize Scalars
        # We normalize some values so they stay roughly within -1 to 1 or 0 to 1
        scalars = np.array([
            car.fixedDeltaTime,
            car.speedMS / 20.0,       # Assuming max speed ~20m/s
            car.steering,          # Already -1 to 1
            car.steeringAngleRad,         # Already -1 to 1
            car.throttle,           # Already 0 to 1
            car.wheelTicks % 1000 / 1000,# Normalize encoder wrap-around
            car.currentWheelRPM / 10000.0,      # Assuming max RPM ~10k
            car.reward              # Include the current reward signal
        ], dtype=np.float32)

        # 3. Extract 3D Vectors
        # (euler, lin_accel, ang_accel)
        vectors = np.array([
            *car.imuAngle,         # x, y, z
            *car.linAccel,         # x, y, z
            *car.angAccel          # x, y, z
        ], dtype=np.float32)

        # 4. Concatenate everything into one flat 1097-length vector
        return np.concatenate([lidar, scalars, vectors])

    def step(self, action):
        steering, throttle = action
        
        # 1. Record the current timestamp to detect a "new" packet
        lastTelemtry = self.bridge.get_telemetry()
        
        # 2. Send control
        self.bridge.send_control(steering, throttle)

        # 3. Synchronous Wait: Loop until Unity advances time
        start_wait = time.time()
        while True:
            car = self.bridge.get_telemetry()
            
            # Detect if a new physics step has been broadcasted
            if car.sequenceId > lastTelemtry.sequenceId:
                break 
            
            # Safety timeout
            if time.time() - start_wait > 1.0:
                print("Sync Warning: Unity heartbeat lost.")
                break
                
            time.sleep(0.001) 

        # 4. Process the data into the 1097-length observation vector
        obs = self._get_obs(car)

        # 5. Native Reward: Take the reward exactly as Unity calculated it
        reward = float(car.reward)
        
        # 6. Termination: Reset the env if Unity says we crashed
        terminated = (car.isCrashed == VehicleStatus.CRASHED)
        
        # Log if we crash for tracking
        if terminated:
            print(f"Episode End | Final Reward: {reward:.2f} | Speed: {car.speedMS:.2f}")

        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Send Reset signal (assuming Unity handles -99 as a teleport)
        self.bridge.send_control(100.0, 100.0)
        time.sleep(0.01) # Give Unity time to reset physics
        
        car = self.bridge.get_telemetry()
        return self._get_obs(car), {}

    def close(self):
        self.bridge.stop()