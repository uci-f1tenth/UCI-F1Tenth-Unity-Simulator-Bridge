import numpy as np
import time
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from MAUnityBridge import MATCPTunnel  # Assuming your bridge is saved here

class MAF110VecEnv(VecEnv):
    def __init__(self, lidar_size=270, port=5005):
        self.lidar_size = lidar_size
        self.bridge = MATCPTunnel(port=port)
        self.bridge.start()

        # 1. Wait for connection and first packet to get agent count
        print("[MAF110VecEnv] Waiting for Unity connection...")
        while not self.bridge.connected or self.bridge.get_telemetry().numberOfAgents == 0:
            time.sleep(0.5)

        self.num_agents = self.bridge.get_telemetry().numberOfAgents
        print(f"[MAF110VecEnv] Connected! Detected {self.num_agents} agents.")

        # 2. Define Spaces
        action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32), 
            high=np.array([1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )

        self.obs_size = self.lidar_size + 7 + 9
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.obs_size,), 
            dtype=np.float32
        )

        # Explicitly satisfy SB3's internal render_mode checks
        self.render_mode = None
        self.render_modes = [None] * self.num_agents
        
        # Initialize base VecEnv
        super().__init__(self.num_agents, observation_space, action_space)

        # Tracking metrics
        self.ups_counter = 0
        self.last_status_time = time.time()
        self.start_seq_id = 0

    def _get_batched_obs(self, hub):
        """Vectorized extraction of all agent data into a single 2D NumPy array."""
        N = self.num_agents

        # Handle edge case if lidar isn't fully initialized yet
        if hub.lidarRanges.shape[1] != self.lidar_size:
            lidar_data = np.zeros((N, self.lidar_size), dtype=np.float32)
        else:
            lidar_data = hub.lidarRanges

        # Stack 1D scalar arrays into a 2D array [N, 7]
        scalars = np.column_stack([
            np.full(N, hub.fixedDeltaTime, dtype=np.float32),
            hub.speedMS,
            hub.steering,
            hub.steeringAngleDeg,
            hub.throttle,
            (hub.wheelTicks % 10000) / 10000.0,
            hub.currentWheelRPM / 10000.0
        ])

        # Stack 2D vector arrays [N, 9]
        vectors = np.hstack([hub.imuAngleDeg, hub.linAccelMS, hub.angAccelDeg])

        # Combine all into final observation block [N, obs_size]
        obs = np.hstack([lidar_data, scalars, vectors]).astype(np.float32)
        return obs

    def step_async(self, actions):
        """Sends the batched actions to Unity."""
        # actions shape is [num_agents, 2]
        steering = actions[:, 0]
        throttle = actions[:, 1]
        
        self.start_seq_id = self.bridge.get_telemetry().sequenceId
        self.bridge.send_action_batch(steering, throttle)

    def step_wait(self):
        """Waits for Unity to process the step and returns the batched results."""
        # Sync Wait (Spin-lock)
        timeout = time.time() + 1.0
        while True:
            hub = self.bridge.get_telemetry()
            if hub.sequenceId > self.start_seq_id:
                break
            if time.time() > timeout:
                break
            time.sleep(0) # Yield CPU

        obs = self._get_batched_obs(hub)
        rewards = hub.reward
        
        # In Unity, isCrashed should be 1 if crashed, 0 otherwise
        dones = (hub.isCrashed == 1) 

        # SB3 requires an info dict per agent. We must handle auto-reset terminal obs.
        infos = []
        for i in range(self.num_agents):
            info = {}
            if dones[i]:
                # SB3 expects terminal_observation to be populated when an episode ends
                info["terminal_observation"] = obs[i].copy()
            infos.append(info)

        # UPS Tracking Logging (optional, tracking agent 0 for reference)
        self.ups_counter += 1
        now = time.time()
        if now - self.last_status_time >= 1.0:
            ups = self.ups_counter / (now - self.last_status_time)
            print(f"📊 [UPS: {ups:5.1f}] | Seq: {hub.sequenceId:6d} | Agent 0 Speed: {hub.speedMS[0]:5.2f} m/s | Mean Rwd: {np.mean(rewards):6.2f}")
            self.ups_counter = 0
            self.last_status_time = now

        return obs, rewards, dones, infos

    def reset(self):
        """Kickstarts Unity's manual physics engine with an initial command."""
        # Note: If your Unity C# script specifically looks for "100.0" to trigger 
        # a hard reset, change these np.zeros to np.full(..., 100.0)
        dummy_steer = np.zeros(self.num_agents, dtype=np.float32)
        dummy_throttle = np.zeros(self.num_agents, dtype=np.float32)
        
        self.start_seq_id = self.bridge.get_telemetry().sequenceId
        self.bridge.send_action_batch(dummy_steer, dummy_throttle)
        
        # Sync Wait: Spin until Unity processes the manual step and populates Lidar
        timeout = time.time() + 2.0
        while True:
            hub = self.bridge.get_telemetry()
            
            # Ensure sequence ID advanced AND Lidar array is fully populated
            if hub.sequenceId > self.start_seq_id and hub.lidarRanges.shape[1] == self.lidar_size:
                break
                
            if time.time() > timeout:
                print("⚠️ [Warning] Reset timed out waiting for Unity Manual Step.")
                break
                
            time.sleep(0.01) # Yield CPU slightly to prevent thread locking

        return self._get_batched_obs(hub)

    def close(self):
        self.bridge.stop()

    # --- Required VecEnv Boilerplate ---
    def get_attr(self, attr_name, indices=None):
        """Fixes the SB3 TypeError by properly handling internal attribute checks."""
        if attr_name == "render_mode":
            return [None] * self.num_agents
            
        # SB3 relies on AttributeError being raised if an attribute isn't found
        raise AttributeError(f"Attribute '{attr_name}' not found in custom VecEnv.")

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_agents