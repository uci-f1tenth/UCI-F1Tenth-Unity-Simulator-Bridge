import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SingleAgent"))
from UnityBridge import UnityTCPBridge, VehicleStatus

if __name__ == "__main__":
    bridge = UnityTCPBridge(port=5005)
    bridge.start()

    # Tracking variables for UPS
    ups_counter = 0
    last_time = time.time()

    try:
        while True:
            # Drive straight at half throttle
            bridge.send_control(0.0, 0.5)
            obs = bridge.get_telemetry()
            ups_counter += 1

            # --- STATUS UPDATE (Every 1 Second) ---
            current_time = time.time()
            if current_time - last_time >= 1.0:
                # Calculate frequency
                ups = ups_counter / (current_time - last_time)

                # Print stats
                if bridge.connected:
                    print(f"📊 [UPS: {ups:5.1f}] | Seq: {obs.sequenceId:6d} | Speed: {obs.speedMS:5.2f} m/s | Rew: {obs.reward:6.2f}")
                else:
                    print("📊 Waiting for Unity connection...")

                # Reset counters
                ups_counter = 0
                last_time = current_time

            # --- GAME LOGIC ---
            if obs.isCrashed == VehicleStatus.CRASHED:
                print("💥 Crash Detected!")

            time.sleep(0.02)

    except KeyboardInterrupt:
        bridge.stop()
