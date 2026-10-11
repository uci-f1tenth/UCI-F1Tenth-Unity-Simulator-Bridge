import time
import sys
import os
import numpy as np
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SingleAgent"))
from UnityBridge import UnityTCPBridge, VehicleStatus

# --- Configuration ---
# Match this to the IP and Port fields in the simulator
UNITY_IP = "127.0.0.1"
UNITY_PORT = 5005

# The LiDAR covers 270 degrees, so trimming 1/6 off each side leaves the front 180 degrees
min_angle: float = -np.pi / 2.0  # radians
max_angle: float = np.pi / 2.0  # radians
bubble_fraction: float = 0.15  # width of the gap the car aims for, as a fraction of the scan

def index_to_angle(index: int, num_points: int) -> float:
    angle_increment = (max_angle - min_angle) / (num_points - 1)
    angle = min_angle + index * angle_increment
    return angle


def find_best_point(lidar_range_array: np.ndarray[Any]) -> int:
    bubble_size = max(1, int(len(lidar_range_array) * bubble_fraction))
    best_index = 0
    best_min_distance = 0.0

    for i in range(len(lidar_range_array) - bubble_size + 1):
        window = lidar_range_array[i : i + bubble_size]
        min_distance = np.min(window)

        if min_distance > best_min_distance:
            best_min_distance = min_distance
            best_index = i

    return best_index + bubble_size // 2


def compute_speed(target_distance: float) -> float:
    if target_distance < 5.0:
        return target_distance / 50.0
    return 0.125

def main():
    # Initialize the bridge
    bridge = UnityTCPBridge(ip=UNITY_IP, port=UNITY_PORT)
    bridge.start()

    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            # 1. Capture the current state
            car = bridge.get_telemetry()

            if len(car.lidarRanges) > 0 and car.isCrashed == VehicleStatus.OK:
                n = len(car.lidarRanges)
                lidar_range_array = car.lidarRanges[n // 6 : -(n // 6)]

                best_point_index = find_best_point(lidar_range_array)

                best_point_angle = index_to_angle(best_point_index, len(lidar_range_array))
                steering = best_point_angle / (np.pi / 2.0)

                target_distance = lidar_range_array[best_point_index]
                throttle = compute_speed(target_distance)

                # The scan runs left to right, so a gap at a positive angle is on the right (positive steering)
                bridge.send_control(steering, throttle)
            else:
                # Keep stepping the simulation (needed in Manual time step mode)
                bridge.send_control(0.0, 0.0)

            # Run at roughly 50Hz
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\nStopping Bridge...")
        bridge.stop()
        print("Done.")

if __name__ == "__main__":
    main()
