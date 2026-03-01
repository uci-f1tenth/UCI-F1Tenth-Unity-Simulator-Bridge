import time
import sys
import os
import numpy as np
from typing import Any
from UnityBridge import UnityTelemetryBridge, VehicleStatus

# --- Configuration ---
# Match these to your UDPTunnel component in the Unity Inspector
UNITY_IP = "127.0.0.1"
PYTHON_RX_PORT = 5005  # Unity 'Send Port'
PYTHON_TX_PORT = 5006  # Unity 'Recv Port'

# Initialize vehicle(s)
min_angle: float = -np.pi / 2.0  # radians
max_angle: float = np.pi / 2.0  # radians
bubble_size: float = 160  # lidar points

def index_to_angle(index: int, num_points: int) -> float:
    angle_increment = (max_angle - min_angle) / (num_points - 1)
    angle = min_angle + index * angle_increment
    return angle


def find_best_point(lidar_range_array: np.ndarray[Any]) -> int:
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
    bridge = UnityTelemetryBridge(
        ip=UNITY_IP, 
        rx_port=PYTHON_RX_PORT, 
        tx_port=PYTHON_TX_PORT
    )
    bridge.start()

    print(f"--- F1-10th Telemetry Dashboard Active ---")
    print(f"Listening on: {PYTHON_RX_PORT} | Sending to: {PYTHON_TX_PORT}")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            # 1. Capture the current state
            car = bridge.get_telemetry()

            if len(car.lidar_ranges) > 0 and car.status == VehicleStatus.OK:
                lidar_range_array = car.lidar_ranges[
                    len(car.lidar_ranges) // 6 : -len(car.lidar_ranges)
                    // 6
                ]

                best_point_index = find_best_point(lidar_range_array)

                best_point_angle = index_to_angle(best_point_index, len(lidar_range_array))
                steering = best_point_angle / (np.pi / 2.0)

                target_distance = lidar_range_array[best_point_index]
                throttle = compute_speed(target_distance)

                # 5. Heartbeat: Send neutral controls so Unity keeps the socket open
                bridge.send_control(-steering, throttle)

            # Run at roughly 50Hz
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\nStopping Bridge...")
        bridge.stop()
        print("Done.")

if __name__ == "__main__":
    # If on Windows, enable ANSI escape sequences for the 'static' dashboard effect
    if os.name == 'nt':
        os.system('color')
    main()