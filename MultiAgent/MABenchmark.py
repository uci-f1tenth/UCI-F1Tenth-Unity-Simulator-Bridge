import time
import numpy as np
from MAUnityBridge import MATCPTunnel  # Ensure your provided code is saved as ma_bridge.py

def run_benchmark():
    # 1. Initialize and start the tunnel
    tunnel = MATCPTunnel(ip="127.0.0.1", port=5005)
    tunnel.start()

    print("⏳ Waiting for Unity to connect on 127.0.0.1:5005...")
    
    # Wait until the background thread establishes a connection
    while not tunnel.connected:
        time.sleep(0.1)

    print("✅ Connected! Starting Hz benchmark...\n")

    # Benchmarking state variables
    last_seq_id = -1
    updates_received = 0
    start_time = time.time()
    last_print_time = start_time

    try:
        while True:
            # Safely grab the latest telemetry
            telemetry = tunnel.get_telemetry()

            if telemetry.sequenceId >= 0:
                # Instantly send a response back (Ping-Pong)
                num_agents = telemetry.numberOfAgents
                if num_agents > 0:
                    # Send zeroed-out steering and throttle commands
                    dummy_steering = np.zeros(num_agents, dtype=np.float32)
                    dummy_throttle = np.zeros(num_agents, dtype=np.float32)
                    tunnel.send_action_batch(dummy_steering, dummy_throttle)
                    
                # If we received a NEW packet
                if telemetry.sequenceId != last_seq_id:
                    last_seq_id = telemetry.sequenceId
                    updates_received += 1

            # Calculate and print the frequency (Hz) every 1.0 seconds
            current_time = time.time()
            elapsed_print_time = current_time - last_print_time
            
            if elapsed_print_time >= 1.0:
                hz = updates_received / elapsed_print_time
                print(f"[T+{current_time - start_time:.1f}s] Connection Speed: {hz:.2f} Updates/Sec (Hz)")
                
                # Reset counters for the next second
                updates_received = 0
                last_print_time = current_time

            # A microscopic sleep prevents this Python `while` loop from locking up a 
            # CPU core at 100%, but is fast enough to not bottleneck Unity's framerate.
            time.sleep(0.00001) 

    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt detected. Stopping benchmark...")
    finally:
        tunnel.stop()
        print("Tunnel shut down successfully.")

if __name__ == "__main__":
    run_benchmark()