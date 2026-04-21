import time
from UnityBridge import UnityTCPBridge  # Ensure your provided code is saved as single_bridge.py

def run_single_benchmark():
    # 1. Initialize and start the bridge
    bridge = UnityTCPBridge(ip="127.0.0.1", port=5005)
    bridge.start()

    print("⏳ Waiting for Unity to connect on 127.0.0.1:5005...")
    
    # Wait until the background thread establishes a connection
    while not bridge.connected:
        time.sleep(0.1)

    print("✅ Connected! Starting Hz benchmark...\n")

    # Benchmarking state variables
    last_seq_id = -1
    updates_received = 0
    start_time = time.time()
    last_print_time = start_time

    try:
        while bridge.connected:
            # Safely grab the latest telemetry
            telemetry = bridge.get_telemetry()
            bridge.send_control(0.0, 0.0)

            # If we received a NEW packet
            if telemetry.sequenceId != last_seq_id and telemetry.sequenceId > 0:
                delta = telemetry.sequenceId - last_seq_id
                if delta > 1:
                    print(delta)
                last_seq_id = telemetry.sequenceId
                updates_received += 1

                # Instantly send a response back (Ping-Pong)
                # Sending 0.0 for both steering and throttle
                #bridge.send_control(0.0, 0.0)

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
            time.sleep(0) 

    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt detected. Stopping benchmark...")
    finally:
        bridge.stop()
        print("Bridge shut down successfully.")

if __name__ == "__main__":
    run_single_benchmark()