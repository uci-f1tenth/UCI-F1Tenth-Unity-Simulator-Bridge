import socket
import struct
import time
import random

PORT_IN, PORT_OUT = 5005, 5006
PACKET_FORMAT = f'<If{360}f'
PACKET_SIZE = struct.calcsize(PACKET_FORMAT) 

rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rx.bind(("127.0.0.1", PORT_IN))
rx.setblocking(False)

tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"--- Python Telemetry Dashboard ---")
print(f"Listening on {PORT_IN} | Press Ctrl+C to stop\n")

packet_count = 0
last_pps_time = time.perf_counter()
current_pps = 0

try:
    while True:
        last_packet = None
        while True:
            try:
                data, addr = rx.recvfrom(PACKET_SIZE + 100)
                last_packet = data
                packet_count += 1
            except BlockingIOError:
                break

        if last_packet and len(last_packet) == PACKET_SIZE:
            unity_ts = struct.unpack('<f', last_packet[4:8])[0]
            
            # Simulated AI Logic
            steer_val = random.random() * 2.0 - 1.0 # Placeholder
            throttle_val = random.random() * 2.0 - 1.0 # Placeholder
            
            # Pong back
            pong_payload = struct.pack('<fff', unity_ts, steer_val, throttle_val)
            tx.sendto(pong_payload, ("127.0.0.1", PORT_OUT))

            # PPS Counter Logic
            now = time.perf_counter()
            if now - last_pps_time >= 1.0:
                current_pps = packet_count
                packet_count = 0
                last_pps_time = now

            # --- THE INSTANT DASHBOARD ---
            # Using \r to overwrite the same line
            # :<20 ensures we overwrite any old characters if the string gets shorter
            stats = f"PPS: {current_pps:<4} | UnityTS: {unity_ts:<8.2f} | Out: S:{steer_val:.2f} T:{throttle_val:.2f}"
            print(f"\r{stats}", end="", flush=True)

        #time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\nReceiver stopped by user.")
finally:
    rx.close()