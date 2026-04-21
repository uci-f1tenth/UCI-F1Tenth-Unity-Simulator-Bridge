import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from UnityBridge import UnityTCPBridge, VehicleStatus

# --- Configuration ---
PORT = 5005
LIDAR_FOV = 270        # Degrees
LIDAR_MODE = "CCW"     # Options: "CCW" (Standard) or "CW" (Inverted)
                       # Must match the 'Scan Direction' setting in Unity Inspector!

def main():
    print(f"🔌 Connecting to Unity Bridge on port {PORT}...")
    bridge = UnityTCPBridge(port=PORT)
    bridge.start()

    # Wait for connection
    while not bridge.connected:
        print("⏳ Waiting for Unity client to connect...")
        time.sleep(1.0)
    print("✅ Connected! Starting Visualization...")

    # --- Setup Matplotlib Figure ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 7))
    fig.canvas.manager.set_window_title("UnityBridge Telemetry Visualizer")

    # 1. Lidar Plot (Polar)
    ax_lidar = fig.add_subplot(1, 2, 1, projection='polar')
    ax_lidar.set_title(f"Lidar Scan ({LIDAR_MODE})", color='white', pad=20)
    
    # Setup Orientation: 0 is North (Front), Angles go Clockwise
    ax_lidar.set_theta_zero_location("N")  
    ax_lidar.set_theta_direction(-1)       
    
    ax_lidar.set_ylim(0, bridge.lidar_scale * 65535.0) # Max range
    
    # Initialize empty plot line
    lidar_line, = ax_lidar.plot([], [], '.', markersize=2, color='cyan', alpha=0.6)

    # 2. Text Data Panel
    ax_text = fig.add_subplot(1, 2, 2)
    ax_text.axis('off')
    
    # Template for text data
    text_template = (
        "BASIC INFO\n"
        "--------------------------------\n"
        "Sequence ID:    {seq}\n"
        "Time:           {time:.2f} s\n"
        "Delta Time:     {fps:.1f} ({dt:.4f}s)\n\n"
        
        "VEHICLE STATE\n"
        "--------------------------------\n"
        "Speed:          {speed:.2f} m/s\n"
        "Throttle:       {thr:.2f} %\n"
        "Steering:       {str:.2f} %\n"
        "Steering Ang:   {str_ang:.2f} deg\n"
        "Crashed:        {crashed}\n\n"
        
        "WHEELS & IMU\n"
        "--------------------------------\n"
        "RPM:            {rpm:.1f}\n"
        "Wheel Ticks:    {ticks}\n"
        "IMU Angles:      {imu} deg\n"
        "Lin Accel:      {lin} m/s^2\n"
        "Ang Accel:      {ang} deg/s^2\n\n"
        
        "LIDAR DATA\n"
        "--------------------------------\n"
        "Scan Mode:      {l_mode}\n"
        "Point Count:    {lidar_count}\n"
        "Reward:         {rew:.4f}"
    )
    
    info_text = ax_text.text(0.05, 0.95, "Waiting for data...", 
                             transform=ax_text.transAxes, 
                             fontsize=11, family='monospace', va='top', color='lime')

    # --- Precompute Angles ---
    # We can't precompute the full array because the size depends on received data,
    # but we define the logic here.
    min_rad = np.radians(-LIDAR_FOV / 2)
    max_rad = np.radians(LIDAR_FOV / 2)

    # --- Update Function ---
    def update(frame):
        if not bridge.connected:
            info_text.set_text("⚠️ Bridge Disconnected")
            return [lidar_line, info_text]

        # Send a "Stay Still" command to keep connection alive if needed
        # bridge.send_control(0, 0) 
        
        data = bridge.get_telemetry()
        
        if data.sequenceId == 0 and len(data.lidarRanges) == 0:
            return [lidar_line, info_text]

        # 1. Update Lidar
        ranges = np.array(data.lidarRanges)
        if len(ranges) > 0:
            # Determine angle distribution based on Mode
            if LIDAR_MODE == "CCW":
                # Standard: -135 to +135
                angles = np.linspace(min_rad, max_rad, len(ranges))
            else:
                # CW: +135 to -135
                angles = np.linspace(max_rad, min_rad, len(ranges))
            
            lidar_line.set_data(angles, ranges)

        # 2. Update Text
        status_str = "CRASHED!" if data.isCrashed == VehicleStatus.CRASHED else "OK"
        fps = 1.0 / data.fixedDeltaTime if data.fixedDeltaTime > 0 else 0
        
        display_str = text_template.format(
            seq=data.sequenceId,
            time=data.timeSinceStart,
            dt=data.fixedDeltaTime,
            fps=fps,
            speed=data.speedMS,
            thr=data.throttle,
            str=data.steering,
            str_ang=data.steeringAngleDeg,
            crashed=status_str,
            rpm=data.currentWheelRPM,
            ticks=data.wheelTicks,
            imu=tuple(round(x, 2) for x in data.imuAngleDeg),
            lin=tuple(round(x, 2) for x in data.linAccel),
            ang=tuple(round(x, 2) for x in data.angAccelDeg),
            rew=data.reward,
            lap=data.lapCount,
            lidar_count=data.lidarCount,
            l_mode=LIDAR_MODE
        )
        
        info_text.set_text(display_str)
        
        # Color code crash status
        if data.isCrashed == VehicleStatus.CRASHED:
            info_text.set_color('red')
        else:
            info_text.set_color('lime')

        return [lidar_line, info_text]

    # Run Animation
    # Interval=20ms (50FPS refresh rate for the plot)
    ani = FuncAnimation(fig, update, interval=20, blit=True, cache_frame_data=False)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        print("🛑 Closing Bridge...")
        bridge.stop()

if __name__ == "__main__":
    main()