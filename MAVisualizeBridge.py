import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from MAUnityBridge import MATCPTunnel, MAVehicleStatus # Ensure names match your saved bridge file

# --- Configuration ---
PORT = 5005
LIDAR_FOV = 270        # Degrees
LIDAR_MODE = "CCW"     # Options: "CCW" (Standard) or "CW" (Inverted)
                       # Must match the 'Scan Direction' setting in Unity Inspector!

# --- Global State for Visualization ---
selected_agent_index = 0
total_agents = 0

def on_key_press(event):
    """Handles keyboard input to switch between agents dynamically."""
    global selected_agent_index, total_agents
    if total_agents == 0:
        return
        
    if event.key == 'up':
        selected_agent_index = (selected_agent_index + 1) % total_agents
        print(f"🎯 Switched view to Agent [{selected_agent_index}]")
    elif event.key == 'down':
        selected_agent_index = (selected_agent_index - 1) % total_agents
        print(f"🎯 Switched view to Agent [{selected_agent_index}]")

def main():
    global selected_agent_index, total_agents
    
    print(f"🔌 Connecting to Multi-Agent Unity Bridge on port {PORT}...")
    bridge = MATCPTunnel(port=PORT)
    bridge.start()

    # Wait for connection
    while not bridge.connected:
        print("⏳ Waiting for Unity client to connect...")
        time.sleep(1.0)
    print("✅ Connected! Starting Visualization...")
    print("⌨️  Use UP/DOWN arrow keys on the plot window to cycle through agents.")

    # --- Setup Matplotlib Figure ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 7))
    fig.canvas.manager.set_window_title("MA Telemetry Visualizer")
    
    # Register keyboard event listener
    fig.canvas.mpl_connect('key_press_event', on_key_press)

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
        "ENVIRONMENT INFO\n"
        "--------------------------------\n"
        "Sequence ID:    {seq}\n"
        "Total Agents:   {agents_count}\n"
        "Sim Time:       {time:.2f} s\n"
        "Sim FPS:        {fps:.1f} ({dt:.4f}s)\n\n"
        
        "AGENT [{agent_id}] STATE\n"
        "--------------------------------\n"
        "Speed:          {speed:.2f} m/s\n"
        "Throttle:       {thr:.2f} %\n"
        "Steering:       {str:.2f} %\n"
        "Steering Ang:   {str_ang:.2f} deg\n"
        "Crashed:        {crashed}\n\n"
        
        "AGENT [{agent_id}] WHEELS & IMU\n"
        "--------------------------------\n"
        "RPM:            {rpm:.1f}\n"
        "Torque:         {trq:.2f} Nm\n"
        "IMU Angles:     {imu} deg\n"
        "Lin Accel:      {lin} m/s^2\n"
        "Ang Accel:      {ang} deg/s^2\n\n"
        
        "AGENT [{agent_id}] LIDAR DATA\n"
        "--------------------------------\n"
        "Scan Mode:      {l_mode}\n"
        "Point Count:    {lidar_count}\n"
        "1st Point Dist: {lidar_dist}\n"
        "Reward:         {rew:.4f}"
    )
    
    info_text = ax_text.text(0.05, 0.95, "Waiting for data...", 
                             transform=ax_text.transAxes, 
                             fontsize=11, family='monospace', va='top', color='lime')

    # --- Precompute Angles ---
    min_rad = np.radians(-LIDAR_FOV / 2)
    max_rad = np.radians(LIDAR_FOV / 2)

    # --- Update Function ---
    def update(frame):
        global selected_agent_index, total_agents
        
        if not bridge.connected:
            info_text.set_text("⚠️ Bridge Disconnected")
            return [lidar_line, info_text]
        
        data = bridge.get_telemetry()
        total_agents = data.numberOfAgents
        
        # Guard clause: Wait until data is populated
        if total_agents == 0 or len(data.speedMS) != total_agents:
            return [lidar_line, info_text]

        # Ensure index doesn't go out of bounds if agent count drops mid-sim
        if selected_agent_index >= total_agents:
            selected_agent_index = 0

        # Shortcut for currently selected agent index
        i = selected_agent_index

        # 1. Update Lidar for Selected Agent
        if data.lidarRanges.shape[0] > 0 and data.lidarRanges.shape[1] > 0:
            ranges = data.lidarRanges[i]
            
            # Determine angle distribution based on Mode
            if LIDAR_MODE == "CW":
                angles = np.linspace(min_rad, max_rad, len(ranges))
            else:
                angles = np.linspace(max_rad, min_rad, len(ranges))
            
            lidar_line.set_data(angles, ranges)

        # 2. Update Text Data for Selected Agent
        is_crashed = bool(data.isCrashed[i])
        status_str = "CRASHED!" if is_crashed else "OK"
        fps = 1.0 / data.fixedDeltaTime if data.fixedDeltaTime > 0 else 0
        
        display_str = text_template.format(
            seq=data.sequenceId,
            agents_count=total_agents,
            time=data.timeSinceStart,
            dt=data.fixedDeltaTime,
            fps=fps,
            agent_id=i,
            speed=data.speedMS[i],
            thr=data.throttle[i],
            str=data.steering[i],
            str_ang=data.steeringAngleDeg[i],
            crashed=status_str,
            rpm=data.currentWheelRPM[i],
            trq=data.appliedWheelTorque[i],
            imu=tuple(round(x, 2) for x in data.imuAngleDeg[i]),
            lin=tuple(round(x, 2) for x in data.linAccelMS[i]),
            ang=tuple(round(x, 2) for x in data.angAccelDeg[i]),
            rew=data.reward[i],
            lidar_count=data.lidarRanges.shape[1] if data.lidarRanges.size > 0 else 0,
            lidar_dist=data.lidarRanges[i][0],
            l_mode=LIDAR_MODE
        )
        
        info_text.set_text(display_str)
        
        # Color code crash status
        if is_crashed:
            info_text.set_color('red')
            lidar_line.set_color('red') # Make lidar red when crashed too
        else:
            info_text.set_color('lime')
            lidar_line.set_color('cyan')

        return [lidar_line, info_text]

    # Run Animation (50FPS refresh rate)
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