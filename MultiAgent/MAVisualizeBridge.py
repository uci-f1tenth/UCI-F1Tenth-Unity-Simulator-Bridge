import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from MAUnityBridge import MATCPTunnel

# --- Disable Matplotlib Default Hotkeys ---
# This prevents 's' from opening the save menu, and stops arrows/WASD from panning the camera.
conflicting_keys = ['w', 'a', 's', 'd', 'up', 'down', 'left', 'right']

for key, bindings in mpl.rcParams.items():
    if key.startswith('keymap.'):
        # Keep only the bindings that do NOT conflict with our controls
        mpl.rcParams[key] = [b for b in bindings if b not in conflicting_keys]

# --- Configuration ---
PORT = 5005

# --- Global State for Visualization & Control ---
selected_agent_index = 0
total_agents = 0

active_keys = {
    'w': False,
    's': False,
    'a': False,
    'd': False
}

bridge = None

def on_key_press(event):
    global selected_agent_index, total_agents, active_keys
    
    if event.key == 'up' and total_agents > 0:
        selected_agent_index = (selected_agent_index + 1) % total_agents
        print(f"🎯 Controlling Agent [{selected_agent_index}]")
    elif event.key == 'down' and total_agents > 0:
        selected_agent_index = (selected_agent_index - 1) % total_agents
        print(f"🎯 Controlling Agent [{selected_agent_index}]")
        
    if event.key in active_keys:
        active_keys[event.key] = True

def on_key_release(event):
    global active_keys
    if event.key in active_keys:
        active_keys[event.key] = False

def handle_batched_controls():
    """
    Constructs batched Numpy arrays for Steering and Throttle for ALL agents.
    Only the actively selected agent receives WASD inputs. Others receive 0.0.
    """
    if not bridge or not bridge.connected or total_agents == 0:
        return

    # Initialize empty action arrays for the entire batch
    steer_batch = np.zeros(total_agents, dtype=np.float32)
    throttle_batch = np.zeros(total_agents, dtype=np.float32)

    # Calculate inputs for the selected agent
    t_val = 0.0
    s_val = 0.0

    if active_keys['w']: t_val = 1.0
    if active_keys['s']: t_val = -1.0 # Reverse/Brake
    if active_keys['a']: s_val = -1.0 # Left
    if active_keys['d']: s_val = 1.0  # Right

    # Assign inputs only to the agent we are currently watching
    steer_batch[selected_agent_index] = s_val
    throttle_batch[selected_agent_index] = t_val

    # Broadcast batched commands to Unity C# loop
    bridge.send_action_batch(steer_batch, throttle_batch)

def main():
    global selected_agent_index, total_agents, bridge
    
    print(f"🔌 Connecting to Multi-Agent Unity Bridge on port {PORT}...")
    bridge = MATCPTunnel(port=PORT)
    bridge.start()

    while not bridge.connected:
        print("⏳ Waiting to connect to Unity Server...")
        time.sleep(1.0)
    print("✅ Connected! Starting Visualization...")
    print("⌨️  Use UP/DOWN arrows to switch active agent.")
    print("🎮  Use W/A/S/D to drive the selected agent!")

    # --- GUI Setup ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 7))
    fig.canvas.manager.set_window_title("RL Telemetry Visualizer & Remote")
    
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)

    # Lidar Subplot
    ax_lidar = fig.add_subplot(1, 2, 1, projection='polar')
    ax_lidar.set_theta_zero_location("N")  
    ax_lidar.set_theta_direction(-1)       
    lidar_line, = ax_lidar.plot([], [], '.', markersize=2, color='cyan', alpha=0.6)

    # Text Subplot
    ax_text = fig.add_subplot(1, 2, 2)
    ax_text.axis('off')
    
    text_template = (
        "ENVIRONMENT INFO\n"
        "--------------------------------\n"
        "Sequence ID:    {seq}\n"
        "Total Agents:   {agents_count}\n"
        "Sim Time:       {time:.2f} s\n"
        "Sim FPS:        {fps:.1f} ({dt:.4f}s)\n\n"
        
        "AGENT [{agent_id}] STATE (Controlling)\n"
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
        "Wheel Ticks:    {ticks}\n"
        "IMU Angles:      {imu} deg\n"
        "Lin Accel:      {lin} m/s^2\n"
        "Ang Accel:      {ang} deg/s^2\n\n"
        
        "AGENT [{agent_id}] LIDAR DATA\n"
        "--------------------------------\n"
        "FOV Range:      {l_min_ang} to {l_max_ang} deg\n"
        "Scan Mode:      {l_mode}\n"
        "Point Count:    {lidar_count}\n"
        "Reward:         {rew:.4f}"
    )
    
    info_text = ax_text.text(0.05, 0.95, "Waiting for data...", 
                             transform=ax_text.transAxes, 
                             fontsize=11, family='monospace', va='top', color='lime')

    # --- Render Loop ---
    def update(frame):
        global selected_agent_index, total_agents
        
        # Broadcast batched controls every frame
        handle_batched_controls()

        if not bridge.connected:
            info_text.set_text("⚠️ Bridge Disconnected")
            return [lidar_line, info_text]
        
        data = bridge.get_telemetry()
        total_agents = data.numberOfAgents
        
        if total_agents == 0 or len(data.speedMS) != total_agents:
            return [lidar_line, info_text]

        if selected_agent_index >= total_agents:
            selected_agent_index = 0

        i = selected_agent_index

        if data.lidarMaxRange > 0:
            ax_lidar.set_ylim(0, data.lidarMaxRange)
            
        scan_dir_str = "CW" if data.lidarScanDirection == 1 else "CCW"
        ax_lidar.set_title(f"Lidar Scan ({scan_dir_str} | Max: {data.lidarMaxRange}m)", color='white', pad=20)

        # Update Lidar map (Using corrected CW/CCW mapping)
        if data.lidarRanges.shape[0] > 0 and data.lidarRanges.shape[1] > 0:
            ranges = data.lidarRanges[i]
            
            min_rad = np.radians(data.lidarMinAngleDeg)
            max_rad = np.radians(data.lidarMaxAngleDeg)
            
            if scan_dir_str == "CCW":
                angles = np.linspace(max_rad, min_rad, len(ranges)) 
            else:
                angles = np.linspace(min_rad, max_rad, len(ranges))
            
            lidar_line.set_data(angles, ranges)

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
            ticks=data.wheelTicks[i],
            imu=tuple(round(x, 2) for x in data.imuAngleDeg[i]),
            lin=tuple(round(x, 2) for x in data.linAccelMS[i]),
            ang=tuple(round(x, 2) for x in data.angAccelDeg[i]),
            l_min_ang=data.lidarMinAngleDeg,
            l_max_ang=data.lidarMaxAngleDeg,
            rew=data.reward[i],
            lidar_count=data.lidarRanges.shape[1] if data.lidarRanges.size > 0 else 0,
            l_mode=scan_dir_str
        )
        
        info_text.set_text(display_str)
        
        if is_crashed:
            info_text.set_color('red')
            lidar_line.set_color('red') 
        else:
            info_text.set_color('lime')
            lidar_line.set_color('cyan')

        return [lidar_line, info_text]

    # Run Animation
    ani = FuncAnimation(fig, update, interval=20, blit=True, cache_frame_data=False)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        print("🛑 Closing Bridge...")
        if bridge:
            bridge.stop()

if __name__ == "__main__":
    main()