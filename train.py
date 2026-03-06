import os
import argparse
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from F110Env import F110UnityEnv

def main():
    parser = argparse.ArgumentParser(description="Train F110 Unity RL Agent")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--model_path", type=str, default="f110_ppo_optimized", help="Path to load/save model")
    args = parser.parse_args()

    # --- Configuration ---
    LIDAR_SIZE = 270
    STACK_SIZE = 3
    TOTAL_TIMESTEPS = 50_000_000  # Increased for meaningful results

    # OPTIMIZATION 1: CPU is often faster for small MLPs (avoids GPU PCIe latency)
    device = "cpu" 
    print(f"🖥️  Training Device: {device.upper()}")

    # --- Environment Setup ---
    def make_env():
        env = F110UnityEnv(lidar_size=LIDAR_SIZE)
        return Monitor(env)  # Essential for TensorBoard logging

    # Create Vectorized Environment
    env = DummyVecEnv([make_env])
    
    # OPTIMIZATION 2: Frame Stacking (gives the agent a sense of velocity/acceleration)
    env = VecFrameStack(env, n_stack=STACK_SIZE)

    # OPTIMIZATION 3: Normalization (CRITICAL for PPO)
    # This scales observations to mean 0, std 1, and normalizes returns.
    # Without this, PPO gradients can explode or vanish if rewards are too large.
    if args.resume:
        # If resuming, load the running average statistics
        env = VecNormalize.load(f"{args.model_path}_vecnorm.pkl", env)
        env.training = True # Set to False only for inference/testing
        env.norm_reward = True
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # --- Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path="./logs/checkpoints/",
        name_prefix="f110_model"
    )

    # --- Model Setup ---
    # Optimized Hyperparameters for Racing (Continuous Control)
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256]), # Separated networks for Policy/Value
        log_std_init=-2, # Start with lower noise (more precise initial driving)
        ortho_init=False
    )

    if args.resume and os.path.exists(f"{args.model_path}.zip"):
        print(f"📂 Resuming training from {args.model_path}...")
        model = PPO.load(args.model_path, env=env, device=device)
    else:
        print(f"🚀 Starting fresh training...")
        model = PPO(
            "MlpPolicy",
            env,
            device=device,
            verbose=1,
            tensorboard_log="./f110_tensorboard/",
            
            # --- Hyperparameter Optimizations ---
            learning_rate=1e-4,     # Standard stable rate
            n_steps=2048,           # Shorter buffer than 8192 (updates more often)
            batch_size=256,         # Smaller batch = more updates per epoch
            n_epochs=15,            # 10 passes over the data
            gamma=0.99,             # High discount factor (care about crashing 5s later)
            gae_lambda=0.92,        # Bias-variance trade-off
            clip_range=0.2,
            ent_coef=0.005,         # Lower entropy (racing requires precision, not randomness)
            max_grad_norm=0.5,      # Prevent gradient spikes
            policy_kwargs=policy_kwargs,
            use_sde=True,           # State Dependent Exploration (Smooths steering)
            sde_sample_freq=8       # Resample noise every 4 steps (smooth control)
        )

    # --- Training Loop ---
    print(f"🏎️  Model Input Shape: {env.observation_space.shape}")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            tb_log_name="PPO_F110_Opt",
            reset_num_timesteps=False
        )
        
        # Save Model AND Normalization Stats
        model.save(args.model_path)
        env.save(f"{args.model_path}_vecnorm.pkl")
        print("💾 Model and Normalization stats saved.")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted! Saving...")
        model.save(args.model_path)
        env.save(f"{args.model_path}_vecnorm.pkl")
    finally:
        env.close()

if __name__ == "__main__":
    main()