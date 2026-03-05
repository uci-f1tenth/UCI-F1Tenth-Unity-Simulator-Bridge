import os
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor  # <--- Essential for TensorBoard
from stable_baselines3.common.callbacks import CheckpointCallback
from f110_env import F110UnityEnv

def main():
    parser = argparse.ArgumentParser(description="Train F110 Unity RL Agent")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--model_path", type=str, default="f110_ppo_stacked_latest", help="Path to load/save model")
    args = parser.parse_args()

    # --- Configuration ---
    LIDAR_SIZE = 180
    STACK_SIZE = 2
    TOTAL_TIMESTEPS = 1_000_000

    # OPTIMIZATION: Use CPU for MLP (Faster than GPU latency for small nets)
    device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Training Device: {device.upper()} (Optimized for Low Latency)")

    # 1. Create Environment with Monitor Wrapper
    def make_env():
        env = F110UnityEnv(lidar_size=LIDAR_SIZE)
        return Monitor(env)  # <--- This fixes your empty TensorBoard!

    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=STACK_SIZE)

    # 2. Setup Checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path="./logs/checkpoints/",
        name_prefix="f110_model"
    )

    # 3. Load or Create Model
    if args.resume and os.path.exists(f"{args.model_path}.zip"):
        print(f"📂 Resuming training from {args.model_path}...")
        model = PPO.load(args.model_path, env=env, device=device, tensorboard_log="./f110_tensorboard/")
    else:
        print(f"🚀 Starting fresh training...")
        model = PPO(
            "MlpPolicy",
            env,
            device=device,
            policy_kwargs=dict(net_arch=[512, 512]),
            learning_rate=3e-4,     
            n_steps=4096,           
            batch_size=128,         # Smaller batch is often better on CPU
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            use_sde=True,           # Smoother steering
            sde_sample_freq=4,
            verbose=1,
            tensorboard_log="./f110_tensorboard/"
        )

    print(f"Training Started. Input Shape: {env.observation_space.shape}")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            tb_log_name="PPO_F110_Stacked",
            reset_num_timesteps=False
        )
        model.save(args.model_path)
    except KeyboardInterrupt:
        print("\n💾 Saving checkpoint...")
        model.save(args.model_path)
    finally:
        env.close()

if __name__ == "__main__":
    main()