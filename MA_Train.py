import os
import argparse
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from MA_F110VecEnv import MA_F110VecEnv

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Agent F110 Unity RL")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--model_path", type=str, default="ma_f110_ppo_optimized", help="Path to load/save model")
    args = parser.parse_args()

    # --- Configuration ---
    LIDAR_SIZE = 270
    STACK_SIZE = 2
    TOTAL_TIMESTEPS = 50_000_000 

    # For large multi-agent batches, GPU becomes beneficial. 
    # Switch to "cpu" only if running less than 8 agents.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🖥️  Training Device: {device.upper()}")

    # --- Environment Setup ---
    # MA_F110VecEnv is ALREADY a VecEnv. Do not wrap in DummyVecEnv.
    env = MA_F110VecEnv(lidar_size=LIDAR_SIZE)
    
    # Apply Frame Stacking
    env = VecFrameStack(env, n_stack=STACK_SIZE)

    # Apply Normalization
    if args.resume:
        env = VecNormalize.load(f"{args.model_path}_vecnorm.pkl", env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # --- Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path="./logs/checkpoints/",
        name_prefix="ma_f110_model"
    )

    # --- Model Setup ---
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256]), 
        log_std_init=-2, 
        ortho_init=False
    )

    if args.resume and os.path.exists(f"{args.model_path}.zip"):
        print(f"📂 Resuming training from {args.model_path}...")
        model = PPO.load(args.model_path, env=env, device=device)
    else:
        print(f"🚀 Starting fresh multi-agent training...")
        
        # NOTE: batch_size and n_steps are adjusted for Multi-Agent.
        # Total buffer size = n_steps * num_agents. 
        # Example: 2048 steps * 32 agents = 65,536 transitions per update.
        model = PPO(
            "MlpPolicy",
            env,
            device=device,
            verbose=1,
            tensorboard_log="./ma_f110_tensorboard/",
            learning_rate=3e-5,
            n_steps=2048,           # Steps per agent before update
            batch_size=1024,        # Size of mini-batches during PPO update
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.92,
            clip_range=0.1,
            ent_coef=0.00001,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            use_sde=True,
            sde_sample_freq=16
        )

    print(f"🏎️  Model Input Shape: {env.observation_space.shape}")
    print(f"🤖 Number of Agents: {env.num_envs}")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            tb_log_name="PPO_MA_F110",
            reset_num_timesteps=False
        )
        
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