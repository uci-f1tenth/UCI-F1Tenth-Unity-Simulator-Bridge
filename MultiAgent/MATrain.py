import os
import argparse
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from MAF110VecEnv import MAF110VecEnv
from typing import Callable

class EntropyDecayCallback(BaseCallback):
    def __init__(self, initial_ent_coef=0.001, final_ent_coef=0.0, total_timesteps=50_000_000, verbose=0):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Calculate the fraction of training remaining
        fraction = 1.0 - (self.num_timesteps / self.total_timesteps)
        fraction = max(0.0, fraction) # Ensure it doesn't go below 0
        
        # Calculate the new entropy coefficient
        current_ent_coef = fraction * (self.initial_ent_coef - self.final_ent_coef) + self.final_ent_coef
        
        # Inject it directly into the algorithm
        self.model.ent_coef = current_ent_coef
        
        # Log it to TensorBoard so you can track the decay!
        self.logger.record("config/ent_coef", current_ent_coef)
        return True

def linear_schedule(initial_value: float, final_value: float = 0.0) -> Callable[[float], float]:
    """
    Returns a function that computes a linear decay.
    progress_remaining goes from 1.0 (start) to 0.0 (end).
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Agent F110 Unity RL")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--model_path", type=str, default="ma_f110_ppo_optimized", help="Path to load/save model")
    args = parser.parse_args()

    # --- Configuration ---
    LIDAR_SIZE = 180
    STACK_SIZE = 2
    TOTAL_TIMESTEPS = 50_000_000 

    # For large multi-agent batches, GPU becomes beneficial. 
    # Switch to "cpu" only if running less than 8 agents.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🖥️  Training Device: {device.upper()}")

    # --- Environment Setup ---
    # MAF110VecEnv is ALREADY a VecEnv. Do not wrap in DummyVecEnv.
    env = MAF110VecEnv(lidar_size=LIDAR_SIZE)

    env = VecMonitor(env)
    
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

    entropy_callback = EntropyDecayCallback(
        initial_ent_coef=0.001,
        final_ent_coef=0.0,
        total_timesteps=TOTAL_TIMESTEPS
    )

    callback_list = CallbackList([checkpoint_callback, entropy_callback])

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
            n_steps=2048,           
            batch_size=8192,        
            n_epochs=10,            
            gamma=0.995,            
            gae_lambda=0.95,        
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            use_sde=True,
            sde_sample_freq=20,
            
            # --- THE FIX ---
            learning_rate=linear_schedule(3e-4, 1e-5),  # This works perfectly
            clip_range=linear_schedule(0.2, 0.05),      # This works perfectly
            ent_coef=0.001                              # MUST be a float!
        )

    print(f"🏎️  Model Input Shape: {env.observation_space.shape}")
    print(f"🤖 Number of Agents: {env.num_envs}")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback_list,
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