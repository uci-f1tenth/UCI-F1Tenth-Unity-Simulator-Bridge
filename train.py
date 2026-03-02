# train.py
from stable_baselines3 import PPO
from f110_env import F110UnityEnv
import os

def main():
    # 1. Create Env
    env = F110UnityEnv()

    # 2. PPO Configuration
    # We use a smaller learning rate and larger batch for stability in racing sims
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./f110_tensorboard/"
    )

    print("Training Started. Ensure Unity is in Play Mode.")
    
    try:
        # Start learning
        model.learn(
            total_timesteps=500_000,
            tb_log_name="PPO_Baseline_Run",
            reset_num_timesteps=False
        )
        model.save("f110_ppo_final")
        
    except KeyboardInterrupt:
        print("\nSaving partial model...")
        model.save("f110_ppo_interrupted")
    finally:
        env.close()

if __name__ == "__main__":
    main()