import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize
from MAF110VecEnv import MAF110VecEnv

def main():
    parser = argparse.ArgumentParser(description="Drive every car with a trained multi-agent model")
    parser.add_argument("--model_path", type=str, default="models/ma_f110_ppo_optimized", help="Model to load (without .zip)")
    args = parser.parse_args()

    # Must match the settings in MATrain.py
    LIDAR_SIZE = 180
    STACK_SIZE = 2

    env = MAF110VecEnv(lidar_size=LIDAR_SIZE)
    env = VecFrameStack(env, n_stack=STACK_SIZE)

    # Load the normalization stats from training and freeze them
    env = VecNormalize.load(f"{args.model_path}_vecnorm.pkl", env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(args.model_path, env=env, device="cpu")

    obs = env.reset()
    try:
        while True:
            actions, _ = model.predict(obs, deterministic=True)
            obs, _, _, _ = env.step(actions)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    main()
