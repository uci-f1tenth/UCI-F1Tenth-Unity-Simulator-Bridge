import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from F110Env import F110UnityEnv

def main():
    parser = argparse.ArgumentParser(description="Drive the car with a trained model")
    parser.add_argument("--model_path", type=str, default="models/f110_ppo_optimized", help="Model to load (without .zip)")
    args = parser.parse_args()

    # Must match the settings in Train.py
    LIDAR_SIZE = 180
    STACK_SIZE = 2

    env = DummyVecEnv([lambda: F110UnityEnv(lidar_size=LIDAR_SIZE)])
    env = VecFrameStack(env, n_stack=STACK_SIZE)

    # Load the normalization stats from training and freeze them
    env = VecNormalize.load(f"{args.model_path}_vecnorm.pkl", env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(args.model_path, env=env, device="cpu")

    obs = env.reset()
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _ = env.step(action)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    main()
