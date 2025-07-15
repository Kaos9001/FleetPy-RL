import gymnasium as gym

from stable_baselines3 import A2C

from FleetPy_gym_rework import FleetPyEnv

if __name__ == "__main__":
    RL_config = {
        "use_case": "train",
        "start_config_i": 0,
        "cc_file": "large_constant_config_pool.csv",
        "sc_file": "large_pool_test.csv",
    }

    env = FleetPyEnv(RL_config)

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    vec_env = model.get_env()

    observation, info = env.reset()

    for i in range(1000):
        action, _state = model.predict(observation, deterministic=True)
        obs, reward, done, info = vec_env.step(action)




