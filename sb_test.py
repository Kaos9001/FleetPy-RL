import gymnasium as gym

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

from FleetPy_gym_rework import FleetPyEnv

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

import torch as th
import torch.nn as nn

def make_env(RL_config, rank: int, seed: int = 0):
    def _init() -> gym.Env:
        env = FleetPyEnv(RL_config)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

class CNNHead(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

if __name__ == "__main__":
    print("start")
    RL_config = {
        "use_case": "train",
        "start_config_i": 0,
        "cc_file": "large_constant_config_pool.csv",
        "sc_file": "large_pool_test.csv",
    }

    num_cpu = 5  # Number of processes to use
    env = make_vec_env(FleetPyEnv, n_envs=num_cpu, env_kwargs={"rl_config": RL_config}, vec_env_cls=SubprocVecEnv)
    #check_env(env)
    print("env created")

    policy_kwargs = dict(
        features_extractor_class=CNNHead,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False,
    )

    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_tensorboard/", policy_kwargs=policy_kwargs)
    print("model created")
    model.learn(total_timesteps=250_000, progress_bar=True, tb_log_name="first_run")
    print("model learned")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    #vec_env = model.get_env()
    #print("vec_env created")
    #observation, info = env.reset()
    #print("reset")
    #actions = []
    #for i in range(1000):
    #    action, _state = model.predict(observation, deterministic=True)
    #    actions.append(action)
    #    obs, reward, done, info = vec_env.step([action])




