import gymnasium as gym

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

from GymEnvBase import FleetPyEnv

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack

import torch as th
import torch.nn as nn
import numpy as np

from demand_generators import make_gaussian_demand_generator

class CNNHead(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=5, stride=2, padding=0),
            nn.LayerNorm([24,24]),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.LayerNorm([11,11]),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.LayerNorm([5,5]),
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

class CNNHeadMk2(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=64, kernel_size=7, stride=2, padding=0, groups=n_input_channels),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.GroupNorm(num_groups=16, num_channels=128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
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

def cossine_annealling_schedule(initial_value, max_value):
    def func(progress_remaining):
        return initial_value + 0.5 * (max_value - initial_value) * (1 + np.cos(progress_remaining * np.pi))

    return func

if __name__ == "__main__":
    print("start")
    RL_config = {
        "scenario_name": "example_rl_grid_study",
        "use_case": "train",
        "start_config_i": 0,
        "cc_file": "gaussian_rl_constant_config.csv",
        "sc_file": "gaussian_rl_scenario_config.csv",
        "delete_temp_demand": True,
        "demand_generator": make_gaussian_demand_generator(n_hotspots=12,
                                                           baseline_strength=0.005,
                                                           peak_fraction_range=(0.1, 0.9),
                                                           strength_range=(0.05, 0.15),
                                                           temporal_spread_range=(1200, 3600),
                                                           spatial_spread_range=(400, 800), 
                                                           balance_range=(0.3, 0.7), candidate_nodes=[0, 25, 50, 1275, 1325, 2550, 2575, 2600]
                                                          ),
    }

    num_cpu = 50  # Number of processes to use
    env = make_vec_env(FleetPyEnv, n_envs=num_cpu, env_kwargs={"rl_config": RL_config, "seed": -1}, vec_env_cls=SubprocVecEnv)
    #n_stack = 4
    #env = VecFrameStack(env, n_stack=n_stack, channels_order='last')
    
    print("env created")

    policy_kwargs = dict(
        features_extractor_class=CNNHeadMk2,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False,
        #net_arch=[128, 128],
    )

    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="/workspace/ext/fleetpy-rl/results/tensorboard/",
                policy_kwargs=policy_kwargs,
                learning_rate=cossine_annealling_schedule(1e-3, 5e-5),
                exploration_fraction=0.5,
                exploration_initial_eps=1,
                batch_size=128,
                gamma=0.998)
    #model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="/workspace/ext/fleetpy-rl/results/tensorboard/", policy_kwargs=policy_kwargs, learning_rate=5e-3, batch_size=128)
    #model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="/workspace/ext/fleetpy-rl/results/tensorboard/", policy_kwargs=policy_kwargs, learning_rate=5e-3)
    print("model created")

    name = "rq_serveonly_nopen_40pax_candidates_15veh_3600time_180step_dqn"
    
    model.learn(total_timesteps=500_000, progress_bar=True, tb_log_name=name)
    print("model learned")
    model.save(name)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=20)
    #vec_env = model.get_env()
    #print("vec_env created")
    #observation, info = env.reset()
    #print("reset")
    #actions = []
    #for i in range(1000):
    #    action, _state = model.predict(observation, deterministic=True)
    #    actions.append(action)
    #    obs, reward, done, info = vec_env.step([action])




