import os.path
import sys

from GymEnvBase import FleetPyEnv
from demand_generators import make_gaussian_demand_generator

import src.misc.config as config
from src.misc.globals import *
import pandas as pd
from tqdm import tqdm
import time

import torch.nn as nn
import torch as th
import numpy as np

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack

import itertools
import multiprocessing as mp

MAIN_DIR = os.path.dirname(__file__)
MOD_STR = "MoD_0"
MM_STR = "Assertion"
LOG_F = "standard_bugfix.log"


class CNNHeadMk2(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=64, kernel_size=7, stride=2, padding=0,
                      groups=n_input_channels),
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


def run_training(name, sc_params):
    sc_params["scenario_name"] = f"{name}"

    RL_config = {
        "scenario_name": "example_rl_grid_study",
        "use_case": "train",
        "start_config_i": 0,
        "cc_file": "gaussian_rl_constant_config.csv",
        "sc_override": [sc_params],
        "delete_temp_demand": True,
        "demand_generator": make_gaussian_demand_generator(n_hotspots=12,
                                                           baseline_strength=0.005,
                                                           peak_fraction_range=(0.1, 0.9),
                                                           strength_range=(0.05, 0.15),
                                                           temporal_spread_range=(1200, 3600),
                                                           spatial_spread_range=(400, 800),
                                                           balance_range=(0.3, 0.7),
                                                           #candidate_nodes=[0, 25, 50, 1275, 1325, 2550, 2575, 2600]
                                                           ),
    }

    print(f"Starting training {name}...")
    start_time = time.time()

    num_cpu = 50
    env = make_vec_env(FleetPyEnv, n_envs=num_cpu, env_kwargs={"rl_config": RL_config, "seed": -1},
                       vec_env_cls=SubprocVecEnv)
    #n_stack = 4
    #env = VecFrameStack(env, n_stack=n_stack, channels_order='last')

    print("env created")

    policy_kwargs = dict(
        features_extractor_class=CNNHeadMk2,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False,
        #net_arch=[128, 128],
    )

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="/workspace/ext/fleetpy-rl/results/tensorboard/",
                policy_kwargs=policy_kwargs,
                learning_rate=cossine_annealling_schedule(1e-3, 5e-5),
                #exploration_fraction=0.5,
                #exploration_initial_eps=1,
                batch_size=256,
                n_steps=4096,
                stats_window_size=10,
                vf_coef=1.0,
                ent_coef=0.01,
                gamma=0.998)
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="/workspace/ext/fleetpy-rl/results/tensorboard/", policy_kwargs=policy_kwargs, learning_rate=5e-3, batch_size=128)
    # model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="/workspace/ext/fleetpy-rl/results/tensorboard/", policy_kwargs=policy_kwargs, learning_rate=5e-3)
    print("Model created")
    model.learn(total_timesteps=500_000, progress_bar=True, tb_log_name=name)
    print("Model learned")
    print(f"Training {name} done in {time.time() - start_time}s")
    model.save(name)
    return run_experiment(name + "autorun", sc_params, model_type=name)


def run_experiment(name, sc_params, model_type="random", seed=42):
    sc_params["scenario_name"] = f"exp_{name}_{seed}"
    sc_params["seedless_tag"] = f"exp_{name}"

    candidate_nodes = [0, 25, 50, 1275, 1325, 2550, 2575, 2600] if sc_params.get("use_candidates", False) else None

    RL_config = {
        "scenario_name": "example_rl_grid_study",
        "use_case": "devresult",
        "start_config_i": 0,
        "cc_file": "gaussian_rl_constant_config.csv",
        "sc_override": [sc_params],
        "delete_temp_demand": False,
        "demand_generator": make_gaussian_demand_generator(n_hotspots=12,
                                                           baseline_strength=0.005,
                                                           peak_fraction_range=(0.25, 0.83),
                                                           strength_range=(0.05, 0.15),
                                                           temporal_spread_range=(1200, 3600),
                                                           spatial_spread_range=(400, 800),
                                                           balance_range=(0.3, 0.7),
                                                           candidate_nodes=candidate_nodes
                                                           ),
    }

    print(f"Starting experiment {name}...")
    start_time = time.time()

    env = FleetPyEnv(RL_config, seed=seed)

    if model_type not in ["random", "nonrl_rt", "nonrl_free"]:
        try:
            model = DQN.load(model_type)
        except:
            raise ValueError("Unknown model")
    else:
        model = model_type

    observation, info = env.reset()
    episode_over = False
    actions = []
    rewards = []
    total_reward = 0
    #with tqdm(total=env.SF.end_time) as pbar:
    while not episode_over:
        if type(model) != str:
            action = model.predict(observation)[0].item()
        elif model == "random":
            action = int(env.action_space.sample())
        elif model == "nonrl_rt":
            action = -1
        elif model == "nonrl_free":
            action = -2

        actions.append(action)
        t = env.sim_time
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated
        #pbar.update(env.sim_time - t)
        total_reward += reward
        rewards.append(reward)

    env.close(eval_result=True)

    scs_path = os.path.join(os.path.dirname(__file__), "studies", "example_rl_grid_study", "scenarios")
    cc = os.path.join(scs_path, RL_config["cc_file"])
    constant_cfg = config.ConstantConfig(cc)

    const_abs = os.path.abspath(cc)
    study_name = os.path.basename(os.path.dirname(os.path.dirname(const_abs)))

    complete_scenario_cfg = constant_cfg + sc_params
    scenario_name = complete_scenario_cfg[G_SCENARIO_NAME]
    output_dir = os.path.join(MAIN_DIR, "studies", study_name, "results", scenario_name)
    standard_eval_f = os.path.join(output_dir, "standard_eval.csv")
    out_df = pd.read_csv(standard_eval_f, index_col=0)
    out_df.loc['total_reward', MOD_STR] = total_reward
    out_df.loc['model', MOD_STR] = model_type

    for key, value in complete_scenario_cfg.items():
        if type(value) == dict:
            value = str(value)
        out_df.loc[f"config_{key}", MOD_STR] = value


    out_df = out_df.rename(columns={MOD_STR: sc_params["scenario_name"]})

    print(f"Experiment {name} done in {time.time() - start_time}s")

    return out_df
'''
if __name__ == "__main__":
    data = {
        "op_fleet_composition": config.decode_config_str("microvan_vehtype:15"),
        "op_reoptimisation_timestep": 300,
        "op_rl_action_timestep": 180,
        "op_hub_roundtrip_max_duration": 3600,
        "op_hub_max_time_to_midpoint": 1800,
        "op_hub_idle_wait_duration": 300,
    }

    out_df = run_training("ppomk2", data)
    breakpoint()
'''

if __name__ == "__main__":
    veh_nums = [20, 30]
    models = ["cnnmk2a", "random", "nonrl_rt", "nonrl_free"]
    op_reop_ts = [180, 300]
    op_rl_act_ts = [180, 300]
    op_rt_dur = [3600, 5400]
    op_midpoint_dur = [0.5, 1]
    op_hub_idle = [300]
    use_candidates = [False]
    seeds = list(range(20,25))

    def execution(params):
        veh_num, model, reop, rl_act, rt_dur, mid_dur, idle, use_candidates, seed = params

        data = {
            "op_fleet_composition": config.decode_config_str(f"microvan_vehtype:{veh_num}"),
            "op_reoptimisation_timestep": reop,
            "op_rl_action_timestep": rl_act,
            "op_hub_roundtrip_max_duration": rt_dur,
            "op_hub_max_time_to_midpoint": rt_dur * mid_dur,
            "op_hub_idle_wait_duration": idle,
            "use_candidates": use_candidates,
        }
        scenario = f"{veh_num}veh_{model}_re{reop}_rl{rl_act}_rt{rt_dur}_mid{mid_dur}_idle{idle}_candidateslimited{use_candidates}"
        return run_experiment(scenario, data, model_type=model, seed=seed)

    param_grid = list(itertools.product(
        veh_nums, models, op_reop_ts, op_rl_act_ts, op_rt_dur, op_midpoint_dur, op_hub_idle, use_candidates, seeds
    ))

    len_grid = len(param_grid)
    
    processes = 8

    results = []
    counter = 0
    save_every = 20
    save_path = "results/grid_hub_sep25/data/full_5_tempsave.pkl"
    
    with mp.Pool(processes=processes) as pool:
        with tqdm(total=len_grid) as pbar:
            for res in pool.imap_unordered(execution, param_grid):
                results.append(res)
                counter += 1
                pbar.update(1)
                if counter % save_every == 0:
                    try:
                        pd.concat(results, axis=1).to_pickle(save_path)
                        print(f"Saved {counter} results to {save_path}")
                    except Exception as e:
                        print(f"Warning: could not save partial results ({e})")

    combined = pd.concat(results, axis=1)
    combined.to_csv("full_5_exps.csv")
    print(f"Final save with {len(results)} results to {save_path}")
    breakpoint()
