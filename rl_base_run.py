import os.path
import sys

from GymEnvBase import FleetPyEnv
from demand_generators import make_gaussian_demand_generator

import src.misc.config as config
from src.misc.init_modules import load_simulation_environment
from src.misc.globals import *
import pandas as pd
from tqdm import tqdm

from stable_baselines3 import DQN, PPO

MAIN_DIR = os.path.dirname(__file__)
MOD_STR = "MoD_0"
MM_STR = "Assertion"
LOG_F = "standard_bugfix.log"

def read_outputs_for_comparison(constant_csv, scenario_csv):
    """This function reads some output parameters for a test of meaningful results of the test cases.

    :param constant_csv: constant parameter definition
    :param scenario_csv: scenario definition
    :return: list of standard_eval data frames
    :rtype: list[DataFrame]
    """
    constant_cfg = config.ConstantConfig(constant_csv)
    scenario_cfgs = config.ScenarioConfig(scenario_csv)
    const_abs = os.path.abspath(constant_csv)
    study_name = os.path.basename(os.path.dirname(os.path.dirname(const_abs)))
    return_list = []
    for scenario_cfg in scenario_cfgs:
        complete_scenario_cfg = constant_cfg + scenario_cfg
        scenario_name = complete_scenario_cfg[G_SCENARIO_NAME]
        output_dir = os.path.join(MAIN_DIR, "studies", study_name, "results", scenario_name)
        standard_eval_f = os.path.join(output_dir, "standard_eval.csv")
        tmp_df = pd.read_csv(standard_eval_f, index_col=0)
        tmp_df.loc[G_SCENARIO_NAME, MOD_STR] = scenario_name
        return_list.append((tmp_df))
    return return_list

def check_assertions(list_eval_df, all_scenario_assertion_dict):
    """This function checks assertions of scenarios to give a quick impression if results are fitting.

    :param list_eval_df: list of evaluation data frames
    :param all_scenario_assertion_dict: dictionary of scenario id to assertion dictionaries
    :return: list of (scenario_name, mismatch_flag, tmp_df) tuples
    """
    list_result_tuples = []
    for sc_id, assertion_dict in all_scenario_assertion_dict.items():
        tmp_df = list_eval_df[sc_id]
        scenario_name = tmp_df.loc[G_SCENARIO_NAME, MOD_STR]
        print("-"*80)
        mismatch = False
        for k, v in assertion_dict.items():
            if tmp_df.loc[k, MOD_STR] != v:
                tmp_df.loc[k, MM_STR] = v
                mismatch = True
        if mismatch:
            prt_str = f"Scenario {scenario_name} has mismatch with assertions:/n{tmp_df}/n" + "-"*80 + "/n"
        else:
            prt_str = f"Scenario {scenario_name} results match assertions/n" + "-"*80 + "/n"
        print(prt_str)
        with open(LOG_F, "a") as fh:
            fh.write(prt_str)
        list_result_tuples.append((scenario_name, mismatch, tmp_df))
    return list_result_tuples

# Main execution
if __name__ == "__main__":
    RL_config = {
        "scenario_name": "example_rl_grid_study",
        "use_case": "devresult",
        "start_config_i": 0,
        "cc_file": "gaussian_rl_constant_config.csv",
        "sc_file": "gaussian_rl_scenario_config.csv",
        "delete_temp_demand": False,
        "demand_generator": make_gaussian_demand_generator(n_hotspots=12,
                                                           baseline_strength=0.005,
                                                           peak_fraction_range=(0.1, 0.9),
                                                           strength_range=(0.05, 0.15),
                                                           temporal_spread_range=(1200, 3600),
                                                           spatial_spread_range=(400, 800), 
                                                           balance_range=(0.3, 0.7), candidate_nodes=[0, 25, 50, 1275, 1325, 2550, 2575, 2600]
                                                          ),
    }

    env = FleetPyEnv(RL_config)

    observation, info = env.reset(seed=473247562)

    model = DQN.load("state_cnnheadmk2_req_fix_large_penalty_dqn")
    print(model.policy)

    episode_over = False
    actions = []
    total_reward = 0
    with tqdm(total=env.SF.end_time) as pbar:
        while not episode_over:
            action = model.predict(observation)[0].item()
            #action = int(env.action_space.sample())
            #action = -1
            actions.append(action)
            #print(env.rl_adapter.vehs_in_hub)
            
            t = env.sim_time
            observation, reward, terminated, truncated, info = env.step(action)
            #print(f"Reward was: {reward}")
            episode_over = terminated or truncated
            pbar.update(env.sim_time - t)
            total_reward += reward

    print(total_reward)
    print(env.active_demand_f_path)
    env.close(eval_result=True)

    scs_path = os.path.join(os.path.dirname(__file__), "studies", "example_rl_grid_study", "scenarios")
    cc = os.path.join(scs_path, RL_config["cc_file"])
    sc = os.path.join(scs_path, RL_config["sc_file"])
    list_results = read_outputs_for_comparison(cc, sc)
    print(list_results)
    print(actions)
    #all_scenario_assert_dict = {0: {"number users": 0}}
    #check_assertions(list_results, all_scenario_assert_dict)
