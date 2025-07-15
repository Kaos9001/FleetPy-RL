import gymnasium as gym
from gymnasium import spaces
import numpy as np

# import FleetPy modules
from src.misc.globals import *
import src.misc.config as config
from src.misc.init_modules import load_simulation_environment
from src.RLBatchOfferSimulation import RLBatchOfferSimulation
from src.fleetctrl.RLAdapterMixin import RLAdapterMixin

from typing import List
import logging

LOG = logging.getLogger(__name__)


class FleetPyEnv(gym.Env):
    """
    Custom FleetPy environment for Gymnasium API
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, rl_config):
        # Initialize the FleetPy environment
        super(FleetPyEnv, self).__init__()
        use_case: str = rl_config["use_case"]
        #action_no = rl_config["action_no"]
        start_config_i = rl_config["start_config_i"]
        cc_file = rl_config["cc_file"]
        sc_file = rl_config["sc_file"]
        self.use_case: str = use_case
        #self.action_no = action_no
        # Initialize your FleetPy simulation here using the config argument if necessary
        scs_path = os.path.join(os.path.dirname(__file__), "studies", "grid_study", "scenarios")

        cc = os.path.join(scs_path, cc_file)
        # sc = os.path.join(scs_path, "zonal_RL.csv")
        sc = os.path.join(scs_path, sc_file)
        if use_case == "train" or use_case == "baseline" or use_case == "zbaseline" or use_case.endswith("result"):
            log_level = "info"
            # sc = os.path.join(scs_path, "zonal_RL.csv")
        elif use_case == "test" or use_case == "baseline_test" or use_case == "zbaseline_test":
            log_level = "debug"
            # sc = os.path.join(scs_path, "example_test.csv")

        constant_cfg = config.ConstantConfig(cc)
        scenario_cfgs = config.ScenarioConfig(sc)
        const_abs = os.path.abspath(cc)
        study_name = os.path.basename(os.path.dirname(os.path.dirname(const_abs)))

        constant_cfg[G_STUDY_NAME] = study_name
        constant_cfg["n_cpu_per_sim"] = 1
        constant_cfg["evaluate"] = 1
        constant_cfg["log_level"] = log_level
        constant_cfg["keep_old"] = False

        if use_case == "train" or use_case == "baseline" or use_case == "zbaseline":
            constant_cfg["skip_file_writing"] = 1
        else:
            constant_cfg["skip_file_writing"] = 0

        # combine constant and scenario parameters into verbose scenario parameters
        for i, scenario_cfg in enumerate(scenario_cfgs):
            scenario_cfgs[i] = constant_cfg + scenario_cfg
        self.scenario_cfgs = scenario_cfgs
        self.current_config_i = start_config_i

        print(f"Loading simulation environment {self.current_config_i}...")
        self.SF: RLBatchOfferSimulation = load_simulation_environment(self.scenario_cfgs[self.current_config_i])
        self.SF.run(rl_init=True)
        self.sim_time = self.SF.start_time

        self.rl_adapter : RLAdapterMixin = self.SF.operators[0]

        # Define action and observation space
        # They must be gym.spaces objects
        self.rl_adapter.setup_spaces()
        self.action_space = self.rl_adapter.action_space
        self.observation_space = self.rl_adapter.observation_space

    def step(self, action):
        # Execute one RL time step within the environment
        # You should interact with your FleetPy simulation here based on the action
        # and return the next state, reward, done, and info
        # for sim_time in range(self.SF.start_time, self.SF.end_time, self.SF.time_step):

        if self.sim_time > self.SF.end_time:
            raise ValueError("Simulation has ended. Please reset the environment.")

        # Options here: match fleetpy steps with RL steps?
        n_steps = self.scenario_cfgs[self.current_config_i][G_RL_TIME_STEP] // self.SF.time_step

        for i in range(n_steps - 1):
            self.SF.step(self.sim_time)
            self.sim_time += self.SF.time_step
        observation, reward, done, truncated, info = self.SF.step(self.sim_time, rl_action=action)
        self.sim_time += self.SF.time_step

        # skip first 60 minute reward (initialization)
        if self.sim_time <= self.SF.start_time + 60 * 60:
            reward = 0

        return observation, reward, done, truncated, info

    def reset(self, seed=None, options=None, eval_result=False):
        # Reset the state of the environment to an initial state
        # This often involves restarting the FleetPy simulation
        super().reset(seed=seed)

        if eval_result:
            # record stats
            self.SF.record_stats()

            # save final state, record remaining travelers and vehicle tasks
            self.SF.save_final_state()
            self.SF.record_remaining_assignments()
            self.SF.demand.record_remaining_users()
            if not self.SF.skip_output:
                self.SF.evaluate()

        # move run_single_simulation() here to handle scenario iteration
        self.SF: RLBatchOfferSimulation = load_simulation_environment(self.scenario_cfgs[self.current_config_i])
        self.current_config_i += 1
        if self.current_config_i >= len(self.scenario_cfgs):
            self.current_config_i = 0

        self.SF.run(rl_init=True)
        self.sim_time = self.SF.start_time

        observation, reward, done, truncated, info  = self.SF.step(self.sim_time,-1)  # do nothing at first timestep
        # self.sim_time += self.SF.time_step

        return observation, None  # Return the initial observation

    def render(self, mode='human', close=False):
        # Render the environment to the screen or another output. This is optional and may not be needed for FleetPy.
        pass

    def close(self):
        # Perform any cleanup when the environment is closed
        pass
