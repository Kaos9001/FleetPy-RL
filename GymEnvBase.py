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

import random

class FleetPyEnv(gym.Env):
    """
    Custom FleetPy environment for Gymnasium API
    Loads data from constant + scenario files like standard FleetPy workflow
    Has an optional override for random demand generation instead of traditional demand file usage
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, rl_config, seed=None):
        # Initialize the FleetPy environment
        super(FleetPyEnv, self).__init__()

        # Load configs from parameter
        use_case: str = rl_config["use_case"]
        start_config_i = rl_config["start_config_i"]
        cc_file = rl_config["cc_file"]
        sc_file = rl_config["sc_file"]
        self.use_case: str = use_case

        # Setup scenario from config files
        scs_path = os.path.join(os.path.dirname(__file__), "studies", rl_config["scenario_name"], "scenarios")
        cc = os.path.join(scs_path, cc_file)
        sc = os.path.join(scs_path, sc_file)
        if use_case == "train" or use_case == "baseline" or use_case == "zbaseline" or use_case.endswith("result"):
            log_level = "info"
        elif use_case == "test" or use_case == "baseline_test" or use_case == "zbaseline_test":
            log_level = "debug"

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

        # Combine constant and scenario parameters into verbose scenario parameters
        for i, scenario_cfg in enumerate(scenario_cfgs):
            scenario_cfgs[i] = constant_cfg + scenario_cfg
        self.scenario_cfgs = scenario_cfgs
        self.current_config_i = start_config_i

        # Check for seed overwrite from parameter
        self.random_seed_on_reset = False
        if seed is not None:
            # If seed is -1, use random seeds on all resets
            if seed == -1:
                random.seed()
                seed = random.randint(0, 10**7)
                self.random_seed_on_reset = True
            self.scenario_cfgs[self.current_config_i][G_RANDOM_SEED] = seed

        # Check if a demand generator was provided. If so, use it to generate a temporary demand file for this run
        self.demand_generator = rl_config.get("demand_generator", None)
        if self.demand_generator:
            demand_f_name, demand_f_path = self.demand_generator(self.scenario_cfgs[self.current_config_i])
            self.scenario_cfgs[self.current_config_i][G_RQ_FILE] = demand_f_name
            self.active_demand_f_path = demand_f_path
            self.delete_temp_demand = rl_config.get("delete_temp_demand", False)

        # Initialize simulation instance from selected scenario
        self.SF: RLBatchOfferSimulation = load_simulation_environment(self.scenario_cfgs[self.current_config_i])
        self.SF.run(rl_init=True)
        self.sim_time = self.SF.start_time

        # Implemented for single-operator case only: fleet control class must inherit from RL adapter
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

        if self.sim_time > self.SF.end_time:
            raise ValueError("Simulation has ended. Please reset the environment.")

        n_steps = self.scenario_cfgs[self.current_config_i][G_RL_TIME_STEP] // self.SF.time_step #RL timestep has to be a multiple of the SF timestep

        for i in range(n_steps - 1):
            self.SF.step(self.sim_time)
            self.sim_time += self.SF.time_step
        assert self.sim_time % self.rl_adapter.rl_action_time_step == 0
        observation, reward, done, truncated, info = self.SF.step(self.sim_time, rl_action=action)
        self.sim_time += self.SF.time_step


        # skip first 60 minute reward (initialization)
        #if self.sim_time <= self.SF.start_time + 60 * 60:
        #    reward = 0
            
        return observation, reward, done, truncated, info

    def reset(self, seed=None, options=None, eval_result=False):
        # Reset the state of the environment to an initial state
        # This involves restarting the FleetPy simulation
        super().reset(seed=seed)

        # Evaluate if needed and clean up FleetPy
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
        self.current_config_i += 1
        if self.current_config_i >= len(self.scenario_cfgs):
            self.current_config_i = 0

        if seed is not None:
            self.scenario_cfgs[self.current_config_i][G_RANDOM_SEED] = seed
        elif self.random_seed_on_reset:
            self.scenario_cfgs[self.current_config_i][G_RANDOM_SEED] = np.random.randint(0, 10**7)

        # If a demand generator is being used, generate new demand for the next run
        if self.demand_generator:
            demand_f_name, demand_f_path = self.demand_generator(self.scenario_cfgs[self.current_config_i])
            self.scenario_cfgs[self.current_config_i][G_RQ_FILE] = demand_f_name
            if self.active_demand_f_path != demand_f_path and self.delete_temp_demand:
                self.active_demand_f_path.unlink(missing_ok=True)
            self.active_demand_f_path = demand_f_path

        self.SF: RLBatchOfferSimulation = load_simulation_environment(self.scenario_cfgs[self.current_config_i])
        self.rl_adapter: RLAdapterMixin = self.SF.operators[0]
        self.SF.run(rl_init=True)
        self.sim_time = self.SF.start_time

        #n_steps = self.scenario_cfgs[self.current_config_i][G_RL_TIME_STEP] // self.SF.time_step
        #for i in range(n_steps):
        #    observation, reward, done, truncated, info  = self.SF.step(self.sim_time, rl_action=0)  # do nothing at first timesteps
        #    self.sim_time += self.SF.time_step

        observation, reward, done, truncated, info = self.SF.step(self.sim_time, rl_action=0)
        self.sim_time += self.SF.time_step


        return observation, info  # Return the initial observation

    def render(self, mode='human', close=False):
        # Render the environment to the screen or another output. This is optional and may not be needed for FleetPy.
        pass

    def close(self, eval_result=False):
        # Perform any cleanup when the environment is closed
        if eval_result:
            # record stats
            self.SF.record_stats()

            # save final state, record remaining travelers and vehicle tasks
            self.SF.save_final_state()
            self.SF.record_remaining_assignments()
            self.SF.demand.record_remaining_users()
            if not self.SF.skip_output:
                self.SF.evaluate()

        super().close()