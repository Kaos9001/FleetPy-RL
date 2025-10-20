from abc import abstractmethod, ABCMeta
from typing import Dict, List, Any, Tuple, TYPE_CHECKING

from src.misc.globals import *

class RLAdapterMixin(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        self.reward = 0
        self.action_space = None
        self.observation_space = None

        super().__init__(*args, **kwargs)

    def add_init(self, operator_attributes, scenario_parameters):
        self.sim_end_time = scenario_parameters[G_SIM_END_TIME]
        self.rl_action_time_step = operator_attributes[G_RL_TIME_STEP]

        super().add_init(operator_attributes, scenario_parameters)

    def time_trigger(self, simulation_time : int, rl_action=None):
        self._call_time_trigger_rl_step(simulation_time, rl_action=rl_action)
        super().time_trigger(simulation_time)

        done = simulation_time + self.rl_action_time_step >= self.sim_end_time

        if rl_action is not None:
            return self.get_current_state(), self.get_reward(), done, False, {}
        return None

    @abstractmethod
    def setup_spaces(self):
        raise NotImplementedError()

    @abstractmethod
    def get_current_state(self):
        raise NotImplementedError()

    @abstractmethod
    def get_reward(self):
        raise NotImplementedError()

    @abstractmethod
    def _call_time_trigger_rl_step(self, simulation_time: int, rl_action=None):
        raise NotImplementedError()




