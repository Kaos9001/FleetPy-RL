# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
from src.fleetctrl.RidePoolingBatchAssignmentFleetcontrol import RidePoolingBatchAssignmentFleetcontrol
from src.fleetctrl.RLAdapterMixin import RLAdapterMixin

from typing import Dict, List, TYPE_CHECKING, Any, Tuple

# additional module imports (> requirements)
# ------------------------------------------
import numpy as np
import pandas as pd
import shapely
import time
import pyproj
import geopandas as gpd

# src imports
# -----------
from src.simulation.Offers import TravellerOffer
from src.fleetctrl.planning.VehiclePlan import VehiclePlan, PlanStop, RoutingTargetPlanStop
from src.fleetctrl.planning.PlanRequest import PlanRequest
from src.simulation.Legs import VehicleRouteLeg

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
LOG = logging.getLogger(__name__)
LARGE_INT = 100000
# TOL = 0.1

from src.simulation.Vehicles import SimulationVehicle
from src.simulation.StationaryProcess import ChargingProcess


# -------------------------------------------------------------------------------------------------------------------- #
# functions
# ---------
from src.misc.globals import *


import gymnasium as gym

class RLDirectedSingleHubPoolingFleetControl(RLAdapterMixin, RidePoolingBatchAssignmentFleetcontrol):
    def __init__(self, op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters,
                 dir_names, op_charge_depot_infra=None, list_pub_charging_infra=[]):
        """
        Fleet control module based on a single hub from which all vehicles depart and return to. All vehicles start at
        the hub and are redirected there if idle elsewhere.
        User requests must have the hub as either origin or destination.

        ride pooling optimisation is called after every optimisation_time_step and offers are created in the
        time_trigger function
        if "user_max_wait_time_2" is given:
            if the user couldn't be assigned in the first try, it will be considered again in the next opt-step with
            this new max_waiting_time constraint
        if "user_offer_time_window" is given:
            after accepting an offer the pick-up time is constraint around the expected pick-up time with an interval
            of the size of this parameter


        :param op_id: operator id
        :type op_id: int
        :param operator_attributes: dictionary with keys from globals and respective values
        :type operator_attributes: dict
        :param list_vehicles: simulation vehicles; their assigned plans should be instances of the VehicleRouteLeg class
        :type list_vehicles: list
        :param routing_engine: routing engine
        :type routing_engine: Network
        :param scenario_parameters: access to all scenario parameters (if necessary)
        :type scenario_parameters: dict
        :param op_charge_depot_infra: reference to a OperatorChargingAndDepotInfrastructure class (optional)
        (unique for each operator)
        :type op_charge_depot_infra: OperatorChargingAndDepotInfrastructure
        :param list_pub_charging_infra: list of PublicChargingInfrastructureOperator classes (optional)
        (accesible for all agents)
        :type list_pub_charging_infra: list of PublicChargingInfrastructureOperator
        """

        super().__init__(op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters,
                         dir_names=dir_names, op_charge_depot_infra=op_charge_depot_infra,
                         list_pub_charging_infra=list_pub_charging_infra)

        self.sim_start_time = scenario_parameters[G_SIM_START_TIME]

        self.hub_id = (len(routing_engine.nodes) - 1) // 2
        self.hub_pos = self.routing_engine.return_node_position(self.hub_id)

        self.max_time_to_midpoint = operator_attributes.get(G_OP_HUB_MID_DUR, LARGE_INT)
        self.round_trip_max_duration = operator_attributes.get(G_OP_HUB_RT_DUR, LARGE_INT)

        # dict of vehicles in hub (vid -> -1  if not in hub (active), else vid -> timestamp of arrival in hub)
        self.vehs_in_hub = {}

        self.target_directions_dict = {}

        target_directions_f = os.path.join(self.dir_names[G_DIR_INFRA], "action_target_nodes.csv")
        target_directions_df = pd.read_csv(target_directions_f)
        for _, row in target_directions_df.iterrows():
            action_id = row[G_ACTION_ID]
            node_index = row[G_NODE_ID]
            node_pos = self.routing_engine.return_node_position(node_index)
            self.target_directions_dict[action_id] = node_pos

        self.wrong_action_penalty = 100
        self.rw_rejection_penalty = 10
        self.rw_drive_distance_penalty = 1/10000
        self.rw_waiting_time_penalty = 1/600

        # grid specific
        self.n_nodes = self.routing_engine.get_number_network_nodes()
        self.grid_side_length = int(np.sqrt(self.n_nodes))

        self.reward = 0

        self.rejection_counter = 0
        self.driven_distance_dict = {}

        # use moving average here?
        self.waiting_times = []


    def add_init(self, operator_attributes, scenario_parameters):
        # All vehicles start deactivated at the hub
        super().add_init(operator_attributes, scenario_parameters)
        for vid, veh_obj in enumerate(self.sim_vehicles):
            init_state_info = {}
            init_state_info[G_V_INIT_NODE] = self.hub_id
            init_state_info[G_V_INIT_TIME] = scenario_parameters[G_SIM_END_TIME]
            init_state_info[G_V_INIT_SOC] = 0.5 * (1 + np.random.random())
            veh_obj.set_initial_state(self, self.routing_engine, init_state_info,
                                      scenario_parameters[G_SIM_START_TIME], veh_init_blocking=False)
            self.vehs_in_hub[vid] = scenario_parameters[G_SIM_START_TIME]

            veh_obj.status = VRL_STATES.OUT_OF_SERVICE
            plan_stop = RoutingTargetPlanStop(veh_obj.pos, locked=True, duration=self.sim_end_time,
                                              planstop_state=G_PLANSTOP_STATES.INACTIVE)
            self.veh_plans[veh_obj.vid].add_plan_stop(plan_stop, veh_obj, self.sim_start_time, self.routing_engine)
            veh_obj.assigned_route.append(VehicleRouteLeg(VRL_STATES.OUT_OF_SERVICE, veh_obj.pos, {},duration=self.sim_end_time, locked=True))
            veh_obj.start_next_leg_first = True

            self.driven_distance_dict[vid] = veh_obj.cumulative_distance

    def _activate_and_route_vehicle(self, vid : int, simulation_time, rl_action):
        # Activate vehicle
        veh_to_activate = self.sim_vehicles[vid]
        self.vehs_in_hub[vid] = -1
        _, inactive_vrl = veh_to_activate.end_current_leg(simulation_time)
        self.receive_status_update(veh_to_activate.vid, simulation_time, [inactive_vrl])

        #assert self.round_trip_max_duration > self.max_time_to_midpoint

        if type(rl_action) is np.ndarray:
            rl_action = rl_action.item()

        # Set plan to go to midpoint and back
        veh_plan = self.veh_plans[veh_to_activate.vid]
        direction_pos = self.target_directions_dict[rl_action]
        rl_action_stop = RoutingTargetPlanStop(direction_pos, earliest_start_time=simulation_time,
                                     latest_start_time=simulation_time + min(self.max_time_to_midpoint, self.max_time_to_midpoint))
        veh_plan.add_plan_stop(rl_action_stop, veh_to_activate, simulation_time, self.routing_engine)
        return_to_hub_stop = RoutingTargetPlanStop(self.hub_pos, earliest_start_time=simulation_time, locked=True,
                                              latest_start_time=simulation_time + self.round_trip_max_duration)
        veh_plan.add_plan_stop(return_to_hub_stop, veh_to_activate, simulation_time, self.routing_engine)
        deactivate_stop = RoutingTargetPlanStop(self.hub_pos, locked_end=True, duration=self.sim_end_time,
                                                planstop_state=G_PLANSTOP_STATES.INACTIVE)
        veh_plan.add_plan_stop(deactivate_stop, veh_to_activate, simulation_time, self.routing_engine)

        # Hardcoding replacement for assign_vehicle_plan
        veh_plan.update_tt_and_check_plan(veh_to_activate, simulation_time, self.routing_engine, keep_feasible=True)

        new_list_vrls = [
            VehicleRouteLeg(VRL_STATES.ROUTE, direction_pos, {}),
            VehicleRouteLeg(VRL_STATES.REPO_TARGET, direction_pos, {}, locked=True, duration=0),
            VehicleRouteLeg(VRL_STATES.ROUTE, self.hub_pos, {}),
            VehicleRouteLeg(VRL_STATES.OUT_OF_SERVICE, self.hub_pos, {}, locked=True, duration=self.sim_end_time),
        ]
        veh_to_activate.assign_vehicle_plan(new_list_vrls, simulation_time, force_ignore_lock=True)
        self.veh_plans[veh_to_activate.vid] = veh_plan
        self.RPBO_Module.set_assignment(veh_to_activate.vid, veh_plan)
        veh_to_activate.start_next_leg(simulation_time)
        # ----------------------------------------------
        # self.assign_vehicle_plan(veh_to_activate, veh_plan, simulation_time, add_arg=True)

    def receive_status_update(self, vid : int, simulation_time : int, list_finished_VRL : List[VehicleRouteLeg], force_update : bool=True):
        if self.vehs_in_hub[vid] == -1 and len(self.sim_vehicles[vid].assigned_route) == 1:
            self.vehs_in_hub[vid] = 0

        super().receive_status_update(vid, simulation_time, list_finished_VRL, force_update=force_update)

    def _call_time_trigger_rl_step(self, simulation_time : int, rl_action=None):
        if simulation_time % self.rl_action_time_step == 0:
            self.reward = 0
            self.rejection_counter = 0
            if rl_action:
                if len([vid for vid in self.vehs_in_hub if self.vehs_in_hub[vid] != -1]) == 0:
                    if rl_action != 0:
                        self.reward -= self.wrong_action_penalty
                elif rl_action > 0:
                    vid_to_activate = min((vid for vid in self.vehs_in_hub if self.vehs_in_hub[vid] != -1),
                                          key=lambda vid: self.vehs_in_hub[vid])
                    self._activate_and_route_vehicle(vid=vid_to_activate, simulation_time=simulation_time,
                                                     rl_action=rl_action)

    def setup_spaces(self):
        max_requests = 10
        grid_shape = (self.grid_side_length, self.grid_side_length)

        self.observation_space = gym.spaces.Box(low=0, high=max_requests, shape=(4, self.grid_side_length, self.grid_side_length))
        #self.observation_space = gym.spaces.Dict({
        #    "position_grid": gym.spaces.Box(low=0, high=len(self.sim_vehicles), shape=grid_shape),
        #    "inbound_grid": gym.spaces.Box(low=0, high=max_requests, shape=grid_shape),
        #    "outbound_grid": gym.spaces.Box(low=0, high=max_requests, shape=grid_shape),
        #    "route_grid": gym.spaces.Box(low=0, high=len(self.sim_vehicles), shape=grid_shape),
        #})
        self.action_space = gym.spaces.Discrete(len(self.target_directions_dict.keys()) + 1)

    def get_current_state(self):
        '''
        Returns current state
        Four grids:
        1) Vehicle positions
        2) Requested pickups
        3) Requested drop-offs
        4) Vehicle trajectories
        '''

        veh_pos_grid = np.zeros((self.n_nodes, ))
        veh_traj_grid = np.zeros((self.n_nodes, ))
        for vid, veh_obj in enumerate(self.sim_vehicles):
            c_pos = veh_obj.pos
            for i, assigned_VRL in enumerate(veh_obj.assigned_route):
                route = self.routing_engine.return_best_route_1to1(c_pos, assigned_VRL.destination_pos)
                for node in route:
                    veh_traj_grid[node] += 1
                    c_pos = (node, None, None)

        req_inb_grid = np.zeros((self.n_nodes, ))
        req_outb_grid = np.zeros((self.n_nodes, ))

        for rid in self.rq_dict:
            req = self.rq_dict[rid]
            if req.d_pos == self.hub_pos:
                req_inb_grid[req.o_pos[0]] += req.nr_pax
            else:
                req_outb_grid[req.d_pos[0]] += req.nr_pax

        #with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
        #    print(f'---------------------------------------')
        #    print(req_inb_grid.reshape((25, 25)).astype(int))

        state = np.stack((
            veh_pos_grid,
            req_inb_grid,
            req_outb_grid,
            veh_traj_grid
        ), axis=-1).reshape((4, self.grid_side_length, self.grid_side_length)).astype(np.float32)
        state = np.log(state + 1)
        #print(state.max())
        return state

    def _create_rejection(self, prq: PlanRequest, simulation_time: int):
        super()._create_rejection(prq, simulation_time)
        self.rejection_counter += 1

    def acknowledge_boarding(self, rid: Any, vid: int, simulation_time: int):
        super().acknowledge_boarding(rid, vid, simulation_time)
        self.waiting_times.append(simulation_time - self.rq_dict[rid].get_rq_time())

    def get_reward(self):
        '''
        𝑈𝑡 is the number of requests rejected in the last time step,
        𝐷𝑡/𝑘𝑣 is the average distance travelled over the last time step and
        𝑊𝑎𝑣𝑔 is the average time requests have had to wait before pickup
        '''
        reward = self.reward
        reward -= self.rejection_counter * self.rw_rejection_penalty

        total_driven_delta = 0
        for vid, veh_obj in enumerate(self.sim_vehicles):
            total_driven_delta += veh_obj.cumulative_distance - self.driven_distance_dict[vid]
            self.driven_distance_dict[vid] = veh_obj.cumulative_distance
        reward -= total_driven_delta * self.rw_drive_distance_penalty

        avg_wait = sum(self.waiting_times)/max(len(self.waiting_times), 1)
        reward -= avg_wait * self.rw_waiting_time_penalty

        #print(f"Reward is {reward}")
        #print(f"Total driven delta is {total_driven_delta}")
        #print(f"There were {self.rejection_counter} requests rejected in the last time step")
        #print(f"Average waiting time is {avg_wait}s")
        #print("------------")

        return reward


