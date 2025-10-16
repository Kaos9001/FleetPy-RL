import random
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from src.misc.globals import *

def make_basic_demand_generator(n_trips):
    def generate_temp_demand(simulation_params):
        nw_path = Path("data") / Path("networks") / Path(simulation_params[G_NETWORK_NAME])
        base_path = nw_path / Path("base")
        nodes = gpd.read_file(base_path / Path('nodes_all_infos.geojson'))

        infra_path = Path("data") / Path('infra') / Path(simulation_params[G_INFRA_NAME]) / Path(
            simulation_params[G_NETWORK_NAME])
        hubs = pd.read_csv(infra_path / Path('hub_nodes.csv'))

        # Single hub version
        hub = hubs.iloc[0]["node_index"].item()

        np.random.seed(simulation_params[G_RANDOM_SEED])

        generated_trips = []
        for i in range(n_trips):
            if i < n_trips / 2:
                start_node = hub
                end_node = np.random.randint(0, len(nodes) - 1)
            else:
                start_node = np.random.randint(0, len(nodes) - 1)
                end_node = hub
            generated_trips.append({
                "start": start_node,
                "end": end_node,
                "rq_time": random.randint(simulation_params[G_SIM_START_TIME], simulation_params[G_SIM_END_TIME]),
            })

        trips = pd.DataFrame(generated_trips).sort_values(by="rq_time").reset_index(drop=True)

        # print("Creating demand directory in FleetPy data folder...")
        demand_path = Path("data") / Path("demand") / Path(simulation_params[G_DEMAND_NAME]) / Path("matched") / Path(
            simulation_params[G_NETWORK_NAME])
        demand_path.mkdir(exist_ok=True, parents=True)

        trips["request_id"] = trips.index
        out_file_name = f"temp-{n_trips}-{simulation_params[G_SIM_START_TIME]}-to-{simulation_params[G_SIM_END_TIME]}-{simulation_params[G_RANDOM_SEED]}.csv"
        out_file_path = demand_path / Path(out_file_name)
        trips.to_csv(out_file_path, columns=["rq_time", "start", "end", "request_id"], index=False)

        # print(f"Demand file {out_file_name} saved to {demand_path}")
        # print(f"Done in {time.time() - start}s.")
        return out_file_name, out_file_path
    return generate_temp_demand

def make_gaussian_demand_generator(n_hotspots=5,
                                   baseline_strength=0.05,
                                   peak_fraction_range=(0.2, 0.8),
                                   strength_range=(0.05, 0.15),
                                   temporal_spread_range=(1200, 3600),
                                   spatial_spread_range=(100, 400), 
                                   balance_range=(0.0, 1.0),
                                   candidate_nodes=None):
    def generate_hotspot_poisson_demand(simulation_params):
        nw_path = Path("data") / "networks" / simulation_params[G_NETWORK_NAME]
        base_path = nw_path / "base"

        start_time = simulation_params[G_SIM_START_TIME] + 150
        end_time = simulation_params[G_SIM_END_TIME]

        out_file_name = f"hotspot-separate-{start_time}-to-{end_time}-{simulation_params[G_RANDOM_SEED]}.csv"
        out_file_path = Path("data") / "demand" / simulation_params[G_DEMAND_NAME] / "matched" / simulation_params[G_NETWORK_NAME] / out_file_name
        
        if out_file_path.exists():
            return out_file_name, out_file_path
        
        nodes = gpd.read_file(base_path / "nodes_all_infos.geojson")

        infra_path = Path("data") / "infra" / simulation_params[G_INFRA_NAME] / simulation_params[G_NETWORK_NAME]
        hubs = pd.read_csv(infra_path / "hub_nodes.csv")

        hub = hubs.iloc[0]["node_index"].item()

        np.random.seed(simulation_params[G_RANDOM_SEED])

        delta = end_time - start_time
        nonlocal candidate_nodes
        if candidate_nodes is None:
            candidate_nodes = [idx for idx in nodes.index if idx != hub]

        # --- Node coordinates for distance calculation ---
        coords = np.vstack([nodes.geometry.x, nodes.geometry.y]).T

        # --- Create N random hotspots with individual spreads ---
        hotspots = []
        for _ in range(n_hotspots):
            center_node = np.random.choice(candidate_nodes)
            peak_time = np.random.randint(
                start_time + delta * peak_fraction_range[0],
                start_time + delta * peak_fraction_range[1]
            )
            strength = np.random.uniform(*strength_range)
            temporal_spread = np.random.randint(*temporal_spread_range)
            spatial_spread = np.random.randint(*spatial_spread_range)
            outbound_inbound_balance = np.random.uniform(*balance_range)

            # Precompute spatial weights
            center_xy = coords[center_node]
            dists = np.linalg.norm(coords - center_xy, axis=1)
            weights = np.exp(-0.5 * (dists/spatial_spread)**2)
            weights[hub] = 0
            weights /= weights.sum()

            hotspots.append({
                "node": center_node,
                "peak": peak_time,
                "strength": strength,
                "t_spread": temporal_spread,
                "s_spread": spatial_spread,
                "weights": weights,
                "outbound_inbound_balance": outbound_inbound_balance
            })

        # --- Generate Poisson events per hotspot ---
        def generate_hotspot_times(h):
            times = []
            t = start_time
            max_rate = h["strength"]
            while t < end_time:
                t += np.random.exponential(1 / max_rate)
                if t >= end_time:
                    break
                lambda_t = h["strength"] * np.exp(-0.5 * ((t - h["peak"]) / h["t_spread"]) ** 2)
                if np.random.rand() < lambda_t / max_rate:
                    times.append(int(t))
            return times

        generated_trips = []

        # --- Generate trips from each hotspot ---
        for h in hotspots:
            rq_times = generate_hotspot_times(h)
            for rq_time in rq_times:
                dest = np.random.choice(nodes.index, p=h["weights"])
                if np.random.rand() < h["outbound_inbound_balance"]:
                    start_node, end_node = hub, dest
                else:
                    start_node, end_node = dest, hub
                generated_trips.append({
                    "start": start_node,
                    "end": end_node,
                    "rq_time": rq_time,
                })

        # --- Background trips (baseline) ---
        if baseline_strength > 0:
            t = start_time
            while t < end_time:
                t += np.random.exponential(1 / baseline_strength)
                if t >= end_time:
                    break
                if np.random.rand() < 0.5:
                    start_node, end_node = hub, np.random.choice(candidate_nodes)
                else:
                    start_node, end_node = np.random.choice(candidate_nodes), hub
                generated_trips.append({
                    "start": start_node,
                    "end": end_node,
                    "rq_time": int(t),
                })

        # --- Save trips ---
        trips = pd.DataFrame(generated_trips).sort_values(by="rq_time").reset_index(drop=True)
        demand_path = Path("data") / "demand" / simulation_params[G_DEMAND_NAME] / "matched" / simulation_params[
            G_NETWORK_NAME]
        demand_path.mkdir(exist_ok=True, parents=True)

        trips["request_id"] = trips.index
        trips.to_csv(out_file_path, columns=["rq_time", "start", "end", "request_id"], index=False)

        return out_file_name, out_file_path
    return generate_hotspot_poisson_demand