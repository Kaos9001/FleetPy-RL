import os.path
import sys

from FleetPy_gym_rework import FleetPyEnv


# Main execution
if __name__ == "__main__":
    RL_config = {
        "use_case": "devresult",
        "start_config_i": 0,
        "cc_file": "large_constant_config_pool.csv",
        "sc_file": "large_pool_test.csv",
    }

    env = FleetPyEnv(RL_config)

    observation, info = env.reset()

    episode_over = False
    ss = 0
    while not episode_over:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        ss += reward

        episode_over = terminated or truncated
    print(ss)
    env.SF.evaluate()

    env.close()
