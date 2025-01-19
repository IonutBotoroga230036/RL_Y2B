import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

corner_one = [-0.1871, -0.1707, 0.1195]
corner_two = [0.253, 0.2195, 0.2895]

drop_command = 0


# Function to generate a random goal position within the defined bounds
def random_goal_position(corner1, corner2):
    x = random.uniform(min(corner1[0], corner2[0]), max(corner1[0], corner2[0]))
    y = random.uniform(min(corner1[1], corner2[1]), max(corner1[1], corner2[1]))
    z = random.uniform(min(corner1[2], corner2[2]), max(corner1[2], corner2[2]))
    return [x, y, z]


class OT2Env(gym.Env):
    """Custom OpenAI Gym environment for the OT-2 RL task."""

    def __init__(self, render=False, max_steps=1500):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps
        self.sim = Simulation(num_agents=1, render=self.render)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self.steps = 0
        self.prev_distance = None

        self.visitation_count = {}

    def reset(self, seed=None):
        """Resets the environment for a new episode."""
        if seed is not None:
            np.random.seed(seed)

        self.goal_position = random_goal_position(corner_one, corner_two)
        observation = self.sim.reset(num_agents=1)
        self.robot_id = next(
            (key for key in observation.keys() if key.startswith("robotId_")), None
        )

        pipette_position = np.array(
            observation[self.robot_id]["pipette_position"], dtype=np.float32
        )
        observation = np.append(pipette_position, self.goal_position).astype(np.float32)

        self.steps = 0
        self.prev_distance = np.linalg.norm(pipette_position - self.goal_position)
        return observation, {}

    def step(self, action):
        """Executes an action and returns the updated state and reward."""
        action = action + np.random.normal(
            scale=0.2, size=action.shape
        )  # Add Gaussian noise
        action = np.clip(action, -1, 1)  # Ensure actions stay within bounds
        action = np.append(action, drop_command)  # Add drop command for simulation
        observation = self.sim.run([action])

        pipette_position = np.array(
            observation[self.robot_id]["pipette_position"], dtype=np.float32
        )
        observation = np.append(pipette_position, self.goal_position).astype(np.float32)

        # Compute distance to goal
        distance = np.linalg.norm(pipette_position - self.goal_position)

        reward = -distance  # Reward function

        if distance < 0.001:  # Success threshold
            reward += 50.0  # Larger success reward
        reward += 0.1 * (self.prev_distance - distance)  # Encourage progress
        reward -= 0.005  # Step penalty

        self.prev_distance = distance

        # Termination conditions
        terminated = distance < 0.001
        truncated = self.steps >= self.max_steps

        self.steps += 1
        return observation, float(reward), terminated, truncated, {}

    def render(self, mode="human"):
        pass

    def close(self):
        self.sim.close()
