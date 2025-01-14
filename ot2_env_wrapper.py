import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1)

        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)  # x, y, z control for pipette
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)  # pipette (x, y, z) + goal (x, y, z)

        # Keep track of the number of steps
        self.steps = 0

    def reset(self, seed=None):
        # Set the seed for reproducibility if provided
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # Set a random goal position within the working area
        self.goal_position = np.random.uniform(low=-1.0, high=1.0, size=(3,))  # Example: goal within [-1, 1] for each axis
        
        # Call the environment reset function to get initial observation
        observation = self.sim.reset(num_agents=1)  
        # Find the first robot ID key dynamically
        robot_id = next(iter(observation.keys()), None)
        if robot_id is None or 'pipette_position' not in observation[robot_id]:
            raise KeyError("'pipette_position' key not found in the observation structure")

        # Extract pipette_position
        pipette_position = observation[robot_id]['pipette_position']
        observation = np.concatenate((pipette_position, self.goal_position), axis=0).astype(np.float32)

        # Reset the number of steps
        self.steps = 0
        info = {} # we don't need to return any additional information
        return observation, info

    def step(self, action):
        # Execute one time step within the environment
        # Append 0 for the drop action (assuming drop action is not part of the pipette control)
        action = np.concatenate([action, [0]], axis=0)

        # Call the environment step function
        observation = self.sim.run([action])  # Pass the action as a list

        # Extract pipette position
        robot_id = next(iter(observation.keys()), None)
        if robot_id is None or 'pipette_position' not in observation[robot_id]:
            raise KeyError("'pipette_position' key not found in the observation structure")

        pipette_position = observation[robot_id]['pipette_position']
        observation = np.concatenate((pipette_position, self.goal_position), axis=0).astype(np.float32)

        # Calculate the reward (negative distance to goal)
        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        reward = -distance_to_goal  # Negative distance as reward (closer = better)

        # Check if the task is complete (distance below threshold)
        threshold = 0.05  # Example threshold, adjust based on the task's scale
        if distance_to_goal < threshold:
            terminated = True
            reward = 10.0  # Reward for completing the task
        else:
            terminated = False

        # Check if the episode should be truncated (max steps reached)
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        info = {}  # No additional info required

        # Increment the number of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass  # Rendering not required in this case

    def close(self):
        self.sim.close()  # Close the simulation
