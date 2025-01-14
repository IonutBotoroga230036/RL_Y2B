import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import random

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000, goal_range=0.3, threshold=0.001, reward_distance_scale=100, step_penalty=-1, bonus_reward=100):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps
        self.goal_range = goal_range # defines the boundaries in which the goal is set
        self.threshold = threshold # defines the threshold for when the task is considered done
        self.reward_distance_scale = reward_distance_scale # defines how much the distance rewards are scaled by.
        self.step_penalty = step_penalty # defines the penalty to be given for each step
        self.bonus_reward = bonus_reward # defines the reward for when the agent reaches the target

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=self.render)

        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)  # x, y, z control for pipette
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)  # pipette (x, y, z) + goal (x, y, z) + speed

        # Keep track of the number of steps
        self.steps = 0
        self.robotId = 0
        self.prev_dist = None
        self.set_step_2_stop = False
        self.close2goal = False
        self.steps_taken_2_stop = None
        self.stopped_at_goal = False
        self.reward_at_stop = 0

    def reset(self, seed=None):
        # Set the seed for reproducibility if provided
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # Set a random goal position within the specified range
        low_bound = -self.goal_range/2
        high_bound = self.goal_range/2
        self.goal_position = np.array([random.uniform(low_bound, high_bound), 
                                       random.uniform(low_bound, high_bound),
                                       random.uniform(low_bound, high_bound)])
        
        # Call the environment reset function to get initial observation
        observation = self.sim.reset(num_agents=1)  
        self.robotId = int(list(observation.keys())[-1][-1])

        # Extract pipette position and append with goal position and velocity
        pipette_position = np.array(self.sim.get_pipette_position(self.robotId))
        v_abs = 0
        observation = np.concatenate([pipette_position, self.goal_position, [v_abs]], axis=0).astype(np.float32)
        self.prev_dist = np.linalg.norm(observation[:3] - observation[3:6])

        # Reset the number of steps
        self.steps = 0
        self.set_step_2_stop = False
        self.close2goal = False
        self.steps_taken_2_stop = None
        self.stopped_at_goal = False
        self.reward_at_stop = 0
        info = {}  # No additional info required
        return observation, info

    def step(self, action):
        # Execute one time step within the environment
        # Append 0 for the drop action
        action = np.concatenate([action, [0]], axis=0)

        # Call the environment step function
        observation = self.sim.run([action])  # Pass the action as a list
        self.robotId = int(list(observation.keys())[-1][-1])


        # Extract pipette position, velocity and create full observation
        pipette_position = np.array(self.sim.get_pipette_position(self.robotId))
        v_joint_x = observation[f'robotId_{self.robotId}']['joint_states']['joint_0']['velocity']
        v_joint_y = observation[f'robotId_{self.robotId}']['joint_states']['joint_1']['velocity']
        v_joint_z = observation[f'robotId_{self.robotId}']['joint_states']['joint_2']['velocity']
        v_abs = np.abs(v_joint_x) + np.abs(v_joint_y) + np.abs(v_joint_z)

        observation = np.concatenate([pipette_position, self.goal_position, [v_abs]], axis=0).astype(np.float32)


        # Calculate distance to the goal
        distance = np.linalg.norm(observation[:3] - observation[3:6])
        base_reward = (self.prev_dist - distance)*self.reward_distance_scale

        # Calculate the reward (distance-based + step penalty + bonus for completion)
        reward = base_reward + self.step_penalty
        self.prev_dist = distance

        # Check if the task is complete (distance below threshold and staying still for some steps)
        if distance <= self.threshold:
            if self.set_step_2_stop == False:
                self.steps_taken_2_stop = self.steps
                self.set_step_2_stop = True

            if distance <= self.threshold and self.steps - self.steps_taken_2_stop >= 4:
                self.reward_at_stop =  self.bonus_reward
                reward += self.reward_at_stop
                self.stopped_at_goal = True
                terminated = True
            else:
                terminated = False
        else:
            terminated = False
           

        # Check if the episode should be truncated (max steps reached)
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        info = {'d-goal': distance,
                'speed':observation[6],
                 'checks':{f"base reward:": base_reward,
                           f"stopped at: {self.steps_taken_2_stop}": self.reward_at_stop}}

        # Increment the number of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass  # Rendering not required in this case

    def close(self):
        self.sim.close()  # Close the simulation

    