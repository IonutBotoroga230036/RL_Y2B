      
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import random

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000, threshold=0.0001, bonus_reward=100):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps  # The maximum number of steps an episode can last
        self.threshold = threshold  # The distance threshold for considering the task complete
        self.bonus_reward = bonus_reward  # Reward for successfully reaching the goal

        # Define the goal position limits
        self.goal_x_min = -0.1870
        self.goal_x_max = 0.2531
        self.goal_y_min = -0.1705
        self.goal_y_max = 0.2209
        self.goal_z_min = 0.1197
        self.goal_z_max = 0.2209

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=self.render)

        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)  # x, y, z control for pipette
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)  # pipette (x, y, z) + goal (x, y, z) + speed

        # Keep track of the number of steps and other environment variables
        self.steps = 0  # current step
        self.robotId = 0  # id of the robot to be controlled
        self.set_step_2_stop = False  # variable to mark the start of staying on target
        self.close2goal = False  # variable to mark that the agent is close to goal
        self.steps_taken_2_stop = None  # step count since the agent got close to goal
        self.stopped_at_goal = False  # boolean to check if the agent stopped at goal
        self.reward_at_stop = 0  # final reward given at termination when stopped at goal
        self.prev_pos = None # previous position of the pipette
        self.stagnation_step = 0

    def reset(self, seed=None):
        # Set the seed for reproducibility if provided
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # Set a random goal position within the specified range
        self.goal_position = np.array([random.uniform(self.goal_x_min, self.goal_x_max),
                                       random.uniform(self.goal_y_min, self.goal_y_max),
                                       random.uniform(self.goal_z_min, self.goal_z_max)])

        # Call the environment reset function to get initial observation
        observation = self.sim.reset(num_agents=1)
        self.robotId = int(list(observation.keys())[-1][-1])

        # Extract pipette position and append with goal position and velocity
        pipette_position = np.array(self.sim.get_pipette_position(self.robotId))
        v_abs = 0
        observation = np.concatenate([pipette_position, self.goal_position, [v_abs]], axis=0).astype(np.float32)

        # Reset the number of steps
        self.steps = 0
        self.set_step_2_stop = False
        self.close2goal = False
        self.steps_taken_2_stop = None
        self.stopped_at_goal = False
        self.reward_at_stop = 100
        self.prev_pos = None
        info = {}  # No additional info required
        self.last_step = np.linalg.norm(observation[:3] - observation[3:6])
        return observation, info

    def step(self, action):
        # Execute one time step within the environment
        # Append 0 for the drop action
        action = np.concatenate([action, [0]], axis=0)

        # Call the environment step function
        observation = self.sim.run([action])  # Pass the action as a list
        self.robotId = int(list(observation.keys())[-1][-1])

        # get current pipete position
        pipette_position = np.array(self.sim.get_pipette_position(self.robotId))

        #Calculate speed by getting the derivate from previous position
        if self.prev_pos is not None:
            v_abs = np.linalg.norm(pipette_position - self.prev_pos)
        else:
            v_abs = 0

        #create full observation
        observation = np.concatenate([pipette_position, self.goal_position, [v_abs]], axis=0).astype(np.float32)
        self.prev_pos = pipette_position

        # Calculate distance to the goal
        distance = np.linalg.norm(observation[:3] - observation[3:6])
        
        # Calculate the reward based on the distance to the goal
        reward = -distance

        # Check if the task is complete (distance below threshold and staying still for some steps)
        if distance <= self.threshold:
            self.close2goal = True
            reward += 30
            if self.set_step_2_stop == False:
                self.steps_taken_2_stop = self.steps
                self.set_step_2_stop = True
            if self.steps - self.steps_taken_2_stop >= 4:
                self.reward_at_stop =  self.bonus_reward
                reward += self.reward_at_stop
                self.stopped_at_goal = True
                terminated = True
            else:
                terminated = False
        else:
            self.close2goal = False  # if it goes outside threshold, reset steps for stopping logic
            self.set_step_2_stop = False
            terminated = False

        # Check if the episode should be truncated (max steps reached)
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        if distance < self.last_step:
            self.stagnation_step = 0
        else:
            self.stagnation_step += 1

        if self.stagnation_step > 50:
            truncated = True

        # # Check if the task is complete (distance below threshold and staying still for some steps)
        # if distance <= self.threshold:
        #     self.close2goal = True
        #     if self.set_step_2_stop == False:
        #         self.steps_taken_2_stop = self.steps
        #         self.set_step_2_stop = True
        #     if self.steps - self.steps_taken_2_stop >= 4:
        #         self.reward_at_stop =  self.bonus_reward
        #         reward += self.reward_at_stop
        #         self.stopped_at_goal = True
        #         terminated = True
        #     else:
        #         terminated = False
        # else:
        #     self.close2goal = False  # if it goes outside threshold, reset steps for stopping logic
        #     self.set_step_2_stop = False
        #     terminated = False

        # # Check if the episode should be truncated (max steps reached)
        # if self.steps >= self.max_steps:
        #     truncated = True
        # else:
        #     truncated = False

        info = {'distance': distance,
                'speed':observation[6],
                "stopped at reward": self.reward_at_stop}

        # Increment the number of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass  # Rendering not required in this case

    def close(self):
        self.sim.close()  # Close the simulation

    