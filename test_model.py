import numpy as np
import time
import gymnasium as gym
from stable_baselines3 import PPO
from sim_class import Simulation

class OT2EnvTest(gym.Env):
    def __init__(self, render=False, max_steps=1000, threshold=0.0001, bonus_reward=100):
        super(OT2EnvTest, self).__init__()
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
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)  # x, y, z control for pipette
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)  # pipette (x, y, z) + goal (x, y, z) + speed

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
        # Set a fixed goal position for testing purposes
        self.goal_position = np.array([0.2, 0.2, 0.25])

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
    
def test_rl_model(model_path, max_steps=1000, threshold=0.001, stay_steps=50, render=False):
    """
    Tests a trained RL model by moving the pipette tip to a target position and staying there for some time.

    Parameters:
        model_path (str): Path to the trained RL model (.zip file).
        max_steps (int): Maximum number of steps to run the simulation before terminating.
        threshold (float): Threshold value for the task to be considered complete.
        stay_steps (int): The number of steps the robot should stay inside threshold before considering the target reached.
        render (bool): If true the rendering is enabled for the simulation.

    Returns:
        bool: Returns true if the position was achieved within the specified tolerance, false otherwise
    """
    env = OT2EnvTest(render=render, threshold = threshold)
    model = PPO.load(model_path)

    observation, _ = env.reset()
    steps_in_threshold = 0
    first_target_achieved_step = None

    for step in range(max_steps):
        action, _ = model.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = env.step(action)
        distance_to_goal = info['distance'] # Changed to info['distance']
         # Print the current pipette position and the distance to the goal
        print(f"Step: {step}, Pipette Position: {observation[:3]}, Distance to Goal: {distance_to_goal}")


        if distance_to_goal <= threshold:
            steps_in_threshold += 1
            if first_target_achieved_step is None: # set this variable to the first time the target was achieved
                first_target_achieved_step = step
            if steps_in_threshold >= stay_steps:
                print(f"Target achieved at step: {first_target_achieved_step}")
                env.close()
                return True
        else:
            steps_in_threshold = 0
            first_target_achieved_step = None

        if terminated or truncated:
            break # if episode ends break the loop

    print("Maximum steps reached, target not achieved")
    env.close()
    return False

if __name__ == "__main__":
    model_path = "model(1).zip"  # Replace with your model path
    stay_steps = 10
    achieved_target = test_rl_model(model_path, stay_steps=stay_steps, render=True)

    if achieved_target:
        print("The target was successfully achieved")
    else:
        print("The target was NOT achieved")