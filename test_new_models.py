import numpy as np
import time
import gymnasium as gym
import random
from stable_baselines3 import PPO
from sim_class import Simulation

class OT2EnvTest(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2EnvTest, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=self.render)

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = gym.spaces.Box(-1, 1, (3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32) # Removed speed

        # keep track of the number of steps
        self.steps = 0
        self.robotId = 0
        self.goal_position = None # Initialize goal position to None
    
    def set_goal_position(self, goal_position):
        """Sets the goal position for the environment."""
        self.goal_position = np.array(goal_position)

    def reset(self, seed=None):
        # being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # goal position is now being set by set_goal_position before calling reset

        # Call the environment reset function
        observation = self.sim.reset(num_agents=1)
        self.robotId = int(list(observation.keys())[-1][-1])

        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
        
        pipette_coords = np.array(self.sim.get_pipette_position(self.robotId))
        goal_coords = np.array(self.goal_position)
        observation = np.concatenate([pipette_coords, goal_coords]).astype(np.float32) # Removed speed from observation

        d_goal = np.linalg.norm(observation[:3] - observation[3:6])

        info = {}

        # Reset the number of steps
        self.steps = 0
        self.set_step_2_stop = False
        self.close2dish = False
        self.close2goal = False
        self.steps_taken_2_stop = None
        self.stopped_at_goal = False
        self.reward_at_dish = 0
        self.reward_at_target = 0
        self.reward_at_stop = 0
        self.prev_d_goal = d_goal

        return (observation, info)

    def step(self, action):
        terminated = False
        truncated = False
        d_goal_max = 0.60

        # Execute one time step within the environment
        # since we are only controlling the pipette position, we accept 3 values for the action and need to append 0 for the drop action
        action = np.append(action, 0)

        # Call the environment step function
        observation = self.sim.run([action]) # Why do we need to pass the action as a list? Think about the simulation class.
        self.robotId = int(list(observation.keys())[-1][-1])

        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32

        pipette_coords = np.array(self.sim.get_pipette_position(self.robotId))
        goal_coords = np.array(self.goal_position)
        observation = np.concatenate([pipette_coords, goal_coords]).astype(np.float32) # Removed speed from observation

        d_goal = np.linalg.norm(observation[:3] - observation[3:6])

        # Calculate the reward, this is something that you will need to experiment with to get the best results
        reward = ((self.prev_d_goal - d_goal) / d_goal_max) * 100
        
        base_reward = reward
        self.prev_d_goal = d_goal
        # next we need to check if the if the task has been completed and if the episode should be terminated
        # To do this we need to calculate the distance between the pipette position and the goal position and if it is below a certain threshold, we will consider the task complete. 
        # What is a reasonable threshold? Think about the size of the pipette tip and the size of the plants.
        if d_goal < 0.001:
            if self.set_step_2_stop == False:
                self.step_2_stop = self.steps
                self.set_step_2_stop = True

            if d_goal < 0.001 and self.steps - self.step_2_stop >= 2:
                self.reward_at_stop =  50
                reward += self.reward_at_stop
                self.stopped_at_goal = True
                terminated = True
            else:
                terminated = False
    
        
        # next we need to check if the episode should be truncated, we can check if the current number of steps is greater than the maximum number of steps
        if self.steps + 1 == self.max_steps:
            truncated = True
        else:
            truncated = False

        info = {'d-goal': d_goal,
                 'checks':{f"base reward:": base_reward,
                           f"stopped at: {self.steps_taken_2_stop}": self.reward_at_stop}}

        # increment the number of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()
    
def test_rl_model(model_path, max_steps=1000, threshold=0.001, stay_steps=50, render=False, num_tests = 10):
    """
    Tests a trained RL model by moving the pipette tip to multiple target positions and staying there for some time.

    Parameters:
        model_path (str): Path to the trained RL model (.zip file).
        max_steps (int): Maximum number of steps to run the simulation before terminating.
        threshold (float): Threshold value for the task to be considered complete.
        stay_steps (int): The number of steps the robot should stay inside threshold before considering the target reached.
        render (bool): If true the rendering is enabled for the simulation.
        num_tests (int): The number of random positions to be generated

    Returns:
        bool: Returns true if the position was achieved within the specified tolerance, false otherwise
    """
    env = OT2EnvTest(render=render)
    model = PPO.load(model_path)
    
    # Define the goal position limits
    goal_x_min = -0.1870
    goal_x_max = 0.2531
    goal_y_min = -0.1705
    goal_y_max = 0.2209
    goal_z_min = 0.1695
    goal_z_max = 0.2896
    
    cnt = 0
    for i in range(num_tests):
         # Generate a random target position within valid range
        target_pos = [
            random.uniform(goal_x_min, goal_x_max),
            random.uniform(goal_y_min, goal_y_max),
            random.uniform(goal_z_min, goal_z_max),
        ]
        print(f"\n--- Test {i + 1} ---")
        print(f"Target position: {target_pos}")

        env.set_goal_position(target_pos) # Set new target position for the environment
        observation, _ = env.reset()
        
        steps_in_threshold = 0
        first_target_achieved_step = None
        target_achieved = False
        total_reward = 0
        for step in range(max_steps):
            action, _ = model.predict(observation, deterministic=True)

            observation, reward, terminated, truncated, info = env.step(action)
            distance_to_goal = info['d-goal'] # Changed to info['distance']
             # Print the current pipette position and the distance to the goal
            total_reward+= reward
            print(f"Step: {step}, Pipette Position: {observation[:3]}, Distance to Goal: {distance_to_goal}, Reward: {total_reward}")


            if distance_to_goal <= threshold:
                steps_in_threshold += 1
                if first_target_achieved_step is None: # set this variable to the first time the target was achieved
                   first_target_achieved_step = step
                if steps_in_threshold >= stay_steps:
                    print(f"Target achieved at step: {first_target_achieved_step}")
                    target_achieved = True
                    break
            else:
                 steps_in_threshold = 0
                 first_target_achieved_step = None

            if terminated or truncated:
               break # if episode ends break the loop
               
        if target_achieved:
            print("The target was successfully achieved")
            cnt += 1
        else:
             print("Maximum steps reached, target not achieved")
        
    env.close()
    print(f"{cnt} successful")
    return cnt == num_tests


if __name__ == "__main__":
    model_path = "model(4).zip"  # Replace with your model path
    stay_steps = 10
    num_tests = 10
    achieved_all_targets = test_rl_model(model_path, stay_steps=stay_steps, render=True, num_tests=num_tests)

    if achieved_all_targets:
        print("The model successfully achieved all targets.")
    else:
        print("The model did NOT achieve all targets.")