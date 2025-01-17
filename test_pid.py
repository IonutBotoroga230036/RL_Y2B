import numpy as np
import time
from pid_controller import PIDController
from sim_class import Simulation
import random


def test_pid_controller(kp, ki, kd, dt, target_pos, max_steps=1000, threshold=0.001, stay_steps=50, render=False):
    """
    Tests the PID controller by moving the pipette tip to a target position and staying for some steps.

    Parameters:
        kp (float): Proportional gain.
        ki (float): Integral gain.
        kd (float): Derivative gain.
        dt (float): Time step for the controller.
        target_pos (list): Target position [x, y, z] for the pipette tip.
        max_steps (int): Maximum number of steps to run the simulation before terminating.
        threshold (float): Threshold value for the task to be considered complete.
        stay_steps (int): The number of steps the robot should stay inside threshold before considering the target reached.
        render (bool): If true the rendering is enabled for the simulation.

    Returns:
        bool: Returns true if the position was achieved within the specified tolerance, false otherwise
    """
    sim = Simulation(num_agents=1, render=render)  # Create simulation object
    controller = PIDController(kp, ki, kd, dt)  # create controller object

    # Reset the simulation to the initial position, extract the pipette position
    observation = sim.reset(num_agents=1)
    robotId = int(list(observation.keys())[-1][-1])
    current_pos = np.array(sim.get_pipette_position(robotId))

    controller.reset()  # reset the controller variables
    steps_in_threshold = 0  # Initialize counter for steps within the threshold
    first_target_achieved_step = None # initialize the first time the target was achieved.
    for step in range(max_steps):
        # Calculate the control output with the controller
        control_output = controller.calculate(current_pos, target_pos)

        # Limit the action to be between -1 and 1
        control_output = np.clip(control_output, -1, 1)

        # Send action to environment (needs to be a list)
        action = np.concatenate([control_output, [0]])
        observation = sim.run([action])

        # get the new pipette position
        robotId = int(list(observation.keys())[-1][-1])
        current_pos = np.array(sim.get_pipette_position(robotId))

        # check if goal was achieved
        distance_to_goal = np.linalg.norm(current_pos - target_pos)

        # Print the current pipette position and the distance to the goal
        print(f"Step: {step}, Pipette Position: {current_pos}, Distance to Goal: {distance_to_goal}")

        if distance_to_goal <= threshold:
            steps_in_threshold += 1
            if first_target_achieved_step is None: # set this variable to the first time the target was achieved
                first_target_achieved_step = step
            if steps_in_threshold >= stay_steps:
                print(f"Target achieved at step: {first_target_achieved_step}")
                sim.close()
                return True
        else:
            steps_in_threshold = 0  # reset the counter if it is not in threshold
            first_target_achieved_step = None # reset the first step value as we are no longer in target
    print("Maximum steps reached, target not achieved")
    sim.close()
    return False

if __name__ == "__main__":
    # Example Usage:
    kp = 15
    ki = 0
    kd = 0
    dt = 1
    stay_steps = 100
    num_tests = 10

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

        # run the simulation
        achieved_target = test_pid_controller(kp, ki, kd, dt, target_pos, render=True, stay_steps=stay_steps)

        if achieved_target:
            print("The target was successfully achieved")
            cnt += 1
        else:
            print("The target was NOT achieved")

    print(f"{cnt} successful")