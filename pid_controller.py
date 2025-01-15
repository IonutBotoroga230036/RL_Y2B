import numpy as np
import time

class PIDController:
    def __init__(self, kp, ki, kd, dt):
        """
        Initializes the PID controller.

        Parameters:
        kp (float): Proportional gain.
        ki (float): Integral gain.
        kd (float): Derivative gain.
        dt (float): Time step (in seconds) for the controller.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.prev_error = np.zeros(3)
        self.integral = np.zeros(3)

    def calculate(self, current_pos, target_pos):
        """
        Calculates the control output for the PID controller.

        Parameters:
        current_pos (np.array): Current 3D position of the pipette tip.
        target_pos (np.array): Target 3D position for the pipette tip.

        Returns:
        np.array: Control output (3D vector) representing the desired change in position.
        """
        error = np.array(target_pos) - np.array(current_pos)

        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.prev_error = error
        return output

    def reset(self):
      """Resets the integral and previous error to zero"""
      self.prev_error = np.zeros(3)
      self.integral = np.zeros(3)