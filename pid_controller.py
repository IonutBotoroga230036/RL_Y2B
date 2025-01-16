import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, dt):
        """
        Initializes the PID controller with gains and time step.

        Parameters:
            kp (list): Proportional gain [x,y,z].
            ki (list): Integral gain [x,y,z].
            kd (list): Derivative gain [x,y,z].
            dt (float): Time step for the controller.
        """
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.dt = dt
        self.prev_error = np.zeros(3)
        self.integral = np.zeros(3)

    def calculate(self, current_pos, target_pos):
        """
        Calculates the PID control output.

        Parameters:
            current_pos (np.array): Current position [x, y, z].
            target_pos (np.array): Target position [x, y, z].

        Returns:
            np.array: The control output [x, y, z].
        """
        error = np.array(target_pos) - np.array(current_pos)

        # Accumulate integral term
        self.integral += error * self.dt

        # Calculate derivative term
        derivative = (error - self.prev_error) / self.dt

        # Calculate the control output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Store the error for the next iteration
        self.prev_error = error
        return output

    def reset(self):
        """Resets the controller's internal state."""
        self.prev_error = np.zeros(3)
        self.integral = np.zeros(3)