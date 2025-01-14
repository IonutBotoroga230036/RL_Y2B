import pybullet as p
import pybullet_data
import numpy as np
import imageio
from sim_class import Simulation

# Initialize the simulation with a specified number of agents
sim = Simulation(num_agents=1)  # For one robot

# Create a list to store frames
frames = []

# Add a 'render' method to the Simulation instance dynamically
def render_override():
    # Get an image from the PyBullet simulation
    width, height, rgb_img, _, _ = p.getCameraImage(
        width=640, height=480, renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    # Normalize and convert to uint8
    rgb_img = np.reshape(rgb_img, (height, width, 4))  # Convert to NumPy array
    rgb_img = rgb_img[:, :, :3]  # Ignore the alpha channel (transparency)
    rgb_img = np.array(rgb_img, dtype=np.uint8)  # Convert to uint8 type
    return rgb_img

# Override the sim's render attribute
sim.render = render_override

# Define a helper function to capture a frame
def capture_frame():
    frame = sim.render()  # Call the overridden method
    frames.append(frame)

# Define actions and steps
actions_sequence = [
    [[1, 0.5, -0.5, 0], 100],
    [[0, -1, 0, 0], 120],
    [[-1, 0, 0, 0], 120],
    [[0, 1, 0, 0], 120],
    [[1, 0, 0, 0], 120],
]

# Simulate and capture frames
for actions, num_steps in actions_sequence:
    for step in range(num_steps):
        sim.run(actions=[actions], num_steps=1)  # Executes a single step
        capture_frame()

# Save the frames as a GIF
gif_filename = "simulation.gif"
imageio.mimsave(gif_filename, frames, fps=15)  # Adjust FPS as needed
print(f"GIF saved as {gif_filename}")