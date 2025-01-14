from sim_class import Simulation
import csv
# Initialize the simulation with a specified number of agents
sim = Simulation(num_agents=1)  # For one robot

v = [0,-1,0,1,0]

X = []
Y = []
Z = []

velocity_x = 1
velocity_y = 0.5   
velocity_z = -0.5
drop_command = 0
actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

# Run the simulation for a specified number of steps
sim.run(actions, num_steps=100)
positions = []
for i in range(4):
    print(sim.get_pipette_position(1))
    X.append(sim.get_pipette_position(1)[0])
    Y.append(sim.get_pipette_position(1)[1])
    Z.append(sim.get_pipette_position(1)[2])

    positions.append(sim.get_pipette_position(1))
    velocity_x = v[i]
    velocity_y = v[i+1]   
    velocity_z = -1
    drop_command = 0
    actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

    # Run the simulation for a specified number of steps
    sim.run(actions, num_steps=120)

actions = [[0, 0, 2, drop_command]]


# Run the simulation for a specified number of steps
sim.run(actions, num_steps=120)

for i in range(4):
    print(sim.get_pipette_position(1))
    X.append(sim.get_pipette_position(1)[0])
    Y.append(sim.get_pipette_position(1)[1])
    Z.append(sim.get_pipette_position(1)[2])
    positions.append(sim.get_pipette_position(1))    
    velocity_x = v[i]
    velocity_y = v[i+1]   
    velocity_z = 1
    drop_command = 0
    actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

    # Run the simulation for a specified number of steps
    sim.run(actions, num_steps=120)   

csv_file = 'positions.csv'

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the data
    writer.writerow(positions)

print(min(X),max(X))
print(min(Y),max(Y))
print(min(Z),max(Z))

csv_file = 'min_max_positions.csv'

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the data
    writer.writerow([min(X),min(Y),min(Z)])
    writer.writerow([max(X),max(Y),max(Z)])


# velocity_x = 0
# velocity_y = -1   
# velocity_z = 0
# drop_command = 0
# actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

# # Run the simulation for a specified number of steps
# sim.run(actions, num_steps=120)

# velocity_x = -1
# velocity_y = 0   
# velocity_z = 0
# drop_command = 0
# actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

# # Run the simulation for a specified number of steps
# sim.run(actions, num_steps=120)

# velocity_x = 0
# velocity_y = 1   
# velocity_z = 0
# drop_command = 0
# actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

# # Run the simulation for a specified number of steps
# sim.run(actions, num_steps=120)

# velocity_x = 1
# velocity_y = 0   
# velocity_z = 0
# drop_command = 0
# actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

# # Run the simulation for a specified number of steps
# sim.run(actions, num_steps=120)