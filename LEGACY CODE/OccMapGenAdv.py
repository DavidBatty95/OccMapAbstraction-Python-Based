import os
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

# Define map dimensions and resolution
map_size = 50  # Map size in meters
resolution = 0.1  # Resolution in meters
grid_size = int(map_size / resolution)  # Size of the grid (500 x 500 for 50m x 50m with 0.1m resolution)
corridor_width = 1.0  # Corridor width in meters
edge_buffer = 2  # Buffer to prevent placing rooms too close to edges (in meters)

# Initialize an empty occupancy map (1 = free space, 0 = occupied space)
occupancy_map = np.ones((grid_size, grid_size))

# Helper function to add walls (occupy space)
def add_wall(x_min, y_min, x_max, y_max):
    # Convert to grid indices
    x_min = int(x_min / resolution)
    y_min = int(y_min / resolution)
    x_max = int(x_max / resolution)
    y_max = int(y_max / resolution)
    occupancy_map[y_min:y_max, x_min:x_max] = 0  # Mark space as occupied (wall)

# Function to add rooms properly
def add_room(x_min, y_min, room_width, room_height):
    x_max = x_min + room_width
    y_max = y_min + room_height
    add_wall(x_min, y_min, x_max, y_min)  # Top wall
    add_wall(x_max, y_min, x_max, y_max)  # Right wall
    add_wall(x_min, y_max, x_max, y_max)  # Bottom wall
    add_wall(x_min, y_min, x_min, y_max)  # Left wall
    return (x_min + x_max) / 2, (y_min + y_max) / 2  # Return room center

# Function to add doorways (remove part of a wall)
def add_doorway(x_min, y_min, x_max, y_max, orientation="horizontal"):
    x_min = int(x_min / resolution)
    y_min = int(y_min / resolution)
    x_max = int(x_max / resolution)
    y_max = int(y_max / resolution)

    if orientation == "horizontal":
        door_start = random.randint(x_min + 2, x_max - 2)  # Door is placed randomly, but not at edges
        occupancy_map[y_min:y_max, door_start:door_start + int(corridor_width / resolution)] = 1
    else:  # vertical doorway
        door_start = random.randint(y_min + 2, y_max - 2)
        occupancy_map[door_start:door_start + int(corridor_width / resolution), x_min:x_max] = 1

# Function to connect rooms with a hallway and a doorway
def connect_rooms_with_corridor(room1_center, room2_center):
    x1, y1 = room1_center
    x2, y2 = room2_center

    if abs(x1 - x2) > abs(y1 - y2):  # Horizontal corridor
        corridor_y = (y1 + y2) / 2
        add_wall(min(x1, x2), corridor_y - corridor_width / 2, max(x1, x2), corridor_y + corridor_width / 2)
        add_doorway(min(x1, x2), corridor_y - corridor_width / 2, max(x1, x2), corridor_y + corridor_width / 2, orientation="horizontal")
    else:  # Vertical corridor
        corridor_x = (x1 + x2) / 2
        add_wall(corridor_x - corridor_width / 2, min(y1, y2), corridor_x + corridor_width / 2, max(y1, y2))
        add_doorway(corridor_x - corridor_width / 2, min(y1, y2), corridor_x + corridor_width / 2, max(y1, y2), orientation="vertical")

# Parameters for room generation
room_min_size = 8  # Minimum room size in meters
room_max_size = 15  # Maximum room size in meters
room_separation = 1  # Minimum separation between rooms in meters

# Function to generate distinct rooms with proper spacing and edge buffer
def generate_rooms(num_rooms):
    room_centers = []
    for _ in range(num_rooms):
        while True:
            room_width = random.uniform(room_min_size, room_max_size)
            room_height = random.uniform(room_min_size, room_max_size)
            x_min = random.uniform(edge_buffer, map_size - room_width - edge_buffer)
            y_min = random.uniform(edge_buffer, map_size - room_height - edge_buffer)

            # Ensure rooms don't overlap
            if all(np.linalg.norm(np.array([center_x, center_y]) - np.array([(x_min + room_width) / 2, (y_min + room_height) / 2])) > room_max_size + room_separation for center_x, center_y in room_centers):
                room_center = add_room(x_min, y_min, room_width, room_height)
                room_centers.append(room_center)
                break
    return room_centers

# Generate rooms and connect them with corridors
num_rooms = 6  # Number of rooms
room_centers = generate_rooms(num_rooms)

# Connect each room with the next room in the list
for i in range(len(room_centers) - 1):
    connect_rooms_with_corridor(room_centers[i], room_centers[i + 1])

# Save occupancy grid to file
output_folder = 'occupancy_map_output'

# Create folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save the occupancy map as a CSV file
output_file = os.path.join(output_folder, 'occupancy_grid.csv')
pd.DataFrame(occupancy_map).to_csv(output_file, index=False, header=False)

print(f"Occupancy grid saved to {output_file}")

# Display the occupancy map with grid overlay
plt.figure(figsize=(10, 10))
plt.imshow(occupancy_map, cmap="gray", origin="lower", extent=[0, map_size, 0, map_size])

# Overlay a 1m grid for better visualization of dimensions
plt.grid(True, which='both', color='r', linestyle='--', linewidth=0.5)
plt.xticks(np.arange(0, map_size + 1, 1))
plt.yticks(np.arange(0, map_size + 1, 1))

# Add title and labels
plt.title("Occupancy Map Abstraction for ML Exploration")
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")

plt.show()