import os
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

# Define map dimensions and resolution
map_size = 50  # Map size in meters
resolution = 0.05  # Resolution in meters
grid_size = int(map_size / resolution)  # Size of the grid (1000 x 1000 for 50m x 50m with 0.05m resolution)

# Initialize an empty occupancy map (0 = free space, 1 = occupied, 2 = obstacles)
occupancy_map = np.zeros((grid_size, grid_size))

# Helper function to add walls or obstacles (draw rectangles on the grid)
def add_wall(x_min, y_min, x_max, y_max):
    # Correct calculation of grid indices
    x_min = int(x_min / resolution)
    y_min = int(y_min / resolution)
    x_max = int(x_max / resolution)
    y_max = int(y_max / resolution)
    occupancy_map[y_min:y_max, x_min:x_max] = 1  # Set the area to occupied (walls or obstacles)

# Generate a room randomly placed within a section, ensuring it doesn't overlap section boundaries
def add_random_room_in_section(x_min, y_min, section_width, section_height):
    # Random room size between 10m and 14m for width and height
    room_width = random.uniform(10, min(14, section_width - 2))  # Keep some margin for corridors
    room_height = random.uniform(10, min(14, section_height - 2))

    # Random position within the section, ensuring room fits within section bounds
    room_x_min = random.uniform(x_min + 0.5, x_min + section_width - room_width - 0.5)  # Avoid overlap with walls
    room_y_min = random.uniform(y_min + 0.5, y_min + section_height - room_height - 0.5)
    room_x_max = room_x_min + room_width
    room_y_max = room_y_min + room_height

    add_wall(room_x_min, room_y_min, room_x_max, room_y_max)

    # Return the center of the room for corridor placement and its boundaries
    return (room_x_min + room_x_max) / 2, (room_y_min + room_y_max) / 2, room_x_min, room_x_max, room_y_min, room_y_max

# Function to randomly place varying sized obstacles inside the room
def place_obstacles_in_room(room_x_min, room_x_max, room_y_min, room_y_max, entrance_x=None, entrance_y=None):
    num_obstacles = random.randint(1, 5)  # Random number of obstacles between 1 and 5
    safe_zone = 1.5  # Define a 1.5-meter radius safe zone around entrances

    for _ in range(num_obstacles):
        while True:
            # Random object position inside the room
            obj_x_min = random.uniform(room_x_min + 0.5, room_x_max - 2.0)  # Leave margin for object size
            obj_y_min = random.uniform(room_y_min + 0.5, room_y_max - 2.0)
            obj_size = random.uniform(0.5, 2.0)  # Object size between 0.5m and 2m
            obj_x_max = obj_x_min + obj_size
            obj_y_max = obj_y_min + obj_size

            # Ensure the object doesn't block the entrance (within a 1.5-meter safe zone)
            if entrance_x is None or entrance_y is None or (
                    abs(obj_x_min - entrance_x) > safe_zone and abs(obj_y_min - entrance_y) > safe_zone):
                add_wall(obj_x_min, obj_y_min, obj_x_max, obj_y_max)  # Treat objects as walls
                break

# Add outer walls of the building
add_wall(0, 0, 50, 0.5)  # Bottom wall
add_wall(0, 49.5, 50, 50)  # Top wall
add_wall(0, 0, 0.5, 50)  # Left wall
add_wall(49.5, 0, 50, 50)  # Right wall

# Parameters for dividing the map into sections for evenly spaced rooms
num_rows = 4  # 3 rows of rooms
num_cols = 4 # 3 columns of rooms
section_width = map_size / num_cols
section_height = map_size / num_rows

# Determine which 80% of cells will have rooms
total_cells = num_rows * num_cols
cells_to_fill = int(total_cells * 0.8)
filled_cells = random.sample(range(total_cells), cells_to_fill)  # Randomly pick cells to fill

# Store room centers and boundaries
room_centers = []

# Add rooms only to selected cells
for row in range(num_rows):
    for col in range(num_cols):
        cell_index = row * num_cols + col
        if cell_index in filled_cells:
            x_min = col * section_width
            y_min = row * section_height
            center_x, center_y, room_x_min, room_x_max, room_y_min, room_y_max = add_random_room_in_section(x_min,
                                                                                                            y_min,
                                                                                                            section_width,
                                                                                                            section_height)
            room_centers.append((center_x, center_y, room_x_min, room_x_max, room_y_min, room_y_max))

# Function to add wider corridors ensuring connectivity between rooms
def add_corridor(x1, y1, x2, y2):
    corridor_width = 2.0  # Make corridors wider (2 meters)

    # Convert coordinates to grid indices
    x1 = int(x1 / resolution)
    y1 = int(y1 / resolution)
    x2 = int(x2 / resolution)
    y2 = int(y2 / resolution)

    # Add horizontal or vertical corridor
    occupancy_map[min(y1, y2):max(y1, y2) + int(corridor_width / resolution),
    min(x1, x2):max(x1, x2) + int(corridor_width / resolution)] = 1  # Set the area to occupied

# Ensure each room has at least two corridors
def ensure_multiple_corridors_with_branching(room_centers):
    num_rooms = len(room_centers)

    # Connect each room to two other rooms
    for i in range(num_rooms):
        connections = set()
        while len(connections) < 2:
            target_room = random.randint(0, num_rooms - 1)
            if target_room != i:  # Avoid connecting to itself
                connections.add(target_room)

        # Add corridors to the chosen rooms and apply branching logic
        for target in connections:
            x1, y1, room_x_min, room_x_max, room_y_min, room_y_max = room_centers[i]
            x2, y2, _, _, _, _ = room_centers[target]

            # Add corridors with slight offset
            offset_x1 = random.uniform(-1, 1)
            offset_y1 = random.uniform(-1, 1)
            offset_x2 = random.uniform(-1, 1)
            offset_y2 = random.uniform(-1, 1)

            # Horizontal then vertical corridor or vice versa
            if random.random() < 0.5:
                add_corridor(x1 + offset_x1, y1 + offset_y1, x2 + offset_x2, y1 + offset_y1)  # Horizontal corridor
                add_corridor(x2 + offset_x2, y1 + offset_y1, x2 + offset_x2, y2 + offset_y2)  # Vertical corridor
            else:
                add_corridor(x1 + offset_x1, y1 + offset_y1, x1 + offset_x1, y2 + offset_y2)  # Vertical corridor
                add_corridor(x1 + offset_x1, y2 + offset_y2, x2 + offset_x2, y2 + offset_y2)  # Horizontal corridor

# Add multiple corridors to ensure each room is connected to at least two other rooms
ensure_multiple_corridors_with_branching(room_centers)

# Place random obstacles in each room, avoiding entrances
for room_center in room_centers:
    center_x, center_y, room_x_min, room_x_max, room_y_min, room_y_max = room_center
    place_obstacles_in_room(room_x_min, room_x_max, room_y_min, room_y_max, entrance_x=center_x, entrance_y=center_y)

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