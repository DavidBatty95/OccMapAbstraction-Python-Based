import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import csv
from math import sqrt
from scipy.ndimage import distance_transform_edt
from skimage.measure import block_reduce
from sklearn.neighbors import KDTree
import pandas as pd


# ------------------------------------------------------------------------
#                           PART 1: PATH GENERATION
# ------------------------------------------------------------------------

# Output directory
output_dir = "occupancy_map_output"
os.makedirs(output_dir, exist_ok=True)

# Map configuration
map_size = 100         # metres
resolution = 0.05      # metres
grid_size = int(map_size / resolution)

# Create a blank occupancy map (0 = free, 1 = occupied)
occupancy_map = np.zeros((grid_size, grid_size))

# Parameters for room layout
num_rows = 5
num_cols = 5
section_width = map_size / num_cols
section_height = map_size / num_rows

# Corridor and wall settings
corridor_width = 4.0
wall_width = 0.5

# Track room centres and their adjacencies
room_centres = []
connections = {}

def add_wall(x_min, y_min, x_max, y_max):
    """
    Paint a rectangular occupied area on the global occupancy_map.
    """
    x_min_cell = int(x_min / resolution)
    y_min_cell = int(y_min / resolution)
    x_max_cell = int(x_max / resolution)
    y_max_cell = int(y_max / resolution)
    occupancy_map[y_min_cell:y_max_cell, x_min_cell:x_max_cell] = 1

# Build external walls
add_wall(0, 0, map_size, wall_width)                       # Bottom
add_wall(0, map_size - wall_width, map_size, map_size)     # Top
add_wall(0, 0, wall_width, map_size)                       # Left
add_wall(map_size - wall_width, 0, map_size, map_size)     # Right

# Randomly add rooms (with varying size)
for row in range(num_rows):
    for col in range(num_cols):
        room_index = row * num_cols + col
        room_w = random.uniform(8, min(section_width - 1, section_width - 0.1))
        room_h = random.uniform(8, min(section_height - 1, section_height - 0.1))
        cx = col * section_width + section_width / 2
        cy = row * section_height + section_height / 2

        x_min = cx - room_w / 2
        x_max = cx + room_w / 2
        y_min = cy - room_h / 2
        y_max = cy + room_h / 2

        add_wall(x_min, y_min, x_max, y_max)
        room_centres.append((cx, cy))
        connections[room_index] = set()

# Connect rooms in a grid
for row in range(num_rows):
    for col in range(num_cols):
        idx_here = row * num_cols + col
        # Downward connection
        if row < num_rows - 1:
            idx_down = (row + 1) * num_cols + col
            connections[idx_here].add(idx_down)
            connections[idx_down].add(idx_here)
        # Right connection
        if col < num_cols - 1:
            idx_right = row * num_cols + (col + 1)
            connections[idx_here].add(idx_right)
            connections[idx_right].add(idx_here)

# Randomly remove 20% of connections
total_conn = sum(len(nbs) for nbs in connections.values()) // 2
to_remove = int(total_conn * 0.2)
removed = 0
while removed < to_remove:
    r1 = random.choice(list(connections.keys()))
    if len(connections[r1]) > 1:
        r2 = random.choice(list(connections[r1]))
        if len(connections[r2]) > 1:
            connections[r1].remove(r2)
            connections[r2].remove(r1)
            removed += 1

def add_corridor(room1, room2):
    """
    Marks the corridor region as occupied (value=1) between two room centres.
    """
    x1, y1 = room_centres[room1]
    x2, y2 = room_centres[room2]
    x1c = int(x1 / resolution)
    y1c = int(y1 / resolution)
    x2c = int(x2 / resolution)
    y2c = int(y2 / resolution)
    cw = int(corridor_width / resolution)

    if x1c == x2c:
        # Vertical corridor
        ymin = min(y1c, y2c)
        ymax = max(y1c, y2c)
        occupancy_map[ymin:ymax + cw, x1c - cw//2:x1c + cw//2] = 1
    elif y1c == y2c:
        # Horizontal corridor
        xmin = min(x1c, x2c)
        xmax = max(x1c, x2c)
        occupancy_map[y1c - cw//2:y1c + cw//2, xmin:xmax + cw] = 1

# Paint corridors
for r1, nbset in connections.items():
    for r2 in nbset:
        if r1 < r2:
            add_corridor(r1, r2)

# Generate 11 paths ensuring coverage
max_path_length = 50
paths = []
all_visited_rooms = set()

def generate_path(start_room):
    """
    Builds a path from 'start_room', prioritising unvisited rooms.
    """
    path = []
    current = start_room
    local_visited = {current}
    all_visited_rooms.add(current)
    path.append(room_centres[current])

    while len(path) < max_path_length:
        unvisited_nbs = [r for r in connections[current]
                         if r not in local_visited and r not in all_visited_rooms]
        visited_nbs = [r for r in connections[current] if r not in local_visited]

        if unvisited_nbs:
            nxt = random.choice(unvisited_nbs)
        elif visited_nbs:
            nxt = random.choice(visited_nbs)
        else:
            break

        path.append(room_centres[nxt])
        local_visited.add(nxt)
        all_visited_rooms.add(nxt)
        current = nxt

    return path

# Create 11 main paths
for _ in range(11):
    p = generate_path(0)
    paths.append(p)

# Ensure all rooms are visited
remaining_rooms = set(range(num_rows * num_cols)) - all_visited_rooms
while remaining_rooms:
    starter = random.choice(list(remaining_rooms))
    extra_path = generate_path(starter)
    paths.append(extra_path)
    for cxy in extra_path:
        idx_r = room_centres.index(cxy)
        all_visited_rooms.add(idx_r)
    remaining_rooms = set(range(num_rows * num_cols)) - all_visited_rooms

# Create submaps for each path
simulated_maps = []
for i, path in enumerate(paths):
    sim_map = occupancy_map.copy()
    # Rooms not in this path → set them to 0.5
    for (cx, cy) in room_centres:
        if (cx, cy) not in path:
            xm1 = int(cx / resolution - section_width / (2*resolution))
            ym1 = int(cy / resolution - section_height / (2*resolution))
            xm2 = int(cx / resolution + section_width / (2*resolution))
            ym2 = int(cy / resolution + section_height / (2*resolution))
            sim_map[ym1:ym2, xm1:xm2] = 0.5

    # Save CSV
    filename = os.path.join(output_dir, f"path_{i+1}_occupancy_map.csv")
    np.savetxt(filename, sim_map, delimiter=",")
    simulated_maps.append(filename)

# Also save full occupancy map
full_map_path = os.path.join(output_dir, "full_occupancy_map.csv")
np.savetxt(full_map_path, occupancy_map, delimiter=",")

print("\nCreated 11 submaps (and extras if needed) plus the full map in 'occupancy_map_output' folder.")


# ------------------------------------------------------------------------
#                   PART 2: NODE NETWORK ABSTRACTION
# ------------------------------------------------------------------------

# We focus on counting free-space cells, then building a sparse node network.

min_radius_threshold = 1.5  # metres
downsample_factor = 4
filter_distance = 2.0       # metres
connection_radius = 5.0     # metres
min_node_distance = 1.5
initial_node_position = (5, 5)  # anchor node for merged network

def count_freespace_cells(occ_map):
    """
    Counts how many cells are free (value < 1).
    Includes cells that are 0 (fully free) or 0.5 (unvisited rooms).
    """
    return np.count_nonzero(occ_map < 1.0)

def generate_nodes_connections(file_path):
    """
    Loads a submap, counts free-space cells, builds a sparse node network.
    Returns (selected_nodes, connections_dict, num_freespace_cells, time_taken).
    """
    start_t = time.time()
    occ_map = np.loadtxt(file_path, delimiter=",")

    # Count free-space cells
    num_freecells = count_freespace_cells(occ_map)

    # Convert to binary occupancy for distance transform
    occ_bin = np.where(occ_map == 1.0, 1, 0)
    ds_map = block_reduce(occ_bin, block_size=(downsample_factor, downsample_factor), func=np.min)
    dist_map = distance_transform_edt(ds_map) * (resolution * downsample_factor)

    # Identify points above clearance threshold
    eligible_coords = np.argwhere(dist_map >= min_radius_threshold)
    distances = dist_map[eligible_coords[:, 0], eligible_coords[:, 1]]
    sorted_idx = np.argsort(-distances)  # descending
    sorted_points = eligible_coords[sorted_idx] * downsample_factor

    selected_nodes = []
    for pt in sorted_points:
        rr, cc = pt
        x, y = cc * resolution, rr * resolution
        # Ensure spacing from existing nodes
        if all(sqrt((x - sx)**2 + (y - sy)**2) >= filter_distance
               for sx, sy, _ in selected_nodes):
            rad = dist_map[rr // downsample_factor, cc // downsample_factor]
            selected_nodes.append((x, y, rad))

    # Build connections
    node_conns = {}
    if selected_nodes:
        coords_only = [(n[0], n[1]) for n in selected_nodes]
        tree = KDTree(np.array(coords_only))
        for i, (nx, ny, _) in enumerate(selected_nodes):
            idxs = tree.query_radius([[nx, ny]], r=connection_radius)[0]
            node_conns[(nx, ny)] = []
            for j in idxs:
                if j != i:
                    node_conns[(nx, ny)].append(coords_only[j])

    elapsed = time.time() - start_t
    return selected_nodes, node_conns, num_freecells, elapsed

# We process exactly the first 11 submaps
to_abstract = simulated_maps[:11]

submap_results = []  # Will store [ID, freespace_cells, number_of_nodes, time_taken]
all_nodes_list = []
all_conn_dicts = []

# Create a figure to plot the 3×4 grid (11 submaps + 1 combined)
fig, axes = plt.subplots(3, 4, figsize=(32, 24))
axes = axes.flatten()

for idx, submap_file in enumerate(to_abstract):
    try:
        nodes, connections_dict, free_cells, t_elapsed = generate_nodes_connections(submap_file)
    except Exception as e:
        print(f"Error processing {submap_file}: {e}")
        continue

    # Submap ID (e.g. "Submap 1")
    submap_id = f"Submap {idx + 1}"

    # Count nodes
    n_nodes = len(nodes)

    # Store results for eventual CSV
    submap_results.append([submap_id, free_cells, n_nodes, t_elapsed])

    all_nodes_list.extend([(x, y) for (x, y, _) in nodes])
    all_conn_dicts.append(connections_dict)

    # Plot submap + node network
    ax = axes[idx]
    try:
        occ = np.loadtxt(submap_file, delimiter=",")
    except Exception as e:
        print(f"Error loading {submap_file} for plotting: {e}")
        continue

    ax.imshow(occ, cmap="gray", origin="lower",
              extent=[0, map_size, 0, map_size], vmin=0, vmax=1)
    ax.set_title(submap_id, fontsize=14)
    ax.set_xlabel("X (metres)")
    ax.set_ylabel("Y (metres)")

    # Plot nodes
    for (xn, yn, _) in nodes:
        ax.plot(xn, yn, 'ro', markersize=3)

    # Plot connections
    for (x0, y0), neighs in connections_dict.items():
        for (x1, y1) in neighs:
            ax.plot([x0, x1], [y0, y1], 'b-', linewidth=0.5)

# The final subplot for the combined network
ax_combined = axes[-1]
ax_combined.set_title("Combined Node Network", fontsize=14)
ax_combined.set_xlabel("X (metres)")
ax_combined.set_ylabel("Y (metres)")

# Load the full map
try:
    full_occ = np.loadtxt(full_map_path, delimiter=",")
except Exception as e:
    print(f"Error loading full map: {e}")
    full_occ = None

if full_occ is not None:
    ax_combined.imshow(full_occ, cmap="gray", origin="lower",
                       extent=[0, map_size, 0, map_size], vmin=0, vmax=1)

# Merge connections from all submaps
merged_connections = {}
if all_nodes_list:
    # Anchor near 'initial_node_position'
    anchor = min(all_nodes_list,
                 key=lambda nd: sqrt((nd[0] - initial_node_position[0])**2 +
                                     (nd[1] - initial_node_position[1])**2))
    merged_connections[anchor] = []

    for conn_dict in all_conn_dicts:
        for node_xy, nbs_xy in conn_dict.items():
            if node_xy not in merged_connections:
                merged_connections[node_xy] = []
            for nb_xy in nbs_xy:
                if nb_xy not in merged_connections[node_xy]:
                    merged_connections[node_xy].append(nb_xy)

# Plot combined adjacency
for node_xy, nbs in merged_connections.items():
    x0, y0 = node_xy
    colr = 'green' if node_xy == anchor else 'red'
    ax_combined.plot(x0, y0, 'o', color=colr, markersize=3)
    for (x1, y1) in nbs:
        if sqrt((x0 - x1)**2 + (y0 - y1)**2) >= min_node_distance:
            ax_combined.plot([x0, x1], [y0, y1], 'b-', linewidth=0.5)

plt.tight_layout()
plt.show()

# Print results to terminal
df = pd.DataFrame(submap_results,
                  columns=["ID", "Free-Space Cells", "Number of Nodes", "Time Taken (s)"])
print("\nSubmap Processing Results (for the 11 submaps):")
print(df.to_string(index=False))

# ------------------------------------------------------------------------
#                 PART 3: APPEND RESULTS TO CSV
# ------------------------------------------------------------------------

results_file = os.path.join(output_dir, "submap_results.csv")

# Check if the file already exists
file_exists = os.path.exists(results_file)

# Append rows to the CSV file
with open(results_file, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # If it's a new file, write the header
    if not file_exists:
        writer.writerow(["ID", "Free-Space Cells", "Number of Nodes", "Time Taken (s)"])
    # Write each submap's data as a new row
    for row in submap_results:
        writer.writerow(row)

print(f"\nAppended submap results to '{results_file}'.")