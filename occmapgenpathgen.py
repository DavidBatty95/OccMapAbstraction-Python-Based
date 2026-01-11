import os
import time
import random
import numpy as np
import pandas as pd

from math import sqrt
from scipy.ndimage import distance_transform_edt
from skimage.measure import block_reduce
from sklearn.neighbors import KDTree

# ------------------------------------------------------------------------
#                           PART 1: MAP & PATHS
# ------------------------------------------------------------------------

# Map parameters
map_size = 50         # (metres)
resolution = 0.05      # (metres per cell)
grid_size = int(map_size / resolution)

# Corridor/room parameters
num_rows = 3
num_cols = 3
section_width = map_size / num_cols
section_height = map_size / num_rows
corridor_width = 4.0   # (metres)
wall_width = 0.5       # (metres)

# Create blank occupancy map (0=free, 1=occupied)
occupancy_map = np.zeros((grid_size, grid_size))

def add_wall(x_min, y_min, x_max, y_max):
    """
    Paints a rectangular occupied area (wall) on 'occupancy_map'.
    """
    x_min_cell = int(x_min / resolution)
    y_min_cell = int(y_min / resolution)
    x_max_cell = int(x_max / resolution)
    y_max_cell = int(y_max / resolution)
    occupancy_map[y_min_cell:y_max_cell, x_min_cell:x_max_cell] = 1

# Build outer walls
add_wall(0, 0, map_size, wall_width)                       # bottom
add_wall(0, map_size - wall_width, map_size, map_size)     # top
add_wall(0, 0, wall_width, map_size)                       # left
add_wall(map_size - wall_width, 0, map_size, map_size)     # right

# Prepare room centres and adjacency
room_centres = []
connections = {}

# Randomly add rooms in each cell of a 10x10 grid
for row in range(num_rows):
    for col in range(num_cols):
        room_index = row * num_cols + col
        # Random dimensions of the room
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

# Connect rooms in a grid-like fashion
for row in range(num_rows):
    for col in range(num_cols):
        idx_here = row * num_cols + col
        # Connect downward
        if row < num_rows - 1:
            idx_down = (row + 1) * num_cols + col
            connections[idx_here].add(idx_down)
            connections[idx_down].add(idx_here)
        # Connect rightward
        if col < num_cols - 1:
            idx_right = row * num_cols + (col + 1)
            connections[idx_here].add(idx_right)
            connections[idx_right].add(idx_here)

# Randomly remove 20% of these connections
total_conn = sum(len(nbs) for nbs in connections.values()) // 2
to_remove = int(total_conn * 0.2)
removed = 0
while removed < to_remove:
    r1 = random.choice(list(connections.keys()))
    # Only remove if that room has multiple connections
    if len(connections[r1]) > 1:
        r2 = random.choice(list(connections[r1]))
        if len(connections[r2]) > 1:
            connections[r1].remove(r2)
            connections[r2].remove(r1)
            removed += 1

def add_corridor(room1, room2):
    """
    Fills the corridor region with 1s (occupied) between two room centres.
    """
    x1, y1 = room_centres[room1]
    x2, y2 = room_centres[room2]
    x1c = int(x1 / resolution)
    y1c = int(y1 / resolution)
    x2c = int(x2 / resolution)
    y2c = int(y2 / resolution)
    cw = int(corridor_width / resolution)

    if x1c == x2c:
        # vertical corridor
        ymin = min(y1c, y2c)
        ymax = max(y1c, y2c)
        occupancy_map[ymin:ymax + cw, x1c - cw//2:x1c + cw//2] = 1
    elif y1c == y2c:
        # horizontal corridor
        xmin = min(x1c, x2c)
        xmax = max(x1c, x2c)
        occupancy_map[y1c - cw//2:y1c + cw//2, xmin:xmax + cw] = 1

# Paint corridors
for r1, nbset in connections.items():
    for r2 in nbset:
        if r1 < r2:
            add_corridor(r1, r2)

# Function to generate a path from a given start room
max_path_length = 50
def generate_path(start_room):
    """
    Builds a path from 'start_room' by stepping through connected rooms
    until reaching 'max_path_length' or no unvisited neighbours are found.
    """
    path = []
    current = start_room
    visited_here = {current}
    path.append(room_centres[current])

    while len(path) < max_path_length:
        # Potential next steps
        neighbours = list(connections[current])
        if not neighbours:
            break
        # Choose among neighbours that haven't been visited in this path
        unvisited_nbs = [r for r in neighbours if r not in visited_here]
        if unvisited_nbs:
            nxt = random.choice(unvisited_nbs)
        else:
            nxt = random.choice(neighbours)

        path.append(room_centres[nxt])
        visited_here.add(nxt)
        current = nxt

    return path

# Generate 50 random paths (submaps) in memory
paths = []
num_rooms_total = num_rows * num_cols
for _ in range(50):
    # Pick a random room as start
    start_idx = random.randint(0, num_rooms_total - 1)
    p = generate_path(start_idx)
    paths.append(p)

# ------------------------------------------------------------------------
#          PART 2: NODE NETWORK FOR EACH SUBMAP & CSV RESULTS
# ------------------------------------------------------------------------

# Node abstraction parameters
min_radius_threshold = 1.5  # metres
downsample_factor = 4
filter_distance = 2.0       # metres
connection_radius = 5.0     # metres

def count_free_cells(occ_map):
    """
    Counts how many cells are free (i.e. exactly 0.0).
    """
    return np.sum(occ_map == 0.0)

def generate_nodes_connections(submap_occ):
    """
    Takes a submap occupancy array, counts free cells,
    and returns (list_of_nodes, dict_of_connections, free_cell_count, time_taken).
    """
    start_t = time.time()

    # Count how many cells are 0.0 (fully free)
    free_cell_count = count_free_cells(submap_occ)

    # Convert to binary for distance transform (1=occupied, else 0)
    occ_bin = np.where(submap_occ == 1.0, 1, 0)

    # Downsample the binary map
    ds_map = block_reduce(occ_bin, block_size=(downsample_factor, downsample_factor), func=np.min)

    # Distance transform on the downsampled map
    dist_map = distance_transform_edt(ds_map) * (resolution * downsample_factor)

    # Identify points that have clearance >= min_radius_threshold
    eligible_coords = np.argwhere(dist_map >= min_radius_threshold)
    radii = dist_map[eligible_coords[:, 0], eligible_coords[:, 1]]

    # Sort by descending radius (choose largest‐clearance cells first)
    sorted_idx = np.argsort(-radii)
    sorted_points = eligible_coords[sorted_idx] * downsample_factor

    # Select nodes by ensuring they are at least 'filter_distance' away from previously chosen nodes
    chosen_nodes = []
    for rr, cc in sorted_points:
        x, y = cc * resolution, rr * resolution
        if all(sqrt((x - sx)**2 + (y - sy)**2) >= filter_distance
               for sx, sy, _ in chosen_nodes):
            rad_val = dist_map[rr // downsample_factor, cc // downsample_factor]
            chosen_nodes.append((x, y, rad_val))

    # Build node connectivity with KDTree
    node_conn = {}
    if chosen_nodes:
        coords_only = [(xx, yy) for xx, yy, _ in chosen_nodes]
        tree = KDTree(np.array(coords_only))
        for i, (nx, ny, _) in enumerate(chosen_nodes):
            idxs = tree.query_radius([[nx, ny]], r=connection_radius)[0]
            node_conn[(nx, ny)] = []
            for j in idxs:
                if j != i:
                    node_conn[(nx, ny)].append(coords_only[j])

    elapsed = time.time() - start_t
    return chosen_nodes, node_conn, free_cell_count, elapsed

def create_submap(path_coords):
    """
    Returns a copy of the global occupancy_map where
    rooms *not* in 'path_coords' are set to 0.5.
    """
    sm = occupancy_map.copy()
    # For each room centre *not* in this path, set that cell region to 0.5
    for r_idx, (cx, cy) in enumerate(room_centres):
        if (cx, cy) not in path_coords:
            xm1 = int(cx / resolution - section_width / (2*resolution))
            ym1 = int(cy / resolution - section_height / (2*resolution))
            xm2 = int(cx / resolution + section_width / (2*resolution))
            ym2 = int(cy / resolution + section_height / (2*resolution))
            sm[ym1:ym2, xm1:xm2] = 0.5
    return sm

submap_results = []

# Process each of the 50 submaps
for i, path_coords in enumerate(paths, start=1):
    # Create the submap in memory
    submap_occ = create_submap(path_coords)

    # Abstract node network & measure free cells
    nodes, conns, free_cells, t_elapsed = generate_nodes_connections(submap_occ)

    # Collect stats for CSV
    submap_results.append([
        f"Submap {i}",
        free_cells,
        len(nodes),
        t_elapsed
    ])

# Write final results to a single CSV (no other CSVs created)
df = pd.DataFrame(submap_results,
                  columns=["Submap", "Number of Free Cells", "Number of Nodes", "Time Taken (s)"])
df.to_csv("submap_processing_results.csv", index=False)

print("\nFinished processing 50 submaps. Results written to 'submap_processing_results.csv'.")