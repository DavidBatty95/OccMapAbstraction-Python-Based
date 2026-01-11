import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.measure import block_reduce
from math import sqrt
from sklearn.neighbors import KDTree

# Configuration
paths = ['occupancy_map_output/path_1_occupancy_map.csv',
         'occupancy_map_output/path_2_occupancy_map.csv',
         'occupancy_map_output/path_3_occupancy_map.csv',
         'occupancy_map_output/path_4_occupancy_map.csv']
complete_map_path = 'occupancy_map_output/full_occupancy_map.csv'  # Full map path
map_size = 50  # Map size in meters
resolution = 0.05  # Resolution in meters
min_radius_threshold = 1.5  # Minimum distance to wall for node placement (in meters)
downsample_factor = 4  # Downsampling factor to speed up distance calculations
filter_distance = 2.0  # Minimum spacing between nodes (in meters)
connection_radius = 5.0  # Radius within which to connect nodes
min_node_distance = 1.5  # Minimum distance for the final network (less strict)

# Initial node coordinates (center of the bottom-left room)
initial_node_position = (10, 10)  # Approximate center of bottom-left room


def generate_nodes_connections(file_path):
    # Load and preprocess the occupancy map
    occupancy_map = np.loadtxt(file_path, delimiter=",")
    occupancy_map = np.where(occupancy_map == 1.0, 1, 0)
    downsampled_map = block_reduce(occupancy_map, block_size=(downsample_factor, downsample_factor), func=np.min)
    distance_map = distance_transform_edt(downsampled_map) * (resolution * downsample_factor)

    # Identify points with sufficient radius
    eligible_points = np.argwhere(distance_map >= min_radius_threshold)
    eligible_radii = distance_map[eligible_points[:, 0], eligible_points[:, 1]]
    sorted_indices = np.argsort(-eligible_radii)
    sorted_points = eligible_points[sorted_indices] * downsample_factor
    selected_nodes = []

    for point in sorted_points:
        x, y = point[1] * resolution, point[0] * resolution
        if all(sqrt((x - sx) ** 2 + (y - sy) ** 2) >= filter_distance for sx, sy, *_ in selected_nodes):
            selected_nodes.append((x, y, distance_map[point[0] // downsample_factor, point[1] // downsample_factor]))

    # Create dense connections within connection_radius
    connections = {}
    tree = KDTree(np.array([(x, y) for x, y, _ in selected_nodes]))  # KDTree for fast neighbor lookup
    for i, (x0, y0, _) in enumerate(selected_nodes):
        indices = tree.query_radius([[x0, y0]], r=connection_radius)[0]
        connections[(x0, y0)] = [(selected_nodes[j][0], selected_nodes[j][1]) for j in indices if j != i]

    return selected_nodes, connections


# Set up figure with 2x3 grid
fig = plt.figure(figsize=(20, 15), dpi=150)
fig.suptitle("Occupancy Map Abstracted Sparse Node Network Combination for Multipath Robotic Exploration", fontsize=16)

all_nodes = []
all_connections = []

# Create a grid layout
gs = fig.add_gridspec(2, 3)

# Plot each path in a 2x2 grid
for idx, file_path in enumerate(paths):
    nodes, connections = generate_nodes_connections(file_path)
    all_nodes.extend([(x, y) for x, y, _ in nodes])  # Store only (x, y) for merging
    all_connections.append(connections)

    # Define the position for the 2x2 layout within 2x3 grid
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    ax.imshow(np.loadtxt(file_path, delimiter=","), cmap="gray", origin="lower", extent=[0, map_size, 0, map_size],
              vmin=0, vmax=1)
    ax.set_title(f"Robot Path {idx + 1}", fontsize=14)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")

    # Plot nodes without filtering
    for x, y, _ in nodes:
        ax.plot(x, y, 'ro', markersize=5)

    # Plot connections
    for (x0, y0), neighbors in connections.items():
        for (x1, y1) in neighbors:
            ax.plot([x0, x1], [y0, y1], 'b-', linewidth=0.5)

# Merge nodes using the specified initial node as the anchor
combined_connections = {}
initial_node = min(all_nodes, key=lambda node: sqrt(
    (node[0] - initial_node_position[0]) ** 2 + (node[1] - initial_node_position[1]) ** 2))
combined_connections[initial_node] = []

for connections in all_connections:
    for node, neighbors in connections.items():
        if node not in combined_connections:
            combined_connections[node] = []
        for neighbor in neighbors:
            if neighbor not in combined_connections[node]:
                combined_connections[node].append(neighbor)

# Plot the combined map on the larger right side (2x3 grid's third column)
ax_combined = fig.add_subplot(gs[:, 2])  # Merge top and bottom right cells for larger combined map
ax_combined.set_title("Final Output Node Network", fontsize=14)
ax_combined.set_xlabel("X (meters)")
ax_combined.set_ylabel("Y (meters)")

# Rotate the complete map counterclockwise to align correctly and then clockwise
complete_map = np.loadtxt(complete_map_path, delimiter=",")
complete_map = np.rot90(complete_map, k=3)  # Rotating counterclockwise 90 degrees
complete_map = np.rot90(complete_map, k=1)  # Rotating clockwise 90 degrees
ax_combined.imshow(complete_map, cmap="gray", origin="lower", extent=[0, map_size, 0, map_size], vmin=0, vmax=1)

# Plot combined nodes and connections, highlighting the initial node
for node, neighbors in combined_connections.items():
    x, y = node
    color = 'green' if node == initial_node else 'red'
    ax_combined.plot(x, y, 'o', color=color, markersize=5)
    for neighbor in neighbors:
        x1, y1 = neighbor
        # Only check proximity for final output network, allowing for some overlap
        if sqrt((x - x1) ** 2 + (y - y1) ** 2) >= min_node_distance:
            ax_combined.plot([x, x1], [y, y1], 'b-', linewidth=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to minimize white space and center title
plt.show()