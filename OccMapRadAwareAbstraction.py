import numpy as np
import pandas as pd
from skimage.morphology import medial_axis
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import time
from scipy.ndimage import gaussian_gradient_magnitude


# Function to load and crop the occupancy grid and radiation map consistently
def load_and_crop_maps(occupancy_file, radiation_file):
    occupancy_grid = pd.read_csv(occupancy_file, header=None).values
    radiation_map = pd.read_csv(radiation_file, header=None).values

    non_empty_rows = np.any(occupancy_grid == 0, axis=1)
    non_empty_cols = np.any(occupancy_grid == 0, axis=0)

    cropped_occupancy_grid = occupancy_grid[np.ix_(non_empty_rows, non_empty_cols)]
    cropped_radiation_map = radiation_map[np.ix_(non_empty_rows, non_empty_cols)]

    return cropped_occupancy_grid, cropped_radiation_map


# Function to extract skeleton points using medial axis
def extract_skeleton_with_distance(grid):
    free_space = (grid == 1).astype(int)
    skeleton, _ = medial_axis(free_space, return_distance=True)
    skeleton_points = np.column_stack(np.nonzero(skeleton))
    return skeleton_points


# Function to create a radiation gradient map to skew nodes away from high radiation areas
def compute_radiation_gradient(radiation_map, sigma=2):
    gradient_map = gaussian_gradient_magnitude(radiation_map, sigma=sigma)
    return gradient_map


# Function to ensure that nodes are placed within free space and not on walls
def validate_node_position(x, y, occupancy_grid, buffer_size=5):
    if (0 <= x < occupancy_grid.shape[0]) and (0 <= y < occupancy_grid.shape[1]):
        if occupancy_grid[x, y] == 1:
            # Check surrounding area for wall proximity
            if np.all(occupancy_grid[max(0, x - buffer_size):min(x + buffer_size + 1, occupancy_grid.shape[0]),
                                     max(0, y - buffer_size):min(y + buffer_size + 1, occupancy_grid.shape[1])] == 1):
                return True
    return False


# Function to skew nodes based on radiation and free space, avoiding high radiation areas
def skew_nodes_away_from_radiation(nodes, gradient_map, radiation_map, occupancy_grid, buffer_size=5, max_skew_radius=15, skew_factor=6, radiation_threshold=0.3):
    skewed_nodes = []

    for node in nodes:
        x, y = int(node[0]), int(node[1])
        best_x, best_y = x, y
        min_gradient = gradient_map[x, y]
        radiation_value = radiation_map[x, y]

        # Avoid high-radiation areas completely
        if radiation_value < radiation_threshold * np.max(radiation_map):
            # Check a local area around the node for potential skewing
            for dx in range(-max_skew_radius, max_skew_radius + 1):
                for dy in range(-max_skew_radius, max_skew_radius + 1):
                    new_x, new_y = x + dx, y + dy
                    if validate_node_position(new_x, new_y, occupancy_grid, buffer_size):
                        new_gradient = gradient_map[new_x, new_y]
                        if new_gradient < min_gradient:
                            best_x, best_y = new_x, new_y
                            min_gradient = new_gradient

            # Skew nodes more aggressively by multiplying the skewing distance, with dynamic skew_factor adjustment
            skew_factor_dynamic = min(1 + (min_gradient / np.max(gradient_map)), skew_factor)
            skewed_x = x + skew_factor_dynamic * (best_x - x)
            skewed_y = y + skew_factor_dynamic * (best_y - y)

            # Ensure that the skewed node is placed within the free space and respects the buffer
            if validate_node_position(int(skewed_x), int(skewed_y), occupancy_grid, buffer_size):
                skewed_nodes.append([int(skewed_x), int(skewed_y)])
            else:
                skewed_nodes.append([best_x, best_y])
        else:
            continue

    return np.array(skewed_nodes)


# Function to reduce skeleton points using KMeans clustering
def reduce_skeleton_points(skeleton_points, n_clusters=100, occupancy_grid=None, buffer_size=20):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(skeleton_points)
    centroids = kmeans.cluster_centers_

    valid_centroids = []

    for centroid in centroids:
        x, y = int(centroid[0]), int(centroid[1])
        if occupancy_grid is not None and (0 <= x < occupancy_grid.shape[0]) and (0 <= y < occupancy_grid.shape[1]):
            if occupancy_grid[x, y] == 1 and np.all(
                    occupancy_grid[max(0, x - buffer_size):min(x + buffer_size + 1, occupancy_grid.shape[0]),
                    max(0, y - buffer_size):min(y + buffer_size + 1, occupancy_grid.shape[1])] == 1):
                valid_centroids.append([x, y])

    return np.array(valid_centroids)


# Function to apply repulsive smoothing using KD-tree for faster neighborhood search
def apply_repulsive_smoothing_with_kdtree(nodes, occupancy_grid, radiation_map, repulsion_radius=10, buffer_size=20, iterations=3):
    from scipy.spatial import KDTree
    smoothed_nodes = nodes.copy()

    for _ in range(iterations):
        repulsion_forces = np.zeros_like(smoothed_nodes)

        tree = KDTree(smoothed_nodes)  # Build KD-tree for fast neighborhood lookup

        for i, node in enumerate(smoothed_nodes):
            forces = np.zeros(2)
            node_radiation = radiation_map[int(node[0]), int(node[1])]

            # Scale repulsion based on radiation intensity
            scaled_repulsion_strength = 1 + (node_radiation / np.max(radiation_map)) * 2

            # Query nearby nodes using KD-tree
            neighbors = tree.query_ball_point(node, r=repulsion_radius)
            for j in neighbors:
                if i != j:
                    other_node = smoothed_nodes[j]
                    dx = node[0] - other_node[0]
                    dy = node[1] - other_node[1]
                    distance = np.sqrt(dx ** 2 + dy ** 2)
                    if distance > 0:  # Avoid division by zero
                        repulsion = (repulsion_radius - distance) / distance * scaled_repulsion_strength
                        forces += np.array([dx, dy]) * repulsion

            # Apply the repulsion forces and update the node position
            new_position = smoothed_nodes[i] + forces * 0.01

            # Ensure the new position is within free space and respects the buffer
            new_x, new_y = int(new_position[0]), int(new_position[1])
            if validate_node_position(new_x, new_y, occupancy_grid, buffer_size):
                smoothed_nodes[i] = [new_x, new_y]

    return np.array(smoothed_nodes)


# Function to check if two nodes have line of sight (no walls between them) using ray-casting
def has_line_of_sight(node1, node2, grid):
    x1, y1 = map(int, node1)
    x2, y2 = map(int, node2)
    num_samples = 500

    x_vals = np.linspace(x1, x2, num=num_samples)
    y_vals = np.linspace(y1, y2, num=num_samples)

    for x, y in zip(x_vals, y_vals):
        if grid[int(x), int(y)] == 0:
            return False

    return True


# Function to connect nodes based on line of sight using ray-casting
def connect_all_nodes_with_los_raycasting(nodes, grid, max_distance=250):
    connections = []
    nn_model = NearestNeighbors(n_neighbors=len(nodes))
    nn_model.fit(nodes)
    distances, indices = nn_model.kneighbors(nodes)

    for i, neighbors in enumerate(indices):
        for j in range(1, len(neighbors)):
            neighbor_idx = neighbors[j]
            node1 = nodes[i]
            node2 = nodes[neighbor_idx]
            dist = distances[i][j]

            if dist > max_distance:
                continue

            if has_line_of_sight(node1, node2, grid):
                connections.append((i, neighbor_idx))

    return connections


# Function to plot only the final "After Smoothing" result
def plot_after_smoothing(after_nodes, connections_after, grid, radiation_map):
    plt.figure(figsize=(10, 10))
    cmap = 'hot_r'
    plt.imshow(grid, cmap='gray', interpolation='nearest', alpha=0.5)
    plt.imshow(radiation_map, cmap=cmap, interpolation='nearest', alpha=0.5)
    plt.scatter(after_nodes[:, 1], after_nodes[:, 0], c='red', s=20, label="Nodes")
    for connection in connections_after:
        node1, node2 = after_nodes[connection[0]], after_nodes[connection[1]]
        plt.plot([node1[1], node2[1]], [node1[0], node2[0]], 'k-', lw=0.3, alpha=0.3)

    plt.title("Occupancy Map Abstraction for ML Exploration")
    plt.show()


# Main execution
start_time = time.time()

# Load and crop maps
occupancy_grid, radiation_map = load_and_crop_maps('occupancy_map_output/occupancy_grid.csv',
                                                   'occupancy_map_output/radiation_map.csv')

# Extract skeleton points
skeleton_points = extract_skeleton_with_distance(occupancy_grid)

# Reduce skeleton points using KMeans with an increased number of clusters
n_clusters = 500  # Adjust to focus on important areas
initial_nodes = reduce_skeleton_points(skeleton_points, n_clusters=n_clusters, occupancy_grid=occupancy_grid,
                                       buffer_size=18)

# Compute the gradient of the radiation map for skewing
radiation_gradient = compute_radiation_gradient(radiation_map, sigma=2)

# Skew nodes away from high radiation areas
skewed_nodes = skew_nodes_away_from_radiation(initial_nodes, radiation_gradient, radiation_map, occupancy_grid, skew_factor=10, max_skew_radius=20, radiation_threshold=0.6)


# print(f"Computation completed in {end_time - start_time:.2f} seconds.")faster neighborhood search
smoothed_nodes = apply_repulsive_smoothing_with_kdtree(skewed_nodes, occupancy_grid, radiation_map, repulsion_radius=25,
                                                       iterations=5)

# Connect nodes based on line of sight
connections_after = connect_all_nodes_with_los_raycasting(smoothed_nodes, occupancy_grid)

# Plot only the "After Smoothing" result
plot_after_smoothing(smoothed_nodes, connections_after, occupancy_grid, radiation_map)

end_time = time.time()
print(f"Computation completed in {end_time - start_time:.2f} seconds.")