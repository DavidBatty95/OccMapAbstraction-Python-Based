import numpy as np
import pandas as pd
from skimage.morphology import medial_axis
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import math
import time


# Function to load and crop the occupancy grid and radiation map consistently
def load_and_crop_maps(occupancy_file, radiation_file):
    occupancy_grid = pd.read_csv(occupancy_file, header=None).values
    radiation_map = pd.read_csv(radiation_file, header=None).values

    # Find rows and columns that contain any walls (0s), to avoid cropping too much free space
    non_empty_rows = np.any(occupancy_grid == 0, axis=1)
    non_empty_cols = np.any(occupancy_grid == 0, axis=0)

    # Apply the same cropping to both the occupancy grid and the radiation map
    cropped_occupancy_grid = occupancy_grid[np.ix_(non_empty_rows, non_empty_cols)]
    cropped_radiation_map = radiation_map[np.ix_(non_empty_rows, non_empty_cols)]

    return cropped_occupancy_grid, cropped_radiation_map


# Function to extract free space, apply Medial Axis Transformation (MAT)
def extract_skeleton_with_distance(grid):
    free_space = (grid == 1).astype(int)  # Only keep free space (1s) for skeletonization
    skeleton, distance = medial_axis(free_space, return_distance=True)
    skeleton_points = np.column_stack(np.nonzero(skeleton))  # Extract skeleton points
    return skeleton_points


# Function to skew nodes based on free space and lower radiation areas during initial placement
def skew_nodes_based_on_free_space(nodes, radiation_map, occupancy_grid, buffer_size=20, max_skew_radius=10,
                                   skewing_factor=1.0):
    skewed_nodes = []

    for node in nodes:
        x, y = int(node[0]), int(node[1])
        best_x, best_y = x, y

        # Measure the amount of free space around the node
        free_space_count = np.sum(
            occupancy_grid[max(0, x - max_skew_radius):min(x + max_skew_radius + 1, occupancy_grid.shape[0]),
            max(0, y - max_skew_radius):min(y + max_skew_radius + 1, occupancy_grid.shape[1])] == 1)
        total_area = (min(x + max_skew_radius + 1, occupancy_grid.shape[0]) - max(0, x - max_skew_radius)) * \
                     (min(y + max_skew_radius + 1, occupancy_grid.shape[1]) - max(0, y - max_skew_radius))

        free_space_ratio = free_space_count / total_area

        # Skewing radius is proportional to free space and controlled by skewing_factor
        skew_radius = int(max_skew_radius * free_space_ratio * skewing_factor)

        # Search for a better (lower radiation) position within the skew radius
        for dx in range(-skew_radius, skew_radius + 1):
            for dy in range(-skew_radius, skew_radius + 1):
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < radiation_map.shape[0]) and (0 <= new_y < radiation_map.shape[1]):
                    # Check if it's free space and not too close to a wall (within the buffer zone)
                    if (occupancy_grid[new_x, new_y] == 1 and
                            np.all(occupancy_grid[
                                   max(0, new_x - buffer_size):min(new_x + buffer_size + 1, occupancy_grid.shape[0]),
                                   max(0, new_y - buffer_size):min(new_y + buffer_size + 1,
                                                                   occupancy_grid.shape[1])] == 1)):
                        new_radiation = radiation_map[new_x, new_y]
                        # Stronger bias to prefer lower radiation
                        if new_radiation < radiation_map[best_x, best_y] and radiation_map[best_x, best_y] > 0:
                            best_x, best_y = new_x, new_y

        skewed_nodes.append([best_x, best_y])

    return np.array(skewed_nodes)


# Function to apply repulsive smoothing to make nodes evenly spaced
def apply_repulsive_smoothing(nodes, occupancy_grid, radiation_map, repulsion_radius=10, buffer_size=20, iterations=5):
    smoothed_nodes = nodes.copy()

    for _ in range(iterations):
        repulsion_forces = np.zeros_like(smoothed_nodes)

        # Calculate repulsion forces based on nearby nodes
        for i, node in enumerate(smoothed_nodes):
            forces = np.zeros(2)
            node_radiation = radiation_map[int(node[0]), int(node[1])]  # Get radiation for the current node

            # Scale repulsion based on radiation intensity
            scaled_repulsion_strength = 1 + (
                    node_radiation / np.max(radiation_map)) * 2  # Adjust scaling factor as needed

            for j, other_node in enumerate(smoothed_nodes):
                if i != j:
                    dx = node[0] - other_node[0]
                    dy = node[1] - other_node[1]
                    distance = np.sqrt(dx ** 2 + dy ** 2)
                    if distance < repulsion_radius and distance > 0:
                        repulsion = (repulsion_radius - distance) / distance * scaled_repulsion_strength
                        forces += np.array([dx, dy]) * repulsion

            # Apply the repulsion forces and update the node position
            new_position = smoothed_nodes[i] + forces * 0.01  # The 0.01 factor controls the strength of smoothing

            # Ensure the new position is within free space and respects the buffer
            new_x, new_y = int(new_position[0]), int(new_position[1])
            if (0 <= new_x < occupancy_grid.shape[0]) and (0 <= new_y < occupancy_grid.shape[1]) and \
                    (occupancy_grid[new_x, new_y] == 1 and
                     np.all(occupancy_grid[
                            max(0, new_x - buffer_size):min(new_x + buffer_size + 1, occupancy_grid.shape[0]),
                            max(0, new_y - buffer_size):min(new_y + buffer_size + 1, occupancy_grid.shape[1])] == 1)):
                smoothed_nodes[i] = [new_x, new_y]  # Update the node position if valid

    return np.array(smoothed_nodes)


# Function to reduce skeleton points using KMeans
def reduce_skeleton_points(skeleton_points, n_clusters=150, occupancy_grid=None, buffer_size=20):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(skeleton_points)
    centroids = kmeans.cluster_centers_

    valid_centroids = []

    for centroid in centroids:
        x, y = int(centroid[0]), int(centroid[1])
        # Ensure the node is in free space and maintains a buffer from walls
        if occupancy_grid is not None and (0 <= x < occupancy_grid.shape[0]) and (0 <= y < occupancy_grid.shape[1]):
            # Check if the centroid is within free space and far enough from walls
            if occupancy_grid[x, y] == 1 and np.all(
                    occupancy_grid[max(0, x - buffer_size):min(x + buffer_size + 1, occupancy_grid.shape[0]),
                    max(0, y - buffer_size):min(y + buffer_size + 1, occupancy_grid.shape[1])] == 1):
                valid_centroids.append([x, y])

    return np.array(valid_centroids)


# Function to calculate the number of connections for each node
def count_node_connections(nodes, connections):
    connection_counts = np.zeros(len(nodes), dtype=int)
    for connection in connections:
        connection_counts[connection[0]] += 1
        connection_counts[connection[1]] += 1
    return connection_counts


# Function to remove nodes that are too close to each other
def remove_close_nodes(nodes, connections, min_distance=30):
    # Count the number of connections for each node
    connection_counts = count_node_connections(nodes, connections)

    to_remove = set()  # Keep track of nodes to remove
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i != j and j not in to_remove:
                # Calculate the distance between the two nodes
                distance = np.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)
                if distance < min_distance:
                    # If the distance is less than the threshold, remove the node with fewer connections
                    if connection_counts[i] < connection_counts[j]:
                        to_remove.add(i)
                    else:
                        to_remove.add(j)

    # Remove the selected nodes
    filtered_nodes = np.array([node for i, node in enumerate(nodes) if i not in to_remove])

    return filtered_nodes


# Function to check if two nodes have line of sight (no walls between them) using ray-casting
def has_line_of_sight(node1, node2, grid):
    x1, y1 = map(int, node1)
    x2, y2 = map(int, node2)
    num_samples = 500  # Number of samples along the line

    # Generate points between node1 and node2
    x_vals = np.linspace(x1, x2, num=num_samples)
    y_vals = np.linspace(y1, y2, num=num_samples)

    # Check if any point along the line is inside a wall (grid value == 0)
    for x, y in zip(x_vals, y_vals):
        if grid[int(x), int(y)] == 0:  # Ensure proper indexing
            return False  # Line of sight is blocked by a wall

    return True  # Line of sight is clear


# Function to connect nodes based on line of sight using ray-casting
def connect_all_nodes_with_los_raycasting(nodes, grid, max_distance=250):
    connections = []
    nn_model = NearestNeighbors(n_neighbors=len(nodes))  # Consider all possible neighbors
    nn_model.fit(nodes)
    distances, indices = nn_model.kneighbors(nodes)

    # Connect nodes based on distance and line of sight
    for i, neighbors in enumerate(indices):
        for j in range(1, len(neighbors)):  # Start at 1 to skip self
            neighbor_idx = neighbors[j]
            node1 = nodes[i]
            node2 = nodes[neighbor_idx]
            dist = distances[i][j]

            if dist > max_distance:
                continue

            if has_line_of_sight(node1, node2, grid):
                connections.append((i, neighbor_idx))

    return connections


# Function to plot only the "After Smoothing" figure
def plot_after_smoothing(after_nodes, connections_after, grid, radiation_map):
    plt.figure(figsize=(10, 10))

    # Reverse heatmap (use 'hot_r' for reversed heatmap)
    cmap = 'hot_r'

    # Plot after smoothing
    plt.imshow(grid, cmap='gray', interpolation='nearest', alpha=0.5)
    plt.imshow(radiation_map, cmap=cmap, interpolation='nearest', alpha=0.5)
    plt.scatter(after_nodes[:, 1], after_nodes[:, 0], c='red', s=50, label="Nodes")
    for connection in connections_after:
        node1, node2 = after_nodes[connection[0]], after_nodes[connection[1]]
        plt.plot([node1[1], node2[1]], [node1[0], node2[0]], 'k-', lw=0.5, alpha=0.3)

    plt.title("Occupancy Map Abstraction for ML Exploration")  # Adjusted title as per your preference
    plt.show()


# Main execution with adjusted cluster number and single plot
start_time = time.time()  # Start timer

# Step 1: Load and crop the occupancy grid and radiation map consistently
occupancy_grid, radiation_map = load_and_crop_maps('../occupancy_map_output/occupancy_grid.csv',
                                                   'occupancy_map_output/radiation_map.csv')

# Step 2: Extract skeleton points from the free space
skeleton_points = extract_skeleton_with_distance(occupancy_grid)

# Step 3: Increase the number of clusters to improve coverage
n_clusters = 100  # Increased to improve map coverage
initial_nodes = reduce_skeleton_points(skeleton_points, n_clusters=n_clusters, occupancy_grid=occupancy_grid,
                                       buffer_size=10)

# Apply skewing during initial placement with adjustable skewing factor
skewing_factor = 3.0 # Adjust this value to control the level of skewing
skewed_nodes = skew_nodes_based_on_free_space(initial_nodes, radiation_map, occupancy_grid,
                                              skewing_factor=skewing_factor)

# Step 5: Apply repulsive smoothing directly to all nodes
repulsion_radius = 8  # Define the radius within which nodes push each other apart
iterations = 5  # Number of iterations for smoothing
smoothed_nodes = apply_repulsive_smoothing(skewed_nodes, occupancy_grid, radiation_map,
                                           repulsion_radius=repulsion_radius, buffer_size=20, iterations=iterations)

# Step 6: Remove nodes that are too close to each other based on the number of neighbor connections
min_distance_between_nodes = 10  # Set the distance threshold for node removal
connections_before = connect_all_nodes_with_los_raycasting(smoothed_nodes, occupancy_grid, max_distance=250)
filtered_nodes = remove_close_nodes(smoothed_nodes, connections_before, min_distance=min_distance_between_nodes)

# Step 7: Recalculate connections after removing close nodes
connections_after = connect_all_nodes_with_los_raycasting(filtered_nodes, occupancy_grid, max_distance=250)

# Step 8: Plot only the "After Smoothing" result
plot_after_smoothing(filtered_nodes, connections_after, occupancy_grid, radiation_map)

end_time = time.time()  # End timer
print(f"Computation completed in {end_time - start_time:.2f} seconds.")