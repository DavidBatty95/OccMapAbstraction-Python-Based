import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.measure import block_reduce
from skimage.draw import line
from sklearn.neighbors import KDTree
from math import sqrt
import heapq

# =========================
# Configuration Parameters
# =========================
MAP_PATH = 'occupancy_map_output/full_occupancy_map.csv'
MAP_SIZE = 50  # in meters
RESOLUTION = 0.05  # meters per cell
DOWNSAMPLE_FACTOR = 4
FILTER_DISTANCE = 1.25  # meters
CONNECTION_RADIUS = 5.0  # meters
MAX_NEIGHBORS = 15
MIN_RADIUS_THRESHOLD = 2.0
DISTANCE_WEIGHT = 1.0

# Radiation Parameters
RADIATION_THRESHOLD = 20
RADIATION_MAX_INTENSITY = 100
RADIATION_SOURCE_INTENSITY = 100
RADIATION_SOURCE_RADIUS = 3  # meters
RADIATION_WEIGHT = 2

START_POSITION = (6.2, 6.4)  # in meters
MAX_STAGES = 9
LIDAR_RANGE = 30.0  # in meters

# =========================
# Utility Functions
# =========================

def load_occupancy_map(file_path, border_cells=10):
    try:
        occupancy_map = np.loadtxt(file_path, delimiter=",")
        print(f"Occupancy map loaded: {occupancy_map.shape} cells.")
    except Exception as e:
        print(f"Error loading occupancy map: {e}")
        raise e

    # Mark borders as obstacles
    occupancy_map[:border_cells, :] = 0
    occupancy_map[-border_cells:, :] = 0
    occupancy_map[:, :border_cells] = 0
    occupancy_map[:, -border_cells:] = 0

    return np.where(occupancy_map == 1.0, 1, 0)


def place_radiation_sources():
    # Define radiation point sources (x, y) in meters
    return [
        (41.5, 17),
        (25, 17),
        (31, 4),
        (4, 29)
    ], []  # No constant regions defined


def calculate_radiation_map(occupancy_map, sources, constant_regions, resolution):
    radiation_map = np.zeros_like(occupancy_map, dtype=float)

    # Add radiation from point sources
    for x, y in sources:
        source_x, source_y = int(x / resolution), int(y / resolution)
        y_indices, x_indices = np.indices(occupancy_map.shape)
        distances = np.sqrt((source_x - x_indices) ** 2 + (source_y - y_indices) ** 2)
        sigma = RADIATION_SOURCE_RADIUS / resolution
        radiation_intensity = RADIATION_SOURCE_INTENSITY * np.exp(-(distances ** 2) / (2 * sigma ** 2))
        mask = (occupancy_map == 1) & (distances <= (LIDAR_RANGE / resolution))
        radiation_map += radiation_intensity * mask

    # Add constant radiation regions if any (currently empty)
    for region in constant_regions:
        x_min, x_max, y_min, y_max, intensity = region
        radiation_map[y_min:y_max, x_min:x_max] += intensity

    # Clip to max intensity
    radiation_map = np.clip(radiation_map, 0, RADIATION_MAX_INTENSITY)
    print("Radiation map calculated.")
    return radiation_map


def plot_maps(occupancy_map, radiation_map):
    plt.figure(figsize=(8, 8))
    # Display obstacles
    wall_map = np.where(occupancy_map == 0, 1, 0)
    plt.imshow(wall_map, cmap="Greys", origin="lower", extent=[0, MAP_SIZE, 0, MAP_SIZE])

    # Overlay radiation
    plt.imshow(radiation_map, cmap="jet", alpha=0.7, origin="lower", extent=[0, MAP_SIZE, 0, MAP_SIZE])
    plt.colorbar(label="Radiation Intensity")
    plt.title("Occupancy and Radiation Maps")
    plt.show()


def generate_visibility_map(occupancy_map, visited_points, lidar_range, resolution):
    visibility_map = np.zeros_like(occupancy_map)
    num_rays = 1440
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

    for x, y in visited_points:
        start_x, start_y = int(x / resolution), int(y / resolution)
        for angle in angles:
            dx, dy = np.cos(angle), np.sin(angle)
            for r in range(1, int(lidar_range / resolution)):
                xi = int(start_x + dx * r)
                yi = int(start_y + dy * r)
                if xi < 0 or xi >= occupancy_map.shape[1] or yi < 0 or yi >= occupancy_map.shape[0]:
                    break
                if occupancy_map[yi, xi] == 0:
                    break
                visibility_map[yi, xi] = 1
    print("Visibility map generated.")
    return visibility_map


def calculate_distance_map(occupancy_map, visibility_downsampled):
    downsampled_visibility = block_reduce(visibility_downsampled, block_size=(DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR),
                                         func=np.max)
    downsampled_map = block_reduce(occupancy_map, block_size=(DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR), func=np.min)
    distance_map = distance_transform_edt(downsampled_map * downsampled_visibility) * (RESOLUTION * DOWNSAMPLE_FACTOR)
    print("Distance transform calculated.")
    return distance_map, downsampled_visibility


def select_nodes_dynamic(distance_map, visibility_downsampled, radiation_downsampled, dynamic_radius_threshold, dynamic_radiation_threshold, start_pos):
    eligible = np.argwhere((distance_map >= dynamic_radius_threshold) &
                           (visibility_downsampled == 1) &
                           (radiation_downsampled < dynamic_radiation_threshold))
    if eligible.size == 0:
        print("No eligible nodes found after filtering.")
        return []

    # Sort eligible nodes by descending distance
    radii = distance_map[eligible[:, 0], eligible[:, 1]]
    sorted_indices = np.argsort(-radii)
    sorted_points = eligible[sorted_indices] * DOWNSAMPLE_FACTOR  # Scale back to original resolution

    selected_nodes = []
    for point in sorted_points:
        x, y = point[1] * RESOLUTION, point[0] * RESOLUTION
        if all(sqrt((x - sx) ** 2 + (y - sy) ** 2) >= FILTER_DISTANCE for sx, sy, _ in selected_nodes):
            radius = distance_map[point[0] // DOWNSAMPLE_FACTOR, point[1] // DOWNSAMPLE_FACTOR] / RESOLUTION
            selected_nodes.append((round(x, 2), round(y, 2), radius))

    # Ensure start position is included if safe
    current_start_x, current_start_y = start_pos
    start_idx_x = int(current_start_x / (RESOLUTION * DOWNSAMPLE_FACTOR))
    start_idx_y = int(current_start_y / (RESOLUTION * DOWNSAMPLE_FACTOR))
    start_idx_x = min(start_idx_x, radiation_downsampled.shape[1] - 1)
    start_idx_y = min(start_idx_y, radiation_downsampled.shape[0] - 1)
    start_radiation = radiation_downsampled[start_idx_y, start_idx_x]
    if start_radiation < dynamic_radiation_threshold and not any(
            sqrt((current_start_x - x) ** 2 + (current_start_y - y) ** 2) < 1e-6 for x, y, _ in selected_nodes):
        selected_nodes.append((round(current_start_x, 2), round(current_start_y, 2), 0))

    print(f"Selected {len(selected_nodes)} nodes.")
    return selected_nodes


def create_connections(nodes, radiation_map, radius=CONNECTION_RADIUS):
    if not nodes:
        print("No nodes to connect.")
        return {}

    # Convert node coordinates to integer grid indices for consistency
    node_indices = { (int(round(x / RESOLUTION)), int(round(y / RESOLUTION))): (x, y) for (x, y, _) in nodes }
    grid_points = np.array(list(node_indices.keys()))
    tree = KDTree(grid_points)
    connections = { (x, y): [] for (x, y) in node_indices.keys() }

    for idx, (grid_x, grid_y) in enumerate(node_indices.keys()):
        neighbors_idx = tree.query_radius([[grid_x, grid_y]], r=int(radius / RESOLUTION))[0]
        valid_neighbors = []
        for neighbor in neighbors_idx:
            if neighbor == idx:
                continue
            neighbor_grid = tuple(grid_points[neighbor])
            x1, y1 = node_indices[neighbor_grid]
            distance = sqrt((x1 - node_indices[(grid_x, grid_y)][0])**2 + (y1 - node_indices[(grid_x, grid_y)][1])**2)
            if distance < FILTER_DISTANCE:
                continue
            if not is_path_clear(node_indices[(grid_x, grid_y)][0], node_indices[(grid_x, grid_y)][1],
                                 x1, y1, radiation_map):
                continue
            valid_neighbors.append(neighbor_grid)

        # Sort neighbors by distance and limit to MAX_NEIGHBORS
        valid_neighbors = sorted(valid_neighbors, key=lambda n: sqrt(
            (node_indices[n][0] - node_indices[(grid_x, grid_y)][0])**2 +
            (node_indices[n][1] - node_indices[(grid_x, grid_y)][1])**2
        ))[:MAX_NEIGHBORS]

        for neighbor_grid in valid_neighbors:
            connections[(grid_x, grid_y)].append(neighbor_grid)
            # Ensure bidirectional connection
            connections.setdefault(neighbor_grid, []).append((grid_x, grid_y))

    # Check for isolated nodes
    isolated_nodes = [node for node, neighbors in connections.items() if len(neighbors) == 0]
    if isolated_nodes:
        print(f"{len(isolated_nodes)} isolated nodes detected and excluded.")
        for node in isolated_nodes:
            del connections[node]
    else:
        print("All nodes have connections.")

    total_connections = sum(len(neigh) for neigh in connections.values()) // 2
    print(f"Total connections established: {total_connections}")
    return connections


def is_path_clear(x0, y0, x1, y1, radiation_map):
    x0_idx, y0_idx = int(x0 / RESOLUTION), int(y0 / RESOLUTION)
    x1_idx, y1_idx = int(x1 / RESOLUTION), int(y1 / RESOLUTION)
    rr, cc = line(y0_idx, x0_idx, y1_idx, x1_idx)
    rr = np.clip(rr, 0, radiation_map.shape[0] - 1)
    cc = np.clip(cc, 0, radiation_map.shape[1] - 1)
    path_radiation = radiation_map[rr, cc]
    if np.any(path_radiation >= RADIATION_THRESHOLD * 1.5):
        return False
    return True


def ensure_connectivity(nodes, connections):
    if not nodes:
        return nodes

    # Build adjacency list
    adjacency = {node: set(neighbors) for node, neighbors in connections.items()}
    visited = set()
    components = []

    for node in adjacency:
        if node not in visited:
            stack = [node]
            component = []
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    stack.extend(adjacency[current] - visited)
            components.append(component)

    print(f"Connected components found: {len(components)}")
    if not components:
        return nodes

    largest_component = max(components, key=len)
    connected_nodes = [node for node in nodes if (int(round(node[0] / RESOLUTION)), int(round(node[1] / RESOLUTION))) in largest_component]
    print(f"Retained {len(connected_nodes)} nodes from the largest connected component.")
    return connected_nodes


def identify_best_frontier(nodes, visibility_map, current_position):
    best_node = None
    best_score = -float('inf')
    for x, y, radius in nodes:
        point = (int(y / RESOLUTION), int(x / RESOLUTION))
        window_size = max(1, int(radius))
        y_min = max(0, point[0] - window_size)
        y_max = min(point[0] + window_size + 1, visibility_map.shape[0])
        x_min = max(0, point[1] - window_size)
        x_max = min(point[1] + window_size + 1, visibility_map.shape[1])
        local_area = visibility_map[y_min:y_max, x_min:x_max]
        unexplored = np.sum(local_area == 0)
        distance = sqrt((x - current_position[0]) ** 2 + (y - current_position[1]) ** 2)
        score = unexplored - DISTANCE_WEIGHT * distance
        if score > best_score:
            best_score = score
            best_node = (x, y)
    if best_node:
        print(f"Selected best frontier at ({best_node[0]:.2f}, {best_node[1]:.2f}) with score {best_score:.2f}.")
    else:
        print("No suitable frontier found.")
    return best_node


def plot_exploration(stages, radiation_map, occupancy_map):
    num_stages = len(stages)
    if num_stages == 0:
        print("No exploration stages to display.")
        return

    cols = 3
    rows = (num_stages + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    # Threshold radiation for plotting
    radiation_thresh_map = np.where(radiation_map > 2, radiation_map, np.nan)

    for i, (visibility, nodes, connections, path, _, _, title, dynamic_radiation_threshold) in enumerate(stages):
        if i >= len(axes):
            break
        ax = axes[i]

        # Plot obstacles
        wall_map = np.where(occupancy_map == 0, 1, 0)
        ax.imshow(wall_map, cmap="Greys", origin="lower", extent=[0, MAP_SIZE, 0, MAP_SIZE])

        # Plot visibility
        visible_display = np.where(visibility == 1, 1, 0.5)
        ax.imshow(visible_display, cmap="gray", alpha=0.6, origin="lower", extent=[0, MAP_SIZE, 0, MAP_SIZE])

        # Plot radiation
        ax.imshow(radiation_thresh_map, cmap="jet", alpha=0.4, origin="lower", extent=[0, MAP_SIZE, 0, MAP_SIZE])

        # Highlight unreachable areas in red
        unreachable_map = (visibility == 1) & (radiation_map >= dynamic_radiation_threshold) & (occupancy_map == 1)
        ax.imshow(unreachable_map, cmap='Reds', alpha=0.3, origin='lower', extent=[0, MAP_SIZE, 0, MAP_SIZE])

        # Plot nodes
        for x, y, _ in nodes:
            ax.plot(x, y, 'o', color='red', markersize=5, alpha=0.7)

        # Plot connections
        for (x0, y0), neighbors in connections.items():
            for (x1, y1) in neighbors:
                ax.plot([x0 * RESOLUTION, x1 * RESOLUTION], [y0 * RESOLUTION, y1 * RESOLUTION], 'k-', linewidth=0.3, alpha=0.7)

        # Plot path
        if path:
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, 'b-', linewidth=2, label="Path")

        # Plot start and end
        if path:
            ax.plot(path[0][0], path[0][1], 'go', markersize=10, label="Start")
            ax.plot(path[-1][0], path[-1][1], 'bo', markersize=10, label="End")

        ax.set_title(title)
        ax.set_xlim([0, MAP_SIZE])
        ax.set_ylim([0, MAP_SIZE])
        ax.legend(loc="upper right", fontsize="small")

    # Hide any unused subplots
    for j in range(num_stages, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


# ======================================================================================================================
#                                                   A* PATH PLANNER
# ======================================================================================================================
def a_star(start, goal, connections, radiation_map, occupancy_map):
    start_grid = (int(round(start[0] / RESOLUTION)), int(round(start[1] / RESOLUTION)))
    goal_grid = (int(round(goal[0] / RESOLUTION)), int(round(goal[1] / RESOLUTION)))

    if start_grid not in connections or goal_grid not in connections:
        print("Start or goal node is not connected. Pathfinding aborted.")
        return []

    open_set = []
    heapq.heappush(open_set, (0, start_grid))
    came_from = {start_grid: None}
    g_score = {start_grid: 0}
    f_score = {start_grid: heuristic_grid(start_grid, goal_grid)}

    while open_set:
        current_f, current = heapq.heappop(open_set)
        if current == goal_grid:
            # Reconstruct path
            path = []
            while current:
                x, y = current
                path.append((x * RESOLUTION, y * RESOLUTION))
                current = came_from[current]
            path.reverse()
            print(f"Path found: {len(path)} steps.")
            return path

        for neighbor in connections.get(current, []):
            tentative_g = g_score[current] + movement_cost_grid(current, neighbor, radiation_map)
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic_grid(neighbor, goal_grid)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    print("No valid path found.")
    return []


def heuristic_grid(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def movement_cost_grid(a, b, radiation_map):
    distance = heuristic_grid(a, b)
    midpoint_x = int((a[0] + b[0]) / 2)
    midpoint_y = int((a[1] + b[1]) / 2)
    radiation = radiation_map[midpoint_y, midpoint_x] if (0 <= midpoint_x < radiation_map.shape[1] and 0 <= midpoint_y < radiation_map.shape[0]) else 0
    return distance + RADIATION_WEIGHT * radiation


# ======================================================================================================================
#                                                MAIN EXPLORATION LOOP
# ======================================================================================================================

def exploration_loop(occupancy_map, radiation_map, start_pos):
    print("Starting exploration...")
    visited = [start_pos]
    visibility = generate_visibility_map(occupancy_map, visited, LIDAR_RANGE, RESOLUTION)
    stages = []
    total_path_length = 0

    # Downsample radiation map for node selection
    radiation_down = block_reduce(radiation_map, block_size=(DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR), func=np.mean)

    for stage in range(1, MAX_STAGES + 1):
        print(f"\n--- Stage {stage} ---")
        print(f"Current Position: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")

        # Dynamic thresholds
        dynamic_radius_threshold = max(MIN_RADIUS_THRESHOLD - (stage * 0.1), 0.5)
        dynamic_radiation_threshold = min(RADIATION_THRESHOLD + (stage * 5), RADIATION_MAX_INTENSITY)
        dynamic_connection_radius = CONNECTION_RADIUS + stage * 2

        print(f"Radius Threshold: {dynamic_radius_threshold:.2f} m")
        print(f"Radiation Threshold: {dynamic_radiation_threshold}")
        print(f"Connection Radius: {dynamic_connection_radius} m")

        # Update distance map and visibility
        distance_map, visibility_down = calculate_distance_map(occupancy_map, visibility)
        nodes = select_nodes_dynamic(distance_map, visibility_down, radiation_down, dynamic_radius_threshold, dynamic_radiation_threshold, start_pos)

        # Create connections with dynamic radius
        connections = create_connections(nodes, radiation_map, radius=dynamic_connection_radius)

        # Ensure connectivity and filter nodes accordingly
        connected_nodes = ensure_connectivity(nodes, connections)
        if len(connected_nodes) != len(nodes):
            connected_grid_nodes = { (int(round(x / RESOLUTION)), int(round(y / RESOLUTION))) for (x, y, _) in connected_nodes }
            connections = { node: [n for n in neighbors if n in connected_grid_nodes] for node, neighbors in connections.items() if node in connected_grid_nodes }
            print("Filtered connections to maintain connectivity.")
        nodes = connected_nodes  # Update nodes to connected nodes

        if not nodes:
            print("No nodes available for exploration. Terminating exploration.")
            break

        # Identify the best frontier node
        frontier = identify_best_frontier(nodes, visibility, start_pos)

        if not frontier:
            print("No frontier identified. Selecting closest node.")
            # Selecting the node closest to the start position
            frontier = min(nodes, key=lambda n: heuristic_grid_grid(
                (int(round(n[0] / RESOLUTION)), int(round(n[1] / RESOLUTION))),
                (int(round(start_pos[0] / RESOLUTION)), int(round(start_pos[1] / RESOLUTION)))
            ))

        # Path planning
        path = a_star(start_pos, frontier, connections, radiation_map, occupancy_map)
        if not path:
            print("Path planning failed. Skipping this stage.")
            continue

        # Calculate path length
        path_length = sum(
            sqrt((path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2) for i in range(1, len(path))
        )
        total_path_length += path_length
        print(f"Path length: {path_length:.2f} m | Total: {total_path_length:.2f} m")

        # Record stage details, including dynamic_radiation_threshold
        stages.append(
            (visibility.copy(), nodes.copy(), connections.copy(), path.copy(), path[0], path[-1], f"Stage {stage}", dynamic_radiation_threshold)
        )

        # Update for next stage
        start_pos = path[-1]
        visited.append(start_pos)
        visibility = generate_visibility_map(occupancy_map, visited, LIDAR_RANGE, RESOLUTION)

    print(f"\nExploration complete. Total path length: {total_path_length:.2f} meters.")
    plot_exploration(stages, radiation_map, occupancy_map)


def heuristic_grid_grid(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ======================================================================================================================
#                                                   MAIN FUNCTION
# ======================================================================================================================


if __name__ == "__main__":
    occupancy = load_occupancy_map(MAP_PATH)
    sources, constants = place_radiation_sources()
    radiation = calculate_radiation_map(occupancy, sources, constants, RESOLUTION)
    plot_maps(occupancy, radiation)
    exploration_loop(occupancy, radiation, START_POSITION)