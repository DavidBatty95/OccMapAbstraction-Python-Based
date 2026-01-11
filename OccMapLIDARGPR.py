# ======================================================================================================================
# TITLE: OCC MAP ABSTRACTION EXPLORATION w. ONLINE GPR
# ======================================================================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.measure import block_reduce
from skimage.draw import line
from sklearn.neighbors import KDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from math import sqrt
import heapq
import time

# ======================================================================================================================
#                                                   PARAMETERS
# ======================================================================================================================
# Configuration Parameters
MAP_PATH = 'Maps/3x3mapmain.csv'
MAP_SIZE = 50
RESOLUTION = 0.05

# Map Abstraction Parameters
DOWNSAMPLE_FACTOR = 8
FILTER_DISTANCE = 1.9
CONNECTION_RADIUS = 5.0
MAX_NEIGHBORS = 15

# Radiation Source Parameters
MIN_RADIUS_THRESHOLD = 1.6
DISTANCE_WEIGHT = 1.0
RADIATION_MAX_INTENSITY = 100
RADIATION_SOURCE_INTENSITY = 100
RADIATION_SOURCE_RADIUS = 3.5

# Other Simulation Parameters
START_POSITION = (6.2, 6.4)
MAX_STAGES = 10
LIDAR_RANGE = 30.0
SPEED = 0.5

# GPR Parameters
GPR_KERNEL = C(1.0, (1e-3, 3e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 3e2))
GPR_RESTART_ITERATIONS = 5
GPR_SAMPLING_INTERVAL = 0.5  # meters
MAX_GPR_POINTS = 10000
GPR_SAFETY_FACTOR = 1.0  # Increased for more conservative estimates


# ======================================================================================================================
#                                                   OCCUPANCY MAP HANDLER
# ======================================================================================================================
class OccupancyMap:
    @staticmethod
    def load(file_path, border_cells=10):
        occupancy_map = np.loadtxt(file_path, delimiter=",")
        # Occuapncy Map cropping based on random generation code
        occupancy_map[:border_cells, :] = 0
        occupancy_map[-border_cells:, :] = 0
        occupancy_map[:, :border_cells] = 0
        occupancy_map[:, -border_cells:] = 0
        return np.where(occupancy_map == 1.0, 1, 0)


class RadiationMap:
    @staticmethod
    def place_sources():
        return [
            (25.0, 34.0),
            (4.0, 46.0),
            (42.0, 15.0),
        ], []

    @staticmethod
    def calculate(occupancy_map, sources, constant_regions, resolution, lidar_range):
        radiation_map = np.zeros_like(occupancy_map, dtype=float)
        for x, y in sources:
            source_x, source_y = int(x / resolution), int(y / resolution)
            y_indices, x_indices = np.indices(occupancy_map.shape)
            distances = np.sqrt((source_x - x_indices) ** 2 + (source_y - y_indices) ** 2)
            sigma = RADIATION_SOURCE_RADIUS / resolution
            radiation_intensity = RADIATION_SOURCE_INTENSITY * np.exp(-(distances ** 2) / (2 * sigma ** 2))
            mask = (occupancy_map == 1) & (distances <= (lidar_range / resolution))
            radiation_map += radiation_intensity * mask
        for region in constant_regions:
            x_min, x_max, y_min, y_max, intensity = region
            radiation_map[y_min:y_max, x_min:x_max] += intensity
        radiation_map = np.clip(radiation_map, 0, RADIATION_MAX_INTENSITY)
        return radiation_map

# ======================================================================================================================
#                                                   GPR MODEL CODE
# ======================================================================================================================


class GPRModelHandler:
    def __init__(self, model, radiation_threshold, safety_factor=GPR_SAFETY_FACTOR, max_points=MAX_GPR_POINTS):
        self.model = model
        self.max_points = max_points
        self.positions = []
        self.readings = []
        self.radiation_threshold = radiation_threshold
        self.safety_factor = safety_factor

    def add_readings(self, new_positions, new_readings):
        self.positions.extend(new_positions)
        self.readings.extend(new_readings)
        if len(self.positions) > self.max_points:
            self.positions = self.positions[-self.max_points:]
            self.readings = self.readings[-self.max_points:]

    def update(self, radiation_map):
        if len(self.positions) < 1:
            estimated_map = np.zeros_like(radiation_map)
            uncertainty_map = np.zeros_like(radiation_map)
            return estimated_map, uncertainty_map
        X_train = np.array(self.positions)
        y_train = np.array(self.readings)
        try:
            self.model.fit(X_train, y_train)
        except Exception:
            estimated_map = np.zeros_like(radiation_map)
            uncertainty_map = np.zeros_like(radiation_map)
            return estimated_map, uncertainty_map
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, MAP_SIZE, radiation_map.shape[1]),
            np.linspace(0, MAP_SIZE, radiation_map.shape[0])
        )
        X_pred = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        try:
            y_pred, y_std = self.model.predict(X_pred, return_std=True)
            estimated_map = y_pred.reshape(radiation_map.shape)
            uncertainty_map = y_std.reshape(radiation_map.shape)
        except Exception:
            estimated_map = np.zeros_like(radiation_map)
            uncertainty_map = np.zeros_like(radiation_map)
            return estimated_map, uncertainty_map
        conservative_estimated_map = estimated_map + self.safety_factor * uncertainty_map
        conservative_estimated_map = np.clip(conservative_estimated_map, 0, RADIATION_MAX_INTENSITY)
        adjusted_radiation_threshold = self.radiation_threshold
        return conservative_estimated_map, adjusted_radiation_threshold


def plot_map(ax, occupancy_map, overlay_map=None, cmap="seismic", alpha=0.5, title="", colorbar_label="", vmin=0, vmax=100):
    wall_map = np.where(occupancy_map == 0, 1, 0)
    ax.imshow(wall_map, cmap="Greys", origin="lower", extent=[0, MAP_SIZE, 0, MAP_SIZE])
    if overlay_map is not None:
        im = ax.imshow(
            overlay_map,
            cmap=cmap,
            alpha=alpha,
            origin="lower",
            extent=[0, MAP_SIZE, 0, MAP_SIZE],
            vmin=vmin,
            vmax=vmax
        )
        if colorbar_label:
            plt.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_title(title)
    ax.set_xlim([0, MAP_SIZE])
    ax.set_ylim([0, MAP_SIZE])


def plot_exploration(stages, radiation_map, occupancy_map, run_label, use_gpr):
    num_stages = len(stages)
    if num_stages == 0:
        print("No exploration stages to display.")
        return
    panes_per_stage = 2 if use_gpr else 1
    total_panes = panes_per_stage * num_stages
    cols = 4
    rows = (total_panes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()
    pane_idx = 0
    fig.suptitle(run_label, fontsize=16)
    for i, stage in enumerate(stages):
        visibility, nodes, connections, path, _, _, title, dynamic_radiation_threshold, gpr_estimated_map = stage
        if use_gpr:
            ax_gpr = axes[pane_idx]
            plot_map(
                ax_gpr,
                occupancy_map=occupancy_map,
                overlay_map=gpr_estimated_map,
                cmap="seismic",
                alpha=0.5,  # Increased alpha for prominence
                title=f"{title} - GPR Estimated Radiation",
                colorbar_label="Estimated Radiation Intensity",
                vmin=0,
                vmax=100
            )
            if path:
                robot_x, robot_y = path[0]
                ax_gpr.plot(
                    robot_x, robot_y,
                    marker='*',
                    color='red',
                    markersize=15,
                    label='GPR Update Position'
                )
                ax_gpr.legend(loc="upper right", fontsize="small")
            pane_idx += 1
        ax_map = axes[pane_idx]
        wall_map = np.where(occupancy_map == 0, 1, 0)
        ax_map.imshow(wall_map, cmap="Greys", origin="lower", extent=[0, MAP_SIZE, 0, MAP_SIZE])
        ax_map.imshow(
            radiation_map,
            cmap="seismic",  # Changed colormap for better visibility
            alpha=0.5,  # Increased alpha for prominence
            origin="lower",
            extent=[0, MAP_SIZE, 0, MAP_SIZE],
            vmin=0,
            vmax=100
        )
        visible_display = np.where(visibility == 1, 1, 0.5)
        ax_map.imshow(
            visible_display,
            cmap="gray",
            alpha=0.6,
            origin="lower",
            extent=[0, MAP_SIZE, 0, MAP_SIZE]
        )
        if use_gpr and gpr_estimated_map is not None:
            unreachable_map = (
                (visibility == 1) &
                (gpr_estimated_map >= dynamic_radiation_threshold) &
                (occupancy_map == 1)
            )
            ax_map.imshow(
                unreachable_map,
                cmap='Reds',
                alpha=0.3,
                origin='lower',
                extent=[0, MAP_SIZE, 0, MAP_SIZE]
            )
        for (x0, y0), neighbors in connections.items():
            for (x1, y1) in neighbors:
                ax_map.plot(
                    [x0 * RESOLUTION, x1 * RESOLUTION],
                    [y0 * RESOLUTION, y1 * RESOLUTION],
                    'grey',
                    linewidth=0.5,
                    alpha=0.5
                )
        if nodes:
            node_x, node_y, _ = zip(*nodes)
            ax_map.plot(
                node_x,
                node_y,
                'o',
                color='blue',
                markersize=3,
                alpha=0.9,
                label="Nodes"
            )
        if path:
            path_x, path_y = zip(*path)
            ax_map.plot(
                path_x,
                path_y,
                'b-',
                linewidth=2.5,
                label="Path"
            )
        if path:
            ax_map.plot(
                path[0][0],
                path[0][1],
                'go',
                markersize=10,
                label="Start"
            )
            ax_map.plot(
                path[-1][0],
                path[-1][1],
                'ro',
                markersize=10,
                label="End"
            )
        ax_map.set_title(f"{title} - Map & Node Network")
        ax_map.set_xlim([0, MAP_SIZE])
        ax_map.set_ylim([0, MAP_SIZE])
        ax_map.legend(loc="upper right", fontsize="small")
        pane_idx += 1
    for j in range(pane_idx, len(axes)):
        axes[j].axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_cumulative_dose(run_results, speed=SPEED):
    plt.figure(figsize=(12, 6))
    for cumulative_distance, cumulative_dose, label in run_results:
        plt.plot(cumulative_distance, cumulative_dose, label=label)
    plt.xlabel("Distance Travelled (m)")
    plt.ylabel("Cumulative Dose (units)")
    plt.title("Cumulative Radiation Dose vs. Distance Travelled")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def simulate_radiation_readings(path, radiation_map, sensor_positions, sensor_readings):
    new_positions = []
    new_readings = []
    if len(path) < 2:
        return new_positions, new_readings
    for i in range(1, len(path)):
        x_prev, y_prev = path[i - 1]
        x_curr, y_curr = path[i]
        distance = sqrt((x_curr - x_prev) ** 2 + (y_curr - y_prev) ** 2)
        steps = int(distance / GPR_SAMPLING_INTERVAL)
        for step in range(1, steps + 1):
            interp_x = x_prev + (x_curr - x_prev) * step / steps
            interp_y = y_prev + (y_curr - y_prev) * step / steps
            grid_x = int(round(interp_x / RESOLUTION))
            grid_y = int(round(interp_y / RESOLUTION))
            if 0 <= grid_x < radiation_map.shape[1] and 0 <= grid_y < radiation_map.shape[0]:
                reading = radiation_map[grid_y, grid_x]
                position = (interp_x, interp_y)
                if position not in sensor_positions:
                    sensor_positions.append(position)
                    sensor_readings.append(reading)
                    new_positions.append(position)
                    new_readings.append(reading)
    return new_positions, new_readings


def heuristic(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def bresenham_line(x0, y0, x1, y1):
    rr, cc = line(y0, x0, y1, x1)
    return rr, cc


def a_star(start, goal, connections, radiation_map_down, occupancy_map, radiation_threshold, full_radiation_map, radiation_weight):
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
            path = []
            while current:
                x, y = current
                path.append((x * RESOLUTION, y * RESOLUTION))
                current = came_from[current]
            path.reverse()
            if validate_path(path, full_radiation_map, radiation_threshold):
                return path
            else:
                return []
        for neighbor in connections.get(current, []):
            y_down = neighbor[1] // DOWNSAMPLE_FACTOR
            x_down = neighbor[0] // DOWNSAMPLE_FACTOR
            y_down = min(y_down, radiation_map_down.shape[0] - 1)
            x_down = min(x_down, radiation_map_down.shape[1] - 1)
            if radiation_map_down[y_down, x_down] >= radiation_threshold:
                continue
            tentative_g = g_score[current] + movement_cost_grid(neighbor, radiation_map_down, radiation_weight)
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic_grid(neighbor, goal_grid)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    print("No valid path found.")
    return []

def heuristic_grid(a, b):
    return heuristic(a, b)

def movement_cost_grid(a, radiation_map_down, radiation_weight):
    x_down = a[0] // DOWNSAMPLE_FACTOR
    y_down = a[1] // DOWNSAMPLE_FACTOR
    x_down = min(x_down, radiation_map_down.shape[1] - 1)
    y_down = min(y_down, radiation_map_down.shape[0] - 1)
    return 1 + radiation_weight * radiation_map_down[y_down, x_down]

def validate_path(path, radiation_map, threshold):
    for x, y in path:
        grid_x = int(round(x / RESOLUTION))
        grid_y = int(round(y / RESOLUTION))
        if grid_x < 0 or grid_x >= radiation_map.shape[1] or grid_y < 0 or grid_y >= radiation_map.shape[0]:
            return False
        if radiation_map[grid_y, grid_x] >= threshold:
            return False
    return True

def select_nodes_dynamic(distance_map, visibility_downsampled, radiation_estimated_down,
                         dynamic_radius_threshold, dynamic_radiation_threshold, start_pos):
    eligible = np.argwhere(
        (distance_map >= dynamic_radius_threshold) &
        (visibility_downsampled == 1) &
        (radiation_estimated_down < dynamic_radiation_threshold)
    )
    if eligible.size == 0:
        print("No eligible nodes found after filtering.")
        return []
    radii = distance_map[eligible[:, 0], eligible[:, 1]]
    sorted_indices = np.argsort(-radii)
    sorted_points = eligible[sorted_indices] * DOWNSAMPLE_FACTOR
    selected_nodes = []
    for point in sorted_points:
        x, y = point[1] * RESOLUTION, point[0] * RESOLUTION
        if all(
                sqrt((x - sx) ** 2 + (y - sy) ** 2) >= FILTER_DISTANCE
                for sx, sy, _ in selected_nodes
        ):
            radius = distance_map[point[0] // DOWNSAMPLE_FACTOR, point[1] // DOWNSAMPLE_FACTOR] / RESOLUTION
            selected_nodes.append((round(x, 2), round(y, 2), radius))
    current_start_x, current_start_y = start_pos
    start_idx_x = int(current_start_x / (RESOLUTION * DOWNSAMPLE_FACTOR))
    start_idx_y = int(current_start_y / (RESOLUTION * DOWNSAMPLE_FACTOR))
    start_idx_x = min(start_idx_x, radiation_estimated_down.shape[1] - 1)
    start_idx_y = min(start_idx_y, radiation_estimated_down.shape[0] - 1)
    start_radiation = radiation_estimated_down[start_idx_y, start_idx_x]
    if (
            start_radiation < dynamic_radiation_threshold and
            not any(
                sqrt((current_start_x - x) ** 2 + (current_start_y - y) ** 2) < 1e-6
                for x, y, _ in selected_nodes
            )
    ):
        selected_nodes.append((round(current_start_x, 2), round(current_start_y, 2), 0))
    print(f"Selected {len(selected_nodes)} nodes.")
    return selected_nodes

def create_connections(nodes, radiation_map_down, radiation_threshold, radiation_weight, occupancy_map, radius=CONNECTION_RADIUS):
    if not nodes:
        print("No nodes to connect.")
        return {}
    node_indices = {
        (int(round(x / RESOLUTION)), int(round(y / RESOLUTION))): (x, y)
        for (x, y, _) in nodes
    }
    grid_points = np.array(list(node_indices.keys()))
    tree = KDTree(grid_points)
    connections = {node: [] for node in node_indices.keys()}
    for idx, (grid_x, grid_y) in enumerate(node_indices.keys()):
        neighbors_idx = tree.query_radius([[grid_x, grid_y]], r=int(radius / RESOLUTION))[0]
        valid_neighbors = []
        for neighbor in neighbors_idx:
            if neighbor == idx:
                continue
            neighbor_grid = tuple(grid_points[neighbor])
            x1, y1 = node_indices[neighbor_grid]
            distance = heuristic_grid((grid_x, grid_y), neighbor_grid) * RESOLUTION
            if distance < FILTER_DISTANCE:
                continue
            if not is_path_clear(
                    node_indices[(grid_x, grid_y)][0],
                    node_indices[(grid_x, grid_y)][1],
                    x1,
                    y1,
                    radiation_map_down,
                    radiation_threshold,
                    occupancy_map
            ):
                continue
            valid_neighbors.append(neighbor_grid)
        valid_neighbors = sorted(
            valid_neighbors,
            key=lambda n: heuristic_grid((grid_x, grid_y), n)
        )[:MAX_NEIGHBORS]
        for neighbor_grid in valid_neighbors:
            connections[(grid_x, grid_y)].append(neighbor_grid)
            connections.setdefault(neighbor_grid, []).append((grid_x, grid_y))
    for node, neighbors in connections.items():
        connections[node] = list(set(neighbors))
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

def is_path_clear(x0, y0, x1, y1, radiation_map_down, radiation_threshold, occupancy_map):
    x0_idx, y0_idx = int(x0 / RESOLUTION), int(y0 / RESOLUTION)
    x1_idx, y1_idx = int(x1 / RESOLUTION), int(y1 / RESOLUTION)
    rr, cc = bresenham_line(x0_idx, y0_idx, x1_idx, y1_idx)
    for (y, x) in zip(rr, cc):
        if x < 0 or x >= occupancy_map.shape[1] or y < 0 or y >= occupancy_map.shape[0]:
            return False
        if occupancy_map[y, x] == 0:
            return False
    rr_down = rr // DOWNSAMPLE_FACTOR
    cc_down = cc // DOWNSAMPLE_FACTOR
    rr_down = np.clip(rr_down, 0, radiation_map_down.shape[0] - 1)
    cc_down = np.clip(cc_down, 0, radiation_map_down.shape[1] - 1)
    path_radiation = radiation_map_down[rr_down, cc_down]
    if np.any(path_radiation >= radiation_threshold):
        return False
    return True

def ensure_connectivity(nodes, connections):
    if not nodes:
        return nodes
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
    connected_nodes = [
        node for node in nodes
        if (int(round(node[0] / RESOLUTION)), int(round(node[1] / RESOLUTION))) in largest_component
    ]
    print(f"Retained {len(connected_nodes)} nodes from the largest connected component.")
    return connected_nodes

def is_reachable(start, goal, connections, radiation_map_down, occupancy_map, radiation_map, radiation_threshold):
    path = a_star(start, goal, connections, radiation_map_down, occupancy_map, radiation_threshold, radiation_map, radiation_weight=0)
    return len(path) > 0

def identify_best_frontier(nodes, visibility_map, current_position, connections, radiation_map_down, occupancy_map,
                           radiation_map, radiation_threshold):
    best_node = None
    best_score = -float('inf')
    for node in nodes:
        x, y, _ = node
        if not is_reachable(current_position, node, connections, radiation_map_down, occupancy_map,
                           radiation_map, radiation_threshold):
            continue
        point = (int(y / RESOLUTION), int(x / RESOLUTION))
        window_size = max(1, int(node[2]))
        y_min = max(0, point[0] - window_size)
        y_max = min(point[0] + window_size + 1, visibility_map.shape[0])
        x_min = max(0, point[1] - window_size)
        x_max = min(point[1] + window_size + 1, visibility_map.shape[1])
        local_area = visibility_map[y_min:y_max, x_min:x_max]
        unexplored = np.sum(local_area == 0)
        distance = heuristic(current_position, (x, y))
        score = unexplored - DISTANCE_WEIGHT * distance
        if score > best_score:
            best_score = score
            best_node = (x, y)
    if best_node:
        print(f"Selected best frontier at ({best_node[0]:.2f}, {best_node[1]:.2f}) with score {best_score:.2f}.")
    else:
        print("No suitable frontier found.")
    return best_node

def exploration_loop(occupancy_map, radiation_map, start_pos, radiation_threshold, radiation_weight, use_gpr=True):
    print("\nStarting new exploration run...")
    visited = [start_pos]
    visibility = generate_visibility_map(occupancy_map, visited, LIDAR_RANGE, RESOLUTION)
    stages = []
    total_path_length = 0
    gpr_estimated_map = np.zeros_like(radiation_map)
    gpr_uncertainty_map = np.zeros_like(radiation_map)
    if use_gpr:
        gpr_handler = GPRModelHandler(
            model=GaussianProcessRegressor(
                kernel=GPR_KERNEL,
                n_restarts_optimizer=GPR_RESTART_ITERATIONS,
                alpha=1e-2,
                normalize_y=True
            ),
            radiation_threshold=radiation_threshold,
            safety_factor=GPR_SAFETY_FACTOR,
            max_points=MAX_GPR_POINTS
        )
        sensor_positions = []
        sensor_readings = []
        initial_reading = radiation_map[int(round(start_pos[1] / RESOLUTION)), int(round(start_pos[0] / RESOLUTION))]
        sensor_positions.append(start_pos)
        sensor_readings.append(initial_reading)
        gpr_handler.add_readings([start_pos], [initial_reading])
        gpr_estimated_map, _ = gpr_handler.update(radiation_map)
    else:
        gpr_handler = None
        sensor_positions = []
        sensor_readings = []
    cumulative_distance = [0]
    cumulative_dose = [0]
    buffer_readings = []
    for stage in range(1, MAX_STAGES + 1):
        print(f"\n--- Stage {stage} ---")
        print(f"Current Position: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
        if radiation_weight > 0:
            dynamic_radiation_threshold = min(radiation_threshold + (stage * 5), RADIATION_MAX_INTENSITY)
        else:
            dynamic_radiation_threshold = radiation_threshold
        print(f"Radiation Threshold: {dynamic_radiation_threshold}")
        dynamic_connection_radius = CONNECTION_RADIUS + stage * 2
        if use_gpr and buffer_readings:
            print(f"Updating GPR with {len(buffer_readings)} readings from the previous stage.")
            gpr_handler.add_readings([pos for pos, _ in buffer_readings], [reading for _, reading in buffer_readings])
            gpr_estimated_map, _ = gpr_handler.update(radiation_map)
            print("GPR model updated with previous stage readings.")
            buffer_readings = []
        visibility = generate_visibility_map(occupancy_map, visited, LIDAR_RANGE, RESOLUTION)
        distance_map, visibility_down = calculate_distance_map(occupancy_map, visibility)
        if use_gpr and gpr_estimated_map is not None:
            gpr_estimated_down = block_reduce(
                gpr_estimated_map,
                block_size=(DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR),
                func=np.max
            )
        else:
            gpr_estimated_down = np.zeros_like(distance_map)
        nodes = select_nodes_dynamic(
            distance_map,
            visibility_down,
            gpr_estimated_down,
            dynamic_radius_threshold=0.5,
            dynamic_radiation_threshold=dynamic_radiation_threshold,
            start_pos=start_pos
        )
        if not nodes:
            print("No eligible nodes found. Terminating exploration.")
            break
        if use_gpr:
            connections = create_connections(
                nodes,
                gpr_estimated_down,
                radiation_threshold=dynamic_radiation_threshold,
                radiation_weight=radiation_weight,
                occupancy_map=occupancy_map,
                radius=dynamic_connection_radius
            )
        else:
            connections = create_connections(
                nodes,
                gpr_estimated_down,
                radiation_threshold=1000,
                radiation_weight=0,
                occupancy_map=occupancy_map,
                radius=dynamic_connection_radius
            )
        connected_nodes = ensure_connectivity(nodes, connections)
        if len(connected_nodes) != len(nodes):
            connected_grid_nodes = {
                (int(round(x / RESOLUTION)), int(round(y / RESOLUTION)))
                for (x, y, _) in connected_nodes
            }
            connections = {
                node: [n for n in neighbors if n in connected_grid_nodes]
                for node, neighbors in connections.items() if node in connected_grid_nodes
            }
            print("Filtered connections to maintain connectivity.")
        nodes = connected_nodes
        if not nodes:
            print("No nodes available for exploration after connectivity check. Terminating exploration.")
            break
        frontier = identify_best_frontier(nodes, visibility, start_pos, connections, gpr_estimated_down, occupancy_map,
                                          radiation_map, dynamic_radiation_threshold)
        if not frontier:
            print("No frontier identified. Selecting closest node.")
            frontier = min(
                nodes,
                key=lambda n: heuristic(start_pos, (n[0], n[1]))
            )
            if not is_reachable(start_pos, frontier, connections, gpr_estimated_down, occupancy_map, radiation_map,
                               dynamic_radiation_threshold):
                print("Closest node is not reachable safely. Skipping this node.")
                frontier = None
        if not frontier:
            print("No reachable frontier identified. Terminating exploration.")
            break
        if use_gpr:
            path = a_star(
                start_pos,
                frontier,
                connections,
                gpr_estimated_map,
                occupancy_map,
                dynamic_radiation_threshold,
                radiation_map,
                radiation_weight
            )
        else:
            path = a_star(
                start_pos,
                frontier,
                connections,
                np.zeros_like(gpr_estimated_map),
                occupancy_map,
                1000,
                radiation_map,
                radiation_weight=0
            )
        if not path:
            print("Path planning failed or path validation failed. Skipping this stage.")
            continue
        if use_gpr:
            current_stage_readings = simulate_radiation_readings(path, radiation_map, sensor_positions, sensor_readings)
            buffer_readings.extend(zip(current_stage_readings[0], current_stage_readings[1]))
            print(f"Collected {len(current_stage_readings[0])} new radiation readings from the current path.")
        else:
            current_stage_readings = ([], [])
        for i in range(1, len(path)):
            x_prev, y_prev = path[i - 1]
            x_curr, y_curr = path[i]
            distance = heuristic((x_prev, y_prev), (x_curr, y_curr))
            time_hours = distance / SPEED / 3600
            grid_x = int(round(x_curr / RESOLUTION))
            grid_y = int(round(y_curr / RESOLUTION))
            if 0 <= grid_x < radiation_map.shape[1] and 0 <= grid_y < radiation_map.shape[0]:
                radiation = radiation_map[grid_y, grid_x]
            else:
                radiation = 0
            dose = radiation * time_hours
            cumulative_distance.append(cumulative_distance[-1] + distance)
            cumulative_dose.append(cumulative_dose[-1] + dose)
        stages.append(
            (
                visibility.copy(),
                nodes.copy(),
                connections.copy(),
                path.copy(),
                path[0],
                path[-1],
                f"Stage {stage}",
                dynamic_radiation_threshold,
                gpr_estimated_map.copy() if use_gpr else np.zeros_like(radiation_map)
            )
        )
        path_length = sum(
            heuristic(p1, p2)
            for p1, p2 in zip(path[:-1], path[1:])
        )
        total_path_length += path_length
        print(f"Path length: {path_length:.2f} m | Total: {total_path_length:.2f} m")
        start_pos = path[-1]
        visited.append(start_pos)
    print(f"\nExploration complete. Total path length: {total_path_length:.2f} meters.")
    return stages, cumulative_distance, cumulative_dose

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
    downsampled_visibility = block_reduce(
        visibility_downsampled,
        block_size=(DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR),
        func=np.max
    )
    downsampled_map = block_reduce(
        occupancy_map,
        block_size=(DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR),
        func=np.min
    )
    distance_map = distance_transform_edt(downsampled_map * downsampled_visibility) * (RESOLUTION * DOWNSAMPLE_FACTOR)
    print("Distance transform calculated.")
    return distance_map, downsampled_visibility

def main():
    occupancy = OccupancyMap.load(MAP_PATH)
    sources, constants = RadiationMap.place_sources()
    radiation_map = RadiationMap.calculate(occupancy, sources, constants, RESOLUTION, LIDAR_RANGE)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_map(
        ax,
        occupancy_map=occupancy,
        overlay_map=radiation_map,
        cmap="seismic",  # Changed to 'viridis' for better visibility
        alpha=0.5,  # Increased alpha for prominence
        title="Occupancy and Radiation Maps",
        colorbar_label="Radiation Intensity",
        vmin=0,
        vmax=100
    )
    plt.show()
    exploration_scenarios = [
        {
            'label': 'With Radiation Awareness',
            'radiation_threshold': 15,
            'radiation_weight': 5,
            'use_gpr': True
        },
        {
            'label': 'Without Radiation Awareness',
            'radiation_threshold': 1000,
            'radiation_weight': 0,
            'use_gpr': False
        }
    ]
    run_results = []
    run_times = []
    for scenario in exploration_scenarios:
        print(f"\n=== Running: {scenario['label']} ===")
        start_time = time.perf_counter()
        stages, cumulative_distance, cumulative_dose = exploration_loop(
            occupancy_map=occupancy,
            radiation_map=radiation_map,
            start_pos=START_POSITION,
            radiation_threshold=scenario['radiation_threshold'],
            radiation_weight=scenario['radiation_weight'],
            use_gpr=scenario['use_gpr']
        )
        end_time = time.perf_counter()
        duration = end_time - start_time
        run_times.append((scenario['label'], duration))
        run_results.append((cumulative_distance, cumulative_dose, scenario['label']))
        plot_exploration(
            stages,
            radiation_map,
            occupancy,
            run_label=scenario['label'],
            use_gpr=scenario['use_gpr']
        )
    plot_cumulative_dose(run_results, speed=SPEED)
    print("\n=== Run Time Comparison ===")
    for label, duration in run_times:
        print(f"{label}: {duration:.2f} seconds")

if __name__ == "__main__":
    main()