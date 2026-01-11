import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.measure import block_reduce
from skimage.draw import line
from sklearn.neighbors import KDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from math import sqrt
import heapq
import logging

# ======================================================================================================================
# PARAMETERS
# ======================================================================================================================
MAP_PATH = 'Maps/3x3mapmain.csv'
MAP_SIZE = 50.0              # meters
RESOLUTION = 0.05            # m/cell
COVERAGE_THRESHOLD = 95.0    # %
DOWNSAMPLE_FACTOR = 8
FILTER_DISTANCE = 2.0        # meters

CONNECTION_RADIUS = 3.5      # meters
MAX_NEIGHBORS = 12
MIN_RADIUS_THRESHOLD = 2.0   # meters

# Radiation
RADIATION_MAX_INTENSITY = 100
RADIATION_SOURCE_INTENSITY = 100
RADIATION_SOURCE_RADIUS = 3.0
START_POSITION = (6.2, 6.4)  # m
MAX_STAGES = 15

# LIDAR
LIDAR_RANGE = 20.0           # meters
SPEED = 0.5                  # m/s

# GPR
GPR_KERNEL = (
    ConstantKernel(1.0, (1e-6, 1e4))
    * RBF(length_scale=2.0, length_scale_bounds=(1e-3, 1e3))
)
GPR_RESTART_ITERATIONS = 3
GPR_SAMPLING_INTERVAL = 1.0  # m
MAX_GPR_POINTS = 5000
GPR_SAFETY_FACTOR = 1.0
RADIATION_SAMPLING_RADIUS = 0.0
RADIATION_SAMPLING_INTERVAL = 0.1

# Frontier/Exploration
MIN_FRONTIER_DISTANCE = 2.0
NUM_RUNS = 1

USE_SAFETY_FACTOR = False
RADIATION_COST_WEIGHT = 1.0
INFO_GAIN_WEIGHT = 0.5
PROXIMITY_BONUS = 1000.0
epsilon = 1e-3

# Minimal logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# ======================================================================================================================
# STATIC RADIATION SOURCES
# ======================================================================================================================
STATIC_SOURCES = [
    (20.0, 10.0),
    (35.0, 15.0),
    (40.0, 40.0)
]

# ======================================================================================================================
# OCCUPANCY MAP
# ======================================================================================================================
class OccupancyMap:
    @staticmethod
    def load(file_path, border_cells=10):
        occupancy_map = np.loadtxt(file_path, delimiter=",")
        # Clear boundary cells (optional)
        occupancy_map[:border_cells, :] = 0
        occupancy_map[-border_cells:, :] = 0
        occupancy_map[:, :border_cells] = 0
        occupancy_map[:, -border_cells:] = 0
        # 1 => free, 0 => occupied
        return np.where(occupancy_map == 1.0, 1, 0)

# ======================================================================================================================
# RADIATION MAP
# ======================================================================================================================
class RadiationMap:
    @staticmethod
    def calculate(occupancy_map, sources, resolution, lidar_range):
        """
        Create a radiation map with Gaussian-like sources placed at STATIC_SOURCES.
        """
        radiation_map = np.zeros_like(occupancy_map, dtype=float)
        for x, y in sources:
            src_x, src_y = int(x / resolution), int(y / resolution)
            rows, cols = occupancy_map.shape
            y_indices, x_indices = np.indices((rows, cols))
            distances = np.sqrt((src_x - x_indices)**2 + (src_y - y_indices)**2)
            sigma = RADIATION_SOURCE_RADIUS / resolution
            intensity = RADIATION_SOURCE_INTENSITY * np.exp(-(distances**2) / (2 * sigma**2))
            # Only add to free cells within LIDAR range
            mask = (occupancy_map == 1) & (distances <= (lidar_range / resolution))
            radiation_map += intensity * mask
        return np.clip(radiation_map, 0, RADIATION_MAX_INTENSITY)

# ======================================================================================================================
# HELPER FUNCTIONS
# ======================================================================================================================
def heuristic(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def bresenham_line(x0, y0, x1, y1):
    """
    Return the grid points along a line in (row,col) form.
    """
    rr, cc = line(y0, x0, y1, x1)
    return rr, cc

# Simple utility for plotting occupancy map with an optional overlay
def plot_map(ax, occupancy_map, overlay_map=None, cmap="seismic",
             alpha=0.5, title="", colorbar_label="", vmin=0, vmax=100):
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

# ======================================================================================================================
# GPR HANDLER
# ======================================================================================================================
class GPRModelHandler:
    def __init__(self, model, radiation_threshold, safety_factor=GPR_SAFETY_FACTOR, max_points=MAX_GPR_POINTS,
                 use_safety_factor=USE_SAFETY_FACTOR):
        self.model = model
        self.max_points = max_points
        self.positions = []
        self.readings = []
        self.radiation_threshold = radiation_threshold
        self.safety_factor = safety_factor
        self.use_safety_factor = use_safety_factor

    def add_readings(self, new_positions, new_readings):
        self.positions.extend(new_positions)
        self.readings.extend(new_readings)
        # Keep only the most recent max_points
        if len(self.positions) > self.max_points:
            self.positions = self.positions[-self.max_points:]
            self.readings = self.readings[-self.max_points:]

    def update(self, radiation_map):
        if len(self.positions) < 1:
            return np.zeros_like(radiation_map), np.zeros_like(radiation_map)

        X_train = np.array(self.positions)
        y_train = np.array(self.readings)

        # Fit GPR
        try:
            self.model.fit(X_train, y_train)
        except Exception as e:
            logging.warning(f"GPR fitting failed: {e}")
            return np.zeros_like(radiation_map), np.zeros_like(radiation_map)

        # Predict on the entire grid
        rows, cols = radiation_map.shape
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, MAP_SIZE, cols),
            np.linspace(0, MAP_SIZE, rows)
        )
        X_pred = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        try:
            y_pred, y_std = self.model.predict(X_pred, return_std=True)
            estimated_map = y_pred.reshape(radiation_map.shape)
            uncertainty_map = y_std.reshape(radiation_map.shape)
        except Exception as e:
            logging.warning(f"GPR predict failed: {e}")
            return np.zeros_like(radiation_map), np.zeros_like(radiation_map)

        if self.use_safety_factor:
            estimated_map += self.safety_factor * uncertainty_map

        return np.clip(estimated_map, 0, RADIATION_MAX_INTENSITY), self.radiation_threshold

# ======================================================================================================================
# VISIBILITY UTILS
# ======================================================================================================================
def generate_visibility_map(occupancy_map, visited_points, lidar_range, resolution):
    """
    Returns a binary map of 'visible' cells given the visited_points as sensor origins.
    """
    vis_map = np.zeros_like(occupancy_map)
    num_rays = 36
    angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
    max_steps = int(lidar_range / resolution)

    for (xx, yy) in visited_points:
        sx = int(xx / resolution)
        sy = int(yy / resolution)
        for ang in angles:
            dx, dy = np.cos(ang), np.sin(ang)
            for r in range(1, max_steps):
                xi = sx + int(dx * r)
                yi = sy + int(dy * r)
                if (xi < 0 or xi >= occupancy_map.shape[1] or
                    yi < 0 or yi >= occupancy_map.shape[0]):
                    break
                if occupancy_map[yi, xi] == 0:
                    break
                vis_map[yi, xi] = 1
    return vis_map

def calculate_distance_map(occ_map, vis_map):
    """
    Downsample & compute distance transform on free+visible area.
    """
    v_down = block_reduce(vis_map, (DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR), np.max)
    o_down = block_reduce(occ_map, (DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR), np.min)
    dist_map = distance_transform_edt(o_down * v_down) * (RESOLUTION * DOWNSAMPLE_FACTOR)
    return dist_map, v_down

# ======================================================================================================================
# PATH CLEAR CHECK & LINE-OF-SIGHT
# ======================================================================================================================
def is_line_clear(p1, p2, occupancy_map, rad_map, threshold):
    g1x = int(round(p1[0] / RESOLUTION))
    g1y = int(round(p1[1] / RESOLUTION))
    g2x = int(round(p2[0] / RESOLUTION))
    g2y = int(round(p2[1] / RESOLUTION))

    rr, cc = bresenham_line(g1x, g1y, g2x, g2y)
    valid = (cc >= 0) & (cc < occupancy_map.shape[1]) & (rr >= 0) & (rr < occupancy_map.shape[0])
    rr, cc = rr[valid], cc[valid]

    # Check walls
    if np.any(occupancy_map[rr, cc] == 0):
        return False

    # Check radiation
    if np.any(rad_map[rr, cc] >= threshold):
        return False

    return True

def is_path_clear_grid(gx0, gy0, gx1, gy1, rad_down, rad_threshold, occ_map):
    # Bresenham in grid coords
    rr, cc = bresenham_line(gx0, gy0, gx1, gy1)
    valid_idx = (cc >= 0) & (cc < occ_map.shape[1]) & (rr >= 0) & (rr < occ_map.shape[0])
    rr, cc = rr[valid_idx], cc[valid_idx]
    # Occupancy check
    if np.any(occ_map[rr, cc] == 0):
        return False
    # Radiation check
    rr_d = np.clip(rr // DOWNSAMPLE_FACTOR, 0, rad_down.shape[0] - 1)
    cc_d = np.clip(cc // DOWNSAMPLE_FACTOR, 0, rad_down.shape[1] - 1)
    path_rad = rad_down[rr_d, cc_d]
    if np.any(path_rad >= rad_threshold):
        return False
    return True

# ======================================================================================================================
# PATH PLANNING (A-STAR)
# ======================================================================================================================
def heuristic_grid(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def _reconstruct_path(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    return path[::-1]

def movement_cost_grid(node, rad_map_down, rad_weight, rad_cost_weight=RADIATION_COST_WEIGHT):
    """
    Base cost = 1, plus small penalty for higher radiation.
    """
    x_down = node[0] // DOWNSAMPLE_FACTOR
    y_down = node[1] // DOWNSAMPLE_FACTOR
    x_down = min(x_down, rad_map_down.shape[1] - 1)
    y_down = min(y_down, rad_map_down.shape[0] - 1)
    rad_int = rad_map_down[y_down, x_down]
    penalty = rad_cost_weight * (rad_int / RADIATION_MAX_INTENSITY)
    return 1 + penalty

def validate_path(path, rad_map, threshold):
    """
    Ensure every point on the path is below threshold.
    """
    for px, py in path:
        gx = int(round(px / RESOLUTION))
        gy = int(round(py / RESOLUTION))
        if gx < 0 or gx >= rad_map.shape[1] or gy < 0 or gy >= rad_map.shape[0]:
            return False
        if rad_map[gy, gx] >= threshold:
            return False
    return True

def smooth_path(path, occupancy_map, rad_map, threshold):
    """
    Remove intermediate nodes if direct line-of-sight is free.
    """
    if len(path) <= 2:
        return path.copy()

    new_path = [path[0]]
    cur_idx = 0
    while cur_idx < len(path) - 1:
        next_idx = len(path) - 1
        for i in range(len(path) - 1, cur_idx, -1):
            if is_line_clear(new_path[-1], path[i], occupancy_map, rad_map, threshold):
                next_idx = i
                break
        new_path.append(path[next_idx])
        cur_idx = next_idx

    return new_path

def a_star(start, goal, connections, rad_map_down, occ_map, rad_threshold, full_rad_map, rad_weight,
           rad_cost_weight=RADIATION_COST_WEIGHT):
    """
    Graph-based A* using the node connections.
    start, goal in continuous coords; connections is a dict keyed by grid coords.
    """
    start_grid = (int(round(start[0]/RESOLUTION)), int(round(start[1]/RESOLUTION)))
    goal_grid = (int(round(goal[0]/RESOLUTION)), int(round(goal[1]/RESOLUTION)))

    if start_grid not in connections or goal_grid not in connections:
        logging.warning("Start or goal node not in the graph. Pathfinding aborted.")
        return []

    open_set = []
    heapq.heappush(open_set, (0, start_grid))
    came_from = {start_grid: None}
    g_score = {start_grid: 0}
    f_score = {start_grid: heuristic_grid(start_grid, goal_grid)}

    while open_set:
        current_f, current = heapq.heappop(open_set)
        if current == goal_grid:
            path = _reconstruct_path(came_from, current)
            path = [(p[0]*RESOLUTION, p[1]*RESOLUTION) for p in path]
            # Validate
            if validate_path(path, full_rad_map, rad_threshold):
                return smooth_path(path, occ_map, full_rad_map, rad_threshold)
            else:
                logging.warning("Planned path violates radiation constraints.")
                return []

        for neighbor in connections.get(current, []):
            nx_down = neighbor[0] // DOWNSAMPLE_FACTOR
            ny_down = neighbor[1] // DOWNSAMPLE_FACTOR
            nx_down = min(nx_down, rad_map_down.shape[1]-1)
            ny_down = min(ny_down, rad_map_down.shape[0]-1)

            # If neighbor is too high in radiation, skip
            if rad_map_down[ny_down, nx_down] >= rad_threshold:
                continue

            cost = movement_cost_grid(neighbor, rad_map_down, rad_weight, rad_cost_weight)
            tentative_g = g_score[current] + cost
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                n_pos = (neighbor[0]*RESOLUTION, neighbor[1]*RESOLUTION)
                f_val = tentative_g + heuristic(n_pos, goal)
                if neighbor not in f_score or f_val < f_score[neighbor]:
                    f_score[neighbor] = f_val
                    heapq.heappush(open_set, (f_val, neighbor))

    logging.warning("No valid path found.")
    return []

# ======================================================================================================================
# NODE SELECTION
# ======================================================================================================================
def select_nodes_dynamic(distance_map, vis_downsampled, rad_down, dyn_rad_limit, start_pos):
    """
    Identify candidate frontiers based on distance from obstacles, visibility, and radiation constraints.
    """
    eligible = np.argwhere(
        (distance_map >= MIN_RADIUS_THRESHOLD) &
        (vis_downsampled == 1) &
        (rad_down < dyn_rad_limit)
    )
    if eligible.size == 0:
        return []

    # Sort them by distance_map descending (prefer more open regions first)
    radii = distance_map[eligible[:, 0], eligible[:, 1]]
    sorted_idx = np.argsort(-radii)
    sorted_pts = eligible[sorted_idx] * DOWNSAMPLE_FACTOR

    chosen_nodes = []
    used_grids = set()

    for pt in sorted_pts:
        gx, gy = int(pt[1]), int(pt[0])
        if (gx, gy) in used_grids:
            continue
        x_m = gx * RESOLUTION
        y_m = gy * RESOLUTION
        # Skip if too close to existing node
        if any(heuristic((x_m, y_m), (sx, sy)) < FILTER_DISTANCE for sx, sy, _ in chosen_nodes):
            continue

        # Distance from obstacle in meters
        radius_val = distance_map[gy // DOWNSAMPLE_FACTOR, gx // DOWNSAMPLE_FACTOR] / (RESOLUTION * DOWNSAMPLE_FACTOR)
        chosen_nodes.append((round(x_m, 2), round(y_m, 2), radius_val))
        used_grids.add((gx, gy))

    # Force-include robot's start node if it's safe
    start_gx = int(round(start_pos[0] / RESOLUTION))
    start_gy = int(round(start_pos[1] / RESOLUTION))
    if (0 <= start_gx < rad_down.shape[1] and
        0 <= start_gy < rad_down.shape[0] and
        rad_down[start_gy, start_gx] < dyn_rad_limit):
        # Insert if not already present
        if not any(heuristic(start_pos, (nx, ny)) < 1e-6 for nx, ny, _ in chosen_nodes):
            chosen_nodes.append((start_pos[0], start_pos[1], 0.0))

    return chosen_nodes

def create_connections(nodes, rad_down, rad_threshold, rad_weight, occ_map, radius=CONNECTION_RADIUS,
                       rad_cost_weight=RADIATION_COST_WEIGHT):
    """
    Build edges among nodes using a KDTree, checking line-of-sight with occupancy & radiation constraints.
    """
    if not nodes:
        return {}

    node_dict = {
        (int(round(x / RESOLUTION)), int(round(y / RESOLUTION))): (x, y) for (x, y, _) in nodes
    }
    grid_list = np.array(list(node_dict.keys()))
    kd_tree = KDTree(grid_list)

    connections = {key: [] for key in node_dict.keys()}
    max_cells = int(radius / RESOLUTION)

    for i, (gx, gy) in enumerate(node_dict.keys()):
        neighbors_idx = kd_tree.query_radius([[gx, gy]], r=max_cells)[0]
        valids = []
        for ni in neighbors_idx:
            if ni == i:
                continue
            ng = tuple(grid_list[ni])
            dist_m = heuristic_grid((gx, gy), ng) * RESOLUTION
            # Filter by min distance
            if dist_m < FILTER_DISTANCE:
                continue
            if not is_path_clear_grid(gx, gy, ng[0], ng[1], rad_down, rad_threshold, occ_map):
                continue
            valids.append(ng)
        # Sort by ascending distance, limit to MAX_NEIGHBORS
        valids = sorted(valids, key=lambda n: heuristic_grid((gx, gy), n))[:MAX_NEIGHBORS]
        for v in valids:
            connections[(gx, gy)].append(v)
            connections.setdefault(v, []).append((gx, gy))

    for key in list(connections.keys()):
        connections[key] = list(set(connections[key]))
    iso = [k for k, v in connections.items() if len(v) == 0]
    for k in iso:
        del connections[k]

    return connections

def ensure_connectivity(nodes, connections):
    """
    Keep only the largest connected component of the graph.
    """
    if not nodes:
        return nodes
    adjacency = {k: set(v) for k, v in connections.items()}
    visited = set()
    components = []

    for k in adjacency:
        if k not in visited:
            stack = [k]
            comp = []
            while stack:
                cur = stack.pop()
                if cur not in visited:
                    visited.add(cur)
                    comp.append(cur)
                    stack.extend(adjacency[cur] - visited)
            components.append(comp)

    if not components:
        return nodes

    largest = max(components, key=len)

    node_map = {
        (int(round(x / RESOLUTION)), int(round(y / RESOLUTION))): (x, y, rad)
        for (x, y, rad) in nodes
    }

    keep_nodes = []
    for key in largest:
        x, y, rd = node_map[key]
        keep_nodes.append((x, y, rd))

    return keep_nodes

def identify_best_frontier(nodes, visibility_map, curr_pos, connections,
                           rad_down, occ_map, rad_map, rad_threshold):
    """
    Picks the best frontier node based on a heuristic (info gain, distance, radiation).
    """
    best_node = None
    best_score = -float('inf')

    for node in nodes:
        nx, ny, _ = node
        dist_to_robot = heuristic(curr_pos, (nx, ny))
        if dist_to_robot < MIN_FRONTIER_DISTANCE:
            continue

        # We must have a path
        path = a_star(
            curr_pos,
            (nx, ny),
            connections,
            rad_down,
            occ_map,
            rad_threshold,
            rad_map,
            rad_weight=0,
            rad_cost_weight=RADIATION_COST_WEIGHT
        )
        if len(path) == 0:
            continue

        # Info gain in a local neighborhood
        gx = int(round(nx / RESOLUTION))
        gy = int(round(ny / RESOLUTION))
        window_sz = 2  # small local window
        y_min = max(0, gy - window_sz)
        y_max = min(gy + window_sz + 1, visibility_map.shape[0])
        x_min = max(0, gx - window_sz)
        x_max = min(gx + window_sz + 1, visibility_map.shape[1])
        local_vis = visibility_map[y_min:y_max, x_min:x_max]
        unexplored = np.sum(local_vis == 0)

        # A simple radiation penalty from the downsampled map
        yd_min = y_min // DOWNSAMPLE_FACTOR
        yd_max = y_max // DOWNSAMPLE_FACTOR
        xd_min = x_min // DOWNSAMPLE_FACTOR
        xd_max = x_max // DOWNSAMPLE_FACTOR
        sub = rad_down[yd_min:yd_max, xd_min:xd_max]
        avg_rad = np.mean(sub) if sub.size else 0
        penalty = avg_rad * 0.5

        prox_bonus = PROXIMITY_BONUS / (dist_to_robot + epsilon)
        score = (INFO_GAIN_WEIGHT * unexplored) + prox_bonus - (RADIATION_COST_WEIGHT * penalty)

        if score >= 0 and score > best_score:
            best_score = score
            best_node = (nx, ny)

    return best_node

def simulate_radiation_readings_around_robot(robot_position, radiation_map,
                                             sensor_positions, sensor_readings):
    """
    Simulate a single reading at the robot's grid cell (plus any local area if RADIATION_SAMPLING_RADIUS>0).
    """
    new_positions = []
    new_readings = []
    x, y = robot_position
    gx = int(round(x / RESOLUTION))
    gy = int(round(y / RESOLUTION))

    # If RADIATION_SAMPLING_RADIUS > 0, we could sample a small area around the robot
    radius_in_cells = int(np.ceil(RADIATION_SAMPLING_RADIUS / RESOLUTION))
    for dy in range(-radius_in_cells, radius_in_cells + 1):
        for dx in range(-radius_in_cells, radius_in_cells + 1):
            nx, ny = gx + dx, gy + dy
            dist = sqrt(dx**2 + dy**2) * RESOLUTION
            if dist > RADIATION_SAMPLING_RADIUS:
                continue
            if 0 <= nx < radiation_map.shape[1] and 0 <= ny < radiation_map.shape[0]:
                val = radiation_map[ny, nx]
                pos = (nx * RESOLUTION, ny * RESOLUTION)
                # Only add if not already recorded
                if pos not in sensor_positions:
                    sensor_positions.append(pos)
                    sensor_readings.append(val)
                    new_positions.append(pos)
                    new_readings.append(val)

    # If the sampling radius is 0, we still record the single robot cell
    if RADIATION_SAMPLING_RADIUS <= 0:
        if (gx >= 0 and gx < radiation_map.shape[1] and
            gy >= 0 and gy < radiation_map.shape[0]):
            val = radiation_map[gy, gx]
            pos = (gx * RESOLUTION, gy * RESOLUTION)
            if pos not in sensor_positions:
                sensor_positions.append(pos)
                sensor_readings.append(val)
                new_positions.append(pos)
                new_readings.append(val)

    return new_positions, new_readings

# ======================================================================================================================
# EXPLORATION LOOP
# ======================================================================================================================
def exploration_loop(occupancy_map, radiation_map, start_pos, rad_threshold, rad_weight, use_gpr=True):
    visited = [start_pos]

    # GPR initialization
    if use_gpr:
        model = GaussianProcessRegressor(
            kernel=GPR_KERNEL,
            n_restarts_optimizer=GPR_RESTART_ITERATIONS,
            alpha=1e-2,
            normalize_y=True
        )
        gpr_handler = GPRModelHandler(
            model, rad_threshold, use_safety_factor=USE_SAFETY_FACTOR
        )
        initial_val = radiation_map[
            int(round(start_pos[1]/RESOLUTION)),
            int(round(start_pos[0]/RESOLUTION))
        ]
        gpr_handler.add_readings([start_pos], [initial_val])
        gpr_estimated_map, _ = gpr_handler.update(radiation_map)
        sensor_positions = []
        sensor_readings = []
    else:
        gpr_handler = None
        gpr_estimated_map = np.zeros_like(radiation_map)
        sensor_positions = []
        sensor_readings = []

    stages = []
    current_pos = start_pos
    iteration = 0
    path_planning_failures = 0
    max_path_failures = 3
    buffer_readings = []

    while iteration < MAX_STAGES:
        iteration += 1
        # Optional dynamic threshold
        dynamic_threshold = rad_threshold  # Keeping it simple here

        # Update GPR with any buffered sensor readings
        if use_gpr and buffer_readings:
            newp, newr = zip(*buffer_readings) if buffer_readings else ([], [])
            gpr_handler.add_readings(newp, newr)
            gpr_estimated_map, _ = gpr_handler.update(radiation_map)
            buffer_readings = []

        # Update visibility
        vis_map = generate_visibility_map(occupancy_map, visited, LIDAR_RANGE, RESOLUTION)
        dist_map, vis_down = calculate_distance_map(occupancy_map, vis_map)
        if use_gpr:
            gpr_down = block_reduce(gpr_estimated_map, (DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR), np.max)
        else:
            gpr_down = np.zeros_like(dist_map)

        # Select nodes
        nodes = select_nodes_dynamic(dist_map, vis_down, gpr_down, dynamic_threshold, current_pos)
        if not nodes:
            logging.warning("No eligible nodes found. Exploration ends.")
            break

        # Create connections
        graph = create_connections(nodes, gpr_down, dynamic_threshold, rad_weight, occupancy_map)
        connected_nodes = ensure_connectivity(nodes, graph)
        if len(connected_nodes) != len(nodes):
            cdict = {
                (int(round(x / RESOLUTION)), int(round(y / RESOLUTION))): (x, y)
                for (x, y, _) in connected_nodes
            }
            graph = {
                k: [n for n in neigh if n in cdict]
                for k, neigh in graph.items()
                if k in cdict
            }
        nodes = connected_nodes
        if not nodes:
            logging.warning("No connected nodes after connectivity filter. Exploration ends.")
            break

        # Identify best frontier
        frontier = identify_best_frontier(nodes, vis_map, current_pos, graph,
                                          gpr_down, occupancy_map, radiation_map, dynamic_threshold)
        if not frontier:
            logging.warning("No frontier identified. Exploration ends.")
            break

        # A* path to frontier
        path = a_star(
            current_pos,
            frontier,
            graph,
            gpr_down,
            occupancy_map,
            dynamic_threshold,
            radiation_map,
            rad_weight,
            rad_cost_weight=RADIATION_COST_WEIGHT
        )
        if not path:
            logging.warning("Path planning failed. Skipping iteration.")
            path_planning_failures += 1
            if path_planning_failures >= max_path_failures:
                logging.warning("Too many path failures. Aborting.")
                return stages
            continue

        if len(path) <= 1:
            logging.warning("Path length too short. Ending.")
            break

        path_planning_failures = 0

        # Follow path in discrete steps
        for step in path[1:]:
            visited.append(step)
            current_pos = step
            # Simulate GPR reading
            if use_gpr:
                newp, newr = simulate_radiation_readings_around_robot(
                    current_pos, radiation_map,
                    sensor_positions, sensor_readings
                )
                buffer_readings.extend(zip(newp, newr))

        # Record stage
        stages.append((
            vis_map.copy(),
            connected_nodes.copy(),
            graph.copy(),
            path.copy(),
            path[0],
            path[-1],
            f"Iteration {iteration}",
            dynamic_threshold,
            gpr_estimated_map.copy() if use_gpr else np.zeros_like(radiation_map)
        ))

    # Final GPR update
    if use_gpr and buffer_readings:
        newp, newr = zip(*buffer_readings) if buffer_readings else ([], [])
        gpr_handler.add_readings(newp, newr)
        gpr_estimated_map, _ = gpr_handler.update(radiation_map)

    # Add a final stage to store the final GPR map
    if use_gpr:
        stages.append((
            vis_map.copy() if 'vis_map' in locals() else np.zeros_like(occupancy_map),
            connected_nodes.copy() if 'connected_nodes' in locals() else [],
            graph.copy() if 'graph' in locals() else {},
            [current_pos],
            current_pos,
            current_pos,
            "Final",
            dynamic_threshold if 'dynamic_threshold' in locals() else rad_threshold,
            gpr_estimated_map.copy()
        ))

    return stages

# ======================================================================================================================
# MAIN: SINGLE RUN
# ======================================================================================================================
def main():
    logging.warning("=== Single Exploration Run ===")
    # 1) Load occupancy map
    occ_map = OccupancyMap.load(MAP_PATH)

    # 2) Compute radiation map
    rad_map = RadiationMap.calculate(
        occupancy_map=occ_map,
        sources=STATIC_SOURCES,
        resolution=RESOLUTION,
        lidar_range=LIDAR_RANGE
    )

    # 3) Run exploration loop (with GPR)
    radiation_threshold = 30
    radiation_weight = 10
    stages = exploration_loop(
        occupancy_map=occ_map,
        radiation_map=rad_map,
        start_pos=START_POSITION,
        rad_threshold=radiation_threshold,
        rad_weight=radiation_weight,
        use_gpr=True
    )

    if not stages:
        logging.warning("No exploration stages were produced.")
        return

    # Retrieve the final stage
    final_stage = stages[-1]
    # final_stage structure:
    # (visibility_map, nodes, connections, path, start, end, label, dynamic_threshold, gpr_map)
    visibility_map, nodes, connections, final_path, _, _, stage_label, _, final_gpr_map = final_stage

    # 4) Plot a single figure showing:
    #    - Occupancy map
    #    - GPR estimated map as overlay
    #    - Node connections
    #    - Nodes
    #    - Final path
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_map(
        ax,
        occupancy_map=occ_map,
        overlay_map=final_gpr_map,
        cmap="seismic",
        alpha=0.6,
        title="Final GPR Estimate + Node Network",
        colorbar_label="Estimated Radiation",
        vmin=0,
        vmax=RADIATION_MAX_INTENSITY
    )

    # Plot connections in gray
    for (gx0, gy0), neighs in connections.items():
        x0 = gx0 * RESOLUTION
        y0 = gy0 * RESOLUTION
        for (gx1, gy1) in neighs:
            x1 = gx1 * RESOLUTION
            y1 = gy1 * RESOLUTION
            ax.plot([x0, x1], [y0, y1], 'k-', linewidth=0.5, alpha=0.5)

    # Plot nodes on top
    if nodes:
        nx, ny, _ = zip(*nodes)
        ax.scatter(nx, ny, c='cyan', edgecolors='black', s=60, label="Nodes")

    # Plot the final path (blue line)
    if final_path and len(final_path) > 1:
        px, py = zip(*final_path)
        ax.plot(px, py, 'b-', linewidth=2, label="Final Path")

    # Mark start and end
    ax.plot(START_POSITION[0], START_POSITION[1], 'go', markersize=10, label="Start")
    if final_path:
        ax.plot(final_path[-1][0], final_path[-1][1], 'ro', markersize=10, label="End")

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()