import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import time

# Start the timer
start_time = time.time()

# Load occupancy grid
occupancy_grid = np.genfromtxt('../occupancy_map_output/full_occupancy_map.csv', delimiter=',')

# Grid dimensions
grid_height, grid_width = occupancy_grid.shape
map_size = 50  # meters

# Set coordinates so that the origin is at the bottom-left corner
x_min, x_max = 0, map_size
y_min, y_max = 0, map_size

# Parameters
buffer_distance_sources = 30
max_radiation_intensity = 100  # For normalization

def is_free_space(x, y, occupancy_grid):
    """Check if (x, y) corresponds to free space."""
    x_idx = int((x - x_min) * grid_width / (x_max - x_min))
    y_idx = int((y - y_min) * grid_height / (y_max - y_min))
    return 0 <= x_idx < grid_width and 0 <= y_idx < grid_height and occupancy_grid[y_idx, x_idx] == 1.0

def generate_sources(num_sources, occupancy_grid):
    """Generate sources within free space."""
    sources = []
    while len(sources) < num_sources:
        x, y = np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)
        if is_free_space(x, y, occupancy_grid):
            intensity = np.random.uniform(100000, 500000)  # High intensity for sources
            sources.append((x, y, intensity))
    return np.array(sources)

def calculate_radiation_intensity(X, Y, sources, occupancy_grid):
    """Calculate radiation intensity over a grid using vectorized operations."""
    intensities = np.zeros_like(X)
    free_space_mask = occupancy_grid == 1
    for x_src, y_src, intensity in sources:
        distances = np.sqrt((X - x_src) ** 2 + (Y - y_src) ** 2)
        intensities += np.where(free_space_mask, intensity / (distances ** 2 + 1e-9), 0)
    return intensities

# Generate sources
num_sources = 5
sources = generate_sources(num_sources, occupancy_grid)

# Create grid for radiation calculation
x_grid = np.linspace(x_min, x_max, grid_width)
y_grid = np.linspace(y_min, y_max, grid_height)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# Calculate radiation intensity
true_radiation_grid = calculate_radiation_intensity(X_grid, Y_grid, sources, occupancy_grid)
true_radiation_normalized = np.clip(true_radiation_grid / max_radiation_intensity, 0, 1)

# Custom colormap
inferno_r = cm.get_cmap('inferno_r', 256)
newcolors = inferno_r(np.linspace(0, 1, 256))
newcolors[:20, :] = mcolors.to_rgba('white')  # White for low values
custom_cmap = mcolors.ListedColormap(newcolors)

# Plotting
plt.figure(figsize=(8, 6))

# Occupancy grid as background
plt.imshow(np.flipud(occupancy_grid), cmap='gray', extent=[x_min, x_max, y_min, y_max], alpha=0.6)

# Radiation heatmap
plt.imshow(np.flipud(true_radiation_normalized), cmap=custom_cmap, extent=[x_min, x_max, y_min, y_max], alpha=0.6)

# Overlay black walls
wall_mask = np.flipud(occupancy_grid) == 0
plt.imshow(np.where(wall_mask, 0, np.nan), cmap=mcolors.ListedColormap(['black']),
           extent=[x_min, x_max, y_min, y_max], alpha=1)

# Mark sources
for x_src, y_src, _ in sources:
    plt.scatter(x_src, y_src, color='red', s=120, marker='x', label="Source")

# Add colorbar and labels
plt.colorbar(label="True Radiation Intensity")
plt.title("True Radiation Distribution (Known Sources)")
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")

# Set axis limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.tight_layout()
plt.show()

# Execution time
execution_time = time.time() - start_time
print(f"Execution time: {execution_time:.4f} seconds")
