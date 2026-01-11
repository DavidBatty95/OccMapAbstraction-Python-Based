import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

# Start the timer
start_time = time.time()

# Load the occupancy grid from a CSV file (without printing it out)
occupancy_grid = np.genfromtxt(
    '/Users/david/Documents/MLTest/MLExplorationTest/occupancy_map_output/occupancy_grid.csv', delimiter=',')

# Get the grid dimensions
grid_height, grid_width = occupancy_grid.shape
x_min, x_max = -grid_width / 2, grid_width / 2
y_min, y_max = -grid_height / 2, grid_height / 2

# Define buffer distances
buffer_distance_sources = 30
max_radiation_intensity = 100  # Maximum radiation intensity to normalize


def is_free_space(x_idx, y_idx, occupancy_grid):
    """Check if a given index corresponds to free space in the occupancy grid."""
    return occupancy_grid[y_idx, x_idx] == 1.0


# Generate random points with intensity
def generate_sources(num_sources, buffer_distance_sources, occupancy_grid):
    sources = []
    for i in range(num_sources):
        x, y = np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)
        while not is_free_space(int((x - x_min) * grid_width / (x_max - x_min)),
                                int((y - y_min) * grid_height / (y_max - y_min)), occupancy_grid):
            x, y = np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)
        intensity = np.random.uniform(5000, 20000)  # Increased intensity range for wider reach
        sources.append((x, y, intensity))
    return np.array(sources)


# Place 5 sources inside free space with a 30-cell buffer
num_sources = 5
sources = generate_sources(num_sources, buffer_distance_sources, occupancy_grid)


# Vectorized calculation of distances between points
def calculate_radiation_intensity_vectorized(X, Y, sources, occupancy_grid):
    """Calculate radiation intensity over a grid, vectorized for speed."""
    intensities = np.zeros_like(X)
    for source in sources:
        # Calculate the distance between each grid point and the source
        distances = np.sqrt((X - source[0]) ** 2 + (Y - source[1]) ** 2)

        # Avoid recalculating line of sight by masking invalid free space
        free_space_mask = occupancy_grid == 1

        # Apply inverse square law only to points with valid free space
        intensities += np.where(free_space_mask, source[2] / (distances ** 2 + 1e-9), 0)

    return intensities


# Create a grid of points across the entire map for the heatmap
x_grid = np.linspace(x_min, x_max, grid_width)
y_grid = np.linspace(y_min, y_max, grid_height)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# Calculate radiation intensity across the entire grid using vectorization
intensity_grid = calculate_radiation_intensity_vectorized(X_grid, Y_grid, sources, occupancy_grid)

# Normalize the intensity for color mapping
normalized_intensity_grid = np.clip(intensity_grid / max_radiation_intensity, 0, 1)

# Plot the radiation intensity heatmap
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the occupancy grid
ax.imshow(np.flipud(occupancy_grid), cmap='gray', extent=[x_min, x_max, y_min, y_max], alpha=0.6)

# Overlay the radiation intensity heatmap
heatmap = ax.imshow(np.flipud(normalized_intensity_grid), cmap='inferno', extent=[x_min, x_max, y_min, y_max],
                    alpha=0.6)

# Plot sources
for source in sources:
    ax.scatter(source[0], source[1], color='red', s=120, marker='x', label="Source")

# Add a colorbar to indicate intensity
plt.colorbar(heatmap, ax=ax, label="Radiation Intensity")

ax.set_title('Radiation Distribution Based on Point Sources')
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()

# End the timer and print execution time
execution_time = time.time() - start_time
print(f"Execution time: {execution_time:.4f} seconds")