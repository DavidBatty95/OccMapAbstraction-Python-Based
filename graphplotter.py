# ======================================================================================================================
# TITLE: OCC MAP ABSTRACTION EXPLORATION w. ONLINE GPR - Mean Cumulative Radiation Dose Visualization without Trend Lines
# ======================================================================================================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_paired_runs(output_dir):
    # Regular expressions to match filenames
    pattern_with = re.compile(r'^run(\d+)_with_radiation\.csv$')
    pattern_without = re.compile(r'^run(\d+)_without_radiation\.csv$')

    runs_with = set()
    runs_without = set()

    # List all files in the directory
    try:
        files = os.listdir(output_dir)
    except FileNotFoundError:
        print(f"Error: Directory '{output_dir}' not found.")
        return []

    # Identify run numbers for each scenario
    for file in files:
        match_with = pattern_with.match(file)
        match_without = pattern_without.match(file)
        if match_with:
            runs_with.add(int(match_with.group(1)))
        if match_without:
            runs_without.add(int(match_without.group(1)))

    # Find common run numbers that have both files
    paired_runs = sorted(runs_with.intersection(runs_without))

    if not paired_runs:
        print(f"No paired runs found in the directory '{output_dir}'.")
    else:
        print(f"Found {len(paired_runs)} paired run(s) in '{output_dir}'.")

    return paired_runs

def read_csv_data(filepath):
    try:
        df = pd.read_csv(filepath)
        if 'cumulative_distance' not in df.columns or 'cumulative_dose' not in df.columns:
            raise ValueError("CSV file must contain 'cumulative_distance' and 'cumulative_dose' columns.")
        cumulative_distance = df['cumulative_distance'].values
        cumulative_dose = df['cumulative_dose'].values

        # Ensure cumulative_dose is non-negative
        if np.any(cumulative_dose < 0):
            print(f"Warning: Negative cumulative_dose values found in '{filepath}'. These will be set to zero.")
            cumulative_dose = np.clip(cumulative_dose, a_min=0, a_max=None)

        return cumulative_distance, cumulative_dose
    except Exception as e:
        print(f"Error reading '{filepath}': {e}")
        return None, None

def ensure_monotonic(dose_array):
    return np.maximum.accumulate(dose_array)

def interpolate_and_crop(common_distance, run_distance, run_dose, average_length):
    try:
        # Ensure that run_distance is strictly increasing
        if not np.all(np.diff(run_distance) > 0):
            raise ValueError("cumulative_distance must be strictly increasing.")

        # Determine the maximum distance to interpolate (average_length)
        max_distance = average_length

        # Create a cropped distance and dose up to average_length
        mask = run_distance <= max_distance
        if not np.any(mask):
            print("Warning: Run's cumulative_distance does not reach the average_length. Using entire run.")
            cropped_distance = run_distance
            cropped_dose = run_dose
        else:
            last_valid_index = np.max(np.where(mask)[0])
            cropped_distance = run_distance[:last_valid_index + 1]
            cropped_dose = run_dose[:last_valid_index + 1]

        # Interpolate the cropped data onto the common_distance grid
        interpolated_dose = np.interp(common_distance, cropped_distance, cropped_dose, left=0, right=np.nan)

        # Replace NaNs with the last valid dose value using pandas' forward fill
        interpolated_dose = pd.Series(interpolated_dose).ffill().values  # Updated line to eliminate FutureWarning

        # Ensure no negative values
        interpolated_dose = np.clip(interpolated_dose, a_min=0, a_max=None)

        # Ensure monotonicity
        interpolated_dose = ensure_monotonic(interpolated_dose)

        return interpolated_dose
    except Exception as e:
        print(f"Interpolation and cropping error: {e}")
        return None

def moving_average(x, window_size=5):
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(x, window, 'same')

def average_runs(interpolated_doses):
    if not interpolated_doses:
        return None
    dose_matrix = np.array(interpolated_doses)  # Shape: (num_runs, num_points)

    # Check if all slices are NaN
    if np.all(np.isnan(dose_matrix)):
        print("Warning: All dose values are NaN. Cannot compute mean.")
        return None

    average_dose = np.nanmean(dose_matrix, axis=0)
    return average_dose

def validate_data(interpolated_doses, scenario, dataset_label):
    if not interpolated_doses:
        print(f"[{dataset_label}] No data to validate for scenario '{scenario}'.")
        return False
    dose_matrix = np.array(interpolated_doses)
    if np.any(dose_matrix < 0):
        print(f"[{dataset_label}] Warning: Negative dose values found in scenario '{scenario}'. These will be set to zero.")
        dose_matrix = np.clip(dose_matrix, a_min=0, a_max=None)
        interpolated_doses[:] = dose_matrix.tolist()
        return False
    return True

def plot_mean_cumulative_dose(common_distance_with, average_with,
                             common_distance_without, average_without,
                             dataset_label,
                             save_path=None, smooth_window=1):
    plt.figure(figsize=(14, 8))

    # Initialize variables to store dose rates
    dose_rate_with = None
    dose_rate_without = None

    # Plot mean without radiation (red)
    if average_without is not None:
        plt.plot(common_distance_without, average_without, label=f'No Radiation Awareness Mean ({dataset_label})',
                 color='red', linewidth=2)
        # Get the final point
        final_distance_without = common_distance_without[-1]
        final_dose_without = average_without[-1]
        # Plot a large circle at the final point
        plt.scatter(final_distance_without, final_dose_without, color='red', s=100, zorder=5,
                    edgecolors='black', label=f'No Radiation Awareness Total ({dataset_label})')
        # Draw dashed lines to the axes
        plt.plot([final_distance_without, final_distance_without], [0, final_dose_without],
                 color='red', linestyle='--', linewidth=1)
        plt.plot([0, final_distance_without], [final_dose_without, final_dose_without],
                 color='red', linestyle='--', linewidth=1)
        # Calculate and prepare trend line through origin
        # Using linear regression without intercept: slope = sum(x*y) / sum(x^2)
        slope_without = np.sum(common_distance_without * average_without) / np.sum(common_distance_without**2)
        trend_within_without = slope_without * common_distance_without
        # Store dose rate (slope)
        dose_rate_without = slope_without

    # Plot mean with radiation (blue)
    if average_with is not None:
        # Apply smoothing if desired
        smoothed_average_with = moving_average(average_with, window_size=smooth_window)
        plt.plot(common_distance_with, smoothed_average_with, label=f'Radiation Awareness Mean ({dataset_label})',
                 color='blue', linewidth=2)
        # Get the final point
        final_distance_with = common_distance_with[-1]
        final_dose_with = smoothed_average_with[-1]
        # Plot a large circle at the final point
        plt.scatter(final_distance_with, final_dose_with, color='blue', s=100, zorder=5,
                    edgecolors='black', label=f'Radiation Awareness Total ({dataset_label})')
        # Draw dashed lines to the axes
        plt.plot([final_distance_with, final_distance_with], [0, final_dose_with],
                 color='blue', linestyle='--', linewidth=1)
        plt.plot([0, final_distance_with], [final_dose_with, final_dose_with],
                 color='blue', linestyle='--', linewidth=1)
        # Calculate and prepare trend line through origin
        # Using linear regression without intercept: slope = sum(x*y) / sum(x^2)
        slope_with = np.sum(common_distance_with * smoothed_average_with) / np.sum(common_distance_with**2)
        trend_within_with = slope_with * common_distance_with
        # Store dose rate (slope)
        dose_rate_with = slope_with

    # Print the estimated dose rates
    if dose_rate_without is not None:
        print(f"[{dataset_label}] Estimated Average Dose Rate (Without Radiation Avoidance): {dose_rate_without:.3f} units/m")
    if dose_rate_with is not None:
        print(f"[{dataset_label}] Estimated Average Dose Rate (With Radiation Avoidance): {dose_rate_with:.3f} units/m")

    # Enhancements
    plt.title(f'Cumulative Radiation Dose vs. Distance Travelled (Average Runs) - {dataset_label}', fontsize=16)
    plt.xlabel('Distance Travelled (m)', fontsize=14)
    plt.ylabel('Cumulative Dose (units)', fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[{dataset_label}] Mean cumulative dose plot saved to {save_path}")

    plt.show()

    return dose_rate_with, dose_rate_without

def plot_average_std_dev_vs_num_series(scenario_doses, scenario, dataset_label, save_path=None):
    """
    Plots the average standard deviation against the number of data series for a given scenario.

    Parameters:
    - scenario_doses: List of numpy arrays containing interpolated doses for the scenario.
    - scenario: String indicating the scenario name ('with_radiation' or 'without_radiation').
    - dataset_label: String label for the dataset (e.g., 'radvdistruns', '3x3rundata', '3x3rundata2').
    - save_path: Optional path to save the plot.
    """
    if not scenario_doses:
        print(f"[{dataset_label}] No data available to plot standard deviation for scenario '{scenario}'.")
        return

    num_runs = len(scenario_doses)
    average_std_devs = []

    # Convert list to numpy array for easier slicing
    dose_matrix = np.array(scenario_doses)  # Shape: (num_runs, num_points)

    for i in range(1, num_runs + 1):
        # Select the first i runs
        current_runs = dose_matrix[:i, :]
        # Compute standard deviation at each distance point, ignoring NaNs
        std_devs = np.nanstd(current_runs, axis=0)
        # Compute the average standard deviation across all distance points
        avg_std_dev = np.nanmean(std_devs)
        average_std_devs.append(avg_std_dev)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_runs + 1), average_std_devs, marker='o', linestyle='-',
             color='green', label=f'Average Std Dev ({scenario.replace("_", " ").title()})')
    plt.title(f'Average Standard Deviation vs. Number of Data Series\n({scenario.replace("_", " ").title()}) - {dataset_label}', fontsize=14)
    plt.xlabel('Number of Data Series', fontsize=12)
    plt.ylabel('Average Standard Deviation (units)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[{dataset_label}] Average standard deviation plot for '{scenario}' saved to {save_path}")

    plt.show()

def plot_average_std_dev_vs_num_series_combined(scenario_doses_with, scenario_doses_without, dataset_label, save_path=None):
    """
    Plots the average standard deviation against the number of data series for both scenarios on a single plot.

    Parameters:
    - scenario_doses_with: List of numpy arrays for 'with_radiation' scenario.
    - scenario_doses_without: List of numpy arrays for 'without_radiation' scenario.
    - dataset_label: String label for the dataset (e.g., 'radvdistruns', '3x3rundata', '3x3rundata2').
    - save_path: Optional path to save the plot.
    """
    plt.figure(figsize=(10, 6))

    # Plot for 'with_radiation'
    if scenario_doses_with:
        num_runs_with = len(scenario_doses_with)
        average_std_devs_with = []
        dose_matrix_with = np.array(scenario_doses_with)
        for i in range(1, num_runs_with + 1):
            current_runs = dose_matrix_with[:i, :]
            std_devs = np.nanstd(current_runs, axis=0)
            avg_std_dev = np.nanmean(std_devs)
            average_std_devs_with.append(avg_std_dev)
        plt.plot(range(1, num_runs_with + 1), average_std_devs_with, marker='o', linestyle='-',
                 color='blue', label='With Radiation Awareness')

    # Plot for 'without_radiation'
    if scenario_doses_without:
        num_runs_without = len(scenario_doses_without)
        average_std_devs_without = []
        dose_matrix_without = np.array(scenario_doses_without)
        for i in range(1, num_runs_without + 1):
            current_runs = dose_matrix_without[:i, :]
            std_devs = np.nanstd(current_runs, axis=0)
            avg_std_dev = np.nanmean(std_devs)
            average_std_devs_without.append(avg_std_dev)
        plt.plot(range(1, num_runs_without + 1), average_std_devs_without, marker='s', linestyle='--',
                 color='orange', label='Without Radiation Awareness')

    plt.title(f'Average Standard Deviation vs. Number of Data Series - {dataset_label}', fontsize=14)
    plt.xlabel('Number of Data Series', fontsize=12)
    plt.ylabel('Average Standard Deviation (units)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[{dataset_label}] Combined average standard deviation plot saved to {save_path}")

    plt.show()

def plot_combined_mean_cumulative_dose(datasets_averages, save_path=None):
    """
    Plots the combined mean cumulative radiation dose vs. distance travelled for multiple datasets.

    Parameters:
    - datasets_averages: List of dictionaries containing dataset label, common_distance_with, average_with,
                        common_distance_without, and average_without.
    - save_path: Optional path to save the combined plot.
    """
    plt.figure(figsize=(18, 12))

    # Define a color map for different datasets and scenarios
    # Ensure each dataset and scenario has a unique color
    color_map = {
        'radvdistruns': {'with': 'blue', 'without': 'red'},
        '3x3rundata': {'with': 'green', 'without': 'orange'},
        '3x3rundata2': {'with': 'purple', 'without': 'brown'}
    }

    line_style = {
        'with_radiation': '--',
        'without_radiation': '-'
    }

    marker_map = {
        'radvdistruns': {'with': 'o', 'without': 's'},
        '3x3rundata': {'with': '^', 'without': 'D'},
        '3x3rundata2': {'with': 'v', 'without': 'P'}
    }

    for data in datasets_averages:
        label = data['label']
        # Assign colors based on dataset and scenario
        color_with = color_map.get(label, {}).get('with', 'blue')
        color_without = color_map.get(label, {}).get('without', 'red')

        # Assign markers based on dataset and scenario
        marker_with = marker_map.get(label, {}).get('with', 'o')
        marker_without = marker_map.get(label, {}).get('without', 's')

        # Plot 'without_radiation'
        if data['average_without'] is not None:
            plt.plot(data['common_distance_without'], data['average_without'],
                     label=f'No Radiation ({label})', linestyle=line_style['without_radiation'],
                     color=color_without, linewidth=2)
            # Optionally, mark the final point
            plt.scatter(data['common_distance_without'][-1], data['average_without'][-1],
                        color=color_without, marker=marker_without, s=150, zorder=5, edgecolors='black')

        # Plot 'with_radiation'
        if data['average_with'] is not None:
            plt.plot(data['common_distance_with'], data['average_with'],
                     label=f'With Radiation ({label})', linestyle=line_style['with_radiation'],
                     color=color_with, linewidth=2)
            # Optionally, mark the final point
            plt.scatter(data['common_distance_with'][-1], data['average_with'][-1],
                        color=color_with, marker=marker_with, s=150, zorder=5, edgecolors='black')

    plt.title('Combined Cumulative Radiation Dose vs. Distance Travelled', fontsize=20)
    plt.xlabel('Distance Travelled (m)', fontsize=16)
    plt.ylabel('Cumulative Dose (units)', fontsize=16)
    plt.legend(fontsize=14, loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Combined mean cumulative dose plot saved to {save_path}")

    plt.show()

def process_dataset(output_dir, dataset_label, analytics_dir):
    """
    Processes a single dataset: reads data, interpolates, averages, and generates plots.

    Parameters:
    - output_dir: Path to the dataset directory containing CSV files.
    - dataset_label: String label for the dataset (e.g., 'radvdistruns', '3x3rundata', '3x3rundata2').
    - analytics_dir: Path to the 'analytics' directory where output files will be saved.
    """
    print(f"\nProcessing dataset: {dataset_label}")
    # Get paired runs
    paired_runs = get_paired_runs(output_dir)
    if not paired_runs:
        return None

    scenarios = ['with_radiation', 'without_radiation']
    scenario_doses = {scenario: [] for scenario in scenarios}

    # Lists to store total distances and doses for reporting
    total_distances = {scenario: [] for scenario in scenarios}
    total_doses = {scenario: [] for scenario in scenarios}

    # Determine the average run length for each scenario
    average_lengths = {}
    for scenario in scenarios:
        distances = []
        for run in paired_runs:
            filepath = os.path.join(output_dir, f"run{run}_{scenario}.csv")
            run_distance, _ = read_csv_data(filepath)
            if run_distance is not None:
                distances.append(run_distance[-1])
        if distances:
            average_length = np.mean(distances)
            average_lengths[scenario] = average_length
            print(f"[{dataset_label}] Average run length for '{scenario}': {average_length:.2f} meters.")
        else:
            average_lengths[scenario] = 0
            print(f"[{dataset_label}] No valid run lengths found for '{scenario}'.")

    # Define separate common cumulative_distance grids based on average lengths
    step = 0.1  # meters
    common_distance_with = np.arange(0, average_lengths['with_radiation'] + step, step) if average_lengths['with_radiation'] > 0 else None
    common_distance_without = np.arange(0, average_lengths['without_radiation'] + step, step) if average_lengths['without_radiation'] > 0 else None

    # Lists to store individual runs
    individual_runs_with = []
    individual_runs_without = []

    # Collect and interpolate data for each scenario
    for run in paired_runs:
        for scenario in scenarios:
            filepath = os.path.join(output_dir, f"run{run}_{scenario}.csv")
            run_distance, run_dose = read_csv_data(filepath)
            if run_distance is None or run_dose is None:
                print(f"[{dataset_label}] Skipping run {run}, scenario '{scenario}' due to read error.")
                continue
            # Store total distance and dose for reporting
            total_distances[scenario].append(run_distance[-1])
            total_doses[scenario].append(run_dose[-1])
            # Interpolate and crop dose data onto the scenario-specific grid
            if scenario == 'with_radiation' and common_distance_with is not None:
                interpolated_dose = interpolate_and_crop(common_distance_with, run_distance, run_dose, average_lengths[scenario])
                if interpolated_dose is not None:
                    scenario_doses[scenario].append(interpolated_dose)
                    individual_runs_with.append(interpolated_dose)
            elif scenario == 'without_radiation' and common_distance_without is not None:
                interpolated_dose = interpolate_and_crop(common_distance_without, run_distance, run_dose, average_lengths[scenario])
                if interpolated_dose is not None:
                    scenario_doses[scenario].append(interpolated_dose)
                    individual_runs_without.append(interpolated_dose)

    # Validate data to ensure no negative values
    validate_data(scenario_doses['with_radiation'], 'with_radiation', dataset_label)
    validate_data(scenario_doses['without_radiation'], 'without_radiation', dataset_label)

    # Check if data was collected
    for scenario in scenarios:
        if not scenario_doses[scenario]:
            print(f"[{dataset_label}] No data collected for scenario '{scenario}'.")

    # Average the doses
    average_with = average_runs(scenario_doses['with_radiation']) if scenario_doses['with_radiation'] else None
    average_without = average_runs(scenario_doses['without_radiation']) if scenario_doses['without_radiation'] else None

    # Print the average dose information
    print(f"\n=== [{dataset_label}] Average Cumulative Dose Values ===")
    if average_with is not None:
        mean_dose_with = np.nanmean(average_with)
        print(f"[{dataset_label}] With Radiation Avoidance: {mean_dose_with:.2f} units")
    else:
        print(f"[{dataset_label}] With Radiation Avoidance: No data available.")

    if average_without is not None:
        mean_dose_without = np.nanmean(average_without)
        print(f"[{dataset_label}] Without Radiation Avoidance: {mean_dose_without:.2f} units")
    else:
        print(f"[{dataset_label}] Without Radiation Avoidance: No data available.")

    # Plot the mean cumulative dose with final point markers and compute dose rates
    plot_mean_cumulative_dose(
        common_distance_with, average_with,
        common_distance_without, average_without,
        dataset_label=dataset_label,
        save_path=os.path.join(analytics_dir, f'mean_cumulative_dose_plot_{dataset_label}.png'),
        smooth_window=1  # Set window_size=1 to effectively disable smoothing
    )

    # ============================
    # Plot Average Std Dev vs. Number of Series
    # ============================

    # Plot for 'with_radiation' scenario
    plot_average_std_dev_vs_num_series(
        scenario_doses['with_radiation'],
        scenario='with_radiation',
        dataset_label=dataset_label,
        save_path=os.path.join(analytics_dir, f'average_std_dev_with_radiation_{dataset_label}.png')
    )

    # Plot for 'without_radiation' scenario
    plot_average_std_dev_vs_num_series(
        scenario_doses['without_radiation'],
        scenario='without_radiation',
        dataset_label=dataset_label,
        save_path=os.path.join(analytics_dir, f'average_std_dev_without_radiation_{dataset_label}.png')
    )

    # Optionally, you can combine both scenarios into a single plot for comparison
    plot_average_std_dev_vs_num_series_combined(
        scenario_doses['with_radiation'],
        scenario_doses['without_radiation'],
        dataset_label=dataset_label,
        save_path=os.path.join(analytics_dir, f'average_std_dev_combined_{dataset_label}.png')
    )

    # Return the averages and common distances for combined plotting
    return {
        'label': dataset_label,
        'common_distance_with': common_distance_with,
        'average_with': average_with,
        'common_distance_without': common_distance_without,
        'average_without': average_without
    }

def plot_combined_mean_cumulative_dose(datasets_averages, save_path=None):
    """
    Plots the combined mean cumulative radiation dose vs. distance travelled for multiple datasets.

    Parameters:
    - datasets_averages: List of dictionaries containing dataset label, common_distance_with, average_with,
                        common_distance_without, and average_without.
    - save_path: Optional path to save the combined plot.
    """
    plt.figure(figsize=(18, 12))

    # Define a color map for different datasets and scenarios
    # Ensure each dataset and scenario has a unique color
    color_map = {
        'radvdistruns': {'with': 'blue', 'without': 'red'},
        '3x3rundata': {'with': 'green', 'without': 'orange'},
        '3x3rundata2': {'with': 'purple', 'without': 'brown'}
    }

    line_style = {
        'with_radiation': '--',
        'without_radiation': '-'
    }

    marker_map = {
        'radvdistruns': {'with': 'o', 'without': 's'},
        '3x3rundata': {'with': '^', 'without': 'D'},
        '3x3rundata2': {'with': 'v', 'without': 'P'}
    }

    for data in datasets_averages:
        label = data['label']
        # Assign colors based on dataset and scenario
        color_with = color_map.get(label, {}).get('with', 'blue')
        color_without = color_map.get(label, {}).get('without', 'red')

        # Assign markers based on dataset and scenario
        marker_with = marker_map.get(label, {}).get('with', 'o')
        marker_without = marker_map.get(label, {}).get('without', 's')

        # Plot 'without_radiation'
        if data['average_without'] is not None:
            plt.plot(data['common_distance_without'], data['average_without'],
                     label=f'No Radiation ({label})', linestyle=line_style['without_radiation'],
                     color=color_without, linewidth=2)
            # Optionally, mark the final point
            plt.scatter(data['common_distance_without'][-1], data['average_without'][-1],
                        color=color_without, marker=marker_without, s=150, zorder=5, edgecolors='black')

        # Plot 'with_radiation'
        if data['average_with'] is not None:
            plt.plot(data['common_distance_with'], data['average_with'],
                     label=f'With Radiation ({label})', linestyle=line_style['with_radiation'],
                     color=color_with, linewidth=2)
            # Optionally, mark the final point
            plt.scatter(data['common_distance_with'][-1], data['average_with'][-1],
                        color=color_with, marker=marker_with, s=150, zorder=5, edgecolors='black')

    plt.title('Combined Cumulative Radiation Dose vs. Distance Travelled', fontsize=20)
    plt.xlabel('Distance Travelled (m)', fontsize=16)
    plt.ylabel('Cumulative Dose (units)', fontsize=16)
    plt.legend(fontsize=14, loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Combined mean cumulative dose plot saved to {save_path}")

    plt.show()

def main():
    # Define the datasets to process
    datasets = [
        {
            'label': 'radvdistruns',
            'directory': 'radvdistruns'
        },
        {
            'label': '3x3rundata',
            'directory': '3x3rundata'
        },
        {
            'label': '3x3rundata2',
            'directory': '3x3rundata2'
        }
    ]

    # Define the analytics directory
    analytics_dir = 'analytics'

    # Create the 'analytics' directory if it doesn't exist
    if not os.path.exists(analytics_dir):
        try:
            os.makedirs(analytics_dir)
            print(f"Created directory '{analytics_dir}' for saving analytics outputs.")
        except Exception as e:
            print(f"Error creating '{analytics_dir}' directory: {e}")
            return

    # List to store averages for combined plotting
    datasets_averages = []

    # Process each dataset
    for dataset in datasets:
        result = process_dataset(output_dir=dataset['directory'], dataset_label=dataset['label'], analytics_dir=analytics_dir)
        if result:
            datasets_averages.append(result)

    # ============================
    # Combined Plot: Mean Cumulative Dose vs. Distance Travelled
    # ============================

    if datasets_averages:
        plot_combined_mean_cumulative_dose(
            datasets_averages=datasets_averages,
            save_path=os.path.join(analytics_dir, 'combined_mean_cumulative_dose_plot.png')
        )
    else:
        print("No datasets processed successfully. Combined plot will not be generated.")

if __name__ == "__main__":
    main()