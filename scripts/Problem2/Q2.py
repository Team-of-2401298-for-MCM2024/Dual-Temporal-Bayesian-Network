import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from pathlib import Path

# Load the CSV file
root_path = Path(__file__).parents[2]
file_path = root_path / 'data' /'Wimbledon_featured_matches.csv'
df = pd.read_csv(file_path)

# Assuming df is your DataFrame
matches_list = []

for match_id, group in df.groupby('match_id'):
    match_data = {col: group[col].tolist() for col in group}  # Convert each column to a list
    match_instance = match_data
    matches_list.append(match_instance)

from scipy.stats import norm


def Wald_Wolfowitz_Runs_Test(i):
    match_data = pd.DataFrame(matches_list[i])
    # Prepare the binary sequence for the runs test
    binary_sequence_match = match_data['point_victor'].apply(lambda x: 1 if x == 1 else 0)

    # Simplified manual calculation for runs test on a specific match's point sequence
    # This approach simplifies the process to avoid the issues encountered previously

    # Assuming binary_sequence_match is correctly prepared as a binary sequence representing point victories
    # Count the number of runs (a run is a sequence of consecutive wins or losses)
    runs = 1  # Starting with 1 run
    for i in range(1, len(binary_sequence_match)):
        if binary_sequence_match.iloc[i] != binary_sequence_match.iloc[i - 1]:
            runs += 1

    # Calculating n1 and n2 (number of points won by player 1 and player 2 respectively)
    n1 = binary_sequence_match.sum()
    n2 = len(binary_sequence_match) - n1

    # Expected number of runs and its variance (using simplified formulas)
    expected_runs = 1 + (2 * n1 * n2) / (n1 + n2)
    variance_runs = (expected_runs - 1) * (expected_runs - 2) / (len(binary_sequence_match) - 1)

    # Given values
    actual_runs = runs

    # Calculate Z-value
    z_value = (actual_runs - expected_runs) / (variance_runs ** 0.5)

    # Calculate P-value for a two-tailed test
    p_value = 2 * (1 - norm.cdf(abs(z_value)))

    return runs, expected_runs, variance_runs, z_value, p_value


from statsmodels.stats.diagnostic import acorr_ljungbox


def Ljung_Box_Q_test(i):
    match_data = pd.DataFrame(matches_list[i])

    # Ensure correct data type handling and recalculate p1 and p2 with explicit type conversion if necessary

    # Convert 'server' and 'point_victor' columns to strings to match our initial filter conditions explicitly
    match_data['server'] = match_data['server'].astype(str)
    match_data['point_victor'] = match_data['point_victor'].astype(str)

    # Recalculate p1 and p2 with the corrected approach
    p1_serving_wins_final = match_data[(match_data['server'] == '1') & (match_data['point_victor'] == '1')].shape[0]
    p1_serving_total_final = match_data[match_data['server'] == '1'].shape[0]
    p1_final = p1_serving_wins_final / p1_serving_total_final if p1_serving_total_final > 0 else 0

    p2_serving_wins_final = match_data[(match_data['server'] == '2') & (match_data['point_victor'] == '1')].shape[0]
    p2_serving_total_final = match_data[match_data['server'] == '2'].shape[0]
    p2_final = p2_serving_wins_final / p2_serving_total_final if p2_serving_total_final > 0 else 0

    # Construct a binary sequence where 1 represents a point won by player1, and 0 represents a point won by player2
    point_sequence = match_data['point_victor'].apply(lambda x: 1 if x == '1' else 0)

    # Apply the Ljung-Box Q test to the sequence
    # We'll use a lag that considers up to 10 observations to check for autocorrelation
    ljung_box_results = acorr_ljungbox(point_sequence, lags=[10], return_df=True)

    return p1_final, p2_final, ljung_box_results


from matplotlib import pyplot as plt


def plotting(x, y, alpha, y1=None):
    # Create a figure with a specific size
    plt.figure(figsize=(10, 6))

    # Add grid, set alpha for grid
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Plot horizontal significance line
    plt.axhline(y=alpha, color='r', linestyle='--', label='Significance line')

    # Plot the bars with labels, different colors and with a slight alpha
    plt.bar(x, y, label='Wald_Wolfowitz_Runs_Test', color='blue', alpha=0.7, width=0.4)

    # If there is a second series, offset the x to avoid overlap
    if y1 is not None:
        plt.bar(x, y1, label='Ljung_Box_Q_test', color='orange', alpha=0.7, width=0.4)

    # Add labels and title
    plt.xlabel('Match ID')
    plt.ylabel('Value')
    # plt.title('Statistical Significance of Match IDs Using Wald-Wolfowitz and Ljung-Box Tests')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add legend outside of the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Tight layout to make room for the rotated x-axis labels
    plt.tight_layout()

    # Show the plot
    plt.show()


match_ids = []
p_values = []
for i in range(len(matches_list)):
    runs, expected_runs, variance_runs, z_value, p_value = Wald_Wolfowitz_Runs_Test(i)

    print('--' * 20)  # ________________
    print(f'the following is the data of {i}')
    print(f'runs={runs}, expected_runs={expected_runs}, variance_runs={variance_runs}')
    print(f'z_value={z_value}, p_value={p_value}')
    print('--' * 20)  # ________________

    match_ids.append(i)
    p_values.append(p_value)

match_ids = []
p_values2 = []

for i in range(len(matches_list)):
    p1_final, p2_final, ljung_box_results = Ljung_Box_Q_test(i)

    print('--' * 20)  # ________________
    print(f'the following is the data of {i}')
    print(f'p1_final={p1_final}, p2_final={p2_final},ljung_box_results=')
    print(ljung_box_results)
    print('--' * 20)  # ________________

    match_ids.append(i)
    p_values2.append(ljung_box_results['lb_pvalue'].item())

plotting(match_ids, p_values, 0.05, p_values2)


# Simplifying the calculation for immediate win rates by directly iterating over data
def immediate_win_rate(next_step, player_num=1):
    # Initialize counters
    immediate_wins_p1 = immediate_wins_p2 = 0
    immediate_chances_p1 = immediate_chances_p2 = 0

    # Iterate through the dataset
    for i in range(len(wimbledon_data) - next_step):  # Avoid the last row to prevent index out of range
        current_row = wimbledon_data.iloc[i]
        next_row = wimbledon_data.iloc[i + next_step]

        # Ensure the next point is in the same match and same serve
        if (current_row['match_id'] == next_row['match_id']) and (current_row['server'] == next_row['server']):
            # Player 1 serving scenarios
            if current_row['server'] == 1 and current_row['point_victor'] == player_num:
                immediate_chances_p1 += 1
                if next_row['point_victor'] == 1:
                    immediate_wins_p1 += 1

            # Player 2 serving scenarios
            elif current_row['server'] == 2 and current_row['point_victor'] == player_num:
                immediate_chances_p2 += 1
                if next_row['point_victor'] == 1:
                    immediate_wins_p2 += 1

    # Calculate immediate win rates
    immediate_win_rate_p1 = immediate_wins_p1 / immediate_chances_p1 if immediate_chances_p1 > 0 else 0
    immediate_win_rate_p2 = immediate_wins_p2 / immediate_chances_p2 if immediate_chances_p2 > 0 else 0

    return immediate_win_rate_p1, immediate_win_rate_p2


for step in [1, 2]:
    for num in [1, 2]:
        print("step:", step, "num", num, "p:", immediate_win_rate(step, num))
