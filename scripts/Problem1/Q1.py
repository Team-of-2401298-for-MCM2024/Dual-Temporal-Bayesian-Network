import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from pathlib import Path
# Load the CSV file
root_path = Path(__file__).parents[2]
file_path = root_path / 'data' / 'Wimbledon_featured_matches.csv'
df = pd.read_csv(file_path)


class Match:
    """
    A class to represent a detailed view of a tennis match, with properties holding arrays of values.
    """
    def __init__(self, match_id, player1, player2, elapsed_time, set_no, game_no, point_no, p1_sets, p2_sets, p1_games, p2_games, p1_score, p2_score, server, serve_no, point_victor, p1_points_won, p2_points_won, game_victor, set_victor, p1_ace, p2_ace, p1_winner, p2_winner, winner_shot_type, p1_double_fault, p2_double_fault, p1_unf_err, p2_unf_err, p1_net_pt, p2_net_pt, p1_net_pt_won, p2_net_pt_won, p1_break_pt, p2_break_pt, p1_break_pt_won, p2_break_pt_won, p1_break_pt_missed, p2_break_pt_missed, p1_distance_run, p2_distance_run, rally_count, speed_mph, serve_width, serve_depth, return_depth):
        self.match_id = match_id[0]
        self.player1 = player1[0]
        self.player2 = player2[0]
        self.elapsed_time = elapsed_time
        self.set_no = set_no
        self.game_no = game_no
        self.point_no = point_no
        self.p1_sets = p1_sets
        self.p2_sets = p2_sets
        self.p1_games = p1_games
        self.p2_games = p2_games
        self.p1_score = p1_score
        self.p2_score = p2_score
        self.server = server
        self.serve_no = serve_no
        self.point_victor = point_victor
        self.p1_points_won = p1_points_won
        self.p2_points_won = p2_points_won
        self.game_victor = game_victor
        self.set_victor = set_victor
        self.p1_ace = p1_ace
        self.p2_ace = p2_ace
        self.p1_winner = p1_winner
        self.p2_winner = p2_winner
        self.winner_shot_type = winner_shot_type
        self.p1_double_fault = p1_double_fault
        self.p2_double_fault = p2_double_fault
        self.p1_unf_err = p1_unf_err
        self.p2_unf_err = p2_unf_err
        self.p1_net_pt = p1_net_pt
        self.p2_net_pt = p2_net_pt
        self.p1_net_pt_won = p1_net_pt_won
        self.p2_net_pt_won = p2_net_pt_won
        self.p1_break_pt = p1_break_pt
        self.p2_break_pt = p2_break_pt
        self.p1_break_pt_won = p1_break_pt_won
        self.p2_break_pt_won = p2_break_pt_won
        self.p1_break_pt_missed = p1_break_pt_missed
        self.p2_break_pt_missed = p2_break_pt_missed
        self.p1_distance_run = p1_distance_run
        self.p2_distance_run = p2_distance_run
        self.rally_count = rally_count
        self.speed_mph = speed_mph
        self.serve_width = serve_width
        self.serve_depth = serve_depth
        self.return_depth = return_depth
        # 转换elapsed_time为秒
        self.elapsed_time_seconds = [self.convert_time_to_seconds(time_str) for time_str in elapsed_time]


    @staticmethod
    def convert_time_to_seconds(time_str):
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s

    def calculate_difference_value(self, server, point_victor, length):
        # Ensure the length does not exceed the total number of points
        length = min(length, len(point_victor))

        # Initialize counters for serves and points won by each player within the specified length
        player1_server = player2_server = 0
        player1_server_points_won = player2_server_points_won = 0

        # Loop over points up to the specified length
        for i in range(length):
            # Count serves by each player
            if server[i] == 1:
                player1_server += 1
                if point_victor[i] == 1:
                    player1_server_points_won += 1
            elif server[i] == 2:
                player2_server += 1
                if point_victor[i] == 2:
                    player2_server_points_won += 1

        # Calculate difference value within the specified length
        if player1_server > 0 and player2_server > 0:
            difference_value = (player1_server_points_won / player1_server - player2_server_points_won / player2_server + 1) / 2.0
        else:
            difference_value = 0.5  # Default value in case one of the players didn't serve within the specified length
        return difference_value

    def evaluation_performance(self, length=10):
        window = len(self.server)
        difference_value_list = []
        elapsed_time_list = self.elapsed_time_seconds  # Assuming this is already in seconds format

        for start_index in range(window - length + 1):
            current_server = self.server[start_index:start_index + length]
            current_point_victor = self.point_victor[start_index:start_index + length]
            difference_value = self.calculate_difference_value(current_server, current_point_victor, length)
            difference_value_list.append(difference_value)

        # Use elapsed_time_seconds directly as x
        x = np.array(elapsed_time_list[:len(difference_value_list)])  # Ensure matching length
        y = np.array(difference_value_list)

        # Perform cubic interpolation
        f_cubic = interp1d(x, y, kind='cubic', fill_value="extrapolate")

        # Estimate the shift amount based on the average difference in x multiplied by length
        average_diff = np.mean(np.diff(x))
        shift_amount = average_diff * length

        # Create a smooth range of new x values for plotting, shifted to the right
        x_new = np.linspace(min(x) + shift_amount, max(x) + shift_amount, num=10 * len(x), endpoint=True)
        y_cubic = f_cubic(x_new - shift_amount)  # Apply the shift amount inversely to correct the interpolation

        # Apply the same shift to the original x values for plotting the data points
        x_shifted = x + shift_amount

        # Plotting
        plt.figure()
        plt.axhline(y=0.5, color='r', linestyle='--', label='Equilibrium line')
        # plt.plot(x_shifted, y, 'o', label='Original data shifted right')
        plt.plot(x_new, y_cubic, '-', label='Cubic spline interpolation shifted right')
        plt.xlabel('Elapsed Time (seconds)')
        plt.legend(loc='best')
        plt.show()


# Assuming df is your DataFrame
matches_list = []
for match_id, group in df.groupby('match_id'):
    match_data = {col: group[col].tolist() for col in group}  # Convert each column to a list
    match_instance = Match(**match_data)
    matches_list.append(match_instance)


# Example usage:
# Accessing properties of the first match
# for i in range(2):
#     first_match = matches_list[i]
#     Match.evaluation_performance(first_match, 30)

# 示例用法：绘制第一个比赛的elapsed_time对server的散点图
first_match = matches_list[0]
Match.evaluation_performance(first_match)



