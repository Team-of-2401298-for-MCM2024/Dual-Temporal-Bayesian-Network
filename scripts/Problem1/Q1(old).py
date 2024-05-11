import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from pathlib import Path

# Load the CSV file
root_path = Path(__file__).parents[2]
file_path = root_path / 'data' /'Wimbledon_featured_matches.csv'
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

        for start_index in range(window - length + 1):
            current_server = self.server[start_index:start_index + length]
            current_point_victor = self.point_victor[start_index:start_index + length]
            difference_value = self.calculate_difference_value(current_server, current_point_victor, length)
            difference_value_list.append(difference_value)


        # 插值拟合
        x = np.linspace(length // 2, len(difference_value_list) - 1, num=len(difference_value_list), endpoint=True)
        x_new = np.linspace(length // 2, len(difference_value_list) - 1, num=10 * len(difference_value_list), endpoint=True)
        y = difference_value_list


        f_cubic = interp1d(x, y, kind='cubic')

        # 使用插值函数计算新的y值
        y_cubic = f_cubic(x_new)

        # 绘制结果
        plt.figure()
        plt.axhline(y=0.5, color='r', linestyle='--')
        # plt.axhline(y=0.5, color='r', linestyle='--', label='X-axis')
        # plt.plot(x, y, 'o', label='original data')
        plt.plot(x_new, y_cubic, '-', label='probability difference')
        plt.title(f'Scatter Plot of Elapsed Time for Match {self.match_id}')
        plt.legend(loc='best') # 显示图例
        plt.show()




# Assuming df is your DataFrame
matches_list = []
for match_id, group in df.groupby('match_id'):
    match_data = {col: group[col].tolist() for col in group}  # Convert each column to a list
    match_instance = Match(**match_data)
    matches_list.append(match_instance)


# Example usage:
# Accessing properties of the first match
for i in range(3):
    first_match = matches_list[i]
    Match.evaluation_performance(first_match, 20)



