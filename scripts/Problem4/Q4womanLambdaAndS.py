import pandas as pd
import numpy as np
from scipy.optimize import minimize

# 假定CSV文件路径
file_path = 'Wimbledon_women_points.csv'
df = pd.read_csv(file_path)

class Match:
    def __init__(self, match_id, player1, player2, elapsed_time, set_no, p1_games, p2_games, SetWinner, game_no, GameWinner, point_no, point_victor, server, p1_score, p2_score, p1_points_won, p2_points_won, p1_ace, p2_ace, p1_winner, p2_winner, p1_double_fault, p2_double_fault, p1_unf_err, p2_unf_err, p1_net_pt, p2_net_pt, p1_net_pt_won, p2_net_pt_won, p1_break_pt, p2_break_pt, p1_break_pt_won, p2_break_pt_won, ServeNumber):
        self.match_id = match_id[0]
        self.player1 = player1[0]
        self.player2 = player2[0]
        self.elapsed_time = elapsed_time
        self.set_no = set_no
        self.game_no = game_no
        self.point_no = point_no
        # self.p1_sets = p1_sets
        # self.p2_sets = p2_sets
        self.p1_games = p1_games
        self.p2_games = p2_games
        self.p1_score = p1_score
        self.p2_score = p2_score
        self.server = server
        # self.serve_no = serve_no
        self.point_victor = point_victor
        self.p1_points_won = p1_points_won
        self.p2_points_won = p2_points_won
        # self.game_victor = game_victor
        # self.set_victor = set_victor
        self.p1_ace = p1_ace
        self.p2_ace = p2_ace
        self.p1_winner = p1_winner
        self.p2_winner = p2_winner
        # self.winner_shot_type = winner_shot_type
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
        # self.p1_break_pt_missed = p1_break_pt_missed
        # self.p2_break_pt_missed = p2_break_pt_missed
        # self.p1_distance_run = p1_distance_run
        # self.p2_distance_run = p2_distance_run
        # self.rally_count = rally_count
        # self.speed_mph = speed_mph
        # self.serve_width = serve_width
        # self.serve_depth = serve_depth
        # self.return_depth = return_depth


# 根据match_id创建Match实例列表
matches_list = [Match(*[group[col].values for col in group]) for match_id, group in df.groupby('match_id')]

# Load the rankings data
rankings_path = 'women_rankings.csv'
rankings_df = pd.read_csv(rankings_path)

# Sort the dataframe by the 'rank' column to get the order for the competition
rankings_df = rankings_df.sort_values(by='rank').reset_index(drop=True)

# Create a dictionary mapping player names to their relative ranks
player_to_relative_rank = {row['player']: idx + 1 for idx, row in rankings_df.iterrows()}


def get_relative_ranks(player1, player2):
    """
    Given two player names, return their relative ranks in the competition.
    """
    rank1 = player_to_relative_rank.get(player1, None)
    rank2 = player_to_relative_rank.get(player2, None)

    if rank1 is None or rank2 is None:
        raise ValueError("One or both players not found in the rankings.")

    return rank1, rank2

# Assuming a fixed server advantage for demonstration
s = 0.05  # This is a made-up value; you may need to estimate this from your data


# Modified log-likelihood function for a single match
def match_log_likelihood(params, match):
    lmbda, s = params
    total_log_likelihood = 0

    for i, server in enumerate(match.server):
        r_a, r_b = get_relative_ranks(match.player1, match.player2)
        ra_minus_rb = 6 - np.log2(r_a) - (6 - np.log2(r_b))

        # Adjust the probability calculation based on who is serving
        if server == 1:  # Player 1 serving
            p_win_a = np.exp(lmbda * ra_minus_rb) / (1 + np.exp(lmbda * ra_minus_rb)) + s
        else:  # Player 2 serving
            p_win_a = 1 - np.exp(lmbda * (-ra_minus_rb)) / (1 + np.exp(lmbda * (-ra_minus_rb))) - s

        if p_win_a <= 0 or p_win_a >= 1:
            # print(server, lmbda, ra_minus_rb, p_win_a, s)
            return 1000

        # Ensure probability is within [0,1] after adding server advantage
        # p_win_a = min(max(p_win_a, 0), 1)

        # Determine the point victor
        point_victor = match.point_victor[i]

        # Calculate log-likelihood for this point
        if point_victor == 1:  # Player 1 wins the point
            total_log_likelihood += np.log(p_win_a)
        else:  # Player 2 wins the point
            total_log_likelihood += np.log(1 - p_win_a)

    return -total_log_likelihood  # Negative because we will be minimizing


# Aggregate log-likelihood across all matches
def total_log_likelihood(params, matches):
    return sum(match_log_likelihood(params, match) for match in matches)


# print(total_log_likelihood((0.05, 0.28), matches_list))
# exit()

# Initial guesses for lambda and s
initial_params = [0.1, s]

# Minimize the negative log-likelihood
result = minimize(lambda params: total_log_likelihood(params, matches_list), initial_params, method='L-BFGS-B')

# Output the result
if result.success:
    fitted_lambda, fitted_s = result.x
    print(f"Optimal lambda: {fitted_lambda}, Optimal s: {fitted_s}")
else:
    print("Optimization failed.")