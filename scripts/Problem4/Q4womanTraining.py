import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ExpectationMaximization
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

'''
We here use 3 to 1 as subscript to denote the time depth. 
3 is the most past and 1 is the most recent.
We need a program that computes the average of variables in a certain time interval.
For simplicity the time is just the index of points.
'''

# Load the CSV file
file_path = 'Wimbledon_women_points.csv'
df = pd.read_csv(file_path)

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


class Match:
    """
    A class to represent a detailed view of a tennis match, with properties holding arrays of values.
    """
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

        # generated
        self.length = len(self.point_victor)

        self.unf_err_1, self.unf_err_2 = [0 for _ in range(self.length)], [0 for _ in range(self.length)]
        self.double_fault_1, self.double_fault_2 = [0 for _ in range(self.length)], [0 for _ in range(self.length)]
        self.ace_1, self.ace_2 = [0 for _ in range(self.length)], [0 for _ in range(self.length)]
        self.net_pt_won_1, self.net_pt_won_2 = [0 for _ in range(self.length)], [0 for _ in range(self.length)]
        self.score_advantage_1, self.score_advantage_2 = [0 for _ in range(self.length)], [0 for _ in range(self.length)]
        self.residual_effect_1, self.residual_effect_2 = [0 for _ in range(self.length)], [0 for _ in range(self.length)]
        self.break_pt_won_1, self.break_pt_won_2 = [0 for _ in range(self.length)], [0 for _ in range(self.length)]

        # compute residual effect
        r_a, r_b = get_relative_ranks(self.player1, self.player2)
        ra_minus_rb = 6 - np.log2(r_a) - (6 - np.log2(r_b))
        lmbda, s = 0.0466627, 0.0844507
        self.prob_1win_serve = np.exp(lmbda * ra_minus_rb) / (1 + np.exp(lmbda * ra_minus_rb)) + s
        self.prob_1win_not_serve = 1 - np.exp(lmbda * (-ra_minus_rb)) / (1 + np.exp(lmbda * (-ra_minus_rb))) - s

        self.residual_effect = [0 for _ in range(self.length)]
        for _ in range(self.length):
            if self.point_victor[_] == 1:
                if self.server[_] == 1:
                    self.residual_effect[_] = 1 - self.prob_1win_serve
                else:
                    self.residual_effect[_] = 1 - self.prob_1win_not_serve
            else:
                if self.server[_] == 1:
                    self.residual_effect[_] = 0 - self.prob_1win_serve
                else:
                    self.residual_effect[_] = 0 - self.prob_1win_not_serve

        # for visualization
        self.residual_effect_continuous = [0 for _ in range(self.length)]


matches_list = []
for match_id, group in df.groupby('match_id'):
    match_data = {col: group[col].tolist() for col in group}  # Convert each column to a list
    match_instance = Match(**match_data)
    matches_list.append(match_instance)

"""
We first need to discretize the variables.
"""

# Discretize
def discretize(_):
    if _ > 0:
        return 2
    elif _ == 0:
        return 1
    else:
        return 0

# time_3_start, time_3_end = - 56, - 32
# time_2_start, time_2_end = - 40, - 10
time_2_start, time_2_end = - 10, - 10
time_1_start, time_1_end = - 10, 0
predict_window = 10

# Aggregate across time interval
for match in matches_list:
    for current_t in range(-time_2_start, match.length - predict_window):
        match.unf_err_1[current_t] = discretize(sum(match.p1_unf_err[current_t + time_1_start: current_t + time_1_end]) - sum(match.p2_unf_err[current_t + time_1_start: current_t + time_1_end]))
        match.double_fault_1[current_t] = discretize(sum(match.p1_double_fault[current_t + time_1_start: current_t + time_1_end]) - sum(match.p2_double_fault[current_t + time_1_start: current_t + time_1_end]))
        match.ace_1[current_t] = discretize(sum(match.p1_ace[current_t + time_1_start: current_t + time_1_end]) - sum(match.p2_ace[current_t + time_1_start: current_t + time_1_end]))
        match.net_pt_won_1[current_t] = discretize(sum(match.p1_net_pt_won[current_t + time_1_start: current_t + time_1_end]) - sum(match.p2_net_pt_won[current_t + time_1_start: current_t + time_1_end]))
        match.score_advantage_1[current_t] = discretize(sum(match.p1_games[current_t + time_1_start: current_t + time_1_end]) - sum(match.p2_games[current_t + time_1_start: current_t + time_1_end]))
        match.break_pt_won_1[current_t] = discretize(sum(match.p1_break_pt_won[current_t + time_1_start: current_t + time_1_end]) - sum(match.p2_break_pt_won[current_t + time_1_start: current_t + time_1_end]))
        match.residual_effect_1[current_t] = discretize(sum(match.residual_effect[current_t + 1: current_t + predict_window + 1]))

for match in matches_list:
    for current_t in range(-time_2_start, match.length - predict_window):
        match.unf_err_2[current_t] = discretize(sum(match.p1_unf_err[current_t + time_2_start: current_t + time_2_end]) - sum(match.p2_unf_err[current_t + time_2_start: current_t + time_2_end]))
        match.double_fault_2[current_t] = discretize(sum(match.p1_double_fault[current_t + time_2_start: current_t + time_2_end]) - sum(match.p2_double_fault[current_t + time_2_start: current_t + time_2_end]))
        match.ace_2[current_t] = discretize(sum(match.p1_ace[current_t + time_2_start: current_t + time_2_end]) - sum(match.p2_ace[current_t + time_2_start: current_t + time_2_end]))
        match.net_pt_won_2[current_t] = discretize(sum(match.p1_net_pt_won[current_t + time_2_start: current_t + time_2_end]) - sum(match.p2_net_pt_won[current_t + time_2_start: current_t + time_2_end]))
        match.score_advantage_2[current_t] = discretize(sum(match.p1_games[current_t + time_2_start: current_t + time_2_end]) - sum(match.p2_games[current_t + time_2_start: current_t + time_2_end]))
        match.break_pt_won_2[current_t] = discretize(
            sum(match.p1_break_pt_won[current_t + time_2_start: current_t + time_2_end]) - sum(
                match.p2_break_pt_won[current_t + time_2_start: current_t + time_2_end]))
        match.residual_effect_2[current_t] = discretize(sum(match.residual_effect[current_t + time_2_end: time_2_end + predict_window + 1]))


for match in matches_list:
    for current_t in range(-time_2_start, match.length - predict_window):
        match.residual_effect_continuous[current_t] = sum(match.residual_effect[current_t + 1: current_t + predict_window + 1])

# Placeholder for the list of DataFrames created for each match's data
dfs = []

"""
Next we prepare the training data
"""

def get_train_df(num_layer=2, test_server=False):
    for match in matches_list:
        # Determine the slice range based on the conditions
        start_index = max(0, -time_2_start)  # Ensure start_index is not negative
        end_index = match.length - predict_window  # Adjust based on predict_window

        data = {
            'unf_err_1': match.unf_err_1[start_index:end_index],
            'double_fault_1': match.double_fault_1[start_index:end_index],
            'ace_1': match.ace_1[start_index:end_index],
            'net_pt_won_1': match.net_pt_won_1[start_index:end_index],
            'score_advantage_1': match.score_advantage_1[start_index:end_index],
            'residual_effect_1': match.residual_effect_1[start_index:end_index],
            "break_pt_won_1": match.break_pt_won_1[start_index:end_index],
        }

        if test_server:
            data["server"] = match.server[start_index:end_index]

        if num_layer == 2:
            data.update({
                # Long-term memory variables
                'unf_err_2': match.unf_err_2[start_index:end_index],
                'double_fault_2': match.double_fault_2[start_index:end_index],
                'ace_2': match.ace_2[start_index:end_index],
                'net_pt_won_2': match.net_pt_won_2[start_index:end_index],
                'score_advantage_2': match.score_advantage_2[start_index:end_index],
                'residual_effect_2': match.residual_effect_2[start_index:end_index],
                "break_pt_won_2": match.break_pt_won_2[start_index:end_index]
            })

        data_mirror = {
            'unf_err_1': [2 - x for x in match.unf_err_1[start_index:end_index]],
            'double_fault_1': [2 - x for x in match.double_fault_1[start_index:end_index]],
            'ace_1': [2 - x for x in match.ace_1[start_index:end_index]],
            'net_pt_won_1': [2 - x for x in match.net_pt_won_1[start_index:end_index]],
            'score_advantage_1': [2 - x for x in match.score_advantage_1[start_index:end_index]],
            'residual_effect_1': [2 - x for x in match.residual_effect_1[start_index:end_index]],
            "break_pt_won_1": [2 - x for x in match.break_pt_won_1[start_index:end_index]],
        }

        if num_layer == 2:
            data_mirror.update({
                # Long-term memory variables
                'unf_err_2': [2 - x for x in match.unf_err_2[start_index:end_index]],
                'double_fault_2': [2 - x for x in match.double_fault_2[start_index:end_index]],
                'ace_2': [2 - x for x in match.ace_2[start_index:end_index]],
                'net_pt_won_2': [2 - x for x in match.net_pt_won_2[start_index:end_index]],
                'score_advantage_2': [2 - x for x in match.score_advantage_2[start_index:end_index]],
                'residual_effect_2': [2 - x for x in match.residual_effect_2[start_index:end_index]],
                "break_pt_won_2": [2 - x for x in match.break_pt_won_2[start_index:end_index]],
            })

        # Convert the data into a DataFrame
        match_df = pd.DataFrame(data)
        match_df_mirror = pd.DataFrame(data_mirror)

        # Append the DataFrame to the list of DataFrames
        dfs.append(match_df)
        dfs.append(match_df_mirror)

    # Concatenate all the DataFrames together
    final_df = pd.concat(dfs, ignore_index=True)
    return final_df


def train_model(num_layer=2, file_name=None):
    """
    Now we begin to build our model
    """

    # Define the structure of the Bayesian Network
    model_structure = [
        ('unf_err_1', 'self_efficacy_1'),
        ('double_fault_1', 'self_efficacy_1'),
        ("break_pt_won_1", "self_efficacy_1"),
        ('ace_1', 'perception_of_control_1'),
        ('net_pt_won_1', 'perception_of_control_1'),
        ('self_efficacy_1', 'perception_of_control_1'),
        ('self_efficacy_1', 'residual_effect_1'),
        ('perception_of_control_1', 'residual_effect_1'),
        ('score_advantage_1', 'residual_effect_1'),
    ]

    if num_layer == 2:
        model_structure += [
            ('unf_err_2', 'self_efficacy_2'),
            ('double_fault_2', 'self_efficacy_2'),
            ("break_pt_won_2", "self_efficacy_2"),
            ('ace_2', 'perception_of_control_2'),
            ('net_pt_won_2', 'perception_of_control_2'),
            ('self_efficacy_2', 'perception_of_control_2'),
            ('self_efficacy_2', 'residual_effect_2'),
            ('perception_of_control_2', 'residual_effect_2'),
            ('score_advantage_2', 'residual_effect_2'),
        ]

        model_structure += [
            ('perception_of_control_2', 'residual_effect_1'),
        ]

    latents = ["self_efficacy_1", "perception_of_control_1"]
    if num_layer == 2:
        latents += ["self_efficacy_2", "perception_of_control_2"]
    model = BayesianNetwork(model_structure, latents=latents)

    train_df = get_train_df()
    model.fit(train_df, estimator=ExpectationMaximization)
    if file_name:
        model.save(file_name)
    else:
        if num_layer == 2:
            model.save("Model.bif")
        else:
            model.save("SingleLayerModel.bif")


# print(sum(final_df["residual_effect_1"]) / len(final_df))
# exit()

def residual_plot(match_index, plot_type='both', predicted_weight=1., bias=0., scaling=10., index_scope=None):
    plt.figure(figsize=(10, 6))  # Set the figure size (optional)

    if index_scope is None:
        index_scope = (10, None)  # Default scope starts from 10 to the end if not specified

    # Adjust the start index based on the index_scope provided
    start_index = index_scope[0]
    end_index = index_scope[1] if index_scope[1] is not None else -10  # Adjust the end index or default to -10 from the end

    if plot_type in ('predicted', 'both'):
        # Plotting smoothed predicted effects
        model = BayesianNetwork.load("SingleLayerModel.bif")
        inference = VariableElimination(model)
        test_match = matches_list[match_index]

        predicted_effects = []
        for point_index in range(start_index, test_match.length + end_index):
            # Predict match time effect
            predicted_effect = ((predict_match_time(inference, test_match, point_index)[0] + bias) * 2 - 1) * scaling

            # Get actual residual effect for the point
            actual_residual_effect = test_match.residual_effect_continuous[point_index]

            # Calculate weighted sum of predicted effect and actual residual effect
            weighted_effect = predicted_weight * predicted_effect + (1 - predicted_weight) * actual_residual_effect

            predicted_effects.append(weighted_effect)

        Y = np.array(predicted_effects)
        X = np.arange(start_index, start_index + len(Y))

        Y_sg = savgol_filter(Y, 31, 3)  # Apply Savitzky-Golay filter for smoothing
        plt.plot(X, Y_sg, label='Predicted Effect', color='blue')

    if plot_type in ('residual', 'both'):
        # Plotting smoothed residual effect
        residual_effect = matches_list[match_index].residual_effect_continuous[start_index: test_match.length + end_index]
        X_res = np.array(range(start_index, start_index + len(residual_effect)))
        Y_res = np.array(residual_effect)
        Y_res_sg = savgol_filter(Y_res, 31, 3)  # Apply Savitzky-Golay filter for smoothing
        plt.plot(X_res, Y_res_sg, label='Actual Residual Effect', color='red')

    # Final plot adjustments
    # plt.title('Smoothed Effects Over Points')
    plt.xlabel('Point Number')
    plt.ylabel('Effect')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def predict_match_time(inference, given_match, point_index):
    t = point_index
    evidence_values = {
        "double_fault_1": str(given_match.double_fault_1[t]),
        "score_advantage_1": str(given_match.score_advantage_1[t]),
        "unf_err_1": str(given_match.unf_err_1[t]),
        "break_pt_won_1": str(given_match.break_pt_won_1[t]),
        "net_pt_won_1": str(given_match.net_pt_won_1[t]),
        "ace_1": str(given_match.ace_1[t]),
    }

    return inference.query(variables=['residual_effect_1'], evidence=evidence_values).values


if __name__ == "__main__":
    train_model(num_layer=1, file_name="WomanModel.bif")
    # residual_plot(-1, predicted_weight=0.5, bias=0.02, scaling=25, index_scope=(40, -10))
