import pandas as pd
from pathlib import Path

# Load the CSV file
root_path = Path(__file__).parents[1]
# Load the rankings data
rankings_path = rankings_path = root_path / 'data' / 'ranking.csv'
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


# Example usage
player1 = "Carlos Alcaraz"
player2 = "Daniil Medvedev"
rank1, rank2 = get_relative_ranks(player1, player2)
print(f"Relative ranks: {player1} is {rank1}, {player2} is {rank2}")