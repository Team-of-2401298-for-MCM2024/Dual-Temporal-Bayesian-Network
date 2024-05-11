import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path

# Load the CSV file
root_path = Path(__file__).parents[1]
file_path = root_path / 'data' /'Wimbledon_featured_matches.csv'
df = pd.read_csv(file_path)

# Extracting unique player pairs and their match IDs
match_pairs = df[['match_id', 'player1', 'player2']].drop_duplicates()

# Creating a graph
G = nx.Graph()

# Adding nodes and edges
for _, row in match_pairs.iterrows():
    G.add_node(row['player1'])
    G.add_node(row['player2'])
    G.add_edge(row['player1'], row['player2'], match_id=row['match_id'])

# Drawing the graph
plt.figure(figsize=(12, 8))
nx.draw(G, with_labels=True, node_color='lightblue', node_size=200, edge_color='gray')
plt.title("Graph of Players and Matches")
plt.show()
