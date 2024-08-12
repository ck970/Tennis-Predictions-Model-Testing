import pandas as pd
import numpy as np

df = pd.read_csv('match_data/atp_matches_2024.csv')
df = df.drop(['tourney_name', 'draw_size', 'winner_entry', 'winner_name', 'winner_ioc', 'loser_entry', 'loser_name', 'loser_ioc', 'minutes', 'score', 'winner_rank_points', 'loser_rank_points', 'w_SvGms', 'l_SvGms'], axis=1)

# Rename columns to allow for randomization of player 1 and player 2
df.rename(columns=lambda x: x if x == 'winner_id' or x == 'loser_id' else x.replace('winner', 'p1').replace('loser', 'p2').replace('w_', 'p1_').replace('l_', 'p2_'), inplace=True)

# Randomize which player is player 1 and which player is player 2 to avoid bias
df['p1'] = np.where(np.random.rand(len(df)) > 0.5, df['winner_id'], df['loser_id'])
df['p2'] = np.where(df['p1'] == df['winner_id'], df['loser_id'], df['winner_id'])

# Delete the loser_id column
df = df.drop(['loser_id'], axis=1)

# Create aggregate statistics for each player based on their last 12 matches
# Necessary stats: ace% (#aces/#svpt), df% (#df/#svpt), avg_svpt, 1stIn%, (#1stIn/#svpt)
# 1stWon% (#1stWon/#1stIn), 2ndWon% (#2ndWon/(#svpt - #1stIn)), bpSaved% (#bpSaved/#bpFaced), avg_bpFaced

# # Calculate ace%, df%, avg_svpt, 1stIn%, 1stWon%, 2ndWon%, bpSaved%, avg_bpFaced
# df['p1_ace_pct'] = df['p1_ace'] / df['p1_svpt']
# df['p1_df_pct'] = df['p1_df'] / df['p1_svpt']
# df['p1_avg_svpt'] = df['p1_svpt']
# df['p1_1stIn_pct'] = df['p1_1stIn'] / df['p1_svpt']
# df['p1_1stWon_pct'] = df['p1_1stWon'] / df['p1_1stIn']
# df['p1_2ndWon_pct'] = df['p1_2ndWon'] / (df['p1_svpt'] - df['p1_1stIn'])
# df['p1_bpSaved_pct'] = df['p1_bpSaved'] / df['p1_bpFaced']
# df['p1_avg_bpFaced'] = df['p1_bpFaced']

# df['p2_ace_pct'] = df['p2_ace'] / df['p2_svpt']
# df['p2_df_pct'] = df['p2_df'] / df['p2_svpt']
# df['p2_avg_svpt'] = df['p2_svpt']
# df['p2_1stIn_pct'] = df['p2_1stIn'] / df['p2_svpt']
# df['p2_1stWon_pct'] = df['p2_1stWon'] / df['p2_1stIn']
# df['p2_2ndWon_pct'] = df['p2_2ndWon'] / (df['p2_svpt'] - df['p2_1stIn'])
# df['p2_bpSaved_pct'] = df['p2_bpSaved'] / df['p2_bpFaced']
# df['p2_avg_bpFaced'] = df['p2_bpFaced']

# # Delete unnecessary columns
# df = df.drop(['p1_ace', 'p1_df', 'p1_svpt', 'p1_1stIn', 'p1_1stWon', 'p1_2ndWon', 'p1_bpSaved', 'p1_bpFaced', 'p2_ace', 'p2_df', 'p2_svpt', 'p2_1stIn', 'p2_1stWon', 'p2_2ndWon', 'p2_bpSaved', 'p2_bpFaced'], axis=1)

# One-hot encode remaining categorical variables
df = pd.get_dummies(df, columns=['surface', 'tourney_level', 'round', 'p1_hand', 'p2_hand'], dtype=int)

# Replace NaN values with 0
df = df.fillna(0)

# Make a separate dataframe that stores the stats for each player and has a boolean column indicating whether the player won


# Save the new dataframe to a new csv file
df.to_csv('match_data/atp_matches_2024_cleaned.csv', index=False)

# Print the first 5 rows of the new dataframe
print(df.head())

# NEXT STEPS:
# 1. Create aggregate statistics for each player based on last 12 matches
# 2. Plan out the neural network architecture (scikit-learn implementation)
# 3. Train the neural network
# 4. Evaluate the neural network
# 5. Make predictions on future matches