import pandas as pd
import numpy as np

df = pd.read_csv('match_data/atp_matches_2024.csv')
df = df.drop(['tourney_name', 'draw_size', 'winner_entry', 'winner_name', 'winner_ioc', 'loser_entry', 'loser_name', 'loser_ioc', 'minutes', 'winner_rank_points', 'loser_rank_points'], axis=1)

df.rename(columns=lambda x: x.replace('winner', 'p1').replace('loser', 'p2').replace('w_', 'p1_').replace('l_', 'p2_'), inplace=True)

print(df.head())

# NEXT STEPS:
# 1. Save the original winner_id and loser_id columns to a new dataframe to use as a lookup table
# 2. Randomize which player is player 1 and which player is player 2 to avoid bias
# 3. Create aggregate statistics for each player based on last 12 matches