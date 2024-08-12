import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the cleaned dataset
df = pd.read_csv('match_data/atp_matches_2024_cleaned.csv')

# Use lagged statistics to create aggregate statistics for each player based on their last 12 matches

