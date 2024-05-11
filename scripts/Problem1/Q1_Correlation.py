import pandas as pd
# -------------------**Review of the Problem Statement and Data Dictionary**
# Load the data dictionary to understand the variables provided for analysis
from pathlib import Path
root_path = Path(__file__).parents[2]
data_dictionary_path = root_path / 'data' / 'data_dictionary.csv'
data_dictionary = pd.read_csv(data_dictionary_path)

# -------------------**Initial Data Exploration**:
# Load the Wimbledon featured matches dataset
matches_data_path = root_path / 'data' / 'Wimbledon_featured_matches.csv'
matches_data = pd.read_csv(matches_data_path)

# Display basic information about the dataset
info = matches_data.info()

# Display the first few rows to get a sense of the data
sample_data = matches_data.head()

# Check for missing values in the dataset
missing_values = matches_data.isnull().sum()

# Impute missing values for numerical columns with median and categorical columns with mode
# For simplicity and since the missing values are relatively small compared to the dataset size, we use mode for categorical

# Identify numerical and categorical columns with missing values
num_col_missing = ['speed_mph']
cat_col_missing = ['serve_width', 'serve_depth', 'return_depth']

# Impute missing numerical values with median
for col in num_col_missing:
    matches_data[col].fillna(matches_data[col].median(), inplace=True)

# Impute missing categorical values with mode
for col in cat_col_missing:
    matches_data[col].fillna(matches_data[col].mode()[0], inplace=True)






# -------------------**Feature Importance using Random Forest**
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Preparing the dataset for Random Forest
# Exclude non-predictive columns and target variable from features
X = matches_data.drop(columns=['match_id', 'player1', 'player2', 'elapsed_time', 'point_victor'])
# Target variable
y = matches_data['point_victor']

# Encode categorical variables
le = LabelEncoder()
X['p1_score'] = le.fit_transform(X['p1_score'])
X['p2_score'] = le.fit_transform(X['p2_score'])
X['winner_shot_type'] = le.fit_transform(X['winner_shot_type'])
X['serve_width'] = le.fit_transform(X['serve_width'])
X['serve_depth'] = le.fit_transform(X['serve_depth'])
X['return_depth'] = le.fit_transform(X['return_depth'])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on the training data
rf.fit(X_train, y_train)

# Get feature importances
feature_importances = rf.feature_importances_

# Creating a DataFrame to display feature importances
features_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
features_df = features_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)







import seaborn as sns
import matplotlib.pyplot as plt

# Select top features for correlation analysis
top_features = features_df['Feature'].head(10)

# Create a new DataFrame with the top features for correlation matrix
correlation_data = matches_data[top_features.tolist() + ['point_victor']]

# Compute the correlation matrix
corr = correlation_data.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=False, cmap='Greens', vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

# plt.title('Correlation Heatmap of Top Features')
plt.show()