import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("DATASET/coding_learning_dataset.csv")

# ==========================================
# REMOVE MISSING VALUES
# ==========================================

data = data.dropna()

# ==========================================
# REMOVE DUPLICATE VALUES
# ==========================================

data = data.drop_duplicates()

# ==========================================
# ADD NEW COLUMNS
# ==========================================

# Random Time Taken
data['time_taken'] = np.random.randint(
    30,
    600,
    size=len(data)
)

# Random Errors Count
data['errors_count'] = np.random.randint(
    0,
    10,
    size=len(data)
)

# ==========================================
# ENCODE CATEGORICAL COLUMNS
# ==========================================

topic_encoder = LabelEncoder()

data['topic_encoded'] = topic_encoder.fit_transform(
    data['topic']
)

difficulty_encoder = LabelEncoder()

data['difficulty_encoded'] = difficulty_encoder.fit_transform(
    data['difficulty']
)

# ==========================================
# CREATE LEARNING PATTERN
# ==========================================

def learning_pattern(score):

    if score >= 0.75:
        return "Fast Learner"

    elif score >= 0.50:
        return "Average Learner"

    else:
        return "Needs Improvement"


data['learning_pattern'] = data['success_rate'].apply(
    learning_pattern
)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

scaled_features = scaler.fit_transform(
    data[['attempts', 'time_taken', 'errors_count']]
)

# ==========================================
# SAVE CLEANED DATASET
# ==========================================

data.to_csv(
    "processed_data.csv",
    index=False
)

# ==========================================
# OUTPUT
# ==========================================

print("Preprocessing Completed Successfully")

print(data.head())

print(data['learning_pattern'].value_counts())