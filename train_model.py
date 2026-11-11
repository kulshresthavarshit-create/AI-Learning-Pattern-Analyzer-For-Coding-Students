# train_model.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ===============================
# 1. LOAD DATA
# ===============================
data = pd.read_csv("dataset/coding_learning_dataset.csv")

print("Dataset Loaded!")

# ===============================
# 2. HANDLE CATEGORICAL DATA
# ===============================
le = LabelEncoder()

data['topic'] = le.fit_transform(data['topic'])
data['difficulty'] = le.fit_transform(data['difficulty'])

# ===============================
# 3. FEATURE SELECTION
# ===============================
X = data[['attempts', 'success_rate', 'time_taken', 'errors_count', 'topic', 'difficulty']]

# ===============================
# 4. SCALING
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 5. CLUSTERING (KMEANS)
# ===============================
kmeans = KMeans(n_clusters=3, random_state=42)
data['learning_pattern'] = kmeans.fit_predict(X_scaled)

print("Clustering Done!")

# ==========================================
# 6. LINEAR REGRESSION
# ==========================================

from sklearn.linear_model import LinearRegression
import joblib

# Features
X_lr = data[['attempts', 'time_taken', 'errors_count']]

# Target
y_lr = data['success_rate']

# Create model
lr_model = LinearRegression()

# Train model
lr_model.fit(X_lr, y_lr)

# Predict success rate
joblib.dump(lr_model,"success_rate_model.pkl")


print("Success Rate Model Added Successfully")

# ===============================
# 7. RANDOM FOREST CLASSIFIER
# ===============================

# Predict learning pattern
y = data['learning_pattern']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Accuracy
accuracy = rf.score(X_test, y_test)

print("Random Forest Accuracy:", accuracy)

# ===============================
# 7. SAVE OUTPUT
# ===============================
data.to_csv("processed_data.csv", index=False)

print("Final data saved!")

import joblib

#Save trained  odel
joblib.dump(rf,"learning_pattern_model.pkl")
print("Model Saved Successfully")