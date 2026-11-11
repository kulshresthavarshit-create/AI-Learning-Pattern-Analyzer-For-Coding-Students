# visualization.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("dataset/processed_data.csv")
data = pd.read_csv("processed_data.csv")

# Create static folder automatically
if not os.path.exists("static"):
    os.makedirs("static")
# Select features
X = data[['attempts', 'success_rate', 'time_taken', 'errors_count']]

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
data['learning_pattern'] = kmeans.fit_predict(X)
# Topic Names
topic_names = {
    0: "Python Basics",
    1: "Loops",
    2: "Functions",
    3: "Data Structures",
    4: "Algorithms"
}

# Convert topic numbers to names
data['topic_name'] = data['topic'].map(topic_names)

pattern_labels= {
  0:"Needs Improvement",
  1:"Average Learner",
  2:"Fast Learner"  
}
data['learning_pattern_name'] = data['learning_pattern'].map(pattern_labels)
 #Create graph
plt.figure(figsize=(6,4))
print(data[['learning_pattern','learning_pattern_name']].head(10))
sns.countplot(
   x='learning_pattern_name',
   data=data
)
plt.title("Learning Pattern Distribution")
plt.xlabel("Learning Pattern")
plt.ylabel("Count")

plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig("static/learning_pattern_chart.png")
plt.close()
print("Columns in dataset:", data.columns)
# GRAPH 1
plt.figure()
plt.scatter(data['attempts'], data['success_rate'])
plt.title("Attempts vs Success Rate")

# GRAPH 2
plt.figure()
plt.scatter(data['time_taken'], data['success_rate'])
plt.title("Time vs Success Rate")

# GRAPH 3 (if exists)
if 'learning_pattern' in data.columns:
    plt.figure()
    plt.scatter(data['attempts'], data['time_taken'], c=data['learning_pattern'])
    plt.title("Learning Patterns")

# GRAPH 4
plt.figure()
sns.histplot(data['success_rate'])
plt.title("Success Rate Distribution")

# GRAPH 5
plt.figure()
plt.scatter(data['errors_count'], data['success_rate'])
plt.title("Errors vs Success Rate")
topic_mapping = {
  '0':"Arrays",
  '1':"Strings",
  '2':"Graphs",
  '3':"Recursion",
  '4':"Dynamic Programming"
}
data['topic']=data['topic'].map(topic_mapping)
print(data['topic'].dtype)
print(data['topic'].unique())

# GRAPH 6

plt.figure(figsize=(10,5))
ax= sns.barplot(x='topic',y='success_rate',data=data)
ax.set_xticklabels([
  'Arrays',
  'Strings',
  'Graphs',
  'Recursion',
  'Dynamic Programming'
])
plt.title("Weak Topics Graph")

plt.xticks(rotation=45)
plt.savefig("static/ weak_topics_chart.png")
plt.close()

# GRAPH 7
plt.figure()

numeric_data = data.select_dtypes(include='number')

if not numeric_data.empty:
    sns.heatmap(numeric_data.corr(), annot=True)
    plt.title("Correlation Heatmap")
else:
    print("No numeric data for heatmap")
# Learning Pattern Graph
pattern_count = data['learning_pattern'].value_counts()

plt.figure(figsize=(6,5))

pattern_count.plot(kind='bar')

plt.title("Learning Pattern Distribution")

plt.savefig("static/learning_pattern_chart.png")

plt.close()


# Topic Success Graph
topic_avg = data.groupby('topic_name')['success_rate'].mean()

plt.figure(figsize=(6,5))

topic_avg.plot(kind='bar')

plt.title("Topic Success Analysis")

plt.savefig("static/topic_success_chart.png")

plt.close()
# Learning Pattern Graph
pattern_count = data['learning_pattern'].value_counts()

plt.figure(figsize=(6,5))

pattern_count.plot(kind='bar')

plt.title("Learning Pattern Distribution")

plt.savefig("static/learning_pattern_chart.png")

plt.close()


# Topic Success Graph
topic_avg = data.groupby('topic_name')['success_rate'].mean()

plt.figure(figsize=(6,5))

topic_avg.plot(kind='bar')

plt.title("Topic Success Analysis")

plt.savefig("static/topic_success_chart.png")

plt.close()
# Learning Pattern Graph
pattern_count = data['learning_pattern'].value_counts()

plt.figure(figsize=(6,5))

pattern_count.plot(kind='bar')

plt.title("Learning Pattern Distribution")

plt.savefig("static/learning_pattern_chart.png")

plt.close()


# Topic Success Graph
topic_avg = data.groupby('topic_name')['success_rate'].mean()

plt.figure(figsize=(6,5))

topic_avg.plot(kind='bar')

plt.title("Topic Success Analysis")

plt.savefig("static/topic_success_chart.png")

plt.close()

print("All graphs generated successfully")