import pandas as pd

# Load dataset
data = pd.read_csv("processed_data.csv")

# Recommendation Function
def recommend_topic(topic, success_rate):

    if success_rate < 0.5:
        return f"Practice more in {topic}"

    elif success_rate < 0.75:
        return f"Revise concepts of {topic}"

    else:
        return f"You are performing well in {topic}"

# Apply recommendation
data['recommendation'] = data.apply(
    lambda row: recommend_topic(
        row['topic'],
        row['success_rate']
    ),
    axis=1
)

# Save updated dataset
data.to_csv("processed_data.csv", index=False)

print("Recommendation System Added Successfully")
print(data[['topic', 'success_rate', 'recommendation']].head())